"""0.16.5 correctness pass 3: provider adapters' less-travelled paths."""

import json
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletionChunk

from llm_gateway import GatewayConfig, InvalidRequestError
from llm_gateway.errors import ErrorKind, classify, is_model_scoped
from llm_gateway.providers import vertex
from llm_gateway.providers._anthropic_translate import json_object_messages
from llm_gateway.providers.base import (
    StreamDelta,
    ToolIndexNormalizer,
    parse_openai_style_chunk,
)


def _chunk(**fields) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", **fields}
    )


def _tc_chunk(*calls) -> ChatCompletionChunk:
    return _chunk(choices=[{"index": 0, "delta": {"tool_calls": list(calls)}}])


def _call(index, id=None, name=None, args=""):
    out: dict = {"index": index, "function": {"arguments": args}}
    if id:
        out["id"] = id
        out["type"] = "function"
    if name:
        out["function"]["name"] = name
    return out


def _aggregate(deltas: list[StreamDelta]) -> dict[int, dict]:
    calls: dict[int, dict] = {}
    for d in deltas:
        for tc in d.tool_call_deltas or []:
            entry = calls.setdefault(tc["index"], {"id": None, "name": "", "args": ""})
            entry["id"] = tc.get("id") or entry["id"]
            entry["name"] += tc["function"].get("name", "")
            entry["args"] += tc["function"].get("arguments", "")
    return calls


# 1. Tool calls that share an index (Gemini's OpenAI endpoint) -------------------


def test_distinct_tool_calls_sharing_index_zero_are_kept_apart():
    norm = ToolIndexNormalizer()
    chunks = [
        _tc_chunk(_call(0, "call_a", "get_weather", '{"city":')),
        _tc_chunk(_call(0, None, None, '"Paris"}')),
        _tc_chunk(_call(0, "call_b", "get_time", '{"tz":"UTC"}')),
    ]
    deltas = [norm.apply(parse_openai_style_chunk(c, "m")) for c in chunks]
    calls = _aggregate(deltas)
    assert len(calls) == 2
    assert json.loads(calls[0]["args"]) == {"city": "Paris"}
    assert calls[1]["name"] == "get_time" and calls[1]["id"] == "call_b"


def test_interleaved_well_formed_indexes_are_untouched():
    norm = ToolIndexNormalizer()
    chunks = [
        _tc_chunk(_call(0, "a", "f", '{"x":')),
        _tc_chunk(_call(1, "b", "g", '{"y":')),
        _tc_chunk(_call(0, None, None, "1}")),
        _tc_chunk(_call(1, None, None, "2}")),
    ]
    calls = _aggregate([norm.apply(parse_openai_style_chunk(c, "m")) for c in chunks])
    assert json.loads(calls[0]["args"]) == {"x": 1}
    assert json.loads(calls[1]["args"]) == {"y": 2}


def test_a_missing_index_is_treated_as_a_new_call_when_it_carries_an_id():
    norm = ToolIndexNormalizer()
    first = StreamDelta(tool_call_deltas=[_call(0, "a", "f", "{")])
    no_index = {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}}
    second = StreamDelta(tool_call_deltas=[no_index])
    tail = StreamDelta(tool_call_deltas=[{"function": {"arguments": ""}}])
    calls = _aggregate([norm.apply(d) for d in (first, second, tail)])
    assert set(calls) == {0, 1}


# 2. Provider error bodies with a numeric code (OpenRouter) -----------------------


def _api_error(body: dict) -> openai.APIError:
    return openai.APIError("upstream", httpx.Request("POST", "https://x"), body=body)


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        (502, ErrorKind.TRANSIENT),
        (429, ErrorKind.RATE_LIMITED),
        (401, ErrorKind.AUTH),
        (400, ErrorKind.INVALID_REQUEST),
    ],
)
def test_in_stream_error_body_with_numeric_code_is_classified(code, kind):
    body = {
        "error": {
            "code": code,
            "message": "Provider returned error",
            "metadata": {"provider_name": "DeepInfra"},
        }
    }
    assert classify(_api_error(body)) is kind


# 3. Model-not-found spellings that feed the per-model breaker --------------------


def _status_error(status: int, body: dict, message: str = "err") -> openai.APIStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", "https://x"))
    return openai.APIStatusError(message, response=response, body=body)


def test_mistral_invalid_model_400_is_model_scoped_missing():
    err = _status_error(
        400,
        {
            "object": "error",
            "message": "Invalid model: foo",
            "type": "invalid_model",
            "code": "1500",
        },
        "Invalid model: foo",
    )
    assert is_model_scoped(err) == "missing"


def test_openrouter_not_a_valid_model_id_is_model_scoped_missing():
    err = _status_error(
        400,
        {"error": {"message": "foo/bar is not a valid model ID", "code": 400}},
        "foo/bar is not a valid model ID",
    )
    assert is_model_scoped(err) == "missing"


def test_an_ordinary_400_is_not_model_scoped():
    err = _status_error(
        400, {"error": {"message": "max_tokens too large", "code": 400}}, "max_tokens too large"
    )
    assert is_model_scoped(err) is None


# 4. Vertex Claude takes base64 images only ---------------------------------------


def _url_image_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ],
        }
    ]


async def test_vertex_claude_rejects_url_images_before_any_call():
    config = GatewayConfig(vertex_project_id="my-project", vertex_model="claude-haiku-4-5@20251001")
    with patch.object(vertex, "_client", MagicMock()) as client:
        with pytest.raises(InvalidRequestError, match="base64"):
            await vertex.chat(config, _url_image_messages(), None, 100)
        with pytest.raises(InvalidRequestError, match="base64"):
            async for _ in vertex.stream_chat(config, _url_image_messages(), None, 100):
                pass
        client.assert_not_called()


async def test_vertex_claude_accepts_data_uri_images():
    vertex.check_images_are_base64(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ],
            }
        ]
    )


# 5. json_object on Anthropic-family providers is no longer silently ignored ------


def test_json_object_response_format_adds_a_system_instruction():
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
    out = json_object_messages(messages, {"type": "json_object"})
    assert out[-1]["role"] == "system" and "JSON" in out[-1]["content"]
    assert messages[0] in out and len(messages) == 2  # input untouched


def test_other_response_formats_leave_messages_alone():
    messages = [{"role": "user", "content": "hi"}]
    assert json_object_messages(messages, None) is messages
    assert json_object_messages(messages, {"type": "text"}) is messages
    assert json_object_messages(messages, {"type": "json_schema", "json_schema": {}}) is messages
