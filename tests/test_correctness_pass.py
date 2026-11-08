"""0.13.3 correctness pass: usage on a final choice chunk, tool-call indexes,
malformed tool arguments, lenient structured-output parsing, list system
content."""

from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from llm_gateway import GatewayConfig
from llm_gateway.errors import ErrorKind, InvalidOutputError, classify
from llm_gateway.providers import anthropic
from llm_gateway.providers._anthropic_translate import to_anthropic_messages
from llm_gateway.providers.base import parse_openai_style_chunk, parse_openai_style_response
from llm_gateway.structured import OutputSchema
from tests.test_anthropic_provider import _event, _FakeAnthropicStream, _tool_use_block


def _chunk(**fields) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            **fields,
        }
    )


# 1. Usage sent on the final chunk *with* choices (Mistral and other hosts) ---


def test_usage_on_a_chunk_that_also_has_choices_is_kept():
    chunk = _chunk(
        choices=[{"index": 0, "delta": {"content": "!"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    )
    delta = parse_openai_style_chunk(chunk, "m")
    assert delta.content == "!"
    assert delta.finish_reason == "stop"
    assert delta.usage == (7, 3)
    assert delta.usage_details.input_tokens == 7


def test_chunk_without_usage_has_none():
    chunk = _chunk(choices=[{"index": 0, "delta": {"content": "a"}, "finish_reason": None}])
    assert parse_openai_style_chunk(chunk, "m").usage is None


# 2. Anthropic tool-call stream indexes are tool ordinals, not block indexes --


async def test_anthropic_stream_tool_indexes_start_at_zero_after_text():
    events = [
        _event("message_start", message=MagicMock(usage=MagicMock(input_tokens=1))),
        _event("content_block_start", index=0, content_block=MagicMock(type="text")),
        _event("content_block_delta", index=0, delta=MagicMock(type="text_delta", text="ok")),
        _event("content_block_start", index=1, content_block=_tool_use_block("t1", "a")),
        _event(
            "content_block_delta",
            index=1,
            delta=MagicMock(type="input_json_delta", partial_json="{}"),
        ),
        _event("content_block_start", index=2, content_block=_tool_use_block("t2", "b")),
        _event(
            "content_block_delta",
            index=2,
            delta=MagicMock(type="input_json_delta", partial_json="{}"),
        ),
        _event(
            "message_delta",
            delta=MagicMock(stop_reason="tool_use"),
            usage=MagicMock(output_tokens=2),
        ),
    ]
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        deltas = [
            d
            async for d in anthropic.stream_chat(
                GatewayConfig(anthropic_api_key="k"),
                [{"role": "user", "content": "x"}],
                tools=[{"type": "function", "function": {"name": "a"}}],
                max_tokens=10,
            )
        ]
    tool = [d.tool_call_deltas[0] for d in deltas if d.tool_call_deltas]
    assert [(t["index"], t.get("id")) for t in tool] == [
        (0, "t1"),
        (0, None),
        (1, "t2"),
        (1, None),
    ]


# 3. Malformed tool arguments are an invalid output, not an UNKNOWN error -----


def _completion_with_args(arguments: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": arguments},
                            }
                        ],
                    },
                }
            ],
        }
    )


@pytest.mark.parametrize("arguments", ['{"a": ', "[1, 2]", '"s"'])
def test_bad_tool_arguments_raise_invalid_output(arguments):
    with pytest.raises(InvalidOutputError) as excinfo:
        parse_openai_style_response(_completion_with_args(arguments), "m")
    assert classify(excinfo.value) is ErrorKind.INVALID_OUTPUT


def test_good_and_empty_tool_arguments_still_parse():
    result = parse_openai_style_response(_completion_with_args('{"a": 1}'), "m")
    assert result.tool_calls[0].arguments == {"a": 1}
    assert parse_openai_style_response(_completion_with_args(""), "m").tool_calls[0].arguments == {}


# 4. Structured output wrapped in prose / fences ------------------------------


class _Answer(BaseModel):
    answer: int


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": 42}',
        '```json\n{"answer": 42}\n```',
        'Here is the JSON:\n```json\n{"answer": 42}\n```\nHope that helps!',
        'Sure! {"answer": 42}',
        '{"answer": 42}\n\nLet me know if you need anything else.',
        '  ```\n{"answer": 42}\n```  ',
    ],
)
def test_parse_finds_the_json_object_around_prose_and_fences(text):
    assert OutputSchema.from_spec(_Answer).parse(text) == _Answer(answer=42)


@pytest.mark.parametrize(
    "text", ['{"answer": 4', "no json here", '{"answer": "x"} {"answer": 1', ""]
)
def test_parse_still_rejects_unusable_text(text):
    with pytest.raises(InvalidOutputError):
        OutputSchema.from_spec(_Answer).parse(text)


def test_parse_array_schema_with_trailing_text():
    schema = OutputSchema.from_spec({"type": "array"})
    assert schema.parse("Result: [1, 2, 3] done") == [1, 2, 3]


# 5. System messages with content parts --------------------------------------


def test_system_message_with_text_parts_is_joined():
    system, msgs = to_anthropic_messages(
        [
            {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
            {"role": "system", "content": "Be kind."},
            {"role": "user", "content": "hi"},
        ]
    )
    assert system == "Be brief.\nBe kind."
    assert msgs == [{"role": "user", "content": "hi"}]
