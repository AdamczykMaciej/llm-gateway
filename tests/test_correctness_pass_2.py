"""0.15.4 correctness pass 2: finish reasons, content-filter empties, no-choice
replies, inline <think> text, reasoning-model parameters, Anthropic sampling
limits and content parts, OpenRouter cache writes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llm_gateway import EmptyCompletionError, GatewayConfig, InvalidRequestError
from llm_gateway.errors import ErrorKind, InvalidOutputError, classify
from llm_gateway.pricing import ModelPrice, cost_usd
from llm_gateway.providers import anthropic, azure, openai
from llm_gateway.providers._anthropic_translate import (
    from_anthropic_response,
    to_anthropic_messages,
)
from llm_gateway.providers.base import (
    StreamDelta,
    guard_empty_stream,
    normalize_finish_reason,
    parse_openai_style_chunk,
    parse_openai_style_response,
    provider_result_from_openai_style,
    strip_think,
    usage_from_openai_style,
)
from tests.test_anthropic_provider import (
    _config,
    _event,
    _FakeAnthropicStream,
    _text_response,
    _tool_use_block,
)


def _completion(content="hi", finish_reason="stop", usage=None, choices=None) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": choices
            if choices is not None
            else [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            **({"usage": usage} if usage else {}),
        }
    )


def _chunk(**fields) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", **fields}
    )


# 1. One finish-reason vocabulary, chat and stream -----------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "stop"),
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        (None, "stop"),
        ("length", "length"),
        ("max_tokens", "length"),
        ("model_length", "length"),
        ("content_filter", "content_filter"),
        ("SAFETY", "content_filter"),
        ("refusal", "content_filter"),
        ("tool_calls", "tool_calls"),
        ("tool_use", "tool_calls"),
    ],
)
def test_normalize_finish_reason(raw, expected):
    assert normalize_finish_reason(raw) == expected


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_openai_chat_keeps_truncation_and_filter_reasons(reason):
    result = parse_openai_style_response(_completion("partial", reason), "m")
    assert result.finish_reason == reason


def test_openai_chat_stop_stays_stop():
    assert parse_openai_style_response(_completion("ok"), "m").finish_reason == "stop"


def test_openai_stream_finish_reason_is_normalized():
    chunk = _chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "length"}])
    assert parse_openai_style_chunk(chunk, "m").finish_reason == "length"
    chunk = _chunk(choices=[{"index": 0, "delta": {"content": "a"}, "finish_reason": None}])
    assert parse_openai_style_chunk(chunk, "m").finish_reason is None


async def _anthropic_chat(response, **kwargs):
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        return await anthropic.chat(
            _config(), [{"role": "user", "content": "hi"}], None, 100, **kwargs
        )


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [("max_tokens", "length"), ("end_turn", "stop"), ("stop_sequence", "stop")],
)
async def test_anthropic_chat_finish_reason(stop_reason, expected):
    response = _text_response("partial")
    response.stop_reason = stop_reason
    assert (await _anthropic_chat(response)).finish_reason == expected


async def test_anthropic_stream_finish_reason_max_tokens_is_length():
    events = [
        _event("message_start", message=MagicMock(usage=MagicMock(input_tokens=1))),
        _event("content_block_start", index=0, content_block=MagicMock(type="text")),
        _event("content_block_delta", index=0, delta=MagicMock(type="text_delta", text="cut")),
        _event(
            "message_delta",
            delta=MagicMock(stop_reason="max_tokens"),
            usage=MagicMock(output_tokens=2),
        ),
    ]
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        deltas = [
            d
            async for d in anthropic.stream_chat(
                _config(), [{"role": "user", "content": "hi"}], None, 5
            )
        ]
    assert deltas[-1].finish_reason == "length"


def test_anthropic_tool_call_cut_off_by_max_tokens_is_invalid_output():
    block = _tool_use_block("t1", "f")
    block.input = {}
    resp = SimpleNamespace(
        content=[block], usage=MagicMock(input_tokens=1, output_tokens=1), stop_reason="max_tokens"
    )
    with pytest.raises(InvalidOutputError) as excinfo:
        from_anthropic_response(resp, "m")
    assert classify(excinfo.value) is ErrorKind.INVALID_OUTPUT


# 2. A filtered/refused empty reply is the caller's content, not provider health


@pytest.mark.parametrize("reason", ["content_filter", "refusal", "SAFETY", "no_choices"])
def test_empty_reply_from_a_filter_does_not_count_for_the_breaker(reason):
    exc = EmptyCompletionError(provider="p", model="m", finish_reason=reason, reasoning_tokens=None)
    assert classify(exc) is ErrorKind.INVALID_OUTPUT


@pytest.mark.parametrize("reason", ["length", "stop", None])
def test_empty_reply_from_budget_still_counts(reason):
    exc = EmptyCompletionError(provider="p", model="m", finish_reason=reason, reasoning_tokens=5)
    assert classify(exc) is ErrorKind.EMPTY_RESPONSE


# 3. A 200 with no choices (Gemini blocks the prompt) is an empty completion ---


def test_no_choices_is_empty_completion_not_index_error():
    resp = _completion(choices=[])
    with pytest.raises(EmptyCompletionError) as excinfo:
        parse_openai_style_response(resp, "m", provider="gemini")
    assert classify(excinfo.value) is ErrorKind.INVALID_OUTPUT
    with pytest.raises(EmptyCompletionError):
        provider_result_from_openai_style(resp, "m", provider="gemini")


# 4. Inline <think> text never reaches content --------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>hmm</think>\n\nAnswer", "Answer"),
        ("  <think>a\nb</think>Answer", "Answer"),
        ("<think>never closed", ""),
        ("Answer with <think> inside", "Answer with <think> inside"),
        (None, None),
    ],
)
def test_strip_think(text, expected):
    assert strip_think(text) == expected


def test_think_block_is_stripped_from_chat_and_plain_results():
    text = "<think>reasoning</think>\nFinal."
    assert parse_openai_style_response(_completion(text), "m").content == "Final."
    assert provider_result_from_openai_style(_completion(text), "m").text == "Final."


def test_think_only_reply_is_empty():
    with pytest.raises(EmptyCompletionError):
        parse_openai_style_response(_completion("<think>x</think>", "length"), "m")


async def _drain(pieces, **kwargs):
    async def source():
        for piece in pieces:
            yield piece

    return [
        d
        async for d in guard_empty_stream(
            source(), provider="p", model="m", strip_think=True, **kwargs
        )
    ]


@pytest.mark.parametrize(
    "pieces",
    [
        ["<think>reas", "oning</th", "ink>\n\nAn", "swer"],
        ["<", "think>", "x", "</think>", "\n", "Answer"],
        ["<think>x</think>Answer"],
    ],
)
async def test_stream_strips_think_in_any_chunking(pieces):
    deltas = await _drain([StreamDelta(content=p) for p in pieces])
    assert "".join(d.content or "" for d in deltas) == "Answer"


async def test_stream_without_think_passes_text_through_unchanged():
    deltas = await _drain([StreamDelta(content=p) for p in ["<th", "e end ", "of it"]])
    assert "".join(d.content or "" for d in deltas) == "<the end of it"


async def test_think_only_stream_is_empty_so_it_can_fail_over():
    with pytest.raises(EmptyCompletionError):
        await _drain([StreamDelta(content="<think>only"), StreamDelta(finish_reason="length")])


# 5. Reasoning-model parameters -----------------------------------------------


async def _openai_kwargs(model, sampling=None):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_completion("ok"))
    with patch.object(openai, "_client", return_value=client):
        await openai.chat(
            GatewayConfig(openai_api_key="k"),
            [{"role": "user", "content": "hi"}],
            None,
            64,
            model=model,
            sampling=sampling,
        )
        await openai.call(GatewayConfig(openai_api_key="k"), "s", "p", 64, model=model)
    return [c.kwargs for c in client.chat.completions.create.call_args_list]


@pytest.mark.parametrize("model", ["o3", "o4-mini", "gpt-5", "gpt-5-mini", "gpt-5.1"])
async def test_openai_reasoning_models_get_completion_tokens_and_default_sampling(model):
    for kwargs in await _openai_kwargs(
        model, {"temperature": 0.2, "top_p": 0.5, "stop": ["x"], "seed": 3}
    ):
        assert kwargs["max_completion_tokens"] == 64
        assert "max_tokens" not in kwargs
        assert not {"temperature", "top_p", "stop"} & kwargs.keys()
    assert (await _openai_kwargs(model, {"seed": 3}))[0]["seed"] == 3


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5-chat-latest", "gpt-4.1"])
async def test_openai_other_models_are_unchanged(model):
    first, second = await _openai_kwargs(model, {"temperature": 0.2})
    assert (first["max_tokens"], first["temperature"]) == (64, 0.2)
    assert second["max_tokens"] == 64 and "max_completion_tokens" not in second


def test_azure_reasoning_deployment_drops_rejected_sampling():
    sampling = {"temperature": 0.2, "top_p": 0.4, "presence_penalty": 1, "seed": 9}
    with_effort = azure._request_kwargs(
        GatewayConfig(azure_reasoning_effort="low"), None, None, sampling, None
    )
    assert with_effort["seed"] == 9
    assert not {"temperature", "top_p", "presence_penalty"} & with_effort.keys()
    plain = azure._request_kwargs(
        GatewayConfig(azure_reasoning_effort=""), None, None, sampling, None
    )
    assert plain["temperature"] == 0.2 and plain["top_p"] == 0.4


# 6. Anthropic sampling limits -------------------------------------------------


def test_anthropic_temperature_is_clamped_to_one():
    kwargs = anthropic._sampling_kwargs({"temperature": 1.8}, "claude-test-model")
    assert kwargs["extra_body"] == {"temperature": 1.0}


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-5-20250929"])
def test_anthropic_models_rejecting_both_keep_only_temperature(model):
    kwargs = anthropic._sampling_kwargs({"temperature": 0.3, "top_p": 0.9}, model)
    assert kwargs["extra_body"] == {"temperature": 0.3}


def test_anthropic_top_p_alone_and_older_models_are_unchanged():
    assert anthropic._sampling_kwargs({"top_p": 0.9}, "claude-haiku-4-5")["extra_body"] == {
        "top_p": 0.9
    }
    both = anthropic._sampling_kwargs({"temperature": 0.3, "top_p": 0.9}, "claude-test-model")
    assert both["extra_body"] == {"temperature": 0.3, "top_p": 0.9}


# 7. Multi-part content ---------------------------------------------------------


def test_text_and_image_parts_translate_and_empty_text_is_dropped():
    _, msgs = to_anthropic_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
                ],
            }
        ]
    )
    assert [b["type"] for b in msgs[0]["content"]] == ["text", "image"]
    assert msgs[0]["content"][1]["source"]["media_type"] == "image/jpeg"


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}},
        {"type": "file", "file": {"file_id": "f"}},
        {"type": "image_url", "image_url": {}},
    ],
)
def test_unsupported_parts_are_a_clear_invalid_request(part):
    with pytest.raises(InvalidRequestError) as excinfo:
        to_anthropic_messages([{"role": "user", "content": [part]}])
    assert classify(excinfo.value) is ErrorKind.INVALID_REQUEST
    assert isinstance(excinfo.value, ValueError)


def test_bad_tool_arguments_in_history_do_not_count_for_the_breaker():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{bad"}}],
        },
    ]
    with pytest.raises(InvalidRequestError) as excinfo:
        to_anthropic_messages(messages)
    assert classify(excinfo.value) is ErrorKind.INVALID_REQUEST


# 8. OpenRouter cache writes reach usage and cost -----------------------------


def test_cache_write_tokens_are_reported_and_priced():
    usage = usage_from_openai_style(
        SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=200, cache_write_tokens=300),
            completion_tokens_details=None,
        )
    )
    assert (usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (200, 300)
    assert usage.uncached_input_tokens == 500
    price = ModelPrice(
        input_per_mtok=1.0,
        output_per_mtok=5.0,
        cached_input_per_mtok=0.1,
        cache_write_per_mtok=1.25,
    )
    assert cost_usd(price, usage) == pytest.approx((500 * 1.0 + 200 * 0.1 + 300 * 1.25 + 50) / 1e6)


def test_openai_cache_without_writes_is_unchanged():
    usage = usage_from_openai_style(
        SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=1,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4),
            completion_tokens_details=None,
        )
    )
    assert usage.cache_creation_input_tokens == 0
