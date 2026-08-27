"""Tests for providers/anthropic.py's chat() at the SDK-call boundary —
verifying tool_choice="none" hard-blocks tools, response_format is emulated
via a forced tool call and unwrapped transparently, and sampling params
reach the real SDK call. Mocks the Anthropic client's messages.create, never
the translation functions themselves (those have their own direct tests in
test_anthropic_translate.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_gateway import GatewayConfig
from llm_gateway.providers import anthropic

pytestmark = pytest.mark.asyncio


def _config(**overrides) -> GatewayConfig:
    defaults = dict(anthropic_api_key="test-key")
    defaults.update(overrides)
    return GatewayConfig(**defaults)


def _text_response(text: str = "hi"):
    resp = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=1, output_tokens=1)
    return resp


def _tool_use_response(name: str, arguments: dict, tool_id: str = "toolu_1"):
    resp = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = arguments
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=1, output_tokens=1)
    return resp


async def test_tool_choice_none_omits_tools_entirely():
    mock_create = AsyncMock(return_value=_text_response())
    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    with patch("llm_gateway.providers.anthropic._client", return_value=mock_client):
        await anthropic.chat(
            _config(),
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            max_tokens=100,
            tool_choice="none",
        )
    assert "tools" not in mock_create.call_args.kwargs
    assert "tool_choice" not in mock_create.call_args.kwargs


async def test_tool_choice_auto_passes_tools_through():
    mock_create = AsyncMock(return_value=_text_response())
    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    with patch("llm_gateway.providers.anthropic._client", return_value=mock_client):
        await anthropic.chat(
            _config(),
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            max_tokens=100,
            tool_choice="auto",
        )
    assert mock_create.call_args.kwargs["tools"][0]["name"] == "get_weather"
    assert mock_create.call_args.kwargs["tool_choice"] == {"type": "auto"}


async def test_response_format_emulated_as_forced_tool_and_unwrapped():
    mock_create = AsyncMock(
        return_value=_tool_use_response(
            "__llm_gateway_structured_output", {"answer": 42}, tool_id="toolu_9"
        )
    )
    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "x", "schema": {"type": "object"}},
    }
    with patch("llm_gateway.providers.anthropic._client", return_value=mock_client):
        result = await anthropic.chat(
            _config(),
            [{"role": "user", "content": "What is 6*7?"}],
            tools=None,
            max_tokens=100,
            response_format=response_format,
        )

    # The forced tool is invisible to the caller — no tool_calls, JSON text instead.
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.content == '{"answer": 42}'
    sent_tools = mock_create.call_args.kwargs["tools"]
    assert len(sent_tools) == 1
    assert sent_tools[0]["name"] == "__llm_gateway_structured_output"
    assert mock_create.call_args.kwargs["tool_choice"] == {
        "type": "tool",
        "name": "__llm_gateway_structured_output",
    }


async def test_response_format_takes_priority_over_regular_tools():
    mock_create = AsyncMock(
        return_value=_tool_use_response("__llm_gateway_structured_output", {"x": 1})
    )
    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    with patch("llm_gateway.providers.anthropic._client", return_value=mock_client):
        await anthropic.chat(
            _config(),
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            max_tokens=100,
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
        )
    # Only the synthetic structured-output tool is sent, not the caller's real tools.
    sent_tools = mock_create.call_args.kwargs["tools"]
    assert len(sent_tools) == 1
    assert sent_tools[0]["name"] == "__llm_gateway_structured_output"


async def test_sampling_params_reach_the_sdk_call():
    mock_create = AsyncMock(return_value=_text_response())
    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    with patch("llm_gateway.providers.anthropic._client", return_value=mock_client):
        await anthropic.chat(
            _config(),
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=100,
            sampling={"temperature": 0.0, "top_p": 0.5, "stop": "STOP", "seed": 42},
        )
    kwargs = mock_create.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 0.5
    assert kwargs["stop_sequences"] == ["STOP"]
    assert "seed" not in kwargs  # no Anthropic equivalent


# ─── Streaming ─────────────────────────────────────────────────────────────


def _event(type_: str, **kwargs) -> MagicMock:
    e = MagicMock()
    e.type = type_
    for k, v in kwargs.items():
        setattr(e, k, v)
    return e


def _tool_use_block(tool_id: str, name: str) -> MagicMock:
    # MagicMock(name=...) is a classic gotcha: `name` is reserved by the Mock
    # constructor for the mock's own repr, not settable as a real attribute
    # that way — must setattr it afterward.
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    return block


class _FakeAnthropicStream:
    """Minimal stand-in for the Anthropic SDK's `async with client.messages
    .stream(...) as stream: async for event in stream:` pattern."""

    def __init__(self, events: list[MagicMock]):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for e in self._events:
            yield e


def _text_events(text: str, input_tokens=10, output_tokens=5) -> list[MagicMock]:
    return [
        _event("message_start", message=MagicMock(usage=MagicMock(input_tokens=input_tokens))),
        _event(
            "content_block_start",
            index=0,
            content_block=MagicMock(type="text"),
        ),
        _event("content_block_delta", index=0, delta=MagicMock(type="text_delta", text=text)),
        _event("content_block_stop", index=0),
        _event(
            "message_delta",
            delta=MagicMock(stop_reason="end_turn"),
            usage=MagicMock(output_tokens=output_tokens),
        ),
    ]


async def test_stream_text_deltas_and_final_usage():
    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(_text_events("Hi!")))
    with patch("llm_gateway.providers.anthropic._client", return_value=mock_client):
        deltas = [
            d
            async for d in anthropic.stream_chat(
                _config(), [{"role": "user", "content": "hi"}], tools=None, max_tokens=100
            )
        ]
    text_deltas = [d.content for d in deltas if d.content]
    assert text_deltas == ["Hi!"]
    final = deltas[-1]
    assert final.finish_reason == "stop"
    assert final.usage == (10, 5)


async def test_stream_tool_use_emits_id_name_then_argument_fragments():
    events = [
        _event("message_start", message=MagicMock(usage=MagicMock(input_tokens=10))),
        _event(
            "content_block_start",
            index=0,
            content_block=_tool_use_block("toolu_1", "get_weather"),
        ),
        _event(
            "content_block_delta",
            index=0,
            delta=MagicMock(type="input_json_delta", partial_json='{"city":'),
        ),
        _event(
            "content_block_delta",
            index=0,
            delta=MagicMock(type="input_json_delta", partial_json=' "Warsaw"}'),
        ),
        _event("content_block_stop", index=0),
        _event(
            "message_delta",
            delta=MagicMock(stop_reason="tool_use"),
            usage=MagicMock(output_tokens=5),
        ),
    ]
    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
    with patch("llm_gateway.providers.anthropic._client", return_value=mock_client):
        deltas = [
            d
            async for d in anthropic.stream_chat(
                _config(),
                [{"role": "user", "content": "weather?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
                max_tokens=100,
            )
        ]

    tool_deltas = [d.tool_call_deltas[0] for d in deltas if d.tool_call_deltas]
    assert tool_deltas[0]["id"] == "toolu_1"
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    # Subsequent fragments carry only arguments, matching OpenAI's own
    # streaming semantics — no id/name repeated.
    assert "id" not in tool_deltas[1]
    assert tool_deltas[1]["function"]["arguments"] == '{"city":'
    assert tool_deltas[2]["function"]["arguments"] == ' "Warsaw"}'
    assert deltas[-1].finish_reason == "tool_calls"


async def test_stream_structured_output_hides_synthetic_tool_and_emits_json_text():
    from llm_gateway.providers._anthropic_translate import STRUCTURED_OUTPUT_TOOL_NAME

    events = [
        _event("message_start", message=MagicMock(usage=MagicMock(input_tokens=10))),
        _event(
            "content_block_start",
            index=0,
            content_block=_tool_use_block("toolu_1", STRUCTURED_OUTPUT_TOOL_NAME),
        ),
        _event(
            "content_block_delta",
            index=0,
            delta=MagicMock(type="input_json_delta", partial_json='{"answer": 42}'),
        ),
        _event("content_block_stop", index=0),
        _event(
            "message_delta",
            delta=MagicMock(stop_reason="tool_use"),
            usage=MagicMock(output_tokens=5),
        ),
    ]
    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
    with patch("llm_gateway.providers.anthropic._client", return_value=mock_client):
        deltas = [
            d
            async for d in anthropic.stream_chat(
                _config(),
                [{"role": "user", "content": "answer?"}],
                tools=None,
                max_tokens=100,
                response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
            )
        ]

    # No tool_call_deltas ever surfaced — the synthetic tool is invisible.
    assert all(d.tool_call_deltas is None for d in deltas)
    text_deltas = [d.content for d in deltas if d.content]
    assert text_deltas == ['{"answer": 42}']
    # Unwrapped structured output is a normal "stop", not "tool_calls".
    assert deltas[-1].finish_reason == "stop"
