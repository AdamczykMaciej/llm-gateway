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
