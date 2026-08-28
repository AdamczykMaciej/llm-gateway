from unittest.mock import AsyncMock, patch

import pytest

from llm_gateway import GatewayConfig, LLMError, chat, reset_circuit_breakers
from llm_gateway.providers.base import ChatResult, ToolCall

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**overrides) -> GatewayConfig:
    defaults = dict(
        anthropic_api_key="test-anthropic-key",
        groq_api_key="test-groq-key",
        openai_api_key="test-openai-key",
        # No retry by default — keeps these tests fast and deterministic;
        # retry behavior itself is covered by test_retry.py and TestRetry
        # in test_router.py.
        retry_attempts=1,
    )
    defaults.update(overrides)
    return GatewayConfig(**defaults)


_MESSAGES = [{"role": "user", "content": "hi"}]


async def test_text_only_response_no_tools():
    result = ChatResult(content="hello", model="claude-x", input_tokens=1, output_tokens=1)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(return_value=result)}):
        out = await chat(messages=_MESSAGES, config=_config(provider_order="anthropic"))
    assert out.content == "hello"
    assert out.tool_calls == []


async def test_tool_call_response_passes_through():
    result = ChatResult(
        content=None,
        model="claude-x",
        input_tokens=1,
        output_tokens=1,
        tool_calls=[ToolCall(id="t1", name="get_weather", arguments={"city": "Warsaw"})],
        finish_reason="tool_calls",
    )
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(return_value=result)}):
        out = await chat(
            messages=_MESSAGES,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            config=_config(provider_order="anthropic"),
        )
    assert out.finish_reason == "tool_calls"
    assert out.tool_calls[0].name == "get_weather"


async def test_falls_back_to_second_provider_on_failure():
    anthropic_call = AsyncMock(side_effect=RuntimeError("boom"))
    groq_result = ChatResult(content="from groq", model="llama", input_tokens=1, output_tokens=1)
    groq_call = AsyncMock(return_value=groq_result)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": anthropic_call, "groq": groq_call}):
        out = await chat(messages=_MESSAGES, config=_config(provider_order="anthropic,groq"))
    assert out.content == "from groq"
    anthropic_call.assert_awaited_once()


async def test_circuit_breaker_shared_with_router():
    # A provider tripped via router.complete()'s breaker must also be
    # skipped by chat() — they share the same breaker module/state.
    from llm_gateway import breaker

    breaker.record_failure("anthropic", threshold=1, cooldown_seconds=60.0)
    groq_result = ChatResult(content="from groq", model="llama", input_tokens=1, output_tokens=1)
    anthropic_call = AsyncMock(side_effect=AssertionError("should be skipped, breaker is open"))
    groq_call = AsyncMock(return_value=groq_result)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": anthropic_call, "groq": groq_call}):
        out = await chat(messages=_MESSAGES, config=_config(provider_order="anthropic,groq"))
    assert out.content == "from groq"
    anthropic_call.assert_not_called()


async def test_force_provider_bypasses_fallback():
    anthropic_call = AsyncMock(
        return_value=ChatResult(content="a", model="x", input_tokens=0, output_tokens=0)
    )
    groq_call = AsyncMock(side_effect=RuntimeError("should not be called"))
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": anthropic_call, "groq": groq_call}):
        out = await chat(
            messages=_MESSAGES,
            config=_config(provider_order="groq"),
            force_provider="anthropic",
        )
    assert out.content == "a"
    groq_call.assert_not_called()


async def test_all_providers_unconfigured_raises_llm_error():
    with pytest.raises(LLMError):
        await chat(
            messages=_MESSAGES,
            config=_config(anthropic_api_key="", groq_api_key="", openai_api_key=""),
        )
