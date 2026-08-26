from unittest.mock import AsyncMock, patch

import pytest

from llm_gateway import GatewayConfig, LLMError, complete, reset_circuit_breakers
from llm_gateway.providers.base import ProviderResult

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
        breaker_failure_threshold=3,
        breaker_cooldown_seconds=60.0,
    )
    defaults.update(overrides)
    return GatewayConfig(**defaults)


async def test_primary_provider_success():
    result = ProviderResult(text="hello", model="claude-x", input_tokens=1, output_tokens=1)
    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=result)}):
        text = await complete(system="s", prompt="p", config=_config(provider_order="anthropic"))
    assert text == "hello"


async def test_falls_back_to_second_provider_on_failure():
    anthropic_call = AsyncMock(side_effect=RuntimeError("boom"))
    groq_result = ProviderResult(text="from groq", model="llama", input_tokens=1, output_tokens=1)
    groq_call = AsyncMock(return_value=groq_result)
    with patch("llm_gateway.router.CALLS", {"anthropic": anthropic_call, "groq": groq_call}):
        text = await complete(
            system="s", prompt="p", config=_config(provider_order="anthropic,groq")
        )
    assert text == "from groq"
    anthropic_call.assert_awaited_once()
    groq_call.assert_awaited_once()


async def test_provider_order_is_configurable():
    groq_result = ProviderResult(text="from groq", model="llama", input_tokens=1, output_tokens=1)
    anthropic_call = AsyncMock(return_value=ProviderResult("nope", "x", 0, 0))
    groq_call = AsyncMock(return_value=groq_result)
    with patch("llm_gateway.router.CALLS", {"anthropic": anthropic_call, "groq": groq_call}):
        text = await complete(system="s", prompt="p", config=_config(provider_order="groq"))
    assert text == "from groq"
    anthropic_call.assert_not_called()


async def test_all_providers_unconfigured_raises_llm_error():
    with pytest.raises(LLMError):
        await complete(
            system="s",
            prompt="p",
            config=_config(
                anthropic_api_key="",
                groq_api_key="",
                openai_api_key="",
                provider_order="anthropic,groq,openai",
            ),
        )


async def test_all_providers_fail_raises_llm_error():
    failing = AsyncMock(side_effect=RuntimeError("down"))
    with patch("llm_gateway.router.CALLS", {"anthropic": failing, "groq": failing}):
        with pytest.raises(LLMError):
            await complete(system="s", prompt="p", config=_config(provider_order="anthropic,groq"))


async def test_circuit_breaker_skips_provider_after_threshold_failures():
    failing = AsyncMock(side_effect=RuntimeError("down"))
    result = ProviderResult(text="ok", model="llama", input_tokens=1, output_tokens=1)
    groq_call = AsyncMock(return_value=result)
    config = _config(provider_order="anthropic,groq", breaker_failure_threshold=2)

    with patch("llm_gateway.router.CALLS", {"anthropic": failing, "groq": groq_call}):
        for _ in range(2):
            await complete(system="s", prompt="p", config=config)
        assert failing.call_count == 2

        # Breaker should now be open for anthropic — a third call must not
        # attempt it again.
        await complete(system="s", prompt="p", config=config)
        assert failing.call_count == 2


async def test_force_provider_bypasses_fallback():
    anthropic_call = AsyncMock(return_value=ProviderResult("a", "x", 0, 0))
    groq_call = AsyncMock(side_effect=RuntimeError("should not be called"))
    with patch("llm_gateway.router.CALLS", {"anthropic": anthropic_call, "groq": groq_call}):
        text = await complete(
            system="s",
            prompt="p",
            config=_config(provider_order="groq"),
            force_provider="anthropic",
        )
    assert text == "a"
    groq_call.assert_not_called()
