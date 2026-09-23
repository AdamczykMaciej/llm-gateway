from unittest.mock import AsyncMock, patch

import anthropic
import httpx2
import pytest

from llm_gateway import (
    AllProvidersExhaustedError,
    GatewayConfig,
    GatewayNotConfiguredError,
    LLMError,
    complete,
    reset_circuit_breakers,
)
from llm_gateway.providers.base import ProviderResult

pytestmark = pytest.mark.asyncio


def _transient(message: str) -> anthropic.APIConnectionError:
    """A connection error — the kind of failure the gateway retries."""
    return anthropic.APIConnectionError(
        message=message, request=httpx2.Request("POST", "https://api.example.test")
    )


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
        # No retry by default — keeps these tests fast and deterministic;
        # retry behavior itself is covered by TestRetry below.
        retry_attempts=1,
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


async def test_retries_the_same_provider_before_falling_over():
    # First call fails, second (the retry) succeeds — groq must never be tried.
    anthropic_call = AsyncMock(
        side_effect=[_transient("blip"), ProviderResult("recovered", "claude-x", 1, 1)]
    )
    groq_call = AsyncMock(side_effect=AssertionError("groq should not have been called"))
    with (
        patch("llm_gateway.router.CALLS", {"anthropic": anthropic_call, "groq": groq_call}),
        patch("llm_gateway.retry.asyncio.sleep", AsyncMock()),
    ):
        text = await complete(
            system="s",
            prompt="p",
            config=_config(provider_order="anthropic,groq", retry_attempts=2),
        )
    assert text == "recovered"
    assert anthropic_call.await_count == 2
    groq_call.assert_not_called()


async def test_retry_exhaustion_still_only_counts_as_one_breaker_failure():
    # retry_attempts=2 means 2 calls to anthropic per logical request, but
    # the breaker should still trip after breaker_failure_threshold *requests*,
    # not *attempts* — retries are invisible to the breaker.
    anthropic_call = AsyncMock(side_effect=_transient("down"))
    groq_result = ProviderResult(text="ok", model="llama", input_tokens=1, output_tokens=1)
    groq_call = AsyncMock(return_value=groq_result)
    config = _config(provider_order="anthropic,groq", retry_attempts=2, breaker_failure_threshold=1)

    with (
        patch("llm_gateway.router.CALLS", {"anthropic": anthropic_call, "groq": groq_call}),
        patch("llm_gateway.retry.asyncio.sleep", AsyncMock()),
    ):
        text = await complete(system="s", prompt="p", config=config)
        assert text == "ok"
        assert anthropic_call.await_count == 2  # both retry attempts used

        # Breaker threshold is 1 request-level failure — already tripped, so
        # a second call must skip anthropic entirely (no further attempts).
        text = await complete(system="s", prompt="p", config=config)
        assert text == "ok"
        assert anthropic_call.await_count == 2


async def test_unclassified_errors_fail_over_without_a_retry():
    # A non-SDK exception (e.g. a response-parsing bug) is not known to be
    # transient, so it goes straight to the next provider.
    anthropic_call = AsyncMock(side_effect=RuntimeError("bug"))
    groq_call = AsyncMock(return_value=ProviderResult("from groq", "llama", 1, 1))
    with (
        patch("llm_gateway.router.CALLS", {"anthropic": anthropic_call, "groq": groq_call}),
        patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep,
    ):
        text = await complete(
            system="s",
            prompt="p",
            config=_config(provider_order="anthropic,groq", retry_attempts=2),
        )
    assert text == "from groq"
    anthropic_call.assert_awaited_once()
    sleep.assert_not_awaited()


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


# ─── Typed "no provider could serve" errors ─────────────────────────────────
#
# A caller (e.g. the app embedding this library) needs to tell "nothing is
# configured, fix the deployment" apart from "everything is configured but
# temporarily unavailable, retry shortly" without parsing message text.


async def test_unconfigured_chain_raises_gateway_not_configured_error_specifically():
    with pytest.raises(GatewayNotConfiguredError) as info:
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
    assert not isinstance(info.value, AllProvidersExhaustedError)


async def test_every_provider_failing_raises_all_providers_exhausted_error_specifically():
    # The production scenario this models: Anthropic over its spend limit,
    # OpenAI out of credits, Groq rate-limited — all at once. Every provider
    # is configured and gets attempted; none can serve the call.
    failing = AsyncMock(side_effect=RuntimeError("down"))
    with patch("llm_gateway.router.CALLS", {"anthropic": failing, "groq": failing}):
        with pytest.raises(AllProvidersExhaustedError) as info:
            await complete(system="s", prompt="p", config=_config(provider_order="anthropic,groq"))
    assert not isinstance(info.value, GatewayNotConfiguredError)


async def test_every_configured_provider_breaker_open_raises_all_providers_exhausted_error():
    # Once the breaker has already tripped for every configured provider
    # (e.g. the incident above, a second or two later), no provider is even
    # attempted this time — attempted_any is False, same as the unconfigured
    # case. This must still raise AllProvidersExhaustedError, not
    # GatewayNotConfiguredError: the providers *are* configured, they are
    # just cooling down, and the message must not tell an operator to go set
    # API keys that are already set.
    failing = AsyncMock(side_effect=RuntimeError("down"))
    config = _config(provider_order="anthropic,groq", breaker_failure_threshold=1)
    with patch("llm_gateway.router.CALLS", {"anthropic": failing, "groq": failing}):
        with pytest.raises(AllProvidersExhaustedError):
            await complete(system="s", prompt="p", config=config)
        # Both breakers are now open; nothing is attempted on this call.
        failing.reset_mock()
        with pytest.raises(AllProvidersExhaustedError) as info:
            await complete(system="s", prompt="p", config=config)
    failing.assert_not_called()
    assert not isinstance(info.value, GatewayNotConfiguredError)
    assert "ANTHROPIC_API_KEY" not in str(info.value)


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


async def test_force_provider_with_explicit_model_reaches_the_provider_call():
    anthropic_call = AsyncMock(return_value=ProviderResult("a", "claude-sonnet-4-6", 0, 0))
    with patch("llm_gateway.router.CALLS", {"anthropic": anthropic_call}):
        await complete(
            system="s",
            prompt="p",
            config=_config(provider_order="anthropic"),
            force_provider="anthropic",
            model="claude-sonnet-4-6",
        )
    # positional call signature: (config, system, prompt, max_tokens, model)
    assert anthropic_call.await_args.args[-1] == "claude-sonnet-4-6"


async def test_explicit_model_is_ignored_without_force_provider():
    # A model override only makes sense tied to a specific provider — in the
    # auto fallback chain each provider must keep using its own configured
    # default, since a model name from one provider's namespace is
    # meaningless to another.
    anthropic_call = AsyncMock(return_value=ProviderResult("a", "x", 0, 0))
    with patch("llm_gateway.router.CALLS", {"anthropic": anthropic_call}):
        await complete(
            system="s",
            prompt="p",
            config=_config(provider_order="anthropic"),
            model="some-other-providers-model",
        )
    assert anthropic_call.await_args.args[-1] is None
