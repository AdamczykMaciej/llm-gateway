"""Credit-balance / spend-limit / insufficient-quota errors are BILLING:
not retried, and one occurrence opens the provider's breaker."""

import time
from unittest.mock import AsyncMock, patch

import anthropic
import httpx2
import openai
import pytest

from llm_gateway import GatewayConfig, breaker, complete, reset_circuit_breakers
from llm_gateway.errors import ErrorKind, classify, is_billing_error, policy_for
from llm_gateway.providers.base import ProviderResult

pytestmark = pytest.mark.asyncio

_REQUEST = httpx2.Request("POST", "https://api.example.test")


def _anthropic_error(status: int, message: str, *, type_: str = "invalid_request_error"):
    body = {"type": "error", "error": {"type": type_, "message": message}}
    return anthropic.AsyncAnthropic(api_key="k")._make_status_error(
        message, body=body, response=httpx2.Response(status, request=_REQUEST)
    )


def _openai_error(status: int, message: str, code: str = "insufficient_quota"):
    body = {"message": message, "type": code, "code": code}
    return openai.AsyncOpenAI(api_key="k")._make_status_error(
        message, body=body, response=httpx2.Response(status, request=_REQUEST)
    )


CREDIT = "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing"
OPENAI_QUOTA = "You exceeded your current quota, please check your plan and billing details."


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**kw) -> GatewayConfig:
    d = dict(
        anthropic_api_key="a",
        groq_api_key="g",
        openai_api_key="o",
        breaker_failure_threshold=3,
        breaker_cooldown_seconds=60.0,
        retry_attempts=3,
        retry_base_delay_seconds=0.0,
    )
    d.update(kw)
    return GatewayConfig(**d)


def _ok(text="from next"):
    return AsyncMock(
        return_value=ProviderResult(text=text, model="m", input_tokens=1, output_tokens=1)
    )


@pytest.mark.parametrize(
    "exc",
    [
        _anthropic_error(400, CREDIT),
        _anthropic_error(400, "YOUR CREDIT BALANCE IS TOO LOW, add funds"),
        _anthropic_error(
            400, "You have reached your specified API usage limits. Resumes 2026-11-01"
        ),
        _openai_error(429, OPENAI_QUOTA),
        _openai_error(400, "Quota problem", code="insufficient_quota"),
        _openai_error(402, "Payment required: insufficient credits"),
    ],
)
async def test_billing_messages_are_classified_billing(exc):
    assert is_billing_error(exc)
    assert classify(exc) is ErrorKind.BILLING
    policy = policy_for(exc)
    assert policy.trips_breaker and policy.opens_immediately and not policy.retry


@pytest.mark.parametrize(
    "exc",
    [
        _anthropic_error(400, "messages.0.content: Input should be a valid string"),
        _anthropic_error(400, "max_tokens: must be greater than 0"),
        _openai_error(400, "Invalid 'messages': malformed", code="invalid_value"),
        _anthropic_error(500, CREDIT),  # a 5xx is not billing, whatever it says
    ],
)
async def test_genuine_invalid_requests_are_not_billing(exc):
    assert not is_billing_error(exc)
    assert classify(exc) is not ErrorKind.BILLING


async def test_malformed_400_does_not_trip_the_breaker():
    exc = _anthropic_error(400, "messages: field required")
    breaker.record_error("anthropic", exc, threshold=1, cooldown_seconds=60)
    assert not breaker.is_open("anthropic")


async def test_anthropic_credit_balance_trips_breaker_and_falls_through_once():
    failing = AsyncMock(side_effect=_anthropic_error(400, CREDIT))
    nxt = _ok()
    config = _config(provider_order="anthropic,groq")
    with patch("llm_gateway.router.CALLS", {"anthropic": failing, "groq": nxt}):
        assert await complete(system="s", prompt="p", config=config) == "from next"
        # Not retried in a loop (retry_attempts=3) and the breaker opened at once
        # although the failure threshold is 3.
        failing.assert_awaited_once()
        assert breaker.is_open("anthropic")
        # Second request inside the window skips anthropic without calling it.
        assert await complete(system="s", prompt="p", config=config) == "from next"
        failing.assert_awaited_once()
        assert nxt.await_count == 2


async def test_openai_insufficient_quota_behaves_the_same():
    failing = AsyncMock(side_effect=_openai_error(429, OPENAI_QUOTA))
    nxt = _ok()
    config = _config(provider_order="openai,groq")
    with patch("llm_gateway.router.CALLS", {"openai": failing, "groq": nxt}):
        await complete(system="s", prompt="p", config=config)
        await complete(system="s", prompt="p", config=config)
    failing.assert_awaited_once()
    assert nxt.await_count == 2


async def test_malformed_400_does_not_trip_but_falls_over_each_time():
    failing = AsyncMock(side_effect=_anthropic_error(400, "messages: field required"))
    nxt = _ok()
    config = _config(provider_order="anthropic,groq")
    with patch("llm_gateway.router.CALLS", {"anthropic": failing, "groq": nxt}):
        await complete(system="s", prompt="p", config=config)
        await complete(system="s", prompt="p", config=config)
    assert failing.await_count == 2
    assert not breaker.is_open("anthropic")


async def test_half_open_recovery_after_the_window():
    failing = AsyncMock(side_effect=_anthropic_error(400, CREDIT))
    config = _config(provider_order="anthropic,groq", breaker_cooldown_seconds=60.0)
    recovered = _ok("anthropic is back")
    with patch("llm_gateway.router.CALLS", {"anthropic": failing, "groq": _ok()}):
        await complete(system="s", prompt="p", config=config)
    assert breaker.is_open("anthropic")
    real = time.monotonic
    with patch("llm_gateway.breaker.time.monotonic", lambda: real() + 61):
        assert not breaker.is_open("anthropic")
        with patch("llm_gateway.router.CALLS", {"anthropic": recovered, "groq": _ok()}):
            assert await complete(system="s", prompt="p", config=config) == "anthropic is back"
    assert not breaker.is_open("anthropic")


async def test_billing_log_has_kind_and_no_message_body(caplog):
    failing = AsyncMock(side_effect=_anthropic_error(400, CREDIT))
    config = _config(provider_order="anthropic,groq")
    with patch("llm_gateway.router.CALLS", {"anthropic": failing, "groq": _ok()}):
        with caplog.at_level("WARNING", logger="llm_gateway"):
            await complete(system="SECRET-SYSTEM", prompt="SECRET-PROMPT", config=config)
    rec = [r for r in caplog.records if "billing" in r.getMessage()]
    assert rec and "provider=anthropic" in rec[0].getMessage()
    assert "SECRET" not in rec[0].getMessage() and "credit balance" not in rec[0].getMessage()
