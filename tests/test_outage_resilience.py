"""Outage resilience (fakes only): every provider failing at once, retired
models, billing shapes, the breaker snapshot and half-open concurrency."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import anthropic
import httpx2
import openai
import pytest
from fastapi.testclient import TestClient

from llm_gateway import (
    AllProvidersExhaustedError,
    GatewayConfig,
    LLMDeadlineExceeded,
    breaker,
    circuit_breaker_status,
    complete,
    reset_circuit_breakers,
)
from llm_gateway.errors import ErrorKind, classify, failure_class
from llm_gateway.providers.base import ProviderResult

pytestmark = pytest.mark.asyncio
_REQ = httpx2.Request("POST", "https://api.example.test")


def _a(status, message, type_="invalid_request_error"):
    body = {"type": "error", "error": {"type": type_, "message": message}}
    return anthropic.AsyncAnthropic(api_key="k")._make_status_error(
        message, body=body, response=httpx2.Response(status, request=_REQ)
    )


def _o(status, message, code, headers=None):
    body = {"message": message, "type": code, "code": code}
    return openai.AsyncOpenAI(api_key="k")._make_status_error(
        message, body=body, response=httpx2.Response(status, request=_REQ, headers=headers)
    )


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw):
    d = dict(
        anthropic_api_key="sk-ant-SECRETKEY123456",
        groq_api_key="gsk_SECRETKEY123456",
        openai_api_key="sk-SECRETKEY123456",
        provider_order="anthropic,groq,openai",
        retry_attempts=3,
        retry_base_delay_seconds=0.0,
    )
    d.update(kw)
    return GatewayConfig(**d)


def _ok(text="ok"):
    return AsyncMock(
        return_value=ProviderResult(text=text, model="m", input_tokens=1, output_tokens=1)
    )


async def test_all_fail_at_once_gives_classified_aggregate_without_secrets():
    calls = {
        "anthropic": AsyncMock(
            side_effect=_a(
                400, "You have reached your specified API usage limits. sk-ant-SECRETKEY123456"
            )
        ),
        "groq": AsyncMock(side_effect=_o(429, "slow down", "rate_limit_exceeded")),
        "openai": AsyncMock(
            side_effect=_o(404, "The model `gpt-x` does not exist", "model_not_found")
        ),
    }
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(AllProvidersExhaustedError) as info:
            await complete(system="s", prompt="p", config=_cfg())
    kinds = {f.provider: f.kind for f in info.value.failures}
    assert kinds == {"anthropic": "billing", "groq": "rate_limited", "openai": "model_retired"}
    assert "anthropic=billing, groq=rate_limited, openai=model_retired" in str(info.value)
    assert "SECRETKEY" not in str(info.value)
    assert all("SECRETKEY" not in f.detail for f in info.value.failures)


async def test_all_hang_is_bounded_by_the_deadline_not_the_sum():
    async def hang(*a, **k):
        await asyncio.sleep(60)

    calls = {p: hang for p in ("anthropic", "groq", "openai")}
    cfg = _cfg(call_deadline_seconds=0.3, request_timeout_seconds=30.0, retry_attempts=5)
    started = time.monotonic()
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(LLMDeadlineExceeded):
            await complete(system="s", prompt="p", config=cfg)
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda: _a(402, "payment", "billing_error"), "billing"),
        (
            lambda: _o(
                403, "Your account is not active, check billing details", "billing_not_active"
            ),
            "billing",
        ),
        (
            lambda: _o(403, "The billing account for the project is disabled", "forbidden"),
            "billing",
        ),
        (lambda: _o(403, "Invalid API key", "invalid_api_key"), "auth"),
        (
            lambda: _o(400, "The model `x` has been decommissioned", "model_decommissioned"),
            "model_retired",
        ),
        (lambda: _o(404, "model `x` not found", "model_not_found"), "model_retired"),
        (lambda: _o(429, "tokens per day", "rate_limit_exceeded"), "rate_limited"),
        (lambda: _o(503, "unavailable", "server_error"), "network"),
    ],
)
async def test_failure_classes(make, expected):
    assert failure_class(make()) == expected


async def test_402_is_billing_kind():
    assert classify(_a(402, "x", "billing_error")) is ErrorKind.BILLING


async def test_retired_model_is_not_retried_and_opens_long():
    failing = AsyncMock(
        side_effect=_o(400, "model has been decommissioned", "model_decommissioned")
    )
    cfg = _cfg(provider_order="groq,openai", breaker_cooldown_seconds=10.0)
    with patch("llm_gateway.router.CALLS", {"groq": failing, "openai": _ok()}):
        assert await complete(system="s", prompt="p", config=cfg) == "ok"
    assert failing.await_count == 1  # no retry
    (state,) = [s for s in circuit_breaker_status() if s["model"]]
    assert state["state"] == "open"
    assert state["last_error_class"] == "model_retired"
    assert state["open_for_seconds"] > 1000  # long cooldown, not 10 s


async def test_billing_cooldown_is_longer_than_default():
    failing = AsyncMock(side_effect=_a(400, "Your credit balance is too low"))
    cfg = _cfg(provider_order="anthropic,openai", breaker_cooldown_seconds=10.0)
    with patch("llm_gateway.router.CALLS", {"anthropic": failing, "openai": _ok()}):
        await complete(system="s", prompt="p", config=cfg)
    state = next(s for s in circuit_breaker_status() if s["provider"] == "anthropic")
    assert state["open_for_seconds"] > 200
    assert state["last_error_class"] == "billing"


async def test_retry_after_is_honoured_but_capped():
    from llm_gateway.errors import retry_after_seconds

    assert retry_after_seconds(_o(503, "x", "server_error", {"retry-after": "3"})) == 3
    assert retry_after_seconds(_o(503, "x", "server_error", {"retry-after": "99999"})) == 60.0


async def test_half_open_admits_exactly_one_concurrent_probe():
    breaker.record_failure("anthropic", threshold=1, cooldown_seconds=5.0)
    real = time.monotonic
    with patch("llm_gateway.breaker.time.monotonic", lambda: real() + 10):
        results = await asyncio.gather(
            *(asyncio.to_thread(breaker.allow, "anthropic") for _ in range(50))
        )
        assert results.count(True) == 1
    breaker.record_success("anthropic")
    assert breaker.allow("anthropic")


async def test_snapshot_has_no_error_text_and_reports_half_open():
    breaker.record_error(
        "groq", _o(429, "secret-ish text", "rate_limit_exceeded"), threshold=1, cooldown_seconds=5.0
    )
    snap = circuit_breaker_status()
    assert snap[0]["provider"] == "groq" and snap[0]["state"] == "open"
    assert "secret-ish" not in repr(snap)
    real = time.monotonic
    with patch("llm_gateway.breaker.time.monotonic", lambda: real() + 10):
        assert circuit_breaker_status()[0]["state"] == "half_open"


@pytest.mark.filterwarnings("ignore")
def test_service_providers_health_endpoint():
    from llm_gateway.service.app import create_app

    cfg = _cfg()
    breaker.record_error(
        "groq", _o(429, "x", "rate_limit_exceeded"), threshold=1, cooldown_seconds=5.0
    )
    app = create_app(cfg)
    client = TestClient(app)
    r = client.get("/v1/providers/health")
    if r.status_code in (401, 403):
        r = client.get("/v1/providers/health", headers={"Authorization": "Bearer test"})
    assert r.status_code == 200
    groq = next(p for p in r.json()["data"] if p["provider"] == "groq")
    assert groq["breaker"][0]["last_error_class"] == "rate_limited"
    assert "SECRET" not in r.text
