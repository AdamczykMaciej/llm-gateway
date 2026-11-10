"""Retry amplification under a provider outage (fakes only): N concurrent
callers against failing providers make a bounded number of requests."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx2
import openai
import pytest

from llm_gateway import GatewayConfig, breaker, complete, reset_circuit_breakers
from llm_gateway.errors import AllProvidersExhaustedError, InvalidOutputError
from llm_gateway.providers.base import ProviderResult
from llm_gateway.retry import Deadline, call_with_retry

pytestmark = pytest.mark.asyncio
_REQ = httpx2.Request("POST", "https://api.example.test")


def _err(status=503):
    return openai.AsyncOpenAI(api_key="k")._make_status_error(
        "boom",
        body={"message": "boom"},
        response=httpx2.Response(status, request=_REQ),
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


def _patch_calls(fake):
    return patch.dict(
        "llm_gateway.router.CALLS", {n: fake for n in ("anthropic", "groq", "openai")}
    )


async def test_concurrent_callers_make_bounded_requests_during_an_outage():
    n_callers = 20
    started = 0

    async def dead(*a, **k):
        nonlocal started
        started += 1
        await asyncio.sleep(0)
        raise _err(503)

    cfg = _cfg(breaker_failure_threshold=3, breaker_count_retries=True)
    with _patch_calls(dead):
        results = await asyncio.gather(
            *[complete(system="s", prompt="p", config=cfg) for _ in range(n_callers)],
            return_exceptions=True,
        )
    assert all(isinstance(r, AllProvidersExhaustedError) for r in results)
    # Retried failures count: each provider's breaker opens after ~threshold
    # requests and the rest of the herd stops retrying it.
    assert started < n_callers * 3 * 3 // 2
    assert all(breaker.state(p) == "open" for p in ("anthropic", "groq", "openai"))


async def test_max_attempts_per_call_caps_every_request_of_one_call():
    fake = AsyncMock(side_effect=_err(503))
    cfg = _cfg(max_attempts_per_call=4)
    with _patch_calls(fake), pytest.raises(AllProvidersExhaustedError):
        await complete(system="s", prompt="p", config=cfg)
    assert fake.await_count == 4  # not 9


async def test_default_has_no_cap():
    fake = AsyncMock(side_effect=_err(503))
    with _patch_calls(fake), pytest.raises(AllProvidersExhaustedError):
        await complete(system="s", prompt="p", config=_cfg())
    assert fake.await_count == 9


async def test_budget_also_stops_repair_calls():
    ok = ProviderResult(text="not json", model="m", input_tokens=1, output_tokens=1)
    fake = AsyncMock(return_value=ok)
    from llm_gateway import complete_with_usage

    cfg = _cfg(max_attempts_per_call=2, provider_order="anthropic")
    with _patch_calls(fake), pytest.raises(InvalidOutputError):
        await complete_with_usage(
            system="s",
            prompt="p",
            config=cfg,
            validate={"type": "object"},
            repair_attempts=5,
        )
    assert fake.await_count == 2  # first call + one repair, not 6


async def test_retry_stops_when_the_breaker_trips_mid_call():
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        breaker.record_failure("p", threshold=1, cooldown_seconds=60)
        raise _err(503)

    with pytest.raises(openai.APIStatusError):
        await call_with_retry(
            flaky,
            attempts=5,
            base_delay_seconds=0.0,
            should_retry=lambda e: not breaker.tripped("p"),
        )
    assert calls == 1


async def test_probe_lease_is_not_a_tripped_breaker():
    breaker.record_failure("p", threshold=1, cooldown_seconds=0.01)
    await asyncio.sleep(0.03)
    assert breaker.allow("p")  # becomes the probe
    assert not breaker.tripped("p")
    assert not breaker.allow("p")  # exactly one probe


async def test_retry_after_is_jittered_upwards_only():
    err = openai.AsyncOpenAI(api_key="k")._make_status_error(
        "x",
        body={},
        response=httpx2.Response(503, request=_REQ, headers={"retry-after": "7"}),
    )
    delays = []
    for _ in range(20):
        call = AsyncMock(side_effect=[err, "ok"])
        with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep:
            await call_with_retry(call, attempts=2, base_delay_seconds=0.1)
        delays.append(sleep.await_args.args[0])
    assert all(7.0 <= d <= 7.71 for d in delays)
    assert len(set(delays)) > 1


async def test_deadline_attempt_budget_unbounded_by_default():
    d = Deadline(None)
    d.attempts_made = 10_000
    assert not d.attempts_exhausted
