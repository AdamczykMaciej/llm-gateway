"""Reliability pass 7: cancellation, deadline, mid-stream and Retry-After edges.
Fakes only; no provider is ever called."""

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import patch

import anthropic
import httpx2
import pytest

from llm_gateway import GatewayConfig, breaker, complete, reset_circuit_breakers, stream_chat
from llm_gateway.errors import LLMError, retry_after_seconds
from llm_gateway.providers.base import ProviderResult, StreamDelta, Usage

pytestmark = pytest.mark.asyncio

_MSGS = [{"role": "user", "content": "hi"}]
_REQ = httpx2.Request("POST", "https://api.example.test/v1/messages")


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw) -> GatewayConfig:
    kw.setdefault("provider_order", "anthropic,groq")
    kw.setdefault("retry_attempts", 3)
    return GatewayConfig(anthropic_api_key="k", groq_api_key="k", **kw)


def _blip() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(message="blip", request=_REQ)


def _ok(text="ok"):
    async def call(*a, **k):
        return ProviderResult(text=text, usage=Usage(1, 1), model="m", stop_reason="end_turn")

    return call


async def test_cancel_during_backoff_sleep_reraises_and_leaves_breaker_clean():
    async def flaky(*a, **k):
        raise _blip()

    second_calls = []

    async def second(*a, **k):
        second_calls.append(1)
        return await _ok()()

    cfg = _cfg(retry_base_delay_seconds=30.0)
    with patch("llm_gateway.router.CALLS", {"anthropic": flaky, "groq": second}):
        task = asyncio.create_task(complete(system="s", prompt="p", config=cfg))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert second_calls == []
    assert breaker.snapshot() == [] or all(s["state"] == "closed" for s in breaker.snapshot())
    assert breaker.state("anthropic") == "closed"


async def test_no_fallback_after_deadline_passed():
    async def slow_fail(*a, **k):
        await asyncio.sleep(0.2)
        raise ConnectionResetError("x")

    second_calls = []

    async def second(*a, **k):
        second_calls.append(1)
        return await _ok()()

    cfg = _cfg(retry_attempts=1)
    with patch("llm_gateway.router.CALLS", {"anthropic": slow_fail, "groq": second}):
        with pytest.raises(LLMError):
            await complete(system="s", prompt="p", config=cfg, deadline_seconds=0.1)
    assert second_calls == []


async def test_stream_error_after_tokens_does_not_switch_provider_by_default():
    async def dies(*a, **k):
        yield StreamDelta(content="partial")
        raise ConnectionResetError("boom")

    second_calls = []

    async def second(*a, **k):
        second_calls.append(1)
        yield StreamDelta(content="OTHER")

    got = []
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": dies, "groq": second}):
        with pytest.raises(ConnectionResetError):
            async for d in stream_chat(messages=_MSGS, config=_cfg(retry_attempts=1)):
                got.append(d.content)
    assert got == ["partial"] and second_calls == []


async def test_half_open_admits_exactly_one_under_threads():
    breaker.record_failure("p", threshold=1, cooldown_seconds=0.05)
    time.sleep(0.08)
    admitted = []
    barrier = threading.Barrier(32)

    def worker():
        barrier.wait()
        admitted.append(breaker.allow("p"))

    threads = [threading.Thread(target=worker) for _ in range(32)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert admitted.count(True) == 1


def _err(headers):
    class E(Exception):
        response = httpx2.Response(503, request=_REQ, headers=headers)

    return E()


async def test_retry_after_http_date_is_honoured():
    when = datetime.now(UTC) + timedelta(seconds=20)
    got = retry_after_seconds(_err({"retry-after": format_datetime(when, usegmt=True)}))
    assert got is not None and 15 <= got <= 20


async def test_retry_after_http_date_is_capped_and_past_is_zero():
    far = datetime.now(UTC) + timedelta(days=3)
    assert retry_after_seconds(_err({"retry-after": format_datetime(far, usegmt=True)})) == 60.0
    past = datetime.now(UTC) - timedelta(days=1)
    assert retry_after_seconds(_err({"retry-after": format_datetime(past, usegmt=True)})) == 0.0


async def test_retry_after_garbage_ignored():
    for bad in ("", "soon", "nan", "-5", "Mon, 99 Foo 2020"):
        assert retry_after_seconds(_err({"retry-after": bad})) is None


async def test_outer_timeout_mid_stream_closes_stream_and_never_falls_back():
    state = {"closed": False}

    async def slow(*a, **k):
        try:
            yield StreamDelta(content="a")
            await asyncio.sleep(3600)
        finally:
            state["closed"] = True

    second_calls = []

    async def second(*a, **k):
        second_calls.append(1)
        yield StreamDelta(content="OTHER")

    got = []

    async def consume():
        async for d in stream_chat(messages=_MSGS, config=_cfg(), restart_on_failure=True):
            got.append(d.content)

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": slow, "groq": second}):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), 0.1)
    assert got == ["a"] and state["closed"] and second_calls == []
    assert breaker.state("anthropic") == "closed"


async def test_stream_open_backoff_beyond_deadline_gives_up_without_sleeping():
    async def flaky(*a, **k):
        raise _blip()
        yield  # pragma: no cover

    cfg = _cfg(provider_order="anthropic", retry_base_delay_seconds=30.0)
    started = time.monotonic()
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": flaky}):
        with pytest.raises(LLMError):
            async for _ in stream_chat(messages=_MSGS, config=cfg, deadline_seconds=1.0):
                pass
    assert time.monotonic() - started < 1.0


async def test_retry_after_date_without_zone_is_read_as_utc():
    when = datetime.now(UTC) + timedelta(seconds=20)
    got = retry_after_seconds(_err({"retry-after": when.strftime("%d %b %Y %H:%M:%S -0000")}))
    assert got is not None and 15 <= got <= 20


async def test_scrub_http_object_survives_hostile_headers_and_url():
    from llm_gateway.errors import _scrub_http_object

    class Boom:
        def items(self):
            raise RuntimeError("no")

    class Url:
        def __str__(self):
            raise RuntimeError("no")

    class Obj:
        method = "POST"
        headers = Boom()
        url = Url()

    _scrub_http_object(Obj(), None, set())  # best effort: never raises
