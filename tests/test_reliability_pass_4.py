"""Reliability pass 4: cancellation and resource cleanup (fakes only)."""

import asyncio
from unittest.mock import patch

import pytest

from llm_gateway import GatewayConfig, LLMError, breaker, reset_circuit_breakers
from llm_gateway.chat import chat
from llm_gateway.errors import LLMDeadlineExceeded
from llm_gateway.providers.base import StreamDelta
from llm_gateway.retry import call_with_retry
from llm_gateway.router import complete_with_usage
from llm_gateway.streaming import stream_chat
from llm_gateway.tracing import outcome_of

MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw):
    base = dict(
        anthropic_api_key="k1",
        groq_api_key="k2",
        provider_order="anthropic,groq",
        retry_attempts=1,
        breaker_failure_threshold=1,
    )
    base.update(kw)
    return GatewayConfig(**base)


class Hang:
    """A provider stream that waits forever and records that it was closed."""

    def __init__(self, first: StreamDelta | None = None):
        self.first = first
        self.closed = False
        self.started = asyncio.Event()

    def __call__(self, *a, **kw):
        return self._gen()

    async def _gen(self):
        try:
            if self.first is not None:
                yield self.first
            self.started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover
        finally:
            self.closed = True


async def _no_failover(provider_calls):
    return provider_calls


async def test_first_token_timeout_closes_provider_stream():
    hang = Hang()
    calls = []

    def second(*a, **kw):
        calls.append(1)

        async def g():
            yield StreamDelta(content="ok", model="m")

        return g()

    cfg = _cfg(stream_idle_timeout_seconds=0.05)
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": hang, "groq": second}):
        got = [d async for d in stream_chat(messages=MSGS, config=cfg)]
    assert got and hang.closed


async def test_cancel_during_first_token_wait_closes_stream_no_failover_no_breaker():
    hang = Hang()
    calls = []

    def second(*a, **kw):
        calls.append(1)
        raise AssertionError("failover on cancel")

    async def consume():
        async for _ in stream_chat(messages=MSGS, config=_cfg()):
            pass

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": hang, "groq": second}):
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert hang.closed and not calls
    assert not breaker.is_open("anthropic")


async def test_cancel_mid_stream_closes_no_failure_and_cancelled_span():
    hang = Hang(first=StreamDelta(content="a", model="m"))
    spans = []

    async def consume():
        async for _ in stream_chat(messages=MSGS, config=_cfg()):
            pass

    def capture(*a, **kw):
        spans.append(kw.get("error"))

    with (
        patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": hang}),
        patch("llm_gateway.streaming.emit_attempt_span", capture),
    ):
        task = asyncio.create_task(consume())
        await asyncio.wait_for(hang.started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert hang.closed
    assert not breaker.is_open("anthropic")
    assert len(spans) == 1 and isinstance(spans[0], asyncio.CancelledError)
    assert outcome_of(spans[0]) == "cancelled"


async def test_consumer_aclose_mid_stream_is_cancelled_span_not_ok():
    hang = Hang(first=StreamDelta(content="a", model="m"))
    spans = []
    with (
        patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": hang}),
        patch("llm_gateway.streaming.emit_attempt_span", lambda *a, **kw: spans.append(kw)),
    ):
        gen = stream_chat(messages=MSGS, config=_cfg())
        await gen.__anext__()
        await gen.aclose()
    assert hang.closed
    assert outcome_of(spans[0]["error"]) == "cancelled"
    assert not breaker.is_open("anthropic")


async def test_failing_provider_close_does_not_mask_consumer_aclose():
    class BadClose(Hang):
        async def _gen(self):
            try:
                yield StreamDelta(content="a", model="m")
                yield StreamDelta(content="b", model="m")
            finally:
                raise RuntimeError("close blew up")

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": BadClose()}):
        gen = stream_chat(messages=MSGS, config=_cfg())
        await gen.__anext__()
        await gen.aclose()  # must not raise the provider's close error
    assert not breaker.is_open("anthropic")


async def test_idle_timeout_mid_stream_counts_deadline_does_not():
    async def idle(*a, **kw):
        yield StreamDelta(content="a", model="m")
        raise TimeoutError("idle")

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": idle}):
        with pytest.raises(TimeoutError):
            async for _ in stream_chat(messages=MSGS, config=_cfg()):
                pass
    assert breaker.is_open("anthropic")  # provider-level timeout: counted

    reset_circuit_breakers()
    hang = Hang(first=StreamDelta(content="a", model="m"))
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": hang}):
        with pytest.raises(LLMDeadlineExceeded):
            async for _ in stream_chat(messages=MSGS, config=_cfg(call_deadline_seconds=0.1)):
                pass
    assert hang.closed and not breaker.is_open("anthropic")


async def test_retry_backoff_aborts_on_cancel():
    class Boom(Exception):
        pass

    async def fail():
        raise Boom

    task = asyncio.create_task(
        call_with_retry(fail, attempts=3, base_delay_seconds=60, should_retry=lambda e: True)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)


async def _probe_state_after_cancel(run):
    breaker.record_failure("anthropic", threshold=1, cooldown_seconds=0.05, model="m")
    breaker.record_failure("anthropic", threshold=1, cooldown_seconds=0.05)
    await asyncio.sleep(0.08)  # cooldown over: half-open
    started = asyncio.Event()

    async def hang(*a, **kw):
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(run(hang))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_complete_cancel_releases_half_open_probe():
    cfg = _cfg(provider_order="anthropic", anthropic_model="m")

    async def run(hang):
        with patch("llm_gateway.router.CALLS", {"anthropic": hang}):
            await complete_with_usage(system="s", prompt="p", config=cfg, force_provider=None)

    await _probe_state_after_cancel(run)
    assert not breaker.is_open("anthropic")  # lease released, not held for a cooldown


async def test_chat_cancel_releases_half_open_probe_and_emits_span():
    cfg = _cfg(provider_order="anthropic")
    spans = []

    async def run(hang):
        with (
            patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": hang}),
            patch("llm_gateway.chat.emit_attempt_span", lambda *a, **kw: spans.append(kw)),
        ):
            await chat(messages=MSGS, config=cfg)

    await _probe_state_after_cancel(run)
    assert not breaker.is_open("anthropic")
    assert outcome_of(spans[0]["error"]) == "cancelled"


async def test_stream_preflight_cancel_releases_probe():
    cfg = _cfg(provider_order="anthropic")
    hang = Hang()

    async def run(_):
        with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": hang}):
            async for _ in stream_chat(messages=MSGS, config=cfg):
                pass

    breaker.record_failure("anthropic", threshold=1, cooldown_seconds=0.05)
    await asyncio.sleep(0.08)
    task = asyncio.create_task(run(None))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not breaker.is_open("anthropic")


async def test_thousand_cancelled_calls_leak_nothing():
    cfg = _cfg(provider_order="anthropic")
    inflight = 0
    peak = 0

    async def slow(*a, **kw):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            await asyncio.Event().wait()
        finally:
            inflight -= 1

    before = len(asyncio.all_tasks())
    with patch("llm_gateway.router.CALLS", {"anthropic": slow}):
        tasks = [
            asyncio.create_task(complete_with_usage(system="s", prompt="p", config=cfg))
            for _ in range(1000)
        ]
        await asyncio.sleep(0.1)
        assert inflight == 1000
        for t in tasks:
            t.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, asyncio.CancelledError) for r in results)
    assert inflight == 0
    assert not breaker._failures and not breaker._probe_until and not breaker._opened_until
    assert len(asyncio.all_tasks()) == before


async def test_cancel_records_no_cost():
    cfg = _cfg(provider_order="anthropic")
    costs = []

    async def slow(*a, **kw):
        await asyncio.Event().wait()

    with (
        patch("llm_gateway.router.CALLS", {"anthropic": slow}),
        patch("llm_gateway.router.record_cost", lambda *a, **k: costs.append(a)),
    ):
        t = asyncio.create_task(complete_with_usage(system="s", prompt="p", config=cfg))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    assert costs == [] and not breaker.is_open("anthropic")
    assert LLMError  # imported for type context
