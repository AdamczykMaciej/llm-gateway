"""idempotency_key on stream_chat / complete / complete_with_usage, plus the
fallback-reason and breaker-state attempt-span attributes. Fakes only."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_gateway import (
    GatewayConfig,
    InvalidRequestError,
    chat,
    complete,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers.base import ChatResult, ProviderResult, StreamDelta

pytestmark = pytest.mark.asyncio

_MSGS = [{"role": "user", "content": "hi"}]
_PR = ProviderResult("hello", "m", 1, 1)
_OK_CHAT = ChatResult(content="ok", model="m", input_tokens=1, output_tokens=1)


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(order="openai") -> GatewayConfig:
    return GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        openai_api_key="k",
        retry_attempts=1,
        provider_order=order,
    )


def _stream_fn(seen: list | None = None, gate: asyncio.Event | None = None):
    async def gen(config, messages, tools, max_tokens, **kwargs):
        if seen is not None:
            seen.append(kwargs)
        if gate is not None:
            await gate.wait()
        yield StreamDelta(content="a")
        yield StreamDelta(finish_reason="stop", usage=(1, 1))

    return gen


async def _drain(**kw):
    return [d async for d in stream_chat(messages=_MSGS, config=_cfg(), **kw)]


# -- validation


async def test_bad_key_rejected_everywhere():
    with pytest.raises(InvalidRequestError):
        await _drain(idempotency_key="bad key!")
    with pytest.raises(InvalidRequestError):
        await complete(system="s", prompt="p", config=_cfg(), idempotency_key="bad key!")
    with pytest.raises(InvalidRequestError):
        await complete_with_usage(system="s", prompt="p", config=_cfg(), idempotency_key="")


# -- forwarding and AttemptReport


async def test_stream_key_reports_and_forwards_to_openai_only():
    reports = []
    seen_openai, seen_groq = [], []

    async def boom(*a, **k):
        seen_openai.append(k)
        raise RuntimeError("boom")
        yield  # pragma: no cover

    streams = {"openai": boom, "groq": _stream_fn(seen_groq)}
    with patch("llm_gateway.streaming.STREAM_CALLS", streams):
        cfg = _cfg("openai,groq")
        out = [
            d
            async for d in stream_chat(
                messages=_MSGS, config=cfg, idempotency_key="s-1", on_attempt=reports.append
            )
        ]
    assert out
    assert [r.idempotency_key for r in reports] == ["s-1", "s-1"]
    assert seen_openai[0]["idempotency_key"] == "s-1"
    assert "idempotency_key" not in seen_groq[0]


async def test_stream_without_key_unchanged():
    seen = []
    with patch("llm_gateway.streaming.STREAM_CALLS", {"openai": _stream_fn(seen)}):
        await _drain()
    assert "idempotency_key" not in seen[0]


async def test_complete_key_reports_and_forwards_to_openai_only():
    reports = []
    openai_call = AsyncMock(side_effect=RuntimeError("boom"))
    groq_call = AsyncMock(return_value=_PR)
    with patch("llm_gateway.router.CALLS", {"openai": openai_call, "groq": groq_call}):
        text = await complete(
            system="s",
            prompt="p",
            config=_cfg("openai,groq"),
            idempotency_key="c-1",
            on_attempt=reports.append,
        )
    assert text == "hello"
    assert [r.idempotency_key for r in reports] == ["c-1", "c-1"]
    assert openai_call.call_args.kwargs["idempotency_key"] == "c-1"
    assert "idempotency_key" not in groq_call.call_args.kwargs


async def test_complete_without_key_passes_nothing():
    call = AsyncMock(return_value=_PR)
    with patch("llm_gateway.router.CALLS", {"openai": call}):
        await complete_with_usage(system="s", prompt="p", config=_cfg())
    assert "idempotency_key" not in call.call_args.kwargs


async def test_openai_call_and_stream_send_header():
    from unittest.mock import MagicMock

    from llm_gateway.providers import openai as op

    create = AsyncMock(return_value=MagicMock())
    client = MagicMock()
    client.chat.completions.create = create
    with (
        patch.object(op, "_client", return_value=client),
        patch.object(op, "provider_result_from_openai_style", return_value=_PR),
    ):
        await op.call(_cfg(), "s", "p", 10, idempotency_key="h-1")
        assert create.call_args.kwargs["extra_headers"] == {"Idempotency-Key": "h-1"}
        await op.call(_cfg(), "s", "p", 10)
        assert "extra_headers" not in create.call_args.kwargs

    async def fake_chunks():
        return
        yield  # pragma: no cover

    create2 = AsyncMock(return_value=fake_chunks())
    client.chat.completions.create = create2
    with patch.object(op, "_client", return_value=client):
        gen = op._stream_chat(_cfg(), _MSGS, None, 10, idempotency_key="h-2")
        async for _ in gen:
            pass
    assert create2.call_args.kwargs["extra_headers"] == {"Idempotency-Key": "h-2"}


# -- complete dedupe


async def test_concurrent_complete_shares_one_call_and_conflicts():
    gate = asyncio.Event()

    async def slow(*a, **k):
        await gate.wait()
        return _PR

    call = AsyncMock(side_effect=slow)
    with patch("llm_gateway.router.CALLS", {"openai": call}):
        t1 = asyncio.create_task(
            complete_with_usage(system="s", prompt="p", config=_cfg(), idempotency_key="d")
        )
        t2 = asyncio.create_task(
            complete(system="s", prompt="p", config=_cfg(), idempotency_key="d")
        )
        await asyncio.sleep(0.01)
        with pytest.raises(InvalidRequestError):
            await complete(system="s", prompt="OTHER", config=_cfg(), idempotency_key="d")
        gate.set()
        r1, r2 = await asyncio.gather(t1, t2)
        await complete(system="s", prompt="p", config=_cfg(), idempotency_key="d")
    assert r1.text == r2 == "hello"
    assert call.await_count == 2  # one shared call, then a later call (nothing cached)


async def test_same_key_on_chat_and_complete_do_not_collide():
    gate = asyncio.Event()

    async def slow_chat(*a, **k):
        await gate.wait()
        return ChatResult(content="c", model="m", input_tokens=1, output_tokens=1)

    async def slow_complete(*a, **k):
        await gate.wait()
        return _PR

    with (
        patch("llm_gateway.chat.CHAT_CALLS", {"openai": AsyncMock(side_effect=slow_chat)}),
        patch("llm_gateway.router.CALLS", {"openai": AsyncMock(side_effect=slow_complete)}),
    ):
        t1 = asyncio.create_task(chat(messages=_MSGS, config=_cfg(), idempotency_key="x"))
        t2 = asyncio.create_task(
            complete(system="s", prompt="p", config=_cfg(), idempotency_key="x")
        )
        await asyncio.sleep(0.01)
        gate.set()
        assert (await t1).content == "c"
        assert await t2 == "hello"


# -- stream: second concurrent call with the same key is rejected


async def test_concurrent_stream_same_key_is_rejected_and_first_unaffected():
    gate = asyncio.Event()
    seen = []
    with patch("llm_gateway.streaming.STREAM_CALLS", {"openai": _stream_fn(seen, gate)}):
        first = asyncio.create_task(_drain(idempotency_key="st"))
        await asyncio.sleep(0.01)
        with pytest.raises(InvalidRequestError, match="idempotency_key"):
            await _drain(idempotency_key="st")
        gate.set()
        assert len(await first) == 2
        # released afterwards: the same key may stream again
        assert len(await _drain(idempotency_key="st")) == 2
    assert len(seen) == 2  # the rejected call never reached a provider


async def test_stream_key_released_on_early_close_and_error():
    with patch("llm_gateway.streaming.STREAM_CALLS", {"openai": _stream_fn()}):
        gen = stream_chat(messages=_MSGS, config=_cfg(), idempotency_key="e")
        await gen.__anext__()
        await gen.aclose()
        assert len(await _drain(idempotency_key="e")) == 2

    async def boom(*a, **k):
        raise RuntimeError("x")
        yield  # pragma: no cover

    with patch("llm_gateway.streaming.STREAM_CALLS", {"openai": boom}):
        with pytest.raises(Exception):  # noqa: B017
            await _drain(idempotency_key="e2")
    with patch("llm_gateway.streaming.STREAM_CALLS", {"openai": _stream_fn()}):
        assert len(await _drain(idempotency_key="e2")) == 2


async def test_stream_different_keys_run_concurrently():
    gate = asyncio.Event()
    with patch("llm_gateway.streaming.STREAM_CALLS", {"openai": _stream_fn(None, gate)}):
        t1 = asyncio.create_task(_drain(idempotency_key="k1"))
        t2 = asyncio.create_task(_drain(idempotency_key="k2"))
        await asyncio.sleep(0.01)
        gate.set()
        assert len(await t1) == 2 and len(await t2) == 2


# -- span attributes: span, fallback reason, breaker state


def _tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("t"), exporter


def _attempts(exporter):
    return sorted(
        (s for s in exporter.get_finished_spans() if s.name == "llm_gateway.attempt"),
        key=lambda s: s.attributes["llm_gateway.attempt"],
    )


async def test_chat_attempt_spans_have_reason_and_breaker_state():
    tracer, exporter = _tracer()
    calls = {
        "openai": AsyncMock(side_effect=TimeoutError("x")),
        "groq": AsyncMock(
            return_value=ChatResult(content="ok", model="m", input_tokens=1, output_tokens=1)
        ),
    }
    with (
        patch("llm_gateway.chat.get_tracer", return_value=tracer),
        patch("llm_gateway.chat.CHAT_CALLS", calls),
    ):
        await chat(messages=_MSGS, config=_cfg("openai,groq"))
    a, b = _attempts(exporter)
    assert "llm_gateway.fallback_reason" not in a.attributes
    assert a.attributes["llm_gateway.breaker_state"] == "closed"
    assert b.attributes["llm_gateway.fallback_reason"] == "timeout"
    assert b.attributes["llm_gateway.breaker_state"] == "closed"


async def test_stream_and_complete_spans_have_idempotency_reason_and_breaker():
    tracer, exporter = _tracer()

    async def boom(*a, **k):
        raise TimeoutError("x")
        yield  # pragma: no cover

    with (
        patch("llm_gateway.streaming.get_tracer", return_value=tracer),
        patch("llm_gateway.streaming.STREAM_CALLS", {"openai": boom, "groq": _stream_fn()}),
    ):
        async for _ in stream_chat(
            messages=_MSGS, config=_cfg("openai,groq"), idempotency_key="sp-1"
        ):
            pass
    a, b = _attempts(exporter)
    assert a.attributes["llm_gateway.idempotency_key"] == "sp-1"
    assert b.attributes["llm_gateway.idempotency_key"] == "sp-1"
    assert b.attributes["llm_gateway.fallback_reason"] == "timeout"
    assert a.attributes["llm_gateway.breaker_state"] == "closed"
    (root,) = (s for s in exporter.get_finished_spans() if s.name == "llm_gateway.stream_chat")
    assert root.attributes["llm_gateway.idempotency_key"] == "sp-1"

    tracer, exporter = _tracer()
    calls = {
        "openai": AsyncMock(side_effect=TimeoutError("x")),
        "groq": AsyncMock(return_value=_PR),
    }
    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch("llm_gateway.router.CALLS", calls),
    ):
        await complete(system="s", prompt="p", config=_cfg("openai,groq"), idempotency_key="sp-2")
    a, b = _attempts(exporter)
    assert a.attributes["llm_gateway.idempotency_key"] == "sp-2"
    assert b.attributes["llm_gateway.fallback_reason"] == "timeout"
    (root,) = (s for s in exporter.get_finished_spans() if s.name == "llm_gateway.complete")
    assert root.attributes["llm_gateway.idempotency_key"] == "sp-2"


async def test_breaker_probe_state_is_reported():
    from llm_gateway import breaker

    tracer, exporter = _tracer()
    cfg = _cfg("openai")
    breaker.record_failure("openai", threshold=1, cooldown_seconds=0.01)
    await asyncio.sleep(0.05)  # cooldown over -> half-open probe
    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch("llm_gateway.router.CALLS", {"openai": AsyncMock(return_value=_PR)}),
    ):
        await complete(system="s", prompt="p", config=cfg)
    (a,) = _attempts(exporter)
    assert a.attributes["llm_gateway.breaker_state"] == "probe"
