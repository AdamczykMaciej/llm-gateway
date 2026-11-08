"""Opt-in mid-stream restart (`restart_on_failure`). Fakes only: no provider is
ever called."""

import asyncio
from unittest.mock import patch

import pytest

from llm_gateway import (
    AllProvidersExhaustedError,
    GatewayConfig,
    InvalidOutputError,
    LLMDeadlineExceeded,
    StreamTooLargeError,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers.base import StreamDelta

pytestmark = pytest.mark.asyncio

_MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw) -> GatewayConfig:
    kw.setdefault("provider_order", "anthropic,groq,openai")
    return GatewayConfig(
        anthropic_api_key="k", groq_api_key="k", openai_api_key="k", retry_attempts=1, **kw
    )


def _text(*pieces, usage=(3, 4)):
    async def gen(*a, **k):
        for p in pieces:
            yield StreamDelta(content=p)
        yield StreamDelta(finish_reason="stop", usage=usage)

    return gen


def _dies(*pieces, error=None):
    err = error if error is not None else ConnectionResetError("boom")

    async def gen(*a, **k):
        for p in pieces:
            yield StreamDelta(content=p)
        raise err

    return gen


async def _collect(calls, *, kwargs=None, cfg=None):
    out: list[StreamDelta] = []
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        async for d in stream_chat(messages=_MSGS, config=cfg or _cfg(), **(kwargs or {})):
            out.append(d)
    return out


async def test_midstream_death_restarts_on_next_provider_with_marker():
    calls = {"anthropic": _dies("par", "tial"), "groq": _text("full ", "reply")}
    out = await _collect(calls, kwargs={"restart_on_failure": True})
    assert [d.content for d in out if not d.restart] == ["par", "tial", "full ", "reply", None]
    marker_at = [i for i, d in enumerate(out) if d.restart]
    assert marker_at == [2]  # after the void chunks, before the new attempt
    marker = out[2]
    assert marker.content is None and marker.tool_call_deltas is None
    assert marker.provider == "groq" and marker.restart_attempt == 1
    assert marker.failed_provider == "anthropic"
    assert marker.failed_error == "ConnectionResetError"
    assert out[-1].usage == (3, 4) and out[-1].provider == "groq"


async def test_default_is_off_and_unchanged():
    calls = {"anthropic": _dies("par"), "groq": _text("x")}
    out: list[StreamDelta] = []
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(ConnectionResetError):
            async for d in stream_chat(messages=_MSGS, config=_cfg()):
                out.append(d)
    assert [d.content for d in out] == ["par"]
    assert not any(d.restart for d in out)


async def test_config_default_enables_it_and_call_can_disable():
    calls = {"anthropic": _dies("par"), "groq": _text("ok")}
    out = await _collect(calls, cfg=_cfg(stream_restart_on_failure=True))
    assert any(d.restart for d in out)
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(ConnectionResetError):
            async for _ in stream_chat(
                messages=_MSGS,
                config=_cfg(stream_restart_on_failure=True),
                restart_on_failure=False,
            ):
                pass


async def test_consumer_reading_only_content_gets_marker_as_empty():
    calls = {"anthropic": _dies("a"), "groq": _text("b")}
    out = await _collect(calls, kwargs={"restart_on_failure": True})
    # A naive consumer: marker has no content and no tool deltas, so it is inert.
    text = "".join(d.content or "" for d in out)
    assert text == "ab"  # (naive consumers keep the void text: that is why it is opt-in)


async def test_restart_limit_reached_raises_original_error():
    calls = {
        "anthropic": _dies("a"),
        "groq": _dies("b", error=TimeoutError("idle")),
        "openai": _text("c"),
    }
    out: list[StreamDelta] = []
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(TimeoutError, match="idle"):
            async for d in stream_chat(messages=_MSGS, config=_cfg(), restart_on_failure=True):
                out.append(d)
    assert sum(d.restart for d in out) == 1  # default max 1: openai never reached


async def test_max_restarts_two_reaches_third_provider():
    calls = {"anthropic": _dies("a"), "groq": _dies("b"), "openai": _text("c")}
    out = await _collect(
        calls, kwargs={"restart_on_failure": True}, cfg=_cfg(stream_max_restarts=2)
    )
    assert [d.restart_attempt for d in out if d.restart] == [1, 2]
    assert out[-1].provider == "openai"


async def test_no_replacement_available_raises_the_midstream_error():
    calls = {"anthropic": _dies("a"), "groq": _boom_on_open()}
    cfg = _cfg(provider_order="anthropic,groq")
    out: list[StreamDelta] = []
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(ConnectionResetError, match="boom"):
            async for d in stream_chat(messages=_MSGS, config=cfg, restart_on_failure=True):
                out.append(d)
    assert not any(d.restart for d in out)


def _boom_on_open():
    async def gen(*a, **k):
        raise ConnectionResetError("nope")
        yield  # pragma: no cover

    return gen


async def test_cancellation_midstream_never_restarts():
    started = asyncio.Event()
    second_called = []

    async def slow(*a, **k):
        yield StreamDelta(content="a")
        started.set()
        await asyncio.sleep(30)
        yield StreamDelta(content="never")

    def second(*a, **k):
        second_called.append(1)
        return _text("b")()

    async def consume():
        async for _ in stream_chat(messages=_MSGS, config=_cfg(), restart_on_failure=True):
            pass

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": slow, "groq": second}):
        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert second_called == []


@pytest.mark.parametrize(
    "error",
    [
        InvalidOutputError("bad"),
        StreamTooLargeError("big", limit=1),
    ],
)
async def test_invalid_output_and_oversize_never_restart(error):
    calls = {"anthropic": _dies("a", error=error), "groq": _text("b")}
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(type(error)):
            async for _ in stream_chat(messages=_MSGS, config=_cfg(), restart_on_failure=True):
                pass


async def test_content_filter_cutoff_never_restarts():
    async def filtered(*a, **k):
        yield StreamDelta(content="some")
        yield StreamDelta(finish_reason="content_filter", usage=(1, 1))

    calls = {"anthropic": filtered, "groq": _text("b")}
    out = await _collect(calls, kwargs={"restart_on_failure": True})
    assert not any(d.restart for d in out)
    assert out[-1].finish_reason == "content_filter"


async def test_filtered_empty_completion_error_never_restarts():
    from llm_gateway import EmptyCompletionError

    err = EmptyCompletionError(
        provider="anthropic", model="m", finish_reason="content_filter", reasoning_tokens=0
    )
    calls = {"anthropic": _dies("a", error=err), "groq": _text("b")}
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(EmptyCompletionError):
            async for _ in stream_chat(messages=_MSGS, config=_cfg(), restart_on_failure=True):
                pass


async def test_deadline_exhausted_midstream_is_not_restarted():
    async def stalls(*a, **k):
        yield StreamDelta(content="a")
        await asyncio.sleep(30)
        yield StreamDelta(content="never")

    calls = {"anthropic": stalls, "groq": _text("b")}
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(LLMDeadlineExceeded):
            async for _ in stream_chat(
                messages=_MSGS, config=_cfg(call_deadline_seconds=0.2), restart_on_failure=True
            ):
                pass


async def test_no_restart_when_deadline_expired_by_failing_attempt():
    async def slow_die(*a, **k):
        yield StreamDelta(content="a")
        await asyncio.sleep(0.3)
        raise ConnectionResetError("late")

    calls = {"anthropic": slow_die, "groq": _text("b")}
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises((LLMDeadlineExceeded, ConnectionResetError)):
            async for _ in stream_chat(
                messages=_MSGS, config=_cfg(call_deadline_seconds=0.25), restart_on_failure=True
            ):
                pass


async def test_on_attempt_sees_both_attempts_and_breaker_counts_failure():
    reports = []
    calls = {"anthropic": _dies("a"), "groq": _text("b", usage=(5, 6))}
    out = await _collect(
        calls,
        kwargs={"restart_on_failure": True, "on_attempt": reports.append},
    )
    assert [(r.attempt, r.provider, r.outcome, r.is_fallback) for r in reports] == [
        (1, "anthropic", "failed", False),
        (2, "groq", "succeeded", True),
    ]
    assert reports[0].error_type == "ConnectionResetError"
    assert reports[1].usage is not None and reports[1].usage.output_tokens == 6
    assert out[-1].usage == (5, 6)

    from llm_gateway import breaker

    assert breaker._state  # failure was recorded against a breaker


async def test_tool_call_partials_are_voided_by_marker():
    frag = {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": "{"}}

    async def partial_tools(*a, **k):
        yield StreamDelta(tool_call_deltas=[frag])
        raise ConnectionResetError("x")

    full = {
        "index": 0,
        "id": "c2",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }

    async def good(*a, **k):
        yield StreamDelta(tool_call_deltas=[full])
        yield StreamDelta(finish_reason="tool_calls", usage=(1, 1))

    out = await _collect(
        {"anthropic": partial_tools, "groq": good}, kwargs={"restart_on_failure": True}
    )
    after = out[[d.restart for d in out].index(True) + 1 :]
    assert [f["id"] for d in after for f in d.tool_call_deltas or []] == ["c2"]


async def test_empty_replacement_stream_still_emits_marker():
    async def empty(*a, **k):
        return
        yield  # pragma: no cover

    out = await _collect(
        {"anthropic": _dies("a"), "groq": empty}, kwargs={"restart_on_failure": True}
    )
    assert [d.restart for d in out] == [False, True]


async def test_all_still_exhausted_error_when_first_provider_fails_before_chunks():
    calls = {"anthropic": _boom_on_open(), "groq": _boom_on_open(), "openai": _boom_on_open()}
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(AllProvidersExhaustedError):
            async for _ in stream_chat(messages=_MSGS, config=_cfg(), restart_on_failure=True):
                pass
