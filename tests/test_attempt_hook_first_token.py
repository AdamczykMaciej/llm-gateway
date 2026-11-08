"""Per-attempt usage hook (`on_attempt`) and `first_token_timeout_seconds`.
Fakes only: no provider is ever called."""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

from llm_gateway import (
    AllProvidersExhaustedError,
    FirstTokenTimeoutError,
    GatewayConfig,
    ModelPrice,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers.base import ChatResult, ProviderResult, StreamDelta, Usage

pytestmark = pytest.mark.asyncio

_MSGS = [{"role": "user", "content": "SECRET-PROMPT"}]
_PRICES = {"anthropic/m1": ModelPrice(input_per_mtok=1_000_000, output_per_mtok=2_000_000)}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw) -> GatewayConfig:
    kw.setdefault("provider_order", "anthropic,groq")
    return GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        claude_model="m1",
        retry_attempts=1,
        model_prices=_PRICES,
        **kw,
    )


def _ok(model="m1", i=3, o=4):
    return ProviderResult(text="REPLY-TEXT", model=model, input_tokens=i, output_tokens=o)


# ── on_attempt: complete ────────────────────────────────────────────────


async def test_failover_reports_every_attempt_with_cost():
    reports = []
    calls = {
        "anthropic": AsyncMock(side_effect=RuntimeError("boom SECRET-PROMPT")),
        "groq": AsyncMock(return_value=_ok("g1")),
    }
    with patch("llm_gateway.router.CALLS", calls):
        done = await complete_with_usage(
            system="s", prompt="p", config=_cfg(), request_id="rid-1", on_attempt=reports.append
        )
    first, second = reports
    assert (first.provider, first.attempt, first.outcome) == ("anthropic", 1, "failed")
    assert first.error_type == "RuntimeError" and first.usage is None and first.cost_usd is None
    assert first.request_id == "rid-1" and not first.is_fallback
    assert (second.provider, second.model, second.attempt) == ("groq", "g1", 2)
    assert second.outcome == "succeeded" and second.is_fallback
    assert second.usage == done.usage and second.cost_usd == done.cost_usd
    assert "SECRET" not in repr(reports) and "REPLY" not in repr(reports)


async def test_success_cost_from_price_table_and_config_hook_async():
    reports = []

    async def hook(report):
        reports.append(report)

    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=_ok())}):
        await complete_with_usage(system="s", prompt="p", config=_cfg(on_attempt=hook))
    assert reports[0].cost_usd == pytest.approx(3 + 8)


async def test_per_call_hook_replaces_config_hook():
    cfg_calls, call_calls = [], []
    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=_ok())}):
        await complete_with_usage(
            system="s",
            prompt="p",
            config=_cfg(on_attempt=cfg_calls.append),
            on_attempt=call_calls.append,
        )
    assert cfg_calls == [] and len(call_calls) == 1


async def test_failed_schema_validation_reports_paid_usage():
    reports = []
    bad = ProviderResult(text="not json", model="m1", input_tokens=5, output_tokens=6)
    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=bad)}):
        with pytest.raises(AllProvidersExhaustedError):
            await complete_with_usage(
                system="s",
                prompt="p",
                config=_cfg(provider_order="anthropic"),
                output_schema={"type": "object"},
                on_attempt=reports.append,
            )
    assert reports[0].outcome == "failed"
    assert reports[0].usage is not None and reports[0].usage.input_tokens == 5
    assert reports[0].cost_usd == pytest.approx(5 + 12)


async def test_hook_raising_never_breaks_call_and_is_logged_without_content(caplog):
    def hook(report):
        raise ValueError("leaky SECRET-PROMPT")

    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=_ok())}):
        with caplog.at_level(logging.WARNING, logger="llm_gateway"):
            done = await complete_with_usage(system="s", prompt="p", config=_cfg(on_attempt=hook))
    assert done.text == "REPLY-TEXT"
    assert "ValueError" in caplog.text and "SECRET" not in caplog.text


async def test_async_hook_raising_is_swallowed():
    async def hook(report):
        raise RuntimeError("x")

    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=_ok())}):
        done = await complete_with_usage(system="s", prompt="p", config=_cfg(), on_attempt=hook)
    assert done.text == "REPLY-TEXT"


async def test_cancelled_attempt_is_reported_then_cancellation_propagates():
    reports = []
    started = asyncio.Event()

    async def slow(*a, **k):
        started.set()
        await asyncio.sleep(30)

    with patch("llm_gateway.router.CALLS", {"anthropic": slow}):
        task = asyncio.create_task(
            complete_with_usage(system="s", prompt="p", config=_cfg(), on_attempt=reports.append)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert [r.outcome for r in reports] == ["cancelled"]
    assert reports[0].usage is None


# ── on_attempt: chat ────────────────────────────────────────────────────


async def test_chat_reports_failed_then_succeeded():
    reports = []
    good = ChatResult(content="hi", model="m1", input_tokens=2, output_tokens=1)
    calls = {
        "anthropic": AsyncMock(side_effect=RuntimeError("x")),
        "groq": AsyncMock(return_value=good),
    }
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        await chat(messages=_MSGS, config=_cfg(), on_attempt=reports.append)
    assert [(r.provider, r.outcome) for r in reports] == [
        ("anthropic", "failed"),
        ("groq", "succeeded"),
    ]
    assert reports[1].usage.output_tokens == 1


# ── on_attempt: stream_chat ─────────────────────────────────────────────


async def _gen(*items):
    for item in items:
        if isinstance(item, BaseException):
            raise item
        yield item


def _usage_delta(i=7, o=9):
    return StreamDelta(usage=(i, o), usage_details=Usage(reasoning_tokens=2))


async def test_stream_mid_stream_failure_reports_partial_usage():
    reports = []
    items = [StreamDelta(content="a"), _usage_delta(), RuntimeError("cut")]
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": lambda *a, **k: _gen(*items)}):
        with pytest.raises(RuntimeError):
            async for _ in stream_chat(
                messages=_MSGS, config=_cfg(provider_order="anthropic"), on_attempt=reports.append
            ):
                pass
    (r,) = reports
    assert r.outcome == "failed" and r.error_type == "RuntimeError"
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens) == (7, 9, 2)
    assert r.cost_usd == pytest.approx(7 + 18)


async def test_stream_failover_and_success_reports():
    reports = []
    calls = {
        "anthropic": lambda *a, **k: _gen(RuntimeError("down")),
        "groq": lambda *a, **k: _gen(StreamDelta(content="a"), _usage_delta()),
    }
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        async for _ in stream_chat(messages=_MSGS, config=_cfg(), on_attempt=reports.append):
            pass
    assert [(r.provider, r.outcome, r.attempt) for r in reports] == [
        ("anthropic", "failed", 1),
        ("groq", "succeeded", 2),
    ]
    assert reports[1].usage.input_tokens == 7


async def test_stream_early_close_is_cancelled():
    reports = []
    items = [StreamDelta(content="a"), StreamDelta(content="b")]
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": lambda *a, **k: _gen(*items)}):
        stream = stream_chat(
            messages=_MSGS, config=_cfg(provider_order="anthropic"), on_attempt=reports.append
        )
        await stream.__anext__()
        await stream.aclose()
    assert [r.outcome for r in reports] == ["cancelled"]


async def test_stream_hook_raising_does_not_break_stream():
    def hook(report):
        raise RuntimeError("x")

    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": lambda *a, **k: _gen(StreamDelta(content="a"))},
    ):
        out = [
            d
            async for d in stream_chat(
                messages=_MSGS, config=_cfg(provider_order="anthropic", on_attempt=hook)
            )
        ]
    assert out[0].content == "a"


# ── first_token_timeout_seconds ─────────────────────────────────────────


async def _slow_first(*a, **k):
    await asyncio.sleep(30)
    yield StreamDelta(content="late")


async def test_slow_first_chunk_fails_over_to_next_provider():
    reports = []
    calls = {
        "anthropic": _slow_first,
        "groq": lambda *a, **k: _gen(StreamDelta(content="fast")),
    }
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        out = [
            d
            async for d in stream_chat(
                messages=_MSGS,
                config=_cfg(),
                first_token_timeout_seconds=0.05,
                on_attempt=reports.append,
            )
        ]
    assert [d.content for d in out] == ["fast"]
    assert reports[0].outcome == "failed" and reports[0].error_type == "FirstTokenTimeoutError"


async def test_first_token_timeout_from_config_and_error_type():
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": _slow_first}):
        with pytest.raises(AllProvidersExhaustedError) as ei:
            async for _ in stream_chat(
                messages=_MSGS,
                config=_cfg(provider_order="anthropic", first_token_timeout_seconds=0.05),
            ):
                pass
    assert isinstance(ei.value.__cause__, FirstTokenTimeoutError)
    assert isinstance(ei.value.__cause__, TimeoutError)


async def test_default_none_waits_for_slow_first_chunk():
    async def slowish(*a, **k):
        await asyncio.sleep(0.1)
        yield StreamDelta(content="ok")

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": slowish}):
        out = [
            d async for d in stream_chat(messages=_MSGS, config=_cfg(provider_order="anthropic"))
        ]
    assert out[0].content == "ok"


async def test_idle_bound_still_wins_when_tighter_and_is_not_first_token_error():
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": _slow_first}):
        with pytest.raises(AllProvidersExhaustedError) as ei:
            async for _ in stream_chat(
                messages=_MSGS,
                config=_cfg(
                    provider_order="anthropic",
                    stream_idle_timeout_seconds=0.05,
                    first_token_timeout_seconds=10,
                ),
            ):
                pass
    assert not isinstance(ei.value.__cause__, FirstTokenTimeoutError)
    assert isinstance(ei.value.__cause__, TimeoutError)


async def test_mid_stream_stall_is_not_a_first_token_timeout():
    async def stalls(*a, **k):
        yield StreamDelta(content="a")
        await asyncio.sleep(0.3)
        yield StreamDelta(content="b")

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": stalls}):
        out = [
            d
            async for d in stream_chat(
                messages=_MSGS,
                config=_cfg(provider_order="anthropic"),
                first_token_timeout_seconds=0.1,
            )
        ]
    assert [d.content for d in out] == ["a", "b"]
