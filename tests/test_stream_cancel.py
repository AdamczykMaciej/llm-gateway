"""Streaming cancellation: the upstream stream is closed before the metering
hook runs, and an attempt cut short is marked partial. Fakes only."""

import asyncio
from unittest.mock import patch

import pytest

from llm_gateway import GatewayConfig, reset_circuit_breakers, stream_chat
from llm_gateway.providers.base import StreamDelta

pytestmark = pytest.mark.asyncio

_MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw) -> GatewayConfig:
    return GatewayConfig(anthropic_api_key="k", provider_order="anthropic", retry_attempts=1, **kw)


def _tracked(state, *items):
    async def gen(*a, **k):
        try:
            for item in items:
                if isinstance(item, BaseException):
                    raise item
                yield item
            await asyncio.sleep(3600)
        finally:
            state["closed"] = True

    return gen


async def test_upstream_closed_before_hook_runs():
    state = {"closed": False}
    seen = []

    async def hook(report):
        seen.append(state["closed"])
        await asyncio.sleep(0)

    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": _tracked(state, StreamDelta(content="a"), StreamDelta(content="b"))},
    ):
        stream = stream_chat(messages=_MSGS, config=_cfg(), on_attempt=hook)
        await stream.__anext__()
        await stream.aclose()
    assert seen == [True]


async def test_upstream_closed_even_if_hook_is_cancelled():
    state = {"closed": False}

    async def hook(report):
        raise asyncio.CancelledError

    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": _tracked(state, StreamDelta(content="a"))},
    ):
        stream = stream_chat(messages=_MSGS, config=_cfg(), on_attempt=hook)
        await stream.__anext__()
        try:
            await stream.aclose()
        except asyncio.CancelledError:
            pass
    assert state["closed"]


async def test_cancelled_stream_without_usage_is_marked_partial():
    state = {"closed": False}
    reports = []
    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": _tracked(state, StreamDelta(content="abc"), StreamDelta(content="de"))},
    ):
        stream = stream_chat(messages=_MSGS, config=_cfg(), on_attempt=reports.append)
        await stream.__anext__()
        await stream.__anext__()
        await stream.aclose()
    (r,) = reports
    assert r.outcome == "cancelled" and r.usage is None
    assert r.partial is True and r.streamed_chars == 5


async def test_mid_stream_failure_is_partial_and_complete_stream_is_not():
    reports = []
    err = {"anthropic": _tracked({}, StreamDelta(content="ab"), RuntimeError("cut"))}
    with patch("llm_gateway.streaming.STREAM_CALLS", err):
        with pytest.raises(RuntimeError):
            async for _ in stream_chat(messages=_MSGS, config=_cfg(), on_attempt=reports.append):
                pass
    assert reports[0].partial is True and reports[0].streamed_chars == 2

    ok = {
        "anthropic": _tracked(
            {}, StreamDelta(content="ab"), StreamDelta(finish_reason="stop", usage=(1, 2))
        )
    }

    async def finite(*a, **k):
        yield StreamDelta(content="ab")
        yield StreamDelta(finish_reason="stop", usage=(1, 2))

    del ok
    reports.clear()
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": finite}):
        async for _ in stream_chat(messages=_MSGS, config=_cfg(), on_attempt=reports.append):
            pass
    assert reports[0].outcome == "succeeded" and reports[0].partial is False
