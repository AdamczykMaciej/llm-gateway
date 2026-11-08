"""Size caps on streamed output: a hostile or buggy provider cannot make the
gateway buffer unbounded tool-call arguments or text."""

from unittest.mock import patch

import pytest

from llm_gateway import (
    GatewayConfig,
    InvalidOutputError,
    StreamTooLargeError,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers.base import StreamDelta

pytestmark = pytest.mark.asyncio

_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw) -> GatewayConfig:
    return GatewayConfig(anthropic_api_key="k", **kw)


async def _run(deltas, **cfg):
    async def gen():
        for d in deltas:
            yield d

    out = []
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": lambda *a, **k: gen()}):
        async for d in stream_chat(
            messages=_MESSAGES, config=_cfg(**cfg), force_provider="anthropic"
        ):
            out.append(d)
    return out


def _args(index, text, first=False):
    frag = {"index": index, "function": {"arguments": text}}
    if first:
        frag.update(id=f"c{index}", type="function")
        frag["function"]["name"] = "f"
    return StreamDelta(tool_call_deltas=[frag])


async def test_tool_arguments_over_cap_end_stream_with_typed_error():
    deltas = [_args(0, "", first=True)] + [_args(0, "x" * 40) for _ in range(5)]
    with pytest.raises(StreamTooLargeError) as ei:
        await _run(deltas, stream_max_tool_arguments_chars=100)
    assert isinstance(ei.value, InvalidOutputError)
    assert "x" * 10 not in str(ei.value)


async def test_cap_is_per_tool_call():
    deltas = [_args(0, "", first=True), _args(1, "", first=True)]
    deltas += [_args(0, "x" * 80), _args(1, "y" * 80)]
    out = await _run(deltas, stream_max_tool_arguments_chars=100)
    assert len(out) == 4


async def test_total_text_over_cap_raises():
    deltas = [StreamDelta(content="a" * 30) for _ in range(5)]
    with pytest.raises(StreamTooLargeError):
        await _run(deltas, stream_max_total_chars=100)


async def test_within_caps_unchanged():
    out = await _run([StreamDelta(content="a" * 50), StreamDelta(finish_reason="stop")])
    assert [d.content for d in out] == ["a" * 50, None]


async def test_zero_disables():
    out = await _run(
        [StreamDelta(content="a" * 500)],
        stream_max_total_chars=0,
        stream_max_tool_arguments_chars=0,
    )
    assert len(out) == 1
