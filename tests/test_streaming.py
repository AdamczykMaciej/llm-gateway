"""Tests for the streaming orchestrator (llm_gateway.streaming) — the
pre-first-chunk provider fallback semantics are the important thing to get
right here (see the module docstring for why fallback can only happen
before the first chunk reaches the caller)."""

from unittest.mock import patch

import pytest

from llm_gateway import GatewayConfig, LLMError, reset_circuit_breakers, stream_chat
from llm_gateway.providers.base import StreamDelta

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**overrides) -> GatewayConfig:
    defaults = dict(
        anthropic_api_key="test-anthropic-key",
        groq_api_key="test-groq-key",
        openai_api_key="test-openai-key",
    )
    defaults.update(overrides)
    return GatewayConfig(**defaults)


_MESSAGES = [{"role": "user", "content": "hi"}]


async def _gen(*deltas):
    for d in deltas:
        yield d


def _failing_gen(exc: Exception):
    async def _g(*args, **kwargs):
        raise exc
        yield  # pragma: no cover — makes this a generator function; unreachable

    return _g()


async def test_yields_all_deltas_from_first_provider():
    deltas = [
        StreamDelta(content="Hel", model="x"),
        StreamDelta(content="lo", model="x"),
        StreamDelta(finish_reason="stop", model="x", usage=(1, 2)),
    ]
    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": lambda *a, **kw: _gen(*deltas)},
    ):
        collected = [
            d
            async for d in stream_chat(
                messages=_MESSAGES, config=_config(provider_order="anthropic")
            )
        ]
    assert [d.content for d in collected if d.content] == ["Hel", "lo"]
    assert collected[-1].finish_reason == "stop"


async def test_falls_back_before_first_chunk_on_immediate_failure():
    groq_deltas = [StreamDelta(content="from groq", model="llama")]
    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {
            "anthropic": lambda *a, **kw: _failing_gen(RuntimeError("boom")),
            "groq": lambda *a, **kw: _gen(*groq_deltas),
        },
    ):
        collected = [
            d
            async for d in stream_chat(
                messages=_MESSAGES, config=_config(provider_order="anthropic,groq")
            )
        ]
    assert collected[0].content == "from groq"


async def test_no_silent_fallback_after_first_chunk_yielded():
    # Once a chunk has been yielded, a later failure from the SAME provider's
    # generator must propagate as LLMError, not silently retry elsewhere —
    # a partial response can't be un-sent.
    async def _anthropic_gen(*a, **kw):
        yield StreamDelta(content="partial", model="x")
        raise RuntimeError("mid-stream failure")

    groq_call_made = False

    def _groq_gen(*a, **kw):
        nonlocal groq_call_made
        groq_call_made = True
        return _gen(StreamDelta(content="should not appear", model="llama"))

    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": _anthropic_gen, "groq": _groq_gen},
    ):
        collected = []
        with pytest.raises(RuntimeError):
            async for d in stream_chat(
                messages=_MESSAGES, config=_config(provider_order="anthropic,groq")
            ):
                collected.append(d)
    assert collected[0].content == "partial"
    assert groq_call_made is False


async def test_circuit_breaker_shared_across_engines():
    from llm_gateway import breaker

    breaker.record_failure("anthropic", threshold=1, cooldown_seconds=60.0)
    anthropic_called = False

    def _anthropic_gen(*a, **kw):
        nonlocal anthropic_called
        anthropic_called = True
        return _gen(StreamDelta(content="should be skipped", model="x"))

    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {
            "anthropic": _anthropic_gen,
            "groq": lambda *a, **kw: _gen(StreamDelta(content="ok", model="llama")),
        },
    ):
        collected = [
            d
            async for d in stream_chat(
                messages=_MESSAGES, config=_config(provider_order="anthropic,groq")
            )
        ]
    assert anthropic_called is False
    assert collected[0].content == "ok"


async def test_all_providers_unconfigured_raises_llm_error():
    with pytest.raises(LLMError):
        async for _ in stream_chat(
            messages=_MESSAGES,
            config=_config(anthropic_api_key="", groq_api_key="", openai_api_key=""),
        ):
            pass


async def test_all_providers_fail_preflight_raises_llm_error():
    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {
            "anthropic": lambda *a, **kw: _failing_gen(RuntimeError("down")),
            "groq": lambda *a, **kw: _failing_gen(RuntimeError("also down")),
        },
    ):
        with pytest.raises(LLMError):
            async for _ in stream_chat(
                messages=_MESSAGES, config=_config(provider_order="anthropic,groq")
            ):
                pass
