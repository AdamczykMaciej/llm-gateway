"""Accounting pass 8: no negative counts or cost, no float drift on sums, and
a failed stream attempt that reported usage is counted once. Fakes only."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_gateway import GatewayConfig, reset_circuit_breakers, stream_chat
from llm_gateway.pricing import ModelPrice, cost_usd
from llm_gateway.providers._anthropic_translate import usage_from_anthropic
from llm_gateway.providers.base import StreamDelta, Usage, as_int, usage_from_openai_style
from llm_gateway.router import Completion, _merge_attempts

PRICE = ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0)


def test_as_int_never_negative():
    assert as_int(-5) == 0
    assert as_int(7) == 7


def test_negative_provider_counts_cannot_make_a_negative_cost():
    u = usage_from_openai_style(
        SimpleNamespace(prompt_tokens=-10, completion_tokens=-4, prompt_tokens_details=None)
    )
    assert u == Usage()
    assert usage_from_anthropic(5, -3, -2, -1) == Usage(input_tokens=5, output_tokens=0)
    assert cost_usd(PRICE, Usage(10, -5)) == pytest.approx(10 / 1e6)
    assert cost_usd(PRICE, Usage(-10, 5)) == pytest.approx(10 / 1e6)


def test_merged_attempt_cost_has_no_float_drift():
    a = Completion("x", "p", "m", Usage(1, 1), cost_usd=0.1)
    b = Completion("x", "p", "m", Usage(1, 1), cost_usd=0.2)
    assert _merge_attempts(a, b).cost_usd == 0.3


@pytest.mark.asyncio
async def test_stream_cut_after_usage_is_counted_once_across_restart():
    reset_circuit_breakers()

    async def cut(*a, **k):
        yield StreamDelta(content="a")
        yield StreamDelta(usage=(10, 5))
        raise ConnectionResetError("boom")

    async def ok(*a, **k):
        yield StreamDelta(content="b")
        yield StreamDelta(finish_reason="stop", usage=(20, 8))

    seen = []
    cfg = GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        provider_order="anthropic,groq",
        retry_attempts=1,
        on_attempt=seen.append,
    )
    calls = {"anthropic": cut, "groq": ok}
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        async for _ in stream_chat(
            messages=[{"role": "user", "content": "hi"}], config=cfg, restart_on_failure=True
        ):
            pass
    assert [(r.provider, r.outcome, r.usage.input_tokens) for r in seen] == [
        ("anthropic", "failed", 10),
        ("groq", "succeeded", 20),
    ]
