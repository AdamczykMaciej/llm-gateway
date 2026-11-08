import pytest

from llm_gateway.pricing import (
    DEFAULT_PRICES,
    PRICE_SOURCES,
    LongContextTier,
    ModelPrice,
    cost_usd,
    format_price_table,
    lookup_price,
    main,
    worst_case_cost_usd,
)
from llm_gateway.providers.base import Usage

HAIKU = DEFAULT_PRICES["anthropic/claude-haiku-5-5"]


def test_old_construction_round_trips():
    p = ModelPrice(input_per_mtok=1, output_per_mtok=2, cached_input_per_mtok=0.5)
    assert p.long_context is None and p.batch_multiplier is None
    assert ModelPrice.model_validate(p.model_dump()) == p
    assert cost_usd(p, Usage(input_tokens=1_000_000, output_tokens=1_000_000)) == 3.0


def test_tier_boundary():
    at = cost_usd(HAIKU, Usage(input_tokens=100_000, output_tokens=1_000_000))
    over = cost_usd(HAIKU, Usage(input_tokens=100_001, output_tokens=1_000_000))
    assert at == pytest.approx(0.01 + 0.5)
    assert over == pytest.approx(100_001 * 0.5 / 1e6 + 2.5)


def test_tier_counts_cache_tokens_and_scales_cache_rates():
    u = Usage(
        input_tokens=150_000, cache_read_input_tokens=100_000, cache_creation_input_tokens=10_000
    )
    expected = (40_000 * 0.5 + 100_000 * 0.05 + 10_000 * 0.625) / 1e6
    assert cost_usd(HAIKU, u) == pytest.approx(expected)


def test_cache_write_5m_vs_1h():
    sonnet = DEFAULT_PRICES["anthropic/claude-sonnet-5-5"]
    u = Usage(input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)
    assert cost_usd(sonnet, u) == pytest.approx(2.5)
    assert cost_usd(sonnet, u, cache_write_1h_tokens=1_000_000) == pytest.approx(4.0)
    assert cost_usd(sonnet, u, cache_write_1h_tokens=5_000_000) == pytest.approx(4.0)  # clamped
    assert cost_usd(
        sonnet, Usage(input_tokens=10**6, cache_read_input_tokens=10**6)
    ) == pytest.approx(0.10)


def test_batch_multiplier_and_explicit_tier_rates():
    assert cost_usd(HAIKU, Usage(input_tokens=10**5), batch=True) == pytest.approx(0.005)
    p = ModelPrice(
        input_per_mtok=1,
        output_per_mtok=1,
        long_context=LongContextTier(threshold_input_tokens=10, multiplier=2, output_per_mtok=7),
    )
    assert cost_usd(p, Usage(input_tokens=11, output_tokens=10**6)) == pytest.approx(7.000022)
    nobatch = ModelPrice(input_per_mtok=1, output_per_mtok=1)
    assert cost_usd(nobatch, Usage(input_tokens=10**6), batch=True) == 1.0


def test_unknown_model_none_and_override_wins():
    assert lookup_price("anthropic", "nope") is None
    assert cost_usd(None, Usage(input_tokens=5)) is None
    own = ModelPrice(input_per_mtok=9, output_per_mtok=9)
    assert lookup_price("anthropic", "claude-haiku-5-5", {"anthropic/claude-haiku-5-5": own}) is own


def test_vertex_premium_scales_tier():
    p = ModelPrice(
        input_per_mtok=1,
        output_per_mtok=1,
        long_context=LongContextTier(threshold_input_tokens=5, input_per_mtok=3),
    )
    from llm_gateway import pricing

    pricing.DEFAULT_PRICES["vertex/t"] = p
    try:
        got = lookup_price("vertex", "t", vertex_location="europe-west1")
    finally:
        del pricing.DEFAULT_PRICES["vertex/t"]
    assert got is not None and got.long_context is not None
    assert got.input_per_mtok == 1.1 and got.long_context.input_per_mtok == 3.3


def test_worst_case_uses_tier():
    small = worst_case_cost_usd(HAIKU, input_chars=4_000, max_output_tokens=1000)
    big = worst_case_cost_usd(HAIKU, input_chars=4_000_000, max_output_tokens=1000)
    assert small == pytest.approx((1000 * 0.125 + 1000 * 0.5) / 1e6)  # 5m write 0.125 > input 0.10
    assert big == pytest.approx((1_000_000 * 0.625 + 1000 * 2.5) / 1e6)


def test_every_default_row_documented():
    assert set(PRICE_SOURCES) == set(DEFAULT_PRICES)


def test_cli_list(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "anthropic/claude-opus-5-5" in out and "x5" in out and "2026-11-08" in out
    assert format_price_table().startswith("model")
    with pytest.raises(SystemExit):
        main(["--bogus"])
    assert main([]) == 2
