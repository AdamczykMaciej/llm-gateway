"""0.28.0: estimate_max_cost / routable_models. Fakes only: no provider is ever called."""

import pytest

import llm_gateway
from llm_gateway import (
    CostEstimate,
    GatewayConfig,
    ModelPrice,
    ProviderPolicy,
    estimate_max_cost,
    routable_models,
)


def _cfg(**kw) -> GatewayConfig:
    base = {
        "anthropic_api_key": "k",
        "claude_model": "cheap",
        "groq_api_key": "k",
        "groq_model": "mid",
        "provider_order": "anthropic,groq",
        "model_prices": {
            "anthropic/cheap": ModelPrice(input_per_mtok=1, output_per_mtok=2),
            "groq/mid": ModelPrice(input_per_mtok=3, output_per_mtok=4),
        },
    }
    base.update(kw)
    return GatewayConfig(**base)


def test_routable_models_lists_the_chain_in_order():
    got = routable_models(_cfg())
    assert [(m.provider, m.model) for m in got] == [("anthropic", "cheap"), ("groq", "mid")]


def test_max_is_the_worst_routable_model():
    est = estimate_max_cost(_cfg(), input_tokens=1_000_000, max_output_tokens=1_000_000)
    assert isinstance(est, CostEstimate)
    assert est.max_cost_usd == pytest.approx(7.0)  # groq: 3 + 4
    assert est.complete and est.unpriced == ()


def test_unrelated_expensive_model_in_the_price_table_does_not_change_it():
    cfg = _cfg()
    before = estimate_max_cost(cfg, input_tokens=1000, max_output_tokens=500)
    pricey = dict(cfg.model_prices)
    pricey["anthropic/some-unused-fable"] = ModelPrice(input_per_mtok=10, output_per_mtok=50)
    pricey["openai/unused"] = ModelPrice(input_per_mtok=100, output_per_mtok=500)
    after = estimate_max_cost(_cfg(model_prices=pricey), input_tokens=1000, max_output_tokens=500)
    assert after.max_cost_usd == before.max_cost_usd


def test_unconfigured_provider_is_excluded():
    cfg = _cfg(groq_api_key="")
    est = estimate_max_cost(cfg, input_tokens=1_000_000, max_output_tokens=1_000_000)
    assert [m.provider for m in est.models] == ["anthropic"]
    assert est.max_cost_usd == pytest.approx(3.0)


def test_retired_model_is_excluded_unless_blocking_is_off():
    from llm_gateway.lifecycle import LIFECYCLE, is_retired

    retired = next(m for (p, m) in LIFECYCLE if p == "groq" and is_retired(p, m))
    cfg = _cfg(groq_model=retired)
    assert [m.provider for m in routable_models(cfg)] == ["anthropic"]
    assert [
        m.provider for m in routable_models(_cfg(groq_model=retired, block_retired_models=False))
    ] == [
        "anthropic",
        "groq",
    ]


def test_policy_narrows_the_models():
    est = estimate_max_cost(
        _cfg(),
        input_tokens=1_000_000,
        max_output_tokens=1_000_000,
        policy=ProviderPolicy(only=["anthropic"]),
    )
    assert [m.provider for m in est.models] == ["anthropic"]
    assert est.max_cost_usd == pytest.approx(3.0)


def test_unknown_price_is_flagged_never_zero():
    cfg = _cfg(model_prices={"anthropic/cheap": ModelPrice(input_per_mtok=1, output_per_mtok=2)})
    est = estimate_max_cost(cfg, input_tokens=1000, max_output_tokens=500)
    assert est.max_cost_usd is None
    assert est.unpriced == ("groq/mid",)
    assert not est.complete
    assert est.priced_max_cost_usd == est.models[0].cost_usd


def test_nothing_routable_is_none_not_zero():
    est = estimate_max_cost(
        _cfg(anthropic_api_key="", groq_api_key=""), input_tokens=10, max_output_tokens=10
    )
    assert est.models == () and est.max_cost_usd is None and not est.complete


def test_policy_excluding_everything_does_not_raise():
    est = estimate_max_cost(
        _cfg(),
        input_tokens=10,
        max_output_tokens=10,
        policy=ProviderPolicy(only=["openai"]),
    )
    assert est.models == () and est.max_cost_usd is None
    assert est.excluded


def test_negative_tokens_rejected():
    with pytest.raises(ValueError):
        estimate_max_cost(_cfg(), input_tokens=-1, max_output_tokens=1)


def test_exported():
    for name in ("CostEstimate", "RoutableModel", "estimate_max_cost", "routable_models"):
        assert name in llm_gateway.__all__


# 0.30.0: the one-shot claude_model_fallback is a routable model too.
def _fb(**kw) -> GatewayConfig:
    base = {
        "groq_api_key": "",
        "provider_order": "anthropic",
        "claude_model": "cheap",
        "claude_model_fallback": "dear",
        "model_prices": {
            "anthropic/cheap": ModelPrice(input_per_mtok=1, output_per_mtok=2),
            "anthropic/dear": ModelPrice(input_per_mtok=10, output_per_mtok=20),
        },
    }
    base.update(kw)
    return _cfg(**base)


def test_fallback_model_is_routable_after_primary():
    got = routable_models(_fb())
    assert [(m.provider, m.model) for m in got] == [("anthropic", "cheap"), ("anthropic", "dear")]


def test_max_cost_covers_dearer_fallback():
    est = estimate_max_cost(_fb(), input_tokens=1_000_000, max_output_tokens=1_000_000)
    assert est.max_cost_usd == pytest.approx(30.0)


def test_unpriced_fallback_makes_max_cost_none():
    prices = {"anthropic/cheap": ModelPrice(input_per_mtok=1, output_per_mtok=2)}
    est = estimate_max_cost(_fb(model_prices=prices), input_tokens=10, max_output_tokens=10)
    assert est.max_cost_usd is None
    assert est.unpriced == ("anthropic/dear",)


def test_fallback_skipped_when_same_unset_or_explicit_model_but_kept_for_chat():
    assert len(routable_models(_fb(claude_model_fallback="cheap"))) == 1
    assert len(routable_models(_fb(claude_model_fallback=""))) == 1
    assert len(routable_models(_fb(), provider="anthropic", model="x")) == 1  # explicit model
    assert len(routable_models(_fb(), chat=True)) == 2  # chat() retries with it too


def test_fallback_skipped_when_anthropic_not_routable():
    assert routable_models(_fb(anthropic_api_key="")) == ()


def test_policy_max_cost_considers_fallback():
    # primary alone is 3.0 (<= cap), the dearer fallback is 30.0 (> cap)
    est = estimate_max_cost(
        _fb(),
        input_tokens=1_000_000,
        max_output_tokens=1_000_000,
        policy=ProviderPolicy(max_cost_usd=5.0),
    )
    assert est.models == ()
    assert "anthropic" in est.excluded
