"""A reply whose usage the provider did not report is unknown, not free."""

from llm_gateway.pricing import ModelPrice, cost_usd
from llm_gateway.providers.base import Usage, usage_from_openai_style
from llm_gateway.router import Completion

PRICE = ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0)


def test_cost_is_none_when_usage_is_missing():
    assert cost_usd(PRICE, usage_from_openai_style(None)) is None
    assert cost_usd(PRICE, Usage()) is None


def test_cost_is_still_computed_for_reported_usage():
    assert cost_usd(PRICE, Usage(input_tokens=1_000_000)) == 1.0
    assert cost_usd(PRICE, Usage(output_tokens=1_000_000)) == 2.0


def test_completion_flags_missing_usage():
    assert not Completion("x", "openai_compat", "m", Usage()).usage_reported
    assert Completion("x", "openai", "m", Usage(3, 1)).usage_reported
