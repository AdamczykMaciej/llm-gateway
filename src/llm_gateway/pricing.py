"""Model prices and cost accounting.

Prices are USD per 1M tokens, keyed by "provider/model" (the same spelling
as the HTTP service's `model` field). `GatewayConfig.model_prices`
(`MODEL_PRICES`, JSON) is merged over `DEFAULT_PRICES` key by key, so an
override replaces one model's whole entry and a new key adds a model.

The defaults cover only the gateway's default models, at list price. They
don't know about negotiated discounts, batch pricing, data-zone premiums (for
example Anthropic's 1.1x `inference_geo="us"` multiplier) or 1-hour cache
writes: override the table when any of those apply.

One regional premium is applied: Vertex AI charges 10% more for Claude
Sonnet 4.5 and newer (Haiku 4.5 included) on regional and multi-region
endpoints than on the global endpoint. `lookup_price(..., vertex_location=)`
multiplies a *default* `vertex/...` price by `VERTEX_REGIONAL_PREMIUM` when
the location isn't "global". A `MODEL_PRICES` override is used as given,
since it states the price for the location you actually call.
"""

import json
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from .policy import KNOWN_PROVIDERS

if TYPE_CHECKING:
    from .providers.base import Usage

# Heuristic for estimating prompt tokens before a call: about 4 characters
# per token for English text. It can undercount for other languages and for
# code, so `max_output_tokens` (always charged in full) and the input rate
# (the higher of the base and cache-write rates) are what make the
# worst-case estimate conservative. Images aren't counted.
CHARS_PER_TOKEN = 4


class ModelPrice(BaseModel):
    """USD per 1M tokens. Cached-input and cache-write rates are optional:
    without them those tokens are charged at the base input rate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    cached_input_per_mtok: float | None = Field(default=None, ge=0)
    cache_write_per_mtok: float | None = Field(default=None, ge=0)


# List prices, checked 2026-09-14.
_CLAUDE_HAIKU_4_5 = ModelPrice(
    # https://platform.claude.com/docs/en/about-claude/pricing
    # (5-minute cache writes 1.25x, cache hits 0.1x base input)
    input_per_mtok=1.00,
    output_per_mtok=5.00,
    cached_input_per_mtok=0.10,
    cache_write_per_mtok=1.25,
)
# https://cloud.google.com/vertex-ai/generative-ai/pricing ("Anthropic's Claude
# models", Global tab: input $1.00, output $5.00, 5m cache write $1.25, cache
# hit $0.10). The europe-west1 / us-east5 tabs list the same model at 1.1x.
_CLAUDE_HAIKU_4_5_VERTEX_GLOBAL = ModelPrice(
    input_per_mtok=1.00,
    output_per_mtok=5.00,
    cached_input_per_mtok=0.10,
    cache_write_per_mtok=1.25,
)
# Regional and multi-region Vertex endpoints vs. the global endpoint, for
# Claude Sonnet 4.5 and newer models.
VERTEX_REGIONAL_PREMIUM = 1.1

DEFAULT_PRICES: dict[str, ModelPrice] = {
    "anthropic/claude-haiku-4-5-20251001": _CLAUDE_HAIKU_4_5,
    "anthropic/claude-haiku-4-5": _CLAUDE_HAIKU_4_5,
    # Global-endpoint prices; see VERTEX_REGIONAL_PREMIUM.
    "vertex/claude-haiku-4-5@20251001": _CLAUDE_HAIKU_4_5_VERTEX_GLOBAL,
    "vertex/claude-haiku-4-5": _CLAUDE_HAIKU_4_5_VERTEX_GLOBAL,
    # https://developers.openai.com/api/docs/pricing (Standard tier)
    "openai/gpt-4o-mini": ModelPrice(
        input_per_mtok=0.15, output_per_mtok=0.60, cached_input_per_mtok=0.075
    ),
    # https://console.groq.com/docs/models (the default GROQ_MODEL since 0.4.2)
    "groq/openai/gpt-oss-120b": ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60),
    # No default for azure: a deployment's price depends on its model and SKU.
    # groq/llama-3.3-70b-versatile has none either: Groq lists it as "Contact sales".
}


def validate_price_table(prices: Mapping[str, ModelPrice]) -> dict[str, ModelPrice]:
    for key in prices:
        provider, _, model = key.partition("/")
        if not model or provider not in KNOWN_PROVIDERS:
            raise ValueError(
                f"model_prices key {key!r} must be '<provider>/<model>' with provider one of "
                f"{', '.join(KNOWN_PROVIDERS)}"
            )
    return dict(prices)


def _scaled(price: ModelPrice, factor: float) -> ModelPrice:
    def scale(rate: float | None) -> float | None:
        return None if rate is None else round(rate * factor, 6)

    return ModelPrice(
        input_per_mtok=scale(price.input_per_mtok),
        output_per_mtok=scale(price.output_per_mtok),
        cached_input_per_mtok=scale(price.cached_input_per_mtok),
        cache_write_per_mtok=scale(price.cache_write_per_mtok),
    )


def lookup_price(
    provider: str,
    model: str,
    overrides: Mapping[str, ModelPrice] | None = None,
    *,
    vertex_location: str = "global",
) -> ModelPrice | None:
    """The price for `provider/model`: an override as given, else the default,
    with `VERTEX_REGIONAL_PREMIUM` applied to a default vertex price when
    `vertex_location` isn't "global"."""
    key = f"{provider}/{model}"
    if overrides and key in overrides:
        return overrides[key]
    price = DEFAULT_PRICES.get(key)
    if price is not None and provider == "vertex" and vertex_location != "global":
        return _scaled(price, VERTEX_REGIONAL_PREMIUM)
    return price


def cost_usd(price: ModelPrice | None, usage: "Usage") -> float | None:
    """Cost of a served call from its actual usage, or `None` when the model
    has no price (never a guessed 0)."""
    if price is None:
        return None
    cached_rate = price.cached_input_per_mtok
    write_rate = price.cache_write_per_mtok
    total = (
        usage.uncached_input_tokens * price.input_per_mtok
        + usage.cache_read_input_tokens
        * (cached_rate if cached_rate is not None else price.input_per_mtok)
        + usage.cache_creation_input_tokens
        * (write_rate if write_rate is not None else price.input_per_mtok)
        + usage.output_tokens * price.output_per_mtok
    )
    return round(total / 1_000_000, 10)


def estimate_tokens(chars: int) -> int:
    return math.ceil(max(chars, 0) / CHARS_PER_TOKEN)


def worst_case_cost_usd(
    price: ModelPrice | None, *, input_chars: int, max_output_tokens: int
) -> float | None:
    """Upper-bound cost of a call before it is made: the estimated prompt
    tokens at the higher of the base and cache-write input rates, plus
    `max_output_tokens` at the output rate. `None` without a price."""
    if price is None:
        return None
    input_rate = max(price.input_per_mtok, price.cache_write_per_mtok or 0.0)
    total = (
        estimate_tokens(input_chars) * input_rate
        + max(max_output_tokens, 0) * price.output_per_mtok
    )
    return round(total / 1_000_000, 10)


def json_chars(value: object) -> int:
    if not value:
        return 0
    return len(json.dumps(value, separators=(",", ":"), default=str))


def message_chars(messages: list[dict] | None) -> int:
    """Characters of text in OpenAI-wire messages: string content, text
    parts, and tool-call arguments. Image parts are not counted."""
    total = 0
    for message in messages or ():
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(
                len(str(part.get("text", "")))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        total += json_chars(message.get("tool_calls"))
    return total
