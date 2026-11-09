"""Model prices and cost accounting.

Prices are USD per 1M tokens, keyed by "provider/model" (the same spelling
as the HTTP service's `model` field). `GatewayConfig.model_prices`
(`MODEL_PRICES`, JSON) is merged over `DEFAULT_PRICES` key by key, so an
override replaces one model's whole entry and a new key adds a model.

The defaults cover the gateway's default models plus current Claude 5.5
models, at list price. A row may carry a long-context tier (applied per
request by its input tokens), a batch multiplier and a 1-hour cache-write
rate. They don't know about negotiated discounts or data-zone premiums (for
example Anthropic's 1.1x `inference_geo="us"` multiplier): override the table
when those apply. `python -m llm_gateway.pricing --list` prints the table.

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
from typing import TYPE_CHECKING, NamedTuple

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


class LongContextTier(BaseModel):
    """Rates for a request whose input tokens exceed `threshold_input_tokens`.

    The whole request is billed at the tier rates (not just the excess), and
    `input_tokens` counts cache reads and writes, as providers define it.
    Give `multiplier` to scale every base rate, and/or explicit rates; an
    explicit rate wins over the multiplier for that rate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold_input_tokens: int = Field(gt=0)
    multiplier: float | None = Field(default=None, gt=0)
    input_per_mtok: float | None = Field(default=None, ge=0)
    output_per_mtok: float | None = Field(default=None, ge=0)
    cached_input_per_mtok: float | None = Field(default=None, ge=0)
    cache_write_per_mtok: float | None = Field(default=None, ge=0)
    cache_write_1h_per_mtok: float | None = Field(default=None, ge=0)


class ModelPrice(BaseModel):
    """USD per 1M tokens. Cached-input and cache-write rates are optional:
    without them those tokens are charged at the base input rate.

    `cache_write_1h_per_mtok` is the 1-hour cache write rate where the
    provider distinguishes it from the 5-minute one (`cache_write_per_mtok`).
    `long_context` applies per request by its input tokens. `batch_multiplier`
    scales every rate for batch requests (`cost_usd(batch=True)`).
    Where a built-in row came from is recorded in `PRICE_SOURCES`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    cached_input_per_mtok: float | None = Field(default=None, ge=0)
    cache_write_per_mtok: float | None = Field(default=None, ge=0)
    cache_write_1h_per_mtok: float | None = Field(default=None, ge=0)
    long_context: LongContextTier | None = None
    batch_multiplier: float | None = Field(default=None, gt=0)


_CHECKED = "2026-11-09"
_ANTHROPIC_PRICING = "https://platform.claude.com/docs/en/about-claude/pricing"


# List prices. Every row cites its source and the date it was checked; a row
# marked "unverified" could not be confirmed on 2026-11-08 and keeps its
# earlier value.
_CLAUDE_HAIKU_4_5 = ModelPrice(
    # 5-minute cache writes 1.25x, 1-hour 2x, cache hits 0.1x base input.
    input_per_mtok=1.00,
    output_per_mtok=5.00,
    cached_input_per_mtok=0.10,
    cache_write_per_mtok=1.25,
    cache_write_1h_per_mtok=2.00,
    batch_multiplier=0.5,
)
# Rates for prompts up to 100,000 input tokens; a request over 100,000 pays 5x
# on every rate (input 0.50, output 2.50, 5m write 0.625, 1h write 1.00, read
# 0.05). Haiku 5.5 uses a ~30% larger tokenizer than Haiku 4.5.
_CLAUDE_HAIKU_5_5 = ModelPrice(
    input_per_mtok=0.10,
    output_per_mtok=0.50,
    cached_input_per_mtok=0.01,
    cache_write_per_mtok=0.125,
    cache_write_1h_per_mtok=0.20,
    long_context=LongContextTier(threshold_input_tokens=100_000, multiplier=5.0),
    batch_multiplier=0.5,
)
# Sonnet and Opus 5.5 include the full 1M context at standard pricing (no
# tier); cache hits are 0.05x base input.
_CLAUDE_SONNET_5_5 = ModelPrice(
    input_per_mtok=2.00,
    output_per_mtok=10.00,
    cached_input_per_mtok=0.10,
    cache_write_per_mtok=2.50,
    cache_write_1h_per_mtok=4.00,
    batch_multiplier=0.5,
)
# Cache hits are 0.025x base input on Fable 5.1.
_CLAUDE_FABLE_5_1 = ModelPrice(
    input_per_mtok=10.00,
    output_per_mtok=50.00,
    cached_input_per_mtok=0.25,
    cache_write_per_mtok=12.50,
    cache_write_1h_per_mtok=20.00,
    batch_multiplier=0.5,
)
_CLAUDE_OPUS_5_5 = ModelPrice(
    input_per_mtok=4.00,
    output_per_mtok=20.00,
    cached_input_per_mtok=0.20,
    cache_write_per_mtok=5.00,
    cache_write_1h_per_mtok=8.00,
    batch_multiplier=0.5,
)
# https://cloud.google.com/vertex-ai/generative-ai/pricing ("Anthropic's Claude
# models", Global tab: input $1.00, output $5.00, 5m cache write $1.25, cache
# hit $0.10). The europe-west1 / us-east5 tabs list the same model at 1.1x.
# UNVERIFIED on 2026-11-08: the page did not render its Claude tables to the
# fetcher; value kept from 2026-09-14.
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
    "anthropic/claude-haiku-5-5": _CLAUDE_HAIKU_5_5,
    "anthropic/claude-sonnet-5-5": _CLAUDE_SONNET_5_5,
    "anthropic/claude-opus-5-5": _CLAUDE_OPUS_5_5,
    "anthropic/claude-fable-5-1": _CLAUDE_FABLE_5_1,
    # Global-endpoint prices; see VERTEX_REGIONAL_PREMIUM.
    "vertex/claude-haiku-4-5@20251001": _CLAUDE_HAIKU_4_5_VERTEX_GLOBAL,
    "vertex/claude-haiku-4-5": _CLAUDE_HAIKU_4_5_VERTEX_GLOBAL,
    "openai/gpt-4o-mini": ModelPrice(
        input_per_mtok=0.15,
        output_per_mtok=0.60,
        cached_input_per_mtok=0.075,
        batch_multiplier=0.5,  # Batch 0.075 / 0.30
    ),
    # The default GROQ_MODEL since 0.4.2. Groq now lists a 0.075 cached rate.
    "groq/openai/gpt-oss-120b": ModelPrice(
        input_per_mtok=0.15,
        output_per_mtok=0.60,
        cached_input_per_mtok=0.075,
    ),
    # UNVERIFIED on 2026-11-08: mistral.ai/pricing lists no Small rate and the
    # model card gives two unlabelled figures (0.15 / 0.6, Small 4); kept.
    "mistral/mistral-small-latest": ModelPrice(
        input_per_mtok=0.10,
        output_per_mtok=0.30,
    ),
    # OpenRouter passes the upstream host's price through; gpt-oss-120b
    # listings cluster around this. UNVERIFIED (host-dependent, not re-fetched
    # 2026-11-08). Override with MODEL_PRICES for the host you pin.
    "openrouter/openai/gpt-oss-120b": ModelPrice(
        input_per_mtok=0.15,
        output_per_mtok=0.60,
    ),
    # No default for openai_compat: the operator names the host and model.
    # No default for azure: a deployment's price depends on its model and SKU.
    # groq/llama-3.3-70b-versatile has none either: Groq lists it as "Contact sales".
}


# Where each built-in row came from: key -> (source URL, date checked). A
# source ending "(unverified)" could not be confirmed on 2026-11-08 and keeps
# its earlier value and date.
_VERTEX_PAGE = "https://cloud.google.com/vertex-ai/generative-ai/pricing"
PRICE_SOURCES: dict[str, tuple[str, str]] = {
    "anthropic/claude-haiku-4-5-20251001": (_ANTHROPIC_PRICING, _CHECKED),
    "anthropic/claude-haiku-4-5": (_ANTHROPIC_PRICING, _CHECKED),
    "anthropic/claude-haiku-5-5": (_ANTHROPIC_PRICING, _CHECKED),
    "anthropic/claude-sonnet-5-5": (_ANTHROPIC_PRICING, _CHECKED),
    "anthropic/claude-opus-5-5": (_ANTHROPIC_PRICING, _CHECKED),
    "anthropic/claude-fable-5-1": (_ANTHROPIC_PRICING, _CHECKED),
    "vertex/claude-haiku-4-5@20251001": (_VERTEX_PAGE + " (unverified)", "2026-09-14"),
    "vertex/claude-haiku-4-5": (_VERTEX_PAGE + " (unverified)", "2026-09-14"),
    "openai/gpt-4o-mini": ("https://developers.openai.com/api/docs/pricing", _CHECKED),
    "groq/openai/gpt-oss-120b": (
        "https://console.groq.com/docs/model/openai/gpt-oss-120b",
        _CHECKED,
    ),
    "mistral/mistral-small-latest": ("https://mistral.ai/pricing (unverified)", "2026-09-22"),
    "openrouter/openai/gpt-oss-120b": (
        "https://openrouter.ai/openai/gpt-oss-120b (unverified)",
        "2026-09-22",
    ),
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

    tier = price.long_context
    if tier is not None:
        tier = tier.model_copy(
            update={
                "input_per_mtok": scale(tier.input_per_mtok),
                "output_per_mtok": scale(tier.output_per_mtok),
                "cached_input_per_mtok": scale(tier.cached_input_per_mtok),
                "cache_write_per_mtok": scale(tier.cache_write_per_mtok),
                "cache_write_1h_per_mtok": scale(tier.cache_write_1h_per_mtok),
            }
        )
    return price.model_copy(
        update={
            "input_per_mtok": round(price.input_per_mtok * factor, 6),
            "output_per_mtok": round(price.output_per_mtok * factor, 6),
            "cached_input_per_mtok": scale(price.cached_input_per_mtok),
            "cache_write_per_mtok": scale(price.cache_write_per_mtok),
            "cache_write_1h_per_mtok": scale(price.cache_write_1h_per_mtok),
            "long_context": tier,
        }
    )


class _Rates(NamedTuple):
    input: float
    output: float
    cached: float
    write: float
    write_1h: float


def _rates(price: ModelPrice, input_tokens: int, batch: bool) -> _Rates:
    """The per-MTok rates for one request: the long-context tier when its
    input tokens exceed the threshold, then the batch multiplier."""
    inp, out = price.input_per_mtok, price.output_per_mtok
    cached = price.cached_input_per_mtok if price.cached_input_per_mtok is not None else inp
    write = price.cache_write_per_mtok if price.cache_write_per_mtok is not None else inp
    write_1h = price.cache_write_1h_per_mtok if price.cache_write_1h_per_mtok is not None else write
    tier = price.long_context
    if tier is not None and input_tokens > tier.threshold_input_tokens:
        m = tier.multiplier if tier.multiplier is not None else 1.0

        def pick(explicit: float | None, base: float) -> float:
            return explicit if explicit is not None else base * m

        inp = pick(tier.input_per_mtok, inp)
        out = pick(tier.output_per_mtok, out)
        cached = pick(tier.cached_input_per_mtok, cached)
        write = pick(tier.cache_write_per_mtok, write)
        write_1h = pick(tier.cache_write_1h_per_mtok, write_1h)
    if batch and price.batch_multiplier is not None:
        b = price.batch_multiplier
        inp, out, cached, write, write_1h = inp * b, out * b, cached * b, write * b, write_1h * b
    return _Rates(inp, out, cached, write, write_1h)


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


def cost_usd(
    price: ModelPrice | None,
    usage: "Usage",
    *,
    batch: bool = False,
    cache_write_1h_tokens: int = 0,
) -> float | None:
    """Cost of a served call from its actual usage, or `None` when the model
    has no price (never a guessed 0). The long-context tier is chosen by this
    request's `usage.input_tokens`. `batch=True` applies `batch_multiplier`;
    `cache_write_1h_tokens` is the part of the cache writes made with the
    1-hour TTL (the rest are 5-minute writes)."""
    if price is None:
        return None
    r = _rates(price, usage.input_tokens, batch)
    writes = usage.cache_creation_input_tokens
    long_writes = min(max(cache_write_1h_tokens, 0), writes)
    total = (
        usage.uncached_input_tokens * r.input
        + usage.cache_read_input_tokens * r.cached
        + (writes - long_writes) * r.write
        + long_writes * r.write_1h
        + usage.output_tokens * r.output
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
    tokens = estimate_tokens(input_chars)
    r = _rates(price, tokens, False)
    input_rate = max(r.input, r.write)
    total = tokens * input_rate + max(max_output_tokens, 0) * r.output
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


def format_price_table(prices: Mapping[str, ModelPrice] | None = None) -> str:
    """The price table as aligned text, for audits (no network)."""
    rows = [("model", "input", "output", "cache r/w5m/w1h", "tiers", "source", "checked")]
    for key, p in sorted((prices or DEFAULT_PRICES).items()):
        source, checked = PRICE_SOURCES.get(key, ("-", "-"))
        tier = p.long_context
        tiers = []
        if tier is not None:
            what = f"x{tier.multiplier:g}" if tier.multiplier is not None else "explicit"
            tiers.append(f">{tier.threshold_input_tokens} in: {what}")
        if p.batch_multiplier is not None:
            tiers.append(f"batch x{p.batch_multiplier:g}")

        def f(v: float | None) -> str:
            return "-" if v is None else f"{v:g}"

        cache = "/".join(
            f(v)
            for v in (p.cached_input_per_mtok, p.cache_write_per_mtok, p.cache_write_1h_per_mtok)
        )
        rows.append(
            (
                key,
                f(p.input_per_mtok),
                f(p.output_per_mtok),
                cache,
                "; ".join(tiers) or "-",
                source,
                checked,
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m llm_gateway.pricing")
    parser.add_argument("--list", action="store_true", help="print the built-in price table")
    args = parser.parse_args(argv)
    if not args.list:
        parser.print_help()
        return 2
    print(format_price_table())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
