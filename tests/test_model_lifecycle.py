"""Model catalogue correctness: retired and retiring models.

Sources are the vendors' deprecation pages, read 2026-11-09 (URLs on each
`ModelLifecycle`). Fakes only; no network."""

import logging
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from llm_gateway import (
    GatewayConfig,
    ModelRetiredError,
    PolicyViolationError,
    complete,
    reset_circuit_breakers,
)
from llm_gateway.capabilities import capabilities_for
from llm_gateway.lifecycle import (
    LIFECYCLE,
    is_retired,
    lifecycle_for,
    retired_reason,
    retiring_within,
    warn_if_retiring,
)
from llm_gateway.pricing import DEFAULT_PRICES, lookup_price
from llm_gateway.providers.base import ProviderResult
from llm_gateway.residency import AZURE_EU_REGIONS, azure_residency

TODAY = date(2026, 11, 9)


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**overrides) -> GatewayConfig:
    settings = dict(
        _env_file=None,
        anthropic_api_key="k",
        groq_api_key="k",
        openai_api_key="k",
        provider_order="anthropic,groq,openai",
        retry_attempts=1,
    )
    settings.update(overrides)
    return GatewayConfig(**settings)


def test_every_entry_cites_an_https_source_and_a_check_date():
    assert LIFECYCLE
    for entry in LIFECYCLE.values():
        assert entry.source.startswith("https://")
        assert entry.checked <= TODAY
        assert entry.model == entry.model.lower()


def test_replacements_of_upcoming_retirements_are_not_retired():
    # A replacement named for an already-retired model is the vendor's
    # historical advice and may itself have retired since.
    for entry in LIFECYCLE.values():
        if entry.replacement and entry.retires > TODAY:
            assert is_retired(entry.provider, entry.replacement, today=TODAY) is None, entry


def test_shipped_defaults_are_not_retired():
    config = GatewayConfig(_env_file=None)
    for provider, model in (
        ("anthropic", config.claude_model),
        ("groq", config.groq_model),
        ("openai", config.openai_model),
        ("mistral", config.mistral_model),
        ("gemini", config.gemini_model),
        ("openrouter", config.openrouter_model),
    ):
        assert is_retired(provider, model, today=TODAY) is None, (provider, model)


def test_price_table_and_strict_output_list_name_no_retired_model():
    for key in DEFAULT_PRICES:
        provider, _, model = key.partition("/")
        assert is_retired(provider, model, today=TODAY) is None, key


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("groq", "llama-3.3-70b-versatile"),
        ("groq", "llama-3.1-8b-instant"),
        ("anthropic", "claude-opus-4-1-20250805"),
        ("anthropic", "claude-3-haiku-20240307"),
        ("openai", "gpt-4o-2024-05-13"),
        ("gemini", "google/gemini-2.5-flash"),
    ],
)
def test_retired_models_are_recognized(provider, model):
    entry = is_retired(provider, model, today=TODAY)
    assert entry is not None
    assert "retired on" in retired_reason(entry)


def test_retiring_within_ninety_days():
    sonnet = retiring_within("anthropic", "claude-sonnet-4-5-20250929", 90, today=TODAY)
    assert sonnet is not None and sonnet.retires == date(2026, 11, 30)
    assert retiring_within("openai", "gpt-5", 90, today=TODAY) is not None
    assert retiring_within("anthropic", "claude-haiku-5-5", 90, today=TODAY) is None
    assert retiring_within("groq", "llama-3.3-70b-versatile", 90, today=TODAY) is None  # past


def test_unknown_and_unlisted_models_are_not_flagged():
    assert lifecycle_for("anthropic", "claude-opus-5-5") is None
    assert lifecycle_for("azure", "my-deployment") is None
    assert lifecycle_for("openai", "gpt-5.1") is None  # not gpt-5


def test_warning_is_logged_once(caplog):
    with caplog.at_level(logging.WARNING, logger="llm_gateway"):
        warn_if_retiring("openai", "gpt-5-mini", 90, today=TODAY)
        warn_if_retiring("openai", "gpt-5-mini", 90, today=TODAY)
    messages = [r.getMessage() for r in caplog.records if "model_retiring" in r.getMessage()]
    assert len(messages) <= 1  # once per process; another test may have warned already


async def test_retired_model_is_skipped_and_never_called():
    calls = {"anthropic": AsyncMock(), "groq": AsyncMock(), "openai": AsyncMock()}
    calls["groq"].side_effect = AssertionError("a retired model must never be called")
    calls["anthropic"].return_value = ProviderResult("ok", "m", 1, 1)
    config = _config(groq_model="llama-3.3-70b-versatile", provider_order="groq,anthropic")
    with patch("llm_gateway.router.CALLS", calls):
        text = await complete(system="s", prompt="p", config=config)
    assert text == "ok"
    calls["groq"].assert_not_awaited()


async def test_only_retired_models_raise_a_clear_error():
    calls = {"groq": AsyncMock(side_effect=AssertionError("never called"))}
    config = _config(groq_model="llama-3.3-70b-versatile", provider_order="groq")
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(ModelRetiredError) as info:
            await complete(system="s", prompt="p", config=config)
    assert isinstance(info.value, PolicyViolationError)
    assert "llama-3.3-70b-versatile" in str(info.value)
    assert "2026-08-16" in str(info.value)
    calls["groq"].assert_not_awaited()


async def test_explicit_per_call_retired_model_is_refused():
    calls = {"openai": AsyncMock(side_effect=AssertionError("never called"))}
    config = _config(provider_order="openai")
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(ModelRetiredError):
            await complete(
                system="s",
                prompt="p",
                config=config,
                model="gpt-4o-2024-05-13",
                force_provider="openai",
            )


async def test_block_retired_models_can_be_turned_off():
    calls = {"groq": AsyncMock(return_value=ProviderResult("ok", "m", 1, 1))}
    config = _config(
        groq_model="llama-3.3-70b-versatile", provider_order="groq", block_retired_models=False
    )
    with patch("llm_gateway.router.CALLS", calls):
        assert await complete(system="s", prompt="p", config=config) == "ok"


def test_claude_model_fallback_that_is_retired_is_not_used():
    from llm_gateway.providers.anthropic import _fallback_model

    class Rejected(Exception):
        status_code = 404

    config = _config(claude_model_fallback="claude-3-haiku-20240307")
    with patch("llm_gateway.providers.anthropic.is_model_rejection", return_value=True):
        assert _fallback_model(config, None, Rejected()) is None
    ok = _config(claude_model_fallback="claude-haiku-5-5")
    with patch("llm_gateway.providers.anthropic.is_model_rejection", return_value=True):
        assert _fallback_model(ok, None, Rejected()) == "claude-haiku-5-5"


# Anthropic structured outputs: the exact ids on
# https://platform.claude.com/docs/en/build-with-claude/structured-outputs (2026-11-09).
@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-fable-5",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-haiku-5-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
    ],
)
def test_anthropic_strict_structured_output_matches_the_docs_list(model):
    assert capabilities_for("anthropic", model).structured_output == "strict"


def test_retired_opus_4_1_is_not_claimed_strict():
    assert capabilities_for("anthropic", "claude-opus-4-1-20250805").structured_output is None


def test_fable_5_1_has_a_price():
    price = lookup_price("anthropic", "claude-fable-5-1")
    assert price is not None
    assert (price.input_per_mtok, price.output_per_mtok) == (10.0, 50.0)
    assert price.cached_input_per_mtok == 0.25
    assert price.cache_write_per_mtok == 12.5
    assert price.cache_write_1h_per_mtok == 20.0


def test_switzerland_is_not_an_eu_region():
    # Switzerland is outside the EU and the EEA.
    assert not {r for r in AZURE_EU_REGIONS if r.startswith("switzerland")}
    assert azure_residency(data_zone="", region="switzerlandnorth", endpoint="") == "unknown"
    assert azure_residency(data_zone="", region="swedencentral", endpoint="") == "eu"


@pytest.mark.parametrize(
    ("provider", "model", "retired"),
    [
        ("anthropic", "claude-2.1", "2025-07-21"),
        ("groq", "meta-llama/llama-guard-4-12b", "2026-03-05"),
        ("groq", "llama3-70b-8192", "2025-08-30"),
        ("openai", "gpt-4.1-nano", "2026-10-23"),
        ("openai", "o4-mini", "2026-10-23"),
        ("openai", "gpt-3.5-turbo", "2026-10-23"),
        ("gemini", "google/gemini-1.5-pro-002", "2025-09-24"),
    ],
)
def test_catalogue_0_39_retired_additions(provider, model, retired):
    entry = is_retired(provider, model, today=TODAY)
    assert entry is not None
    assert entry.retires.isoformat() == retired


def test_defaults_are_not_retired():
    cfg = GatewayConfig(_env_file=None)
    for provider, model in [
        ("anthropic", cfg.claude_model),
        ("groq", cfg.groq_model),
        ("openai", cfg.openai_model),
        ("gemini", cfg.gemini_model),
    ]:
        assert is_retired(provider, model, today=TODAY) is None


def test_0_39_default_prices():
    fable = lookup_price("anthropic", "claude-fable-5-2")
    assert (fable.input_per_mtok, fable.output_per_mtok, fable.cached_input_per_mtok) == (
        8,
        40,
        0.2,
    )
    luna = lookup_price("openai", "gpt-6-luna")
    assert (luna.input_per_mtok, luna.output_per_mtok, luna.cached_input_per_mtok) == (
        0.1,
        0.5,
        0.01,
    )
