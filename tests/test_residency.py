"""0.11.0 EU residency: the `residency` of each provider derived from its
configuration, `residency_report()`, and the strict `require_residency` mode
that makes routing refuse any provider whose residency is not the required
one (an unknown residency is never allowed). Fakes only; no network."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

import llm_gateway
from llm_gateway import (
    GatewayConfig,
    PolicyViolationError,
    ResidencyViolationError,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
    residency_report,
    stream_chat,
)
from llm_gateway.providers import RESIDENCY, ProviderResult, StreamDelta, vertex
from llm_gateway.providers import mistral as mistral_provider

ALL_KEYS = dict(
    anthropic_api_key="k",
    groq_api_key="k",
    openai_api_key="k",
    mistral_api_key="k",
    vertex_project_id="interviewer-eu",
)
MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**overrides) -> GatewayConfig:
    settings = dict(
        _env_file=None,
        vertex_location="eu",
        vertex_model="claude-haiku-5-5",
        provider_order="vertex,anthropic,groq,openai,mistral",
        retry_attempts=1,
        **ALL_KEYS,
    )
    settings.update(overrides)
    return GatewayConfig(**settings)


def _ok(text: str = "ok") -> AsyncMock:
    return AsyncMock(return_value=ProviderResult(text, "m", 1, 1))


def _never() -> AsyncMock:
    return AsyncMock(side_effect=AssertionError("a non-EU provider must never be called"))


# ─── Residency per provider ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "location, expected",
    [
        ("eu", "eu"),
        ("europe-west1", "eu"),
        ("europe-west4", "eu"),
        ("global", "global"),
        ("us", "us"),
        ("us-east5", "us"),
        ("asia-northeast1", "unknown"),
    ],
)
def test_vertex_and_gemini_residency_follows_the_location(location, expected):
    config = _config(vertex_location=location, gemini_location=location)
    assert RESIDENCY["vertex"](config) == expected
    assert RESIDENCY["gemini"](config) == expected


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, "unknown"),
        ({"azure_endpoint": "https://res.openai.azure.com"}, "unknown"),
        ({"azure_data_zone": "eu"}, "eu"),
        ({"azure_data_zone": "US"}, "us"),
        ({"azure_data_zone": "global"}, "global"),
        ({"azure_region": "swedencentral"}, "eu"),
        ({"azure_region": "westeurope"}, "eu"),
        ({"azure_region": "eastus"}, "unknown"),
        # An explicit data zone wins over the region.
        ({"azure_data_zone": "global", "azure_region": "swedencentral"}, "global"),
        ({"azure_endpoint": "https://res.swedencentral.api.cognitive.microsoft.com"}, "eu"),
    ],
)
def test_azure_residency_comes_from_data_zone_region_or_endpoint(overrides, expected):
    assert RESIDENCY["azure"](_config(**overrides)) == expected


def test_azure_data_zone_is_validated():
    with pytest.raises(ValidationError):
        _config(azure_data_zone="europe west")


def test_residency_table():
    config = _config()
    assert {p: RESIDENCY[p](config) for p in RESIDENCY} == {
        "anthropic": "global",
        "azure": "unknown",
        "gemini": "eu",
        "groq": "us",
        "huggingface": "global",
        "mistral": "eu",
        "openai": "global",
        "openai_compat": "unknown",
        "openrouter": "global",
        "vertex": "eu",
    }


def test_mistral_is_eu_only_for_the_mistral_host():
    assert mistral_provider.BASE_URL == "https://api.mistral.ai/v1"
    assert RESIDENCY["mistral"](_config()) == "eu"
    with patch.object(mistral_provider, "BASE_URL", "https://proxy.example.com/v1"):
        assert RESIDENCY["mistral"](_config()) == "unknown"


def test_every_known_provider_has_a_residency():
    from llm_gateway.policy import KNOWN_PROVIDERS

    assert set(RESIDENCY) == set(KNOWN_PROVIDERS)
    assert vertex.residency(_config(vertex_location="global")) == "global"


def test_residency_report_covers_the_chain_in_order():
    config = _config(provider_order="vertex, mistral ,anthropic,nonsense")
    report = residency_report(config)
    assert list(report) == ["vertex", "mistral", "anthropic", "nonsense"]
    assert report == {"vertex": "eu", "mistral": "eu", "anthropic": "global", "nonsense": "unknown"}
    assert llm_gateway.residency_report is residency_report


def test_residency_report_all_providers():
    report = residency_report(_config(), all_providers=True)
    assert set(report) == set(RESIDENCY)


# ─── Strict mode ───────────────────────────────────────────────────────────


def test_require_residency_config():
    assert _config().require_residency is None
    assert _config(require_residency="").require_residency is None
    assert _config(require_residency=" EU ").require_residency == "eu"
    with pytest.raises(ValidationError):
        _config(require_residency="not a region")


def test_require_residency_reads_the_environment(monkeypatch):
    monkeypatch.setenv("REQUIRE_RESIDENCY", "eu")
    assert GatewayConfig(_env_file=None).require_residency == "eu"


async def test_strict_mode_never_calls_a_non_eu_provider_and_does_not_fall_through():
    calls = {n: _never() for n in ("anthropic", "groq", "openai")}
    calls["vertex"] = AsyncMock(side_effect=RuntimeError("vertex down"))
    calls["mistral"] = AsyncMock(side_effect=RuntimeError("mistral down"))
    config = _config(require_residency="eu")
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(llm_gateway.LLMError) as caught:
            await complete_with_usage(system="s", prompt="p", config=config)
    assert not isinstance(caught.value, PolicyViolationError)  # the EU providers ran and failed
    calls["vertex"].assert_awaited()
    calls["mistral"].assert_awaited()
    for name in ("anthropic", "groq", "openai"):
        calls[name].assert_not_called()


async def test_strict_mode_serves_from_the_eu_provider():
    calls = {"vertex": _ok("from vertex"), **{n: _never() for n in ("anthropic", "groq")}}
    config = _config(require_residency="eu")
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(system="s", prompt="p", config=config)
    assert (result.text, result.provider) == ("from vertex", "vertex")


async def test_strict_mode_raises_a_typed_error_when_only_non_eu_providers_remain():
    calls = {n: _never() for n in ("anthropic", "groq", "openai")}
    config = _config(require_residency="eu", provider_order="anthropic,groq,openai")
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(ResidencyViolationError) as caught:
            await complete_with_usage(system="s", prompt="p", config=config)
    assert isinstance(caught.value, PolicyViolationError)
    assert set(caught.value.exclusions) == {"anthropic", "groq", "openai"}
    assert "residency=eu" in str(caught.value)
    for call in calls.values():
        call.assert_not_called()


async def test_strict_mode_treats_unknown_residency_as_not_allowed():
    calls = {"azure": _never()}
    config = _config(
        require_residency="eu",
        provider_order="azure",
        azure_endpoint="https://res.openai.azure.com",
        azure_auth="api_key",
        azure_api_key="k",
        azure_model="m",
    )
    assert RESIDENCY["azure"](config) == "unknown"
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(ResidencyViolationError) as caught:
            await complete_with_usage(system="s", prompt="p", config=config)
    assert "unknown" in str(caught.value)
    calls["azure"].assert_not_called()


async def test_strict_mode_applies_to_a_mutated_chain_and_forced_provider():
    calls = {"vertex": _ok(), "anthropic": _never()}
    config = _config(require_residency="eu", provider_order="vertex")
    config.provider_order = "anthropic,vertex"  # mutated after construction
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(system="s", prompt="p", config=config)
        assert result.provider == "vertex"
        config.provider_order = "anthropic"
        with pytest.raises(ResidencyViolationError):
            await complete_with_usage(system="s", prompt="p", config=config)
        with pytest.raises(ResidencyViolationError):
            await complete_with_usage(
                system="s", prompt="p", config=config, force_provider="anthropic"
            )
    calls["anthropic"].assert_not_called()


async def test_strict_mode_follows_a_changed_vertex_location():
    calls = {"vertex": _never()}
    config = _config(require_residency="eu", provider_order="vertex")
    config.vertex_location = "global"
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(ResidencyViolationError):
            await complete_with_usage(system="s", prompt="p", config=config)
    calls["vertex"].assert_not_called()


async def test_strict_mode_in_chat_and_stream():
    chat_calls = {n: _never() for n in ("anthropic", "groq")}
    config = _config(require_residency="eu", provider_order="anthropic,groq")
    with patch("llm_gateway.chat.CHAT_CALLS", chat_calls):
        with pytest.raises(ResidencyViolationError):
            await chat(messages=MESSAGES, config=config)

    opened: list[str] = []

    def never_open(*args, **kwargs):
        opened.append("anthropic")
        raise AssertionError("a non-EU provider must never be called")

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": never_open}):
        with pytest.raises(ResidencyViolationError):
            async for _ in stream_chat(
                messages=MESSAGES,
                config=_config(require_residency="eu", provider_order="anthropic"),
            ):
                pass
    assert opened == []


async def test_strict_mode_stream_serves_eu_provider():
    def open_stream(*args, **kwargs):
        async def generator():
            yield StreamDelta(content="eu", provider="vertex", model="m")

        return generator()

    config = _config(require_residency="eu", provider_order="anthropic,vertex")
    with patch("llm_gateway.streaming.STREAM_CALLS", {"vertex": open_stream}):
        deltas = [d async for d in stream_chat(messages=MESSAGES, config=config)]
    assert [d.content for d in deltas] == ["eu"]


async def test_disabled_by_default_keeps_the_chain_unchanged():
    calls = {
        "vertex": AsyncMock(side_effect=RuntimeError("down")),
        "anthropic": _ok("from anthropic"),
    }
    config = _config(provider_order="vertex,anthropic")
    assert config.require_residency is None
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(system="s", prompt="p", config=config)
    assert result.provider == "anthropic"
