"""0.5.0 provider policy through the library engines: operator-asserted
metadata, the fail-closed routing policy, per-call narrowing, capability
filtering, cost accounting and caps, the budget hook and price sorting.

Mocks the provider registries the same way test_router.py, test_chat.py and
test_streaming.py do; no network."""

import json
import logging
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from llm_gateway import (
    GatewayConfig,
    LLMError,
    ModelPrice,
    PolicyViolationError,
    ProviderMetadata,
    ProviderPolicy,
    UnsupportedCapabilityError,
    Usage,
    breaker,
    chat,
    complete,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.capabilities import ModelCapabilities, capabilities_for, missing_capabilities
from llm_gateway.errors import POLICIES, ErrorKind, classify
from llm_gateway.policy import call_budget_check
from llm_gateway.pricing import cost_usd, estimate_tokens, lookup_price, worst_case_cost_usd
from llm_gateway.providers.base import ChatResult, ProviderResult, StreamDelta

EU_COMPLIANT = {"region": "eu", "retention": "zero", "trains_on_data": False, "dpa": True}
SECRET = "Jan Kowalski, PESEL 44051401359, senior engineer CV"
PROVIDERS = ("anthropic", "groq", "openai")


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**overrides) -> GatewayConfig:
    settings = dict(
        _env_file=None,
        anthropic_api_key="k",
        groq_api_key="k",
        openai_api_key="k",
        claude_model="claude-haiku-4-5-20251001",
        groq_model="llama-3.3-70b-versatile",
        openai_model="gpt-4o-mini",
        provider_order="anthropic,groq,openai",
        retry_attempts=1,
    )
    settings.update(overrides)
    return GatewayConfig(**settings)


def _result(text: str = "ok", model: str = "m", **usage) -> ProviderResult:
    return ProviderResult(text, model, usage.pop("input_tokens", 1), 1, **usage)


def _never() -> AsyncMock:
    return AsyncMock(side_effect=AssertionError("an excluded provider must never be called"))


def _stream(opened: list[str], name: str, *deltas: StreamDelta, error: Exception | None = None):
    def open_stream(*args, **kwargs):
        opened.append(name)

        async def generator():
            if error is not None:
                raise error
            for delta in deltas:
                yield delta

        return generator()

    return open_stream


_MESSAGES = [{"role": "user", "content": SECRET}]


# ─── Metadata and configuration ────────────────────────────────────────────


def test_metadata_defaults_are_conservative():
    metadata = ProviderMetadata()
    assert metadata.region == "unknown"
    assert metadata.retention == "unknown"
    assert metadata.trains_on_data is True
    assert metadata.dpa is False
    assert _config().provider_metadata == {}


def test_config_reads_metadata_policy_and_prices_from_env(monkeypatch):
    monkeypatch.setenv(
        "PROVIDER_METADATA",
        json.dumps(
            {
                "azure": {
                    "region": "global",
                    "retention": "abuse_monitoring_30d",
                    "trains_on_data": False,
                    "dpa": True,
                    "notes": "gpt-oss-120b, GlobalStandard",
                },
                "Groq": {"region": "US"},
            }
        ),
    )
    monkeypatch.setenv("POLICY_RESIDENCY", "EU")
    monkeypatch.setenv("POLICY_ONLY", "azure, anthropic")
    monkeypatch.setenv("POLICY_REQUIRE_DPA", "true")
    monkeypatch.setenv("POLICY_MAX_COST_USD", "0.05")
    monkeypatch.setenv("POLICY_SORT", "price")
    monkeypatch.setenv(
        "MODEL_PRICES",
        json.dumps({"azure/gpt-oss-120b": {"input_per_mtok": 0.15, "output_per_mtok": 0.6}}),
    )

    config = GatewayConfig(_env_file=None)

    assert config.provider_metadata["azure"] == ProviderMetadata(
        region="global",
        retention="abuse_monitoring_30d",
        trains_on_data=False,
        dpa=True,
        notes="gpt-oss-120b, GlobalStandard",
    )
    assert config.provider_metadata["groq"] == ProviderMetadata(region="us")
    assert config.policy == ProviderPolicy(
        residency="eu",
        only=("azure", "anthropic"),
        require_dpa=True,
        max_cost_usd=0.05,
        sort="price",
    )
    assert lookup_price("azure", "gpt-oss-120b", config.model_prices) == ModelPrice(
        input_per_mtok=0.15, output_per_mtok=0.6
    )


def test_policy_settings_default_to_no_policy():
    assert _config().policy == ProviderPolicy(sort="order")
    assert _config().policy.describe() == "none"


@pytest.mark.parametrize(
    "settings",
    [
        {"provider_metadata": {"anthropc": {"region": "eu"}}},
        {"provider_metadata": {"anthropic": {"retention": "forever"}}},
        {"provider_metadata": {"anthropic": {"region": "eu west"}}},
        {"provider_metadata": {"anthropic": {"dpa": True, "zdr": True}}},
        {"policy_only": "anthropic,foo"},
        {"policy_ignore": "bar"},
        {"policy_residency": "unknown"},
        {"policy_sort": "latency"},
        {"model_prices": {"gpt-4o-mini": {"input_per_mtok": 1, "output_per_mtok": 1}}},
        {"model_prices": {"openai/gpt-4o-mini": {"input_per_mtok": -1, "output_per_mtok": 1}}},
    ],
)
def test_invalid_policy_configuration_fails_at_startup(settings):
    with pytest.raises(ValidationError):
        GatewayConfig(_env_file=None, **settings)


# ─── Merging a per-call policy into the global one ─────────────────────────


def test_per_call_policy_can_only_narrow_the_global_policy():
    global_policy = ProviderPolicy(
        residency="eu",
        require_dpa=True,
        only=("anthropic", "azure"),
        ignore=("groq",),
        max_cost_usd=0.01,
    )
    per_call = ProviderPolicy(
        require_dpa=False,
        forbid_training=True,
        only=("anthropic", "groq", "openai"),
        ignore=(),
        max_cost_usd=5.0,
    )

    merged = global_policy.narrowed_by(per_call)

    assert merged == ProviderPolicy(
        residency="eu",
        require_dpa=True,
        forbid_training=True,
        only=("anthropic",),
        ignore=("groq",),
        max_cost_usd=0.01,
    )
    assert global_policy.narrowed_by(None) is global_policy


def test_conflicting_residencies_fail_closed():
    with pytest.raises(PolicyViolationError, match="conflicts with the global residency=eu"):
        ProviderPolicy(residency="eu").narrowed_by(ProviderPolicy(residency="us"))


async def test_global_and_per_call_budget_checks_must_both_allow():
    def allow(provider, model, estimated_cost_usd):
        return True

    async def deny(provider, model, estimated_cost_usd):
        return False

    denied = ProviderPolicy(budget_check=allow).narrowed_by(ProviderPolicy(budget_check=deny))
    allowed = ProviderPolicy(budget_check=allow).narrowed_by(ProviderPolicy(budget_check=allow))
    assert await call_budget_check(denied.budget_check, "groq", "m", None) is False
    assert await call_budget_check(allowed.budget_check, "groq", "m", None) is True


# ─── Fail closed ────────────────────────────────────────────────────────────


async def test_conservative_defaults_exclude_every_provider_under_eu_residency():
    calls = {p: _never() for p in PROVIDERS}
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(PolicyViolationError) as info:
            await complete(
                system="Score this CV.", prompt=SECRET, config=_config(policy_residency="eu")
            )

    error = info.value
    assert isinstance(error, LLMError)
    assert set(error.exclusions) == set(PROVIDERS)
    for reasons in error.exclusions.values():
        assert "region=unknown does not match residency=eu" in reasons
    assert "residency=eu" in str(error)
    assert SECRET not in str(error)
    assert "Score this CV" not in str(error)
    for mock in calls.values():
        mock.assert_not_awaited()
    assert classify(error) is ErrorKind.POLICY_VIOLATION
    assert POLICIES[ErrorKind.POLICY_VIOLATION].trips_breaker is False
    assert POLICIES[ErrorKind.POLICY_VIOLATION].retry is False


@pytest.mark.parametrize(
    ("requirement", "anthropic_metadata", "reason"),
    [
        (
            {"policy_require_zero_retention": True},
            {**EU_COMPLIANT, "retention": "abuse_monitoring_30d"},
            "retention=abuse_monitoring_30d, zero retention required",
        ),
        ({"policy_forbid_training": True}, {"dpa": True}, "trains_on_data is not asserted false"),
        ({"policy_require_dpa": True}, {"trains_on_data": False}, "no DPA asserted"),
        ({"policy_only": "openai"}, EU_COMPLIANT, "not listed in only"),
        ({"policy_ignore": "anthropic"}, EU_COMPLIANT, "listed in ignore"),
    ],
)
async def test_each_requirement_excludes_a_provider_that_fails_it(
    caplog, requirement, anthropic_metadata, reason
):
    caplog.set_level(logging.DEBUG, logger="llm_gateway")
    calls = {"anthropic": _never(), "openai": AsyncMock(return_value=_result("from openai"))}
    config = _config(
        provider_order="anthropic,openai",
        provider_metadata={"anthropic": anthropic_metadata, "openai": EU_COMPLIANT},
        **requirement,
    )
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(system="s", prompt="p", config=config)

    assert result.provider == "openai"
    calls["anthropic"].assert_not_awaited()
    assert f"provider=anthropic reasons={reason}" in caplog.text


async def test_failover_never_reaches_an_excluded_provider():
    calls = {
        "anthropic": AsyncMock(side_effect=RuntimeError("down")),
        "groq": _never(),
        "openai": AsyncMock(side_effect=RuntimeError("also down")),
    }
    config = _config(
        policy_residency="eu",
        provider_metadata={"anthropic": EU_COMPLIANT, "openai": EU_COMPLIANT},
    )
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(LLMError) as info:
            await complete(system="s", prompt="p", config=config)

    assert not isinstance(info.value, PolicyViolationError)
    calls["anthropic"].assert_awaited_once()
    calls["openai"].assert_awaited_once()
    calls["groq"].assert_not_awaited()


async def test_only_eligible_provider_without_a_key_still_fails_closed():
    calls = {"anthropic": _never(), "groq": _never()}
    config = _config(
        provider_order="anthropic,groq",
        groq_api_key="",
        policy_residency="eu",
        provider_metadata={"groq": EU_COMPLIANT},
    )
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(PolicyViolationError) as info:
            await complete(system="s", prompt="p", config=config)
    assert set(info.value.exclusions) == {"anthropic"}


def _azure_config(deployment: str, metadata: dict, **overrides) -> GatewayConfig:
    return _config(
        provider_order="azure,anthropic,groq,openai",
        azure_endpoint="https://my-resource.services.ai.azure.com",
        azure_model=deployment,
        azure_auth="api_key",
        azure_api_key="k",
        provider_metadata={"azure": metadata},
        **overrides,
    )


async def test_eu_only_routes_to_an_asserted_eu_azure_deployment():
    calls = {
        "azure": AsyncMock(return_value=_result("from azure", "mistral-medium-3-5")),
        **{p: _never() for p in PROVIDERS},
    }
    config = _azure_config(
        "mistral-medium-3-5",
        {"region": "eu", "retention": "abuse_monitoring_30d", "trains_on_data": False, "dpa": True},
        azure_reasoning_effort="",
        policy_residency="eu",
        policy_require_dpa=True,
        policy_forbid_training=True,
    )
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(system="s", prompt=SECRET, config=config)
    assert result.provider == "azure"
    assert result.cost_usd is None  # no default price for an azure deployment


async def test_global_gpt_oss_azure_deployment_is_excluded_under_eu_residency():
    calls = {"azure": _never(), **{p: _never() for p in PROVIDERS}}
    config = _azure_config(
        "gpt-oss-120b",
        {"region": "global", "trains_on_data": False, "dpa": True},
        policy_residency="eu",
    )
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(PolicyViolationError) as info:
            await complete(system="s", prompt="p", config=config)
    assert info.value.exclusions["azure"] == ("region=global does not match residency=eu",)
    assert set(info.value.exclusions) == {"azure", *PROVIDERS}


async def test_per_call_policy_applies_without_a_global_policy():
    calls = {
        "anthropic": _never(),
        "groq": _never(),
        "openai": AsyncMock(return_value=_result("from openai")),
    }
    config = _config(provider_metadata={"openai": EU_COMPLIANT})
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(
            system="s", prompt="p", config=config, policy=ProviderPolicy(residency="eu")
        )
    assert result.provider == "openai"


async def test_disjoint_only_lists_allow_nothing():
    calls = {p: _never() for p in PROVIDERS}
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(PolicyViolationError, match="only=<none>"):
            await complete(
                system="s",
                prompt="p",
                config=_config(policy_only="anthropic"),
                policy=ProviderPolicy(only=("groq",)),
            )


async def test_force_provider_is_subject_to_the_policy():
    calls = {"anthropic": _never()}
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(PolicyViolationError):
            await complete(
                system="s",
                prompt="p",
                config=_config(policy_require_dpa=True),
                force_provider="anthropic",
                model="claude-sonnet-5",
            )
    calls["anthropic"].assert_not_awaited()


async def test_chat_fails_closed():
    calls = {p: _never() for p in PROVIDERS}
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        with pytest.raises(PolicyViolationError):
            await chat(messages=_MESSAGES, config=_config(policy_forbid_training=True))
    for mock in calls.values():
        mock.assert_not_awaited()


async def test_stream_fails_closed_before_opening_any_provider():
    opened: list[str] = []
    calls = {p: _stream(opened, p, StreamDelta(content="leak")) for p in PROVIDERS}
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(PolicyViolationError):
            async for _ in stream_chat(messages=_MESSAGES, config=_config(policy_residency="eu")):
                pass
    assert opened == []


async def test_stream_failover_skips_excluded_providers():
    opened: list[str] = []
    calls = {
        "anthropic": _stream(opened, "anthropic", error=RuntimeError("down")),
        "groq": _stream(opened, "groq", StreamDelta(content="non-compliant")),
        "openai": _stream(opened, "openai", StreamDelta(content="from openai")),
    }
    config = _config(
        policy_residency="eu",
        provider_metadata={"anthropic": EU_COMPLIANT, "openai": EU_COMPLIANT},
    )
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        deltas = [d async for d in stream_chat(messages=_MESSAGES, config=config)]
    assert opened == ["anthropic", "openai"]
    assert [d.content for d in deltas] == ["from openai"]


async def test_exclusions_are_logged_and_traced_without_prompt_content(caplog):
    caplog.set_level(logging.DEBUG, logger="llm_gateway")
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    calls = {"anthropic": _never(), "groq": AsyncMock(return_value=_result("from groq"))}
    with (
        patch("llm_gateway.router.CALLS", calls),
        patch("llm_gateway.router.get_tracer", return_value=tracer_provider.get_tracer("t")),
    ):
        await complete(
            system="s",
            prompt=SECRET,
            config=_config(provider_order="anthropic,groq", policy_ignore="anthropic"),
        )

    (span,) = exporter.get_finished_spans()
    assert span.attributes["llm_gateway.policy"] == "ignore=anthropic"
    assert span.attributes["llm_gateway.policy.eligible_providers"] == ("groq",)
    assert span.attributes["llm_gateway.policy.excluded_providers"] == ("anthropic",)
    assert span.attributes["llm_gateway.policy.exclusion_reasons"] == (
        "anthropic: listed in ignore",
    )
    assert "provider=anthropic reasons=listed in ignore" in caplog.text
    assert SECRET not in caplog.text
    assert all(SECRET not in str(value) for value in span.attributes.values())


async def test_without_a_policy_the_chain_is_unchanged():
    attempted: list[str] = []

    def failing(name):
        async def call(*args, **kwargs):
            attempted.append(name)
            raise RuntimeError("down")

        return call

    calls = {p: failing(p) for p in PROVIDERS}
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(LLMError) as info:
            await complete(system="s", prompt="p", config=_config())
    assert attempted == list(PROVIDERS)
    assert str(info.value) == "All configured providers failed. Last error: down"


# ─── Capability filter ──────────────────────────────────────────────────────

_SCHEMA = {"type": "object", "required": ["score"]}


async def test_output_schema_on_json_mode_groq_model_is_allowed_by_default():
    groq_call = AsyncMock(return_value=_result('{"score": 1}', "llama-3.3-70b-versatile"))
    with patch("llm_gateway.router.CALLS", {"groq": groq_call}):
        result = await complete_with_usage(
            system="s", prompt="p", config=_config(provider_order="groq"), output_schema=_SCHEMA
        )
    assert result.parsed == {"score": 1}


async def test_require_parameters_skips_non_strict_groq_model_for_output_schema():
    calls = {
        "groq": _never(),
        "openai": AsyncMock(return_value=_result('{"score": 2}', "gpt-4o-mini")),
    }
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(
            system="s",
            prompt="p",
            config=_config(provider_order="groq,openai"),
            output_schema=_SCHEMA,
            policy=ProviderPolicy(require_parameters=True),
        )
    assert result.provider == "openai"


async def test_require_parameters_keeps_strict_groq_models():
    groq_call = AsyncMock(return_value=_result('{"score": 3}', "openai/gpt-oss-120b"))
    with patch("llm_gateway.router.CALLS", {"groq": groq_call}):
        result = await complete_with_usage(
            system="s",
            prompt="p",
            config=_config(
                provider_order="groq",
                groq_model="openai/gpt-oss-120b",
                policy_require_parameters=True,
            ),
            output_schema=_SCHEMA,
        )
    assert result.provider == "groq"


async def test_capability_filter_emptying_the_chain_names_the_missing_capability():
    with patch("llm_gateway.router.CALLS", {"groq": _never()}):
        with pytest.raises(UnsupportedCapabilityError) as info:
            await complete_with_usage(
                system="s",
                prompt="p",
                config=_config(provider_order="groq"),
                output_schema=_SCHEMA,
                policy=ProviderPolicy(require_parameters=True),
            )
    assert isinstance(info.value, PolicyViolationError)
    assert "has only JSON mode, not strict structured output" in str(info.value)


async def test_image_request_skips_text_only_groq_model():
    calls = {
        "groq": _never(),
        "anthropic": AsyncMock(
            return_value=ChatResult(content="a cat", model="m", input_tokens=1, output_tokens=1)
        ),
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        }
    ]
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        result = await chat(messages=messages, config=_config(provider_order="groq,anthropic"))
    assert result.content == "a cat"


def test_capability_table():
    assert capabilities_for("groq", "llama-3.3-70b-versatile").images is False
    assert capabilities_for("groq", "llama-3.3-70b-versatile").structured_output == "json_mode"
    assert capabilities_for("groq", "openai/gpt-oss-20b").structured_output == "strict"
    assert capabilities_for("anthropic", "claude-haiku-4-5-20251001").structured_output == "strict"
    assert capabilities_for("openai", "gpt-4o-mini").structured_output == "strict"
    assert capabilities_for("azure", "mistral-medium-3-5") == ModelCapabilities()
    assert capabilities_for("azure", "gpt-oss-120b") == ModelCapabilities(
        structured_output="strict", tools=True, images=False, streaming=True
    )
    assert missing_capabilities("azure", "dep", ["streaming"], require_parameters=False) == []
    assert missing_capabilities("azure", "dep", ["streaming"], require_parameters=True) == [
        "streaming support is unknown for model dep (require_parameters)"
    ]


# ─── Cost accounting and caps ───────────────────────────────────────────────

_HAIKU = ModelPrice(
    input_per_mtok=1.0, output_per_mtok=5.0, cached_input_per_mtok=0.1, cache_write_per_mtok=1.25
)


def test_cost_usd_charges_each_token_class_at_its_rate():
    usage = Usage(
        input_tokens=4150,
        output_tokens=7,
        cache_read_input_tokens=4000,
        cache_creation_input_tokens=100,
    )
    # 50 uncached × $1 + 4000 cache reads × $0.10 + 100 cache writes × $1.25 + 7 output × $5
    assert cost_usd(_HAIKU, usage) == pytest.approx(610 / 1_000_000)


def test_cost_usd_without_cache_rates_uses_the_input_rate():
    price = ModelPrice(input_per_mtok=0.15, output_per_mtok=0.6)
    usage = Usage(input_tokens=1000, output_tokens=100, cache_read_input_tokens=400)
    assert cost_usd(price, usage) == pytest.approx((1000 * 0.15 + 100 * 0.6) / 1_000_000)


def test_cost_usd_is_none_without_a_price():
    assert cost_usd(None, Usage(input_tokens=10, output_tokens=10)) is None
    assert lookup_price("groq", "llama-3.3-70b-versatile") is None
    assert lookup_price("azure", "gpt-oss-120b") is None


def test_default_prices_cover_the_default_models():
    defaults = GatewayConfig(_env_file=None)
    assert lookup_price("anthropic", defaults.claude_model) == _HAIKU
    assert lookup_price("openai", defaults.openai_model) == ModelPrice(
        input_per_mtok=0.15, output_per_mtok=0.6, cached_input_per_mtok=0.075
    )
    assert lookup_price("groq", defaults.groq_model) == ModelPrice(
        input_per_mtok=0.15, output_per_mtok=0.6
    )


def test_worst_case_estimate_is_chars_over_four_plus_max_tokens():
    assert estimate_tokens(0) == 0
    assert estimate_tokens(9) == 3
    # 1000 input tokens at the higher cache-write rate + 1000 output tokens
    estimate = worst_case_cost_usd(_HAIKU, input_chars=4000, max_output_tokens=1000)
    assert estimate == pytest.approx((1000 * 1.25 + 1000 * 5.0) / 1_000_000)
    assert worst_case_cost_usd(None, input_chars=4000, max_output_tokens=1000) is None


async def test_complete_with_usage_reports_cost_from_actual_usage():
    result = _result("ok", "gpt-4o-mini", input_tokens=1000, cache_read_input_tokens=500)
    with patch("llm_gateway.router.CALLS", {"openai": AsyncMock(return_value=result)}):
        completion = await complete_with_usage(
            system="s", prompt="p", config=_config(provider_order="openai")
        )
    # 500 uncached × $0.15 + 500 cached × $0.075 + 1 output × $0.60, per 1M
    assert completion.cost_usd == pytest.approx((75 + 37.5 + 0.6) / 1_000_000)


async def test_cost_is_none_for_an_unpriced_model_and_set_once_priced():
    result = _result("ok", "llama-3.3-70b-versatile", input_tokens=1_000_000)
    with patch("llm_gateway.router.CALLS", {"groq": AsyncMock(return_value=result)}):
        unpriced = await complete_with_usage(
            system="s", prompt="p", config=_config(provider_order="groq")
        )
        priced = await complete_with_usage(
            system="s",
            prompt="p",
            config=_config(
                provider_order="groq",
                model_prices={
                    "groq/llama-3.3-70b-versatile": {"input_per_mtok": 0.5, "output_per_mtok": 1}
                },
            ),
        )
    assert unpriced.cost_usd is None
    assert priced.cost_usd == pytest.approx(0.5 + 1 / 1_000_000)


async def test_max_cost_usd_skips_providers_over_the_cap_or_without_a_price():
    calls = {
        "anthropic": _never(),
        "groq": _never(),
        "openai": AsyncMock(return_value=_result("from openai", "gpt-4o-mini")),
    }
    # 400 chars ≈ 100 input tokens, max_tokens=2000:
    # anthropic ≈ $0.010125, groq unpriced, openai ≈ $0.001215.
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(
            system="",
            prompt="x" * 400,
            max_tokens=2000,
            config=_config(),
            policy=ProviderPolicy(max_cost_usd=0.005),
        )
    assert result.provider == "openai"

    with patch("llm_gateway.router.CALLS", {p: _never() for p in PROVIDERS}):
        with pytest.raises(PolicyViolationError) as info:
            await complete(
                system="",
                prompt="x" * 400,
                max_tokens=2000,
                config=_config(policy_max_cost_usd=0.0001),
            )
    reasons = info.value.exclusions
    assert "exceeds max_cost_usd $0.0001" in reasons["anthropic"][0]
    assert reasons["groq"] == ("no price for groq/llama-3.3-70b-versatile to check max_cost_usd",)
    assert "exceeds max_cost_usd" in reasons["openai"][0]


async def test_budget_check_denial_skips_the_provider_without_tripping_its_breaker():
    seen = []

    def budget(provider, model, estimated_cost_usd):
        seen.append((provider, model, estimated_cost_usd))
        return provider != "anthropic"

    calls = {"anthropic": _never(), "groq": AsyncMock(return_value=_result("from groq"))}
    config = _config(provider_order="anthropic,groq", breaker_failure_threshold=1)
    with patch("llm_gateway.router.CALLS", calls):
        for _ in range(3):
            result = await complete_with_usage(
                system="",
                prompt="x" * 400,
                max_tokens=100,
                config=config,
                policy=ProviderPolicy(budget_check=budget),
            )
            assert result.provider == "groq"

    assert not breaker.is_open("anthropic")
    assert breaker._failures.get("anthropic", 0) == 0
    assert seen[0] == (
        "anthropic",
        "claude-haiku-4-5-20251001",
        pytest.approx((100 * 1.25 + 100 * 5.0) / 1_000_000),
    )
    assert seen[1] == ("groq", "llama-3.3-70b-versatile", None)


async def test_async_budget_check_denying_every_provider_fails_closed():
    async def deny(provider, model, estimated_cost_usd):
        return False

    calls = {p: _never() for p in PROVIDERS}
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(PolicyViolationError) as info:
            await complete(
                system="s", prompt="p", config=_config(), policy=ProviderPolicy(budget_check=deny)
            )
    assert info.value.exclusions == {p: ("denied by budget_check",) for p in PROVIDERS}
    assert breaker._failures == {}


async def test_budget_denial_after_a_provider_failure_is_reported_not_bypassed():
    calls = {"anthropic": AsyncMock(side_effect=RuntimeError("down")), "groq": _never()}

    def deny_groq(provider, model, estimated_cost_usd):
        return provider != "groq"

    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(LLMError) as info:
            await complete(
                system="s",
                prompt="p",
                config=_config(provider_order="anthropic,groq"),
                policy=ProviderPolicy(budget_check=deny_groq),
            )
    assert not isinstance(info.value, PolicyViolationError)
    assert "denied by budget_check): groq" in str(info.value)
    calls["groq"].assert_not_awaited()


async def test_stream_budget_denial_mid_failover_never_opens_the_denied_provider():
    opened: list[str] = []
    calls = {
        "anthropic": _stream(opened, "anthropic", error=RuntimeError("down")),
        "groq": _stream(opened, "groq", StreamDelta(content="denied")),
    }

    async def deny_groq(provider, model, estimated_cost_usd):
        return provider != "groq"

    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        with pytest.raises(LLMError):
            async for _ in stream_chat(
                messages=_MESSAGES,
                config=_config(provider_order="anthropic,groq"),
                policy=ProviderPolicy(budget_check=deny_groq),
            ):
                pass
    assert opened == ["anthropic"]


# ─── sort=price ─────────────────────────────────────────────────────────────


def _recording_failures(attempted: list[str]):
    def failing(name):
        async def call(*args, **kwargs):
            attempted.append(name)
            raise RuntimeError("down")

        return call

    return {p: failing(p) for p in PROVIDERS}


@pytest.mark.parametrize(
    ("prices", "expected"),
    [
        # Unpriced groq goes last; openai is cheaper than anthropic.
        ({}, ["openai", "anthropic", "groq"]),
        (
            {"groq/llama-3.3-70b-versatile": {"input_per_mtok": 0.05, "output_per_mtok": 0.08}},
            ["groq", "openai", "anthropic"],
        ),
    ],
)
async def test_sort_price_tries_the_cheapest_estimated_provider_first(prices, expected):
    attempted: list[str] = []
    with patch("llm_gateway.router.CALLS", _recording_failures(attempted)):
        with pytest.raises(LLMError):
            await complete(
                system="s",
                prompt="p",
                config=_config(model_prices=prices),
                policy=ProviderPolicy(sort="price"),
            )
    assert attempted == expected


async def test_per_call_sort_order_overrides_a_global_price_sort():
    attempted: list[str] = []
    with patch("llm_gateway.router.CALLS", _recording_failures(attempted)):
        with pytest.raises(LLMError):
            await complete(
                system="s",
                prompt="p",
                config=_config(policy_sort="price"),
                policy=ProviderPolicy(sort="order"),
            )
    assert attempted == list(PROVIDERS)
