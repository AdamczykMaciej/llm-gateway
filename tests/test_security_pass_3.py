"""Regression tests for the 0.16.8 security pass (redirects, metadata-host URLs,
error text bounds, span exception events)."""

from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from llm_gateway import AllProvidersExhaustedError, GatewayConfig, chat, reset_circuit_breakers
from llm_gateway.errors import redact_secrets
from llm_gateway.providers import anthropic, azure, groq, openai, openai_compat

KEY = "sk-test-SECRET123"
MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


# ── redirects ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ssl_verify", [True, False])
def test_sdk_clients_do_not_follow_redirects(ssl_verify):
    """A redirect to another host would carry x-api-key / api-key headers
    (httpx only strips Authorization across origins)."""
    config = GatewayConfig(
        anthropic_api_key=KEY,
        openai_api_key=KEY,
        groq_api_key=KEY,
        azure_endpoint="https://example.openai.azure.com",
        azure_api_key=KEY,
        azure_model="m",
        ssl_verify=ssl_verify,
    )
    for provider in (anthropic, openai, groq):
        provider._clients.clear()
        assert provider._client(config)._client.follow_redirects is False
        provider._clients.clear()
    azure._clients.clear()
    assert azure._client(config)._client.follow_redirects is False
    azure._clients.clear()


def test_openai_compatible_client_does_not_follow_redirects():
    config = GatewayConfig(
        openai_compat_base_url="https://llm.example.com/v1", openai_compat_api_key=KEY
    )
    openai_compat._clients.clear()
    assert openai_compat._client(config)._client.follow_redirects is False
    openai_compat._clients.clear()


# ── metadata / link-local hosts ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest",
        "https://169.254.169.254",
        "http://169.254.170.2/v2",
        "http://[fe80::1]/v1",
        "http://[fd00:ec2::254]/v1",
        "http://[::ffff:169.254.169.254]/v1",
        "http://2852039166/v1",  # 169.254.169.254 as one decimal number
        "http://0xa9fea9fe/v1",
        "http://metadata.google.internal/v1",
        "http://metadata.google.internal./v1",
        "http://METADATA.GOOGLE.INTERNAL/v1",
    ],
)
@pytest.mark.parametrize(
    "field", ["openai_compat_base_url", "huggingface_base_url", "azure_endpoint"]
)
def test_metadata_hosts_are_rejected(field, url):
    with pytest.raises(ValidationError) as exc:
        GatewayConfig(**{field: url})
    assert "169.254" not in str(exc.value) or field in str(exc.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080/v1",
        "http://127.0.0.1:8000/v1",
        "http://10.0.0.5/v1",
        "http://[::1]:11434/v1",
        "https://api.example.com/v1",
        "http://llm.internal.example/v1",
    ],
)
def test_ordinary_hosts_are_still_accepted(url):
    assert GatewayConfig(openai_compat_base_url=url).openai_compat_base_url == url


# ── bounded error text ───────────────────────────────────────────────────────


async def test_exhausted_error_text_is_length_capped():
    boom = RuntimeError("upstream said: " + "x" * 1_000_000)
    config = GatewayConfig(anthropic_api_key=KEY, provider_order="anthropic", retry_attempts=1)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(side_effect=boom)}):
        with pytest.raises(AllProvidersExhaustedError) as exc:
            await chat(messages=MSGS, config=config)
    assert len(str(exc.value)) < 4000


def test_redact_secrets_max_chars():
    out = redact_secrets("a" * 5000, max_chars=100)
    assert len(out) <= 120 and out.endswith("[truncated]")
    assert redact_secrets("short", max_chars=100) == "short"


# ── span exception events ────────────────────────────────────────────────────


async def test_span_exception_event_does_not_carry_provider_error_text():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    from llm_gateway import tracing

    tracer = provider.get_tracer("llm_gateway")
    # an unconfigured-by-value secret in the provider error: only pattern-redactable
    cause = RuntimeError("401 for key sk-live-ABCDEFGH12345678 and prompt: my SSN is 123-45-6789")
    config = GatewayConfig(anthropic_api_key=KEY, provider_order="anthropic", retry_attempts=1)
    with (
        patch.object(tracing, "_tracer", tracer),
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(side_effect=cause)}),
    ):
        with pytest.raises(AllProvidersExhaustedError):
            await chat(messages=MSGS, config=config)
    spans = [s for s in exporter.get_finished_spans() if s.name == "llm_gateway.chat"]
    assert spans
    blob = repr([(e.name, dict(e.attributes or {})) for s in spans for e in s.events])
    blob += repr([s.status.description for s in spans])
    assert [s.status.description for s in spans] == ["AllProvidersExhaustedError"]
    assert "sk-live-ABCDEFGH12345678" not in blob
    assert "123-45-6789" not in blob
    assert trace is not None
