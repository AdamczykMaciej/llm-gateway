"""Security pass 4: broader secret scrubbing, config repr, https-only token endpoints."""

import logging

import pytest
from pydantic import ValidationError

from llm_gateway import GatewayConfig
from llm_gateway.errors import redact_secrets


@pytest.mark.parametrize(
    "text, leaked",
    [
        ("GET https://g/v1:stream?key=AIzaSyA1234567890abcdefghijklmnopqrstuv&alt=sse", "AIzaSy"),
        ("bad key AIzaSyA1234567890abcdefghijklmnopqrstuv", "AIzaSy"),
        ("access_token=ya29.a0AbCdEfGhIjK", "ya29"),
        ("refresh_token=1//0gabcdefg", "0gabcdefg"),
        ("client_secret=zzzzzzzz", "zzzzzzzz"),
        ("Ocp-Apim-Subscription-Key: abcdef0123456789abcdef", "abcdef0123"),
        ("?api-key=abcdef123456&x=1", "abcdef123456"),
    ],
)
def test_redact_secrets_scrubs_more_shapes(text: str, leaked: str):
    assert leaked not in redact_secrets(text)


def test_redact_secrets_keeps_harmless_text():
    assert redact_secrets("model not found: key length exceeded") == (
        "model not found: key length exceeded"
    )


def test_config_repr_hides_otlp_credentials():
    c = GatewayConfig(
        otel_exporter_otlp_endpoint="https://o.example/x",
        otel_exporter_otlp_headers="Authorization=Bearer SECRETHDR1",
    )
    assert "SECRETHDR1" not in repr(c)
    assert "SECRETHDR1" not in str(c)
    assert "SECRETHDR1" not in redact_secrets("failed SECRETHDR1", c)


@pytest.mark.parametrize("field", ["azure_endpoint", "huggingface_base_url"])
def test_token_bearing_endpoints_require_https(field: str):
    with pytest.raises(ValidationError):
        GatewayConfig(**{field: "http://example.com/v1"})


def test_open_service_logs_a_warning(caplog: pytest.LogCaptureFixture):
    pytest.importorskip("fastapi")
    from llm_gateway.service.app import create_app

    with caplog.at_level(logging.WARNING, logger="llm_gateway"):
        create_app(GatewayConfig(gateway_api_keys=""))
    assert any("GATEWAY_API_KEYS" in r.getMessage() for r in caplog.records)
