"""Security pass 6: credentials stay out of config dumps."""

import pytest

from llm_gateway import GatewayConfig

_SECRETS = {
    "anthropic_api_key": "SECRET-anthropic-1",
    "groq_api_key": "SECRET-groq-2",
    "openai_api_key": "SECRET-openai-3",
    "azure_api_key": "SECRET-azure-4",
    "mistral_api_key": "SECRET-mistral-5",
    "openrouter_api_key": "SECRET-openrouter-6",
    "openai_compat_api_key": "SECRET-compat-7",
    "huggingface_api_key": "SECRET-hf-8",
    "gateway_api_keys": "SECRET-gateway-9",
    "otel_exporter_otlp_headers": "Authorization=Bearer SECRET-otlp-10",
    "otel_exporter_otlp_endpoint": "https://o.example/SECRET-endpoint-11",
}


@pytest.mark.parametrize("mode", ["python", "json"])
def test_model_dump_omits_credentials(mode: str):
    config = GatewayConfig(_env_file=None, **_SECRETS)  # type: ignore[arg-type]
    dumped = str(config.model_dump(mode=mode)) + config.model_dump_json()
    assert "SECRET" not in dumped


def test_credentials_still_usable_after_model_copy():
    config = GatewayConfig(_env_file=None, anthropic_api_key="SECRET-anthropic-1")
    copy = config.model_copy(update={"claude_model": "x"})
    assert copy.anthropic_api_key == "SECRET-anthropic-1"
