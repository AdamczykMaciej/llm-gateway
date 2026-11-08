"""Regression tests for the 0.13.1 security pass (secrets in reprs/errors,
hostile JSON, auth edge cases, request-size accounting)."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from llm_gateway import GatewayConfig, LLMError, reset_circuit_breakers
from llm_gateway.chat import chat
from llm_gateway.errors import InvalidOutputError, redact_secrets
from llm_gateway.providers._anthropic_translate import to_anthropic_messages
from llm_gateway.service import create_app
from llm_gateway.service.rate_limit import reset as reset_rate_limits
from llm_gateway.service.usage import reset as reset_usage
from llm_gateway.structured import OutputSchema

SECRETS = {
    "anthropic_api_key": "sk-ant-LEAKME-anthropic-0001",
    "groq_api_key": "gsk_LEAKME-groq-0002",
    "openai_api_key": "sk-LEAKME-openai-0003",
    "azure_api_key": "LEAKME-azure-0004",
    "mistral_api_key": "LEAKME-mistral-0005",
    "openrouter_api_key": "LEAKME-openrouter-0006",
    "openai_compat_api_key": "LEAKME-compat-0007",
    "huggingface_api_key": "hf_LEAKME-hf-0008",
    "gateway_api_keys": "LEAKME-gw-0009,LEAKME-gw-0010",
}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    reset_rate_limits()
    reset_usage()
    yield
    reset_circuit_breakers()
    reset_rate_limits()
    reset_usage()


def test_config_repr_and_str_never_contain_secrets():
    config = GatewayConfig(**SECRETS)
    for text in (
        repr(config),
        str(config),
        repr(config.model_dump_json) + str(config.__repr_args__()),
    ):
        assert "LEAKME" not in text


def test_redact_secrets_masks_known_values_and_token_shapes():
    config = GatewayConfig(**SECRETS)
    raw = (
        "401 Incorrect API key sk-LEAKME-openai-0003; Authorization: Bearer abc.def-123 "
        "x-api-key: topsecretvalue1 token=hf_abcdefghijklmnop1234 sk-ant-api03-ABCDEFGHIJKL"
    )
    out = redact_secrets(raw, config)
    for needle in (
        "LEAKME",
        "abc.def-123",
        "topsecretvalue1",
        "hf_abcdefghijklmnop1234",
        "ABCDEFGHIJKL",
    ):
        assert needle not in out
    assert "Incorrect API key" in out  # the diagnostic text survives


async def test_all_providers_failed_message_and_span_do_not_leak_keys():
    config = GatewayConfig(
        anthropic_api_key=SECRETS["anthropic_api_key"], provider_order="anthropic"
    )
    boom = AsyncMock(
        side_effect=RuntimeError(
            f"auth failed for key {SECRETS['anthropic_api_key']} "
            "Authorization: Bearer zzz-secret-tok"
        )
    )
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": boom}):
        with pytest.raises(LLMError) as exc:
            await chat(messages=[{"role": "user", "content": "hi"}], config=config)
    msg = str(exc.value)
    assert "LEAKME" not in msg and "zzz-secret-tok" not in msg
    assert "auth failed" in msg


def _client(**overrides) -> TestClient:
    defaults = dict(anthropic_api_key="test-key", provider_order="anthropic")
    defaults.update(overrides)
    return TestClient(create_app(GatewayConfig(**defaults)))


def test_non_ascii_bearer_is_401_not_500():
    client = _client(gateway_api_keys="good-key")
    resp = client.get("/v1/usage", headers={"Authorization": "Bearer caf\xe9".encode("latin-1")})
    assert resp.status_code == 401


def test_service_does_not_echo_upstream_secrets():
    client = _client()
    boom = AsyncMock(side_effect=RuntimeError("bad key Bearer leaked-token-xyz at http://10.0.0.5"))
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": boom}):
        resp = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    assert resp.status_code == 503
    assert "leaked-token-xyz" not in resp.text


def test_tool_definitions_and_tool_call_arguments_count_toward_prompt_ceiling():
    client = _client(max_prompt_chars=1000)
    big = "x" * 5000
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "type": "function", "function": {"name": "f", "arguments": big}}
                    ],
                },
            ]
        },
    )
    assert resp.status_code == 400
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "description": big}}],
        },
    )
    assert resp.status_code == 400


@pytest.mark.parametrize("value", [0, -5])
def test_non_positive_max_tokens_rejected(value):
    resp = _client().post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": value},
    )
    assert resp.status_code == 422


def test_structured_parse_survives_deeply_nested_model_output():
    so = OutputSchema.from_spec({"type": "object"})
    with pytest.raises(InvalidOutputError):
        so.parse("[" * 200_000)


def test_tool_call_arguments_hostile_json_raises_clean_value_error():
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "[" * 200_000},
                }
            ],
        }
    ]
    with pytest.raises(ValueError):
        to_anthropic_messages(msgs)
    msgs[0]["tool_calls"][0]["function"]["arguments"] = "{not json"
    with pytest.raises(ValueError):
        to_anthropic_messages(msgs)
