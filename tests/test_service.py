from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from llm_gateway import GatewayConfig, reset_circuit_breakers
from llm_gateway.service import create_app


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _client(**config_overrides) -> TestClient:
    defaults = dict(anthropic_api_key="test-key", provider_order="anthropic")
    defaults.update(config_overrides)
    app = create_app(GatewayConfig(**defaults))
    return TestClient(app)


def test_health_requires_no_auth():
    client = _client(gateway_api_keys="secret-key")
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_internal_healthz_alias_for_cloud_run_probe():
    # /healthz itself is reserved platform-wide on Cloud Run (external
    # requests to it 404 at the edge regardless of app routes) — this is
    # what the startup probe targets instead. See infra/terraform/cloud_run.tf.
    client = _client(gateway_api_keys="secret-key")
    resp = client.get("/_internal/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_completions_rejects_missing_auth():
    client = _client(gateway_api_keys="secret-key")
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401


def test_chat_completions_rejects_wrong_key():
    client = _client(gateway_api_keys="secret-key")
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401


def test_chat_completions_success_with_valid_key():
    client = _client(gateway_api_keys="secret-key")
    with patch("llm_gateway.service.app.complete", AsyncMock(return_value="Hello there!")):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-key"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello there!"
    assert body["object"] == "chat.completion"


def test_chat_completions_open_when_no_keys_configured():
    # gateway_api_keys unset — deliberately open (documented behavior).
    client = _client()
    with patch("llm_gateway.service.app.complete", AsyncMock(return_value="ok")):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200


def test_chat_completions_forces_provider_from_model_prefix():
    client = _client(gateway_api_keys="secret-key")
    mock_complete = AsyncMock(return_value="forced")
    with patch("llm_gateway.service.app.complete", mock_complete):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": "Bearer secret-key"},
        )
    assert resp.status_code == 200
    assert mock_complete.await_args.kwargs["force_provider"] == "anthropic"
    assert mock_complete.await_args.kwargs["model"] == "claude-sonnet-4-6"


def test_models_endpoint_lists_configured_models():
    client = _client(gateway_api_keys="secret-key")
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "auto" in ids


def test_chat_completions_requires_a_user_message():
    client = _client(gateway_api_keys="secret-key")
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "be nice"}]},
        headers={"Authorization": "Bearer secret-key"},
    )
    assert resp.status_code == 400
