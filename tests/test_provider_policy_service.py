"""0.5.0 provider policy through the HTTP service: the request's `policy`
object narrows the server's global policy and can never loosen it. Runs the
real chat()/stream_chat() engines over mocked provider registries."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from llm_gateway import GatewayConfig, reset_circuit_breakers
from llm_gateway.providers.base import ChatResult, StreamDelta
from llm_gateway.service import create_app
from llm_gateway.service.rate_limit import reset as reset_rate_limits
from llm_gateway.service.usage import reset as reset_usage

EU_COMPLIANT = {"region": "eu", "retention": "zero", "trains_on_data": False, "dpa": True}


@pytest.fixture(autouse=True)
def _reset_state():
    reset_circuit_breakers()
    reset_rate_limits()
    reset_usage()
    yield
    reset_circuit_breakers()
    reset_rate_limits()
    reset_usage()


def _client(**overrides) -> TestClient:
    settings = dict(
        _env_file=None,
        anthropic_api_key="k",
        groq_api_key="k",
        provider_order="anthropic,groq",
        retry_attempts=1,
    )
    settings.update(overrides)
    return TestClient(create_app(GatewayConfig(**settings)))


def _post(client: TestClient, **body):
    return client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], **body}
    )


def _never() -> AsyncMock:
    return AsyncMock(side_effect=AssertionError("an excluded provider must never be called"))


def _served(content: str) -> AsyncMock:
    return AsyncMock(
        return_value=ChatResult(content=content, model="m", input_tokens=1, output_tokens=1)
    )


def test_request_policy_cannot_turn_off_a_server_requirement():
    client = _client(policy_require_dpa=True, provider_metadata={"groq": EU_COMPLIANT})
    calls = {"anthropic": _never(), "groq": _served("from groq")}
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        resp = _post(
            client,
            policy={"require_dpa": False, "only": ["anthropic", "groq"], "ignore": []},
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "from groq"
    calls["anthropic"].assert_not_awaited()


def test_request_cannot_unignore_a_provider():
    client = _client(policy_ignore="anthropic")
    calls = {"anthropic": _never(), "groq": _served("from groq")}
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        resp = _post(client, policy={"ignore": [], "only": ["anthropic"]})
    # only=[anthropic] ∩ server ignore=anthropic leaves nothing: fail closed.
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    calls["anthropic"].assert_not_awaited()
    calls["groq"].assert_not_awaited()


def test_request_residency_conflicting_with_the_server_is_rejected():
    client = _client(policy_residency="eu", provider_metadata={"groq": EU_COMPLIANT})
    calls = {"anthropic": _never(), "groq": _never()}
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        resp = _post(client, policy={"residency": "us"})
    assert resp.status_code == 400
    assert "conflicts with the global residency=eu" in resp.json()["error"]["message"]


def test_request_policy_can_narrow_the_chain():
    client = _client()
    calls = {"anthropic": _never(), "groq": _served("from groq")}
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        resp = _post(client, policy={"only": ["groq"]})
    assert resp.status_code == 200
    calls["anthropic"].assert_not_awaited()


def test_server_policy_excluding_every_provider_returns_400_without_calling():
    client = _client(policy_residency="eu")
    calls = {"anthropic": _never(), "groq": _never()}
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        resp = _post(client)
    assert resp.status_code == 400
    body = resp.json()
    assert "residency=eu" in body["error"]["message"]
    assert "hi" not in body["error"]["message"].split("Excluded:")[1]


def test_forced_provider_model_is_subject_to_the_server_policy():
    client = _client(policy_residency="eu", provider_metadata={"groq": EU_COMPLIANT})
    calls = {"anthropic": _never(), "groq": _never()}
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        resp = _post(client, model="anthropic/claude-sonnet-5")
    assert resp.status_code == 400
    calls["anthropic"].assert_not_awaited()


def test_unknown_policy_field_is_rejected_not_ignored():
    client = _client()
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": _never(), "groq": _never()}):
        resp = _post(client, policy={"zdr": True})
    assert resp.status_code == 422


def test_unknown_provider_id_in_policy_is_400():
    client = _client()
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": _never(), "groq": _never()}):
        resp = _post(client, policy={"only": ["mistral"]})
    assert resp.status_code == 400
    assert "unknown provider id" in resp.json()["error"]["message"]


def test_stream_policy_violation_emits_an_error_event_without_opening_a_provider():
    opened: list[str] = []

    def open_stream(name):
        def fn(*args, **kwargs):
            opened.append(name)

            async def generator():
                yield StreamDelta(content="leak", model="m")

            return generator()

        return fn

    client = _client(policy_residency="eu")
    calls = {"anthropic": open_stream("anthropic"), "groq": open_stream("groq")}
    with patch("llm_gateway.streaming.STREAM_CALLS", calls):
        resp = _post(client, stream=True, policy={"require_dpa": False})

    assert resp.status_code == 200
    lines = [line for line in resp.text.split("\n\n") if line.strip()]
    assert lines[-1] == "data: [DONE]"
    error = json.loads(lines[0].removeprefix("data: "))["error"]
    assert error["code"] == 400
    assert error["type"] == "invalid_request_error"
    assert opened == []
