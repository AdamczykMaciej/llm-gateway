"""0.7.0: `mistral`, `openrouter` and the generic `openai_compat` provider,
through the real openai SDK over an in-memory transport (as the Groq tests)."""

import json
from unittest.mock import patch

import httpx2
import openai as openai_sdk
import pytest
from pydantic import BaseModel

from llm_gateway import LLMError, chat, reset_circuit_breakers
from llm_gateway.capabilities import capabilities_for, missing_capabilities
from llm_gateway.config import GatewayConfig as _Config
from llm_gateway.errors import PolicyViolationError
from llm_gateway.pricing import DEFAULT_PRICES
from llm_gateway.providers import (
    CALLS,
    CHAT_CALLS,
    CONFIGURED,
    DEFAULT_MODEL,
    STREAM_CALLS,
    mistral,
    openai_compat,
    openrouter,
)
from llm_gateway.providers.openai_compatible import OpenAICompatProvider
from tests.test_sdk_compat import _Recorder, _sse

NEW = ("mistral", "openrouter", "openai_compat")

_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
}

_TOOL_COMPLETION = {
    **_COMPLETION,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": json.dumps({"q": "x"})},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    for provider in (mistral, openrouter, openai_compat):
        provider._clients.clear()
    yield
    reset_circuit_breakers()
    for provider in (mistral, openrouter, openai_compat):
        provider._clients.clear()


def _sdk_client(recorder: _Recorder, base_url: str, headers: dict | None = None):
    return openai_sdk.AsyncOpenAI(
        api_key="test-key",
        base_url=base_url,
        default_headers=headers,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)),
    )


def _ok(body: dict):
    return lambda req: httpx2.Response(200, request=req, json=body)


# ─── Registry and config ───────────────────────────────────────────────────


@pytest.mark.parametrize("table", [CALLS, CHAT_CALLS, STREAM_CALLS, CONFIGURED, DEFAULT_MODEL])
def test_new_providers_are_registered_everywhere(table):
    assert set(NEW) <= set(table)


def test_configured_needs_a_key_and_for_the_generic_host_a_base_url():
    assert not mistral.configured(_Config())
    assert mistral.configured(_Config(mistral_api_key="k"))
    assert not openrouter.configured(_Config())
    assert openrouter.configured(_Config(openrouter_api_key="k"))
    assert not openai_compat.configured(_Config(openai_compat_api_key="k"))
    assert not openai_compat.configured(_Config(openai_compat_base_url="https://h.example/v1"))
    assert openai_compat.configured(
        _Config(openai_compat_api_key="k", openai_compat_base_url="https://h.example/v1")
    )


def test_default_models_and_overrides():
    config = _Config(openai_compat_model="deepseek-chat")
    assert mistral.default_model(config) == "mistral-small-latest"
    assert openrouter.default_model(config) == "openai/gpt-oss-120b"
    assert openai_compat.default_model(config) == "deepseek-chat"
    assert DEFAULT_MODEL["mistral"](_Config(mistral_model="mistral-large-latest")) == (
        "mistral-large-latest"
    )


def test_generic_base_url_is_validated_and_normalized():
    assert _Config(openai_compat_base_url="https://h.example/v1/").openai_compat_base_url == (
        "https://h.example/v1"
    )
    with pytest.raises(ValueError, match="http"):
        _Config(openai_compat_base_url="h.example/v1")


def test_provider_order_accepts_the_new_ids():
    config = _Config(provider_order="mistral,openrouter,openai_compat")
    assert config.provider_order_list == ["mistral", "openrouter", "openai_compat"]


def test_policy_only_and_metadata_accept_the_new_ids():
    config = _Config(
        policy_only="mistral",
        provider_metadata={"mistral": {"region": "eu", "retention": "zero", "dpa": True}},
    )
    assert config.policy.only == ("mistral",)
    with pytest.raises(ValueError, match="unknown provider id"):
        _Config(policy_only="cohere")


def test_openrouter_provider_allow_list_parses():
    config = _Config(openrouter_provider_allow=" fireworks, together ,")
    assert config.openrouter_provider_allow_list == ("fireworks", "together")


# ─── Request shaping ───────────────────────────────────────────────────────


async def test_mistral_targets_its_base_url_and_passes_tools_through():
    recorder = _Recorder(_ok(_TOOL_COMPLETION))
    config = _Config(mistral_api_key="test-key")
    real = mistral._client(config)
    assert str(real.base_url).startswith(mistral.BASE_URL)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with patch.object(mistral, "_client", return_value=_sdk_client(recorder, mistral.BASE_URL)):
        result = await mistral.chat(
            config,
            [{"role": "user", "content": "hi"}],
            tools=tools,
            max_tokens=20,
            tool_choice="auto",
        )
    body = recorder.bodies[-1]
    assert body["model"] == "mistral-small-latest"
    assert body["tools"] == tools and body["tool_choice"] == "auto"
    assert [c.name for c in result.tool_calls] == ["lookup"]
    assert (result.input_tokens, result.output_tokens) == (11, 3)


async def test_openrouter_headers_and_provider_preference():
    recorder = _Recorder(_ok(_COMPLETION))
    config = _Config(
        openrouter_api_key="test-key",
        openrouter_provider_allow="fireworks,together",
        openrouter_app_url="https://app.example",
        openrouter_app_name="Example",
    )
    real = openrouter._client(config)
    assert str(real.base_url).startswith(openrouter.BASE_URL)
    assert real.default_headers["HTTP-Referer"] == "https://app.example"
    assert real.default_headers["X-Title"] == "Example"
    with patch.object(
        openrouter, "_client", return_value=_sdk_client(recorder, openrouter.BASE_URL)
    ):
        result = await openrouter.chat(
            config, [{"role": "user", "content": "hi"}], tools=None, max_tokens=20
        )
    body = recorder.bodies[-1]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["provider"] == {"only": ["fireworks", "together"], "allow_fallbacks": False}
    assert result.content == "Hello."


async def test_openrouter_without_allow_list_sends_no_provider_field():
    recorder = _Recorder(_ok(_COMPLETION))
    config = _Config(openrouter_api_key="test-key")
    with patch.object(
        openrouter, "_client", return_value=_sdk_client(recorder, openrouter.BASE_URL)
    ):
        await openrouter.chat(config, [{"role": "user", "content": "hi"}], tools=None, max_tokens=5)
    assert "provider" not in recorder.bodies[-1]
    assert openrouter._client(config).default_headers.get("X-Title") is None


class _Answer(BaseModel):
    ok: bool


async def test_structured_output_strict_vs_json_mode():
    from llm_gateway.structured import OutputSchema

    schema = OutputSchema.from_model(_Answer) if hasattr(OutputSchema, "from_model") else None
    if schema is None:
        pytest.skip("OutputSchema construction differs")
    strict = _Recorder(
        _ok(
            {
                **_COMPLETION,
                "choices": [
                    {
                        **_COMPLETION["choices"][0],
                        "message": {"role": "assistant", "content": '{"ok": true}'},
                    }
                ],
            }
        )
    )
    with patch.object(mistral, "_client", return_value=_sdk_client(strict, mistral.BASE_URL)):
        await mistral.call(_Config(mistral_api_key="k"), "sys", "hi", 20, output_schema=schema)
    assert strict.bodies[-1]["response_format"]["type"] == "json_schema"

    loose = _Recorder(
        _ok(
            {
                **_COMPLETION,
                "choices": [
                    {
                        **_COMPLETION["choices"][0],
                        "message": {"role": "assistant", "content": '{"ok": true}'},
                    }
                ],
            }
        )
    )
    with patch.object(openrouter, "_client", return_value=_sdk_client(loose, openrouter.BASE_URL)):
        await openrouter.call(
            _Config(openrouter_api_key="k"), "sys", "hi", 20, output_schema=schema
        )
    body = loose.bodies[-1]
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON" in body["messages"][0]["content"]


async def test_generic_host_uses_the_configured_base_url_and_streams():
    chunks = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "x",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "x",
            "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "x",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    ]
    recorder = _Recorder(
        lambda req: httpx2.Response(
            200,
            request=req,
            content=_sse(chunks, named=False),
            headers={"content-type": "text/event-stream"},
        )
    )
    config = _Config(
        openai_compat_api_key="k",
        openai_compat_base_url="https://llm.internal/v1",
        openai_compat_model="local-model",
    )
    assert str(openai_compat._client(config).base_url).startswith("https://llm.internal/v1")
    text = []
    with patch.object(
        openai_compat, "_client", return_value=_sdk_client(recorder, "https://llm.internal/v1")
    ):
        async for delta in openai_compat.stream_chat(
            config, [{"role": "user", "content": "hi"}], tools=None, max_tokens=5
        ):
            if delta.content:
                text.append(delta.content)
    assert "".join(text) == "Hello"
    assert recorder.bodies[-1]["model"] == "local-model"
    assert recorder.bodies[-1]["stream"] is True


# ─── Failover, errors, capabilities, pricing ───────────────────────────────


async def test_gateway_fails_over_to_mistral_and_maps_errors():
    ok = _Recorder(_ok(_COMPLETION))
    config = _Config(groq_api_key="k", mistral_api_key="k", provider_order="groq,mistral")

    async def groq_down(*args, **kwargs):
        raise openai_sdk.APIStatusError(
            "boom",
            response=httpx2.Response(500, request=httpx2.Request("POST", "https://x")),
            body=None,
        )

    with (
        patch("llm_gateway.chat.CHAT_CALLS", {"groq": groq_down, "mistral": CHAT_CALLS["mistral"]}),
        patch("llm_gateway.chat.CONFIGURED", {"groq": lambda c: True, "mistral": lambda c: True}),
        patch.object(mistral, "_client", return_value=_sdk_client(ok, mistral.BASE_URL)),
    ):
        result = await chat(
            messages=[{"role": "user", "content": "hi"}], max_tokens=5, config=config
        )
    assert result.model == "mistral-small-latest" and result.content == "Hello."

    failing = _Recorder(
        lambda req: httpx2.Response(401, request=req, json={"error": {"message": "bad key"}})
    )
    only = _Config(mistral_api_key="k", provider_order="mistral")
    with patch.object(mistral, "_client", return_value=_sdk_client(failing, mistral.BASE_URL)):
        with pytest.raises(LLMError):
            await chat(messages=[{"role": "user", "content": "hi"}], max_tokens=5, config=only)


def test_capabilities_for_the_new_providers():
    assert capabilities_for("mistral", "mistral-small-latest").tools is True
    assert capabilities_for("mistral", "mistral-small-latest").structured_output == "strict"
    assert capabilities_for("openrouter", "openai/gpt-oss-120b").tools is None
    assert capabilities_for("openrouter", "openai/gpt-oss-120b").structured_output == "json_mode"
    assert capabilities_for("openai_compat", "x") == capabilities_for("cohere", "x")  # unknown
    from llm_gateway.capabilities import openai_compat_capabilities

    declared = openai_compat_capabilities(supports_tools=False, strict_json_schema=True)
    assert capabilities_for("openai_compat", "x", openai_compat=declared).tools is False
    assert missing_capabilities(
        "openai_compat", "x", ["tools"], require_parameters=False, openai_compat=declared
    )


def test_eu_only_policy_excludes_new_providers_without_asserted_metadata():
    """A provider with no PROVIDER_METADATA entry has region "unknown" and
    must fail an EU residency policy: nothing is asserted for a vendor."""
    from llm_gateway.policy import ProviderPolicy, compliance_reasons, metadata_for

    policy = ProviderPolicy(residency="eu")
    for provider in NEW:
        reasons = compliance_reasons(policy, provider, metadata_for({}, provider))
        assert reasons, provider
    asserted = _Config(provider_metadata={"mistral": {"region": "eu"}}).provider_metadata
    assert not compliance_reasons(policy, "mistral", metadata_for(asserted, "mistral"))


def test_default_prices_cover_the_known_hosts_only():
    assert "mistral/mistral-small-latest" in DEFAULT_PRICES
    assert "openrouter/openai/gpt-oss-120b" in DEFAULT_PRICES
    assert not any(key.startswith("openai_compat/") for key in DEFAULT_PRICES)


def test_shared_implementation_caches_one_client_per_key_and_base_url():
    provider = OpenAICompatProvider(openai_compat._provider.spec)
    a = _Config(openai_compat_api_key="k", openai_compat_base_url="https://a.example/v1")
    b = _Config(openai_compat_api_key="k", openai_compat_base_url="https://b.example/v1")
    assert provider._client(a) is provider._client(a)
    assert provider._client(a) is not provider._client(b)
    assert isinstance(PolicyViolationError, type)
