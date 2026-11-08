"""Gaps found by a branch-coverage pass: Groq/OpenAI request building,
orchestrator "nothing was attempted" paths and tracing's never-raise rule."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_gateway import (
    AllProvidersExhaustedError,
    GatewayConfig,
    LLMError,
    breaker,
    chat,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers import groq, openai
from llm_gateway.tracing import set_call_attributes, set_chat_attributes

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]
FORMAT = {"type": "json_object"}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    groq._clients.clear()
    openai._clients.clear()
    yield
    reset_circuit_breakers()
    groq._clients.clear()
    openai._clients.clear()


def _cfg(**kw):
    return GatewayConfig(groq_api_key="g", openai_api_key="o", retry_attempts=1, **kw)


def _completion():
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning=None)
    return SimpleNamespace(
        model="m",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )


def _chunk(content=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning=None)
    choices = [SimpleNamespace(delta=delta, finish_reason=finish)] if usage is None else []
    return SimpleNamespace(model="m", choices=choices, usage=usage)


class _Client:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not kwargs.get("stream"):
            return _completion()

        async def gen():
            yield _chunk("hi")
            yield _chunk(finish="stop")
            yield _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2))

        return gen()


@pytest.mark.parametrize("module", [groq, openai])
async def test_request_options_reach_the_sdk_for_chat_and_stream(module):
    client = _Client()
    with patch.object(module, "_client", return_value=client):
        await module.chat(
            _cfg(), MESSAGES, TOOLS, 50, tool_choice="required", response_format=FORMAT
        )
        deltas = [
            d
            async for d in module._stream_chat(
                _cfg(), MESSAGES, TOOLS, 50, tool_choice="required", response_format=FORMAT
            )
        ]
        # Without tools, tool_choice must be dropped (providers reject it alone).
        await module.chat(_cfg(), MESSAGES, None, 50, tool_choice="required")
    plain, streamed, no_tools = client.calls
    for call in (plain, streamed):
        assert call["tools"] == TOOLS
        assert call["tool_choice"] == "required"
        assert call["response_format"] == FORMAT
    assert streamed["stream"] is True
    assert "tools" not in no_tools and "tool_choice" not in no_tools
    assert "response_format" not in no_tools
    assert "".join(d.content or "" for d in deltas) == "hi"


@pytest.mark.parametrize("module", [groq, openai])
def test_client_is_cached_and_ssl_verify_false_builds(module):
    config = _cfg(ssl_verify=False)
    assert module._client(config) is module._client(config)
    assert module._client(_cfg()) is not module._client(config)


async def test_chat_skips_providers_with_open_breaker_then_reports_exhausted():
    config = _cfg(provider_order="groq")
    breaker.record_failure("groq", threshold=1, cooldown_seconds=60.0)
    assert breaker.is_open("groq")
    with pytest.raises(AllProvidersExhaustedError):
        await chat(messages=MESSAGES, config=config)
    with pytest.raises(AllProvidersExhaustedError):
        async for _ in stream_chat(messages=MESSAGES, config=config):
            pass


async def test_chat_with_no_registered_call_raises_llm_error():
    with patch("llm_gateway.chat.CHAT_CALLS", {}), pytest.raises(LLMError):
        await chat(messages=MESSAGES, config=_cfg(provider_order="groq"))


class _BrokenSpan:
    def set_attribute(self, *a, **k):
        raise RuntimeError("exporter down")


def test_tracing_attribute_helpers_never_raise():
    config = _cfg(trace_include_prompts=True)
    set_call_attributes(
        _BrokenSpan(),
        config=config,
        provider="groq",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        fallback=False,
        system="groq",
        prompt="hello",
        error_code="x",
    )
    set_chat_attributes(
        _BrokenSpan(),
        config=config,
        provider="groq",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        fallback=False,
        tool_call_count=0,
        finish_reason="stop",
        error_code="x",
    )


# --- privacy: message mapping reaches every text-bearing field -------------


def _shout(text: str) -> str:
    return text.upper()


def test_map_message_covers_parts_and_tool_call_arguments():
    from llm_gateway.privacy import _map_json_string, _map_message

    msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": "a"}, {"type": "image_url"}, "raw"],
        "tool_calls": [
            {"id": "1", "function": {"name": "f", "arguments": '{"k": "v", "n": 1}'}},
            {"id": "2", "function": {"name": "f", "arguments": {"k": "w"}}},
            {"id": "3", "function": {"name": "f", "arguments": 5}},
            "opaque",
        ],
    }
    out = _map_message(msg, _shout)
    assert out["content"] == [{"type": "text", "text": "A"}, {"type": "image_url"}, "raw"]
    calls = out["tool_calls"]
    assert calls[0]["function"]["arguments"] == '{"k": "V", "n": 1}'
    assert calls[1]["function"]["arguments"] == {"k": "W"}
    assert calls[2]["function"]["arguments"] == 5
    assert calls[3] == "opaque"
    # The input is not mutated and a non-JSON argument string is masked as text.
    assert msg["content"][0]["text"] == "a"
    assert _map_json_string("not json", _shout) == "NOT JSON"


def test_rebuild_parsed_passes_through_non_model_values():
    from llm_gateway.privacy import _rebuild_parsed

    class _Out:
        def apply_json(self, data):
            return data

    assert _rebuild_parsed(42, _Out()) == 42
    assert _rebuild_parsed({"a": 1}, _Out()) == {"a": 1}


# --- vertex: credential loader chosen by credential type -------------------


@pytest.mark.parametrize(
    ("info", "module", "attr"),
    [
        (
            {
                "type": "external_account",
                "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
            },
            "aws",
            "Credentials.from_info",
        ),
        (
            {"type": "external_account", "credential_source": {"executable": {"command": "x"}}},
            "pluggable",
            "Credentials.from_info",
        ),
        (
            {"type": "external_account", "credential_source": {"file": "/x"}},
            "identity_pool",
            "Credentials.from_info",
        ),
        (
            {"type": "external_account_authorized_user"},
            "external_account_authorized_user",
            "Credentials.from_info",
        ),
        (
            {"type": "impersonated_service_account"},
            "impersonated_credentials",
            "Credentials.from_impersonated_service_account_info",
        ),
    ],
)
def test_vertex_credentials_dispatch_on_type(info, module, attr):
    import importlib

    from llm_gateway.providers import vertex

    mod = importlib.import_module(f"google.auth.{module}")
    cls_name, method = attr.split(".")
    with patch.object(getattr(mod, cls_name), method, return_value="creds") as loader:
        assert vertex._credentials_from_info(info) == "creds"
    assert (
        loader.call_args.kwargs["scopes"] == [vertex.CLOUD_PLATFORM_SCOPE]
        or loader.call_args.args[1:]
    )
