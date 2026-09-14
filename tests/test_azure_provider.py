"""0.4.2: the azure provider (Azure AI Foundry / Azure OpenAI v1 API), through
the *real* openai SDK over an in-memory httpx2 transport.

No network and no real credentials. Entra auth runs azure-identity's real async
`get_bearer_token_provider` against a fake azure-core credential; those tests
are skipped when the `azure` extra isn't installed, while the import-safety and
missing-extra tests run either way.
"""

import contextlib
import importlib.util
import json
import logging
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import openai as openai_sdk
import pydantic
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from llm_gateway import (
    EmptyCompletionError,
    GatewayConfig,
    LLMError,
    ProviderAuthError,
    Usage,
    breaker,
    chat,
    complete,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.errors import ErrorKind, classify
from llm_gateway.providers import CONFIGURED, azure, groq, openai
from tests.test_groq_reasoning_empty_content import _groq_client
from tests.test_sdk_compat import _openai_client, _Recorder, _sse


def _has_identity() -> bool:
    try:
        return all(importlib.util.find_spec(name) for name in ("azure.identity", "aiohttp"))
    except ModuleNotFoundError:
        return False


needs_identity = pytest.mark.skipif(not _has_identity(), reason="needs the azure extra")

ENDPOINT = "https://contoso-eu.services.ai.azure.com"
DEPLOYMENT = "gpt-oss-120b-prod"
API_KEY = "azure-test-key"
SCOPE = "https://ai.azure.com/.default"
PROMPT = "Tell me about Jan Kowalski, jan@example.com"

_ENV = (
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_MODEL",
    "PROVIDER_ORDER",
    "AZURE_API_KEY",
    "AZURE_MODEL",
    "AZURE_ENDPOINT",
    "AZURE_AUTH",
    "AZURE_MANAGED_IDENTITY_CLIENT_ID",
    "AZURE_REASONING_EFFORT",
    "AZURE_MAX_TOKENS_PARAM",
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(azure, "_missing_identity_logged", False)
    reset_circuit_breakers()
    azure._clients.clear()
    azure._credentials.clear()
    yield
    reset_circuit_breakers()
    azure._clients.clear()
    azure._credentials.clear()


class _Wire:
    """MockTransport handler that keeps whole requests (URL, headers, body).
    Replies with `responses` in order, repeating the last one."""

    def __init__(self, *responses):
        self.responses = responses
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        respond = self.responses[min(len(self.requests), len(self.responses)) - 1]
        return respond(request)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content or b"{}") for r in self.requests]


@contextlib.contextmanager
def _azure(wire: _Wire):
    """The provider's own `_client()` (base URL, auth, options), with only
    the HTTP transport swapped for the in-memory one."""

    def http_client(config):
        return httpx2.AsyncClient(transport=httpx2.MockTransport(wire))

    with patch.object(azure, "_http_client", side_effect=http_client):
        yield


def _config(**overrides) -> GatewayConfig:
    settings = dict(
        _env_file=None,
        azure_endpoint=ENDPOINT,
        azure_model=DEPLOYMENT,
        azure_auth="api_key",
        azure_api_key=API_KEY,
        provider_order="azure",
        retry_attempts=2,
        retry_base_delay_seconds=0.0,
        breaker_failure_threshold=1,
    )
    settings.update(overrides)
    return GatewayConfig(**settings)


def _chain(**overrides) -> GatewayConfig:
    return _config(**{"provider_order": "azure,openai", "openai_api_key": "k", **overrides})


USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 90,
    "total_tokens": 1290,
    "prompt_tokens_details": {"cached_tokens": 1024},
    "completion_tokens_details": {"reasoning_tokens": 60},
}


def _completion(content="Hello from Azure.", *, finish_reason="stop", **message) -> dict:
    return {
        "id": "chatcmpl-az",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-oss-120b",
        "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}],
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, **message},
                "content_filter_results": {},
            }
        ],
        "usage": USAGE,
    }


def _ok(body: dict):
    return lambda req: httpx2.Response(200, request=req, json=body)


def _status(code: int, body: dict):
    return lambda req: httpx2.Response(code, request=req, json=body)


def _stream(chunks: list[dict]):
    return lambda req: httpx2.Response(
        200,
        request=req,
        headers={"content-type": "text/event-stream"},
        content=_sse(chunks, named=False),
    )


def _openai_fallback(content="from openai") -> _Recorder:
    return _Recorder(_ok(_completion(content)))


@contextlib.contextmanager
def _with_fallback(wire: _Wire, fallback: _Recorder):
    with (
        _azure(wire),
        patch.object(openai, "_client", return_value=_openai_client(fallback)),
    ):
        yield


_CHUNK = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-oss-120b"}
# Azure sends prompt filter results in a first chunk with no choices.
_PROMPT_FILTER_CHUNK = {
    "id": "",
    "object": "",
    "created": 0,
    "model": "",
    "choices": [],
    "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}],
}
TEXT_CHUNKS = [
    _PROMPT_FILTER_CHUNK,
    {**_CHUNK, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
    {**_CHUNK, "choices": [{"index": 0, "delta": {"content": "Hello"}}]},
    {**_CHUNK, "choices": [{"index": 0, "delta": {"content": " there"}}]},
    {**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    {**_CHUNK, "choices": [], "usage": USAGE},
]

# The documented Azure OpenAI body for a prompt blocked by content filtering.
CONTENT_FILTER_400 = {
    "error": {
        "message": "The response was filtered due to the prompt triggering Azure OpenAI's "
        "content management policy. Please modify your prompt and retry.",
        "type": None,
        "param": "prompt",
        "code": "content_filter",
        "status": 400,
        "innererror": {
            "code": "ResponsibleAIPolicyViolation",
            "content_filter_result": {
                "hate": {"filtered": False, "severity": "safe"},
                "jailbreak": {"filtered": True, "detected": True},
                "self_harm": {"filtered": False, "severity": "safe"},
                "sexual": {"filtered": False, "severity": "safe"},
                "violence": {"filtered": False, "severity": "safe"},
            },
        },
    }
}


class Verdict(BaseModel):
    score: int
    summary: str


# ─── Configuration ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://contoso.openai.azure.com", "https://contoso.openai.azure.com/openai/v1/"),
        ("https://contoso.openai.azure.com/", "https://contoso.openai.azure.com/openai/v1/"),
        (
            "https://contoso.openai.azure.com/openai/v1",
            "https://contoso.openai.azure.com/openai/v1/",
        ),
        (
            "https://contoso.openai.azure.com/openai/v1/",
            "https://contoso.openai.azure.com/openai/v1/",
        ),
        (
            "https://contoso.services.ai.azure.com",
            "https://contoso.services.ai.azure.com/openai/v1/",
        ),
        (
            " https://contoso.services.ai.azure.com/openai/v1// ",
            "https://contoso.services.ai.azure.com/openai/v1/",
        ),
        (
            "https://contoso.services.ai.azure.com/openai/",
            "https://contoso.services.ai.azure.com/openai/v1/",
        ),
        (
            "https://contoso.services.ai.azure.com/api/projects/interviews",
            "https://contoso.services.ai.azure.com/api/projects/interviews/openai/v1/",
        ),
    ],
)
def test_endpoint_is_normalised_to_the_v1_base_url(endpoint, expected):
    assert azure.base_url(endpoint) == expected


async def test_requests_go_to_the_v1_path_without_api_version():
    wire = _Wire(_ok(_completion()))
    config = _config(azure_endpoint="https://contoso.openai.azure.com/openai/v1")
    with _azure(wire):
        await complete(system="s", prompt="p", config=config)
    url = wire.requests[0].url
    assert str(url) == "https://contoso.openai.azure.com/openai/v1/chat/completions"
    assert "api-version" not in url.params


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (dict(azure_endpoint=ENDPOINT, azure_model=DEPLOYMENT), True),  # entra by default
        (dict(azure_endpoint=ENDPOINT, azure_model=DEPLOYMENT, azure_auth="entra"), True),
        (
            dict(
                azure_endpoint=ENDPOINT,
                azure_model=DEPLOYMENT,
                azure_auth="api_key",
                azure_api_key="k",
            ),
            True,
        ),
        (dict(azure_endpoint=ENDPOINT, azure_model=DEPLOYMENT, azure_auth="api_key"), False),
        (dict(azure_model=DEPLOYMENT, azure_api_key="k"), False),
        (dict(azure_endpoint=ENDPOINT, azure_api_key="k"), False),
        (dict(azure_endpoint="  ", azure_model=DEPLOYMENT), False),
        (dict(azure_endpoint=ENDPOINT, azure_model=" "), False),
        ({}, False),
    ],
)
def test_configured_truth_table(settings, expected):
    config = GatewayConfig(_env_file=None, **settings)
    assert azure.configured(config) is expected
    assert CONFIGURED["azure"](config) is expected


def test_defaults_env_names_and_normalisation(monkeypatch):
    defaults = GatewayConfig(_env_file=None)
    assert (defaults.azure_endpoint, defaults.azure_model, defaults.azure_api_key) == ("", "", "")
    assert (defaults.azure_auth, defaults.azure_reasoning_effort) == ("entra", "low")
    assert defaults.azure_managed_identity_client_id == ""
    assert defaults.azure_max_tokens_param == "max_completion_tokens"

    monkeypatch.setenv("AZURE_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("AZURE_MODEL", DEPLOYMENT)
    monkeypatch.setenv("AZURE_AUTH", " API_KEY ")
    monkeypatch.setenv("AZURE_API_KEY", API_KEY)
    monkeypatch.setenv("AZURE_MANAGED_IDENTITY_CLIENT_ID", "client-id")
    monkeypatch.setenv("AZURE_REASONING_EFFORT", "High")
    monkeypatch.setenv("AZURE_MAX_TOKENS_PARAM", " MAX_TOKENS ")
    config = GatewayConfig(_env_file=None)
    assert (config.azure_endpoint, config.azure_model, config.azure_api_key) == (
        ENDPOINT,
        DEPLOYMENT,
        API_KEY,
    )
    assert (config.azure_auth, config.azure_reasoning_effort) == ("api_key", "high")
    assert config.azure_managed_identity_client_id == "client-id"
    assert config.azure_max_tokens_param == "max_tokens"


@pytest.mark.parametrize("value", ["", "aad", "key", "managed_identity", "API-KEY"])
def test_invalid_azure_auth_is_rejected(value):
    with pytest.raises(pydantic.ValidationError, match="azure_auth must be one of"):
        GatewayConfig(_env_file=None, azure_auth=value)


@pytest.mark.parametrize("value", ["none", "minimal", "xhigh", "max", "on"])
def test_invalid_azure_reasoning_effort_is_rejected(value):
    with pytest.raises(pydantic.ValidationError, match="azure_reasoning_effort"):
        GatewayConfig(_env_file=None, azure_reasoning_effort=value)


async def test_client_honours_timeouts_retries_ssl_verify_and_is_cached():
    config = _config(request_timeout_seconds=12, sdk_max_retries=0, ssl_verify=False)
    client = azure._client(config)
    assert str(client.base_url) == f"{ENDPOINT}/openai/v1/"
    assert client.max_retries == 0
    assert (client.timeout.read, client.timeout.connect) == (12, 5.0)
    assert isinstance(client._client, httpx2.AsyncClient)
    assert azure._client(config) is client
    assert azure._client(_config(azure_api_key="rotated-key")) is not client


# ─── Request shape ────────────────────────────────────────────────────────


async def test_api_key_auth_sends_the_key_as_a_bearer_token():
    wire = _Wire(_ok(_completion()))
    with _azure(wire):
        text = await complete(system="s", prompt="p", max_tokens=321, config=_config())

    assert text == "Hello from Azure."
    request = wire.requests[0]
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert "api-key" not in request.headers
    body = wire.bodies[0]
    assert body["model"] == DEPLOYMENT
    assert body["max_completion_tokens"] == 321
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "low"
    assert body["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "p"},
    ]


def _json_or_stream(request: httpx2.Request) -> httpx2.Response:
    if json.loads(request.content).get("stream"):
        return _stream(TEXT_CHUNKS)(request)
    return _ok(_completion())(request)


@pytest.mark.parametrize("effort", ["low", "medium", "high", ""])
async def test_reasoning_effort_follows_config_on_every_engine(effort):
    wire = _Wire(_json_or_stream)
    config = _config(azure_reasoning_effort=effort)
    messages = [{"role": "user", "content": "hi"}]
    with _azure(wire):
        await complete(system="s", prompt="p", config=config)
        await chat(messages=messages, config=config)
        [d async for d in stream_chat(messages=messages, config=config)]

    assert len(wire.bodies) == 3
    for body in wire.bodies:
        assert body["model"] == DEPLOYMENT
        if effort:
            assert body["reasoning_effort"] == effort
        else:
            # "" omits the parameter, for deployments that reject it.
            assert "reasoning_effort" not in body


@pytest.mark.parametrize(
    ("param", "absent"),
    [
        (None, "max_tokens"),  # the default
        ("max_completion_tokens", "max_tokens"),
        ("max_tokens", "max_completion_tokens"),
    ],
)
async def test_token_budget_field_follows_config_on_every_engine(param, absent):
    wire = _Wire(_json_or_stream)
    config = _config() if param is None else _config(azure_max_tokens_param=param)
    expected = param or "max_completion_tokens"
    messages = [{"role": "user", "content": "hi"}]
    with _azure(wire):
        await complete(system="s", prompt="p", max_tokens=111, config=config)
        await chat(messages=messages, max_tokens=222, config=config)
        [d async for d in stream_chat(messages=messages, max_tokens=333, config=config)]

    assert [body[expected] for body in wire.bodies] == [111, 222, 333]
    assert all(absent not in body for body in wire.bodies)
    assert wire.bodies[2]["stream"] is True


@pytest.mark.parametrize("value", ["", "max_output_tokens", "maxTokens", "tokens"])
def test_invalid_azure_max_tokens_param_is_rejected(value):
    with pytest.raises(pydantic.ValidationError, match="azure_max_tokens_param must be one of"):
        GatewayConfig(_env_file=None, azure_max_tokens_param=value)


async def test_structured_output_uses_strict_json_schema():
    wire = _Wire(_ok(_completion(json.dumps({"score": 7, "summary": "Clear."}))))
    with _azure(wire):
        result = await complete_with_usage(
            system="s", prompt="p", config=_config(), output_schema=Verdict
        )

    assert result.parsed == Verdict(score=7, summary="Clear.")
    response_format = wire.bodies[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Verdict"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["score", "summary"]


async def test_invalid_structured_output_fails_over_without_counting_for_the_breaker():
    wire = _Wire(_ok(_completion("not json")))
    fallback = _openai_fallback(json.dumps({"score": 1, "summary": "From OpenAI."}))
    with _with_fallback(wire, fallback):
        result = await complete_with_usage(
            system="s", prompt="p", config=_chain(), output_schema=Verdict
        )
    assert (result.provider, result.parsed.summary) == ("openai", "From OpenAI.")
    assert len(wire.requests) == 1
    assert not breaker.is_open("azure")


async def test_chat_tool_calls_with_null_content_are_served():
    reply = _completion(
        None,
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": "x"}'},
            }
        ],
    )
    wire = _Wire(_ok(reply))
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with _azure(wire):
        result = await chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="auto",
            sampling={"temperature": 0.2},
            config=_config(),
        )
    assert result.content is None
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("lookup", {"q": "x"})]
    body = wire.bodies[0]
    assert (body["tools"], body["tool_choice"], body["temperature"]) == (tools, "auto", 0.2)


async def test_usage_reports_reasoning_and_cached_tokens_under_provider_azure(caplog):
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    wire = _Wire(_ok(_completion()))
    with (
        _azure(wire),
        patch("llm_gateway.router.get_tracer", return_value=tracer_provider.get_tracer("t")),
        caplog.at_level(logging.DEBUG, logger="llm_gateway"),
    ):
        result = await complete_with_usage(system="s", prompt=PROMPT, config=_config())

    assert (result.provider, result.model, result.stop_reason) == ("azure", DEPLOYMENT, "stop")
    assert result.usage == Usage(
        input_tokens=1200, output_tokens=90, cache_read_input_tokens=1024, reasoning_tokens=60
    )
    (span,) = exporter.get_finished_spans()
    assert span.attributes["gen_ai.system"] == "azure"
    assert span.attributes["gen_ai.request.model"] == DEPLOYMENT
    assert span.attributes["llm_gateway.usage.reasoning_tokens"] == 60
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 1024
    assert "gen_ai.prompt" not in span.attributes
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert API_KEY not in logged and PROMPT not in logged


# ─── Failures ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [
        ("", "stop"),
        (None, "length"),
        ("  \n", "stop"),
        ("", "content_filter"),
        (None, "content_filter"),
    ],
)
async def test_empty_reply_fails_over_counts_for_breaker_and_is_not_retried(content, finish_reason):
    wire = _Wire(_ok(_completion(content, finish_reason=finish_reason)))
    fallback = _openai_fallback()
    with _with_fallback(wire, fallback):
        result = await complete_with_usage(system="s", prompt="p", config=_chain())

    assert (result.provider, result.text) == ("openai", "from openai")
    assert len(wire.requests) == 1
    assert breaker.is_open("azure")

    with _azure(_Wire(_ok(_completion(content, finish_reason=finish_reason)))):
        with pytest.raises(EmptyCompletionError) as excinfo:
            await azure.call(_config(), "s", "p", 50)
    assert classify(excinfo.value) is ErrorKind.EMPTY_RESPONSE
    assert (excinfo.value.provider, excinfo.value.model) == ("azure", DEPLOYMENT)
    assert excinfo.value.finish_reason == finish_reason


async def test_chat_empty_reply_fails_over():
    wire = _Wire(_ok(_completion("", finish_reason="content_filter")))
    with _with_fallback(wire, _openai_fallback()):
        result = await chat(messages=[{"role": "user", "content": "hi"}], config=_chain())
    assert result.content == "from openai"
    assert breaker.is_open("azure")


@pytest.mark.parametrize(
    ("status", "body", "requests", "kind", "counted"),
    [
        (400, CONTENT_FILTER_400, 1, ErrorKind.INVALID_REQUEST, False),
        (
            401,
            {"error": {"code": "401", "message": "Access denied due to invalid subscription key."}},
            1,
            ErrorKind.AUTH,
            True,
        ),
        (
            403,
            {"error": {"code": "PermissionDenied", "message": "The principal lacks the role."}},
            1,
            ErrorKind.AUTH,
            True,
        ),
        (
            429,
            {"error": {"code": "429", "message": "Requests exceeded the token rate limit."}},
            1,
            ErrorKind.RATE_LIMITED,
            True,
        ),
        (
            500,
            {"error": {"code": "InternalServerError", "message": "The server had an error."}},
            2,
            ErrorKind.TRANSIENT,
            True,
        ),
    ],
)
async def test_status_errors_fail_over_as_classified(status, body, requests, kind, counted):
    wire = _Wire(_status(status, body))
    with _with_fallback(wire, _openai_fallback()):
        result = await complete_with_usage(system="s", prompt="p", config=_chain())

    assert result.provider == "openai"
    assert len(wire.requests) == requests  # only the 500 is retried, once
    assert breaker.is_open("azure") is counted

    with _azure(_Wire(_status(status, body))), pytest.raises(openai_sdk.APIStatusError) as excinfo:
        await azure.call(_config(), "s", "p", 50)
    assert classify(excinfo.value) is kind


async def test_content_filtered_prompt_on_a_stream_fails_over_without_breaker_count():
    wire = _Wire(_status(400, CONTENT_FILTER_400))
    fallback = _Recorder(_stream(TEXT_CHUNKS))
    with _with_fallback(wire, fallback):
        deltas = [
            d
            async for d in stream_chat(
                messages=[{"role": "user", "content": "hi"}], config=_chain()
            )
        ]
    assert "".join(d.content or "" for d in deltas) == "Hello there"
    assert len(wire.requests) == 1
    assert not breaker.is_open("azure")


# ─── Streaming ────────────────────────────────────────────────────────────


async def test_stream_text_usage_and_request_options():
    wire = _Wire(_stream(TEXT_CHUNKS))
    config = _config(stream_idle_timeout_seconds=17)
    with _azure(wire):
        deltas = [
            d
            async for d in stream_chat(
                messages=[{"role": "user", "content": "hi"}], max_tokens=64, config=config
            )
        ]

    assert "".join(d.content or "" for d in deltas) == "Hello there"
    body = wire.bodies[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert (body["model"], body["max_completion_tokens"], body["reasoning_effort"]) == (
        DEPLOYMENT,
        64,
        "low",
    )
    assert wire.requests[0].extensions["timeout"]["read"] == 17
    final = next(d for d in deltas if d.usage)
    assert final.provider == "azure"
    assert final.usage == (1200, 90)
    assert final.usage_details.reasoning_tokens == 60


async def test_empty_stream_is_guarded_and_fails_over():
    filtered = [
        _PROMPT_FILTER_CHUNK,
        {**_CHUNK, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]},
        {**_CHUNK, "choices": [], "usage": USAGE},
    ]
    wire = _Wire(_stream(filtered))
    fallback = _Recorder(
        _stream([{**_CHUNK, "choices": [{"index": 0, "delta": {"content": "from openai"}}]}])
    )
    with _with_fallback(wire, fallback):
        deltas = [
            d
            async for d in stream_chat(
                messages=[{"role": "user", "content": "hi"}], config=_chain()
            )
        ]

    assert "".join(d.content or "" for d in deltas) == "from openai"
    assert len(wire.requests) == 1
    assert breaker.is_open("azure")


# ─── Routing ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("azure_settings", "served"),
    [
        (
            dict(
                azure_endpoint=ENDPOINT,
                azure_model=DEPLOYMENT,
                azure_auth="api_key",
                azure_api_key="k",
            ),
            "azure",
        ),
        (dict(azure_endpoint=ENDPOINT, azure_model=DEPLOYMENT, azure_auth="api_key"), "groq"),
        (dict(azure_model=DEPLOYMENT, azure_auth="api_key", azure_api_key="k"), "groq"),
    ],
)
async def test_provider_order_uses_azure_only_when_configured(azure_settings, served):
    config = GatewayConfig(
        _env_file=None,
        provider_order="anthropic,azure,groq,openai",
        groq_api_key="g",
        openai_api_key="o",
        **azure_settings,
    )
    wire = _Wire(_ok(_completion("from azure")))
    groq_recorder = _Recorder(_ok(_completion("from groq")))
    with _azure(wire), patch.object(groq, "_client", return_value=_groq_client(groq_recorder)):
        result = await complete_with_usage(system="s", prompt="p", config=config)

    assert (result.provider, result.text) == (served, f"from {served}")
    assert len(wire.requests) == (1 if served == "azure" else 0)
    assert len(groq_recorder.bodies) == (0 if served == "azure" else 1)


# ─── Entra ID ─────────────────────────────────────────────────────────────


class FakeCredential:
    """Stands in for an azure.identity.aio credential: no Azure network."""

    def __init__(self, token="fake-entra-token", error: Exception | None = None):
        self.token = token
        self.error = error
        self.scopes: list[tuple] = []
        self.closed = False

    async def get_token(self, *scopes, **kwargs):
        from azure.core.credentials import AccessToken

        self.scopes.append(scopes)
        if self.error is not None:
            raise self.error
        return AccessToken(self.token, int(time.time()) + 3600)

    async def close(self):
        self.closed = True


def test_credential_prefers_the_user_assigned_managed_identity():
    identity = SimpleNamespace(
        ManagedIdentityCredential=lambda **kwargs: ("managed", kwargs),
        DefaultAzureCredential=lambda **kwargs: ("default", kwargs),
    )
    entra = dict(azure_auth="entra", azure_api_key="")
    config = _config(**entra, azure_managed_identity_client_id=" client-id ")
    assert azure._credential(config, identity) == ("managed", {"client_id": "client-id"})
    assert azure._credential(_config(**entra), identity) == ("default", {})
    assert azure._credential(_config(**entra, ssl_verify=False), identity) == (
        "default",
        {"connection_verify": False},
    )


@needs_identity
async def test_entra_sends_a_bearer_token_for_the_ai_azure_scope():
    credential = FakeCredential()
    wire = _Wire(_ok(_completion()))
    config = _config(azure_auth="entra", azure_api_key="never-sent")
    with _azure(wire), patch.object(azure, "_credential", return_value=credential) as build:
        await complete(system="s", prompt="p", config=config)
        await complete(system="s", prompt="p", config=config)

    assert [r.headers["authorization"] for r in wire.requests] == ["Bearer fake-entra-token"] * 2
    assert "api-key" not in wire.requests[0].headers
    assert credential.scopes == [(SCOPE,)]  # cached by the token provider until near expiry
    assert build.call_count == 1


@needs_identity
async def test_changed_client_id_makes_a_new_credential_and_aclose_closes_everything():
    credentials: list[FakeCredential] = []

    def build(config, identity):
        credential = FakeCredential(token=f"token-{config.azure_managed_identity_client_id}")
        credentials.append(credential)
        return credential

    wire = _Wire(_ok(_completion()))
    with _azure(wire), patch.object(azure, "_credential", side_effect=build):
        for client_id in ("id-a", "id-a", "id-b"):
            config = _config(azure_auth="entra", azure_managed_identity_client_id=client_id)
            await complete(system="s", prompt="p", config=config)

    assert [r.headers["authorization"] for r in wire.requests] == [
        "Bearer token-id-a",
        "Bearer token-id-a",
        "Bearer token-id-b",
    ]
    assert len(credentials) == 2
    await azure.aclose()
    assert all(c.closed for c in credentials)
    assert not azure._clients and not azure._credentials


@needs_identity
@pytest.mark.parametrize("error_name", ["ClientAuthenticationError", "CredentialUnavailableError"])
async def test_entra_token_failure_is_auth_and_fails_over(error_name):
    from azure.core.exceptions import ClientAuthenticationError
    from azure.identity import CredentialUnavailableError

    error_type = {
        "ClientAuthenticationError": ClientAuthenticationError,
        "CredentialUnavailableError": CredentialUnavailableError,
    }[error_name]
    credential = FakeCredential(error=error_type(message="AADSTS7000215: detail from Entra"))
    wire = _Wire(_ok(_completion()))
    with (
        _with_fallback(wire, _openai_fallback()),
        patch.object(azure, "_credential", return_value=credential),
    ):
        result = await complete_with_usage(
            system="s", prompt="p", config=_chain(azure_auth="entra", azure_api_key="")
        )

    assert result.provider == "openai"
    assert wire.requests == []  # nothing was sent without a token
    assert len(credential.scopes) == 1  # AUTH is not retried
    assert breaker.is_open("azure")

    reset_circuit_breakers()
    with (
        _azure(wire),
        patch.object(azure, "_credential", return_value=credential),
        pytest.raises(LLMError) as excinfo,
    ):
        await complete(system="s", prompt="p", config=_config(azure_auth="entra"))
    cause = excinfo.value.__cause__
    assert isinstance(cause, ProviderAuthError)
    assert classify(cause) is ErrorKind.AUTH
    assert error_name in str(cause) and "Cognitive Services OpenAI User" in str(cause)
    assert "AADSTS" not in str(excinfo.value)


async def test_entra_without_azure_identity_is_a_logged_auth_failure_that_fails_over(caplog):
    blocked = dict.fromkeys(
        ("azure", "azure.identity", "azure.identity.aio", "azure.core", "azure.core.exceptions")
    )
    wire = _Wire(_ok(_completion()))
    config = _chain(azure_auth="entra", azure_api_key="", breaker_failure_threshold=5)
    with (
        patch.dict(sys.modules, blocked),
        _with_fallback(wire, _openai_fallback()),
        caplog.at_level(logging.ERROR, logger="llm_gateway"),
    ):
        first = await complete_with_usage(system="s", prompt="p", config=config)
        second = await complete_with_usage(system="s", prompt="p", config=config)
        with pytest.raises(LLMError) as excinfo:
            await complete(system="s", prompt="p", config=_config(azure_auth="entra"))

    assert (first.provider, second.provider) == ("openai", "openai")
    assert wire.requests == []
    assert azure._credentials == {}
    errors = [r for r in caplog.records if "azure-identity" in r.getMessage()]
    assert len(errors) == 1  # logged once, not per call
    cause = excinfo.value.__cause__
    assert isinstance(cause, ProviderAuthError)
    assert classify(cause) is ErrorKind.AUTH
    assert "llm-gateway[azure]" in str(cause)


def test_gateway_imports_and_serves_without_azure_identity():
    code = "\n".join(
        [
            "import sys",
            "for name in ('azure', 'azure.identity', 'azure.core', 'aiohttp'):",
            "    sys.modules[name] = None",
            "import llm_gateway",
            "from llm_gateway.providers import CONFIGURED, azure",
            "from llm_gateway.service.app import create_app",
            "create_app(llm_gateway.GatewayConfig(_env_file=None))",
            "assert 'azure' in CONFIGURED",
            "print('ok')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
