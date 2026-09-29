"""0.9.0: Gemini on Vertex AI through the OpenAI-compatible endpoint, with the
`vertex` provider's Google credentials and a per-request refreshed token.

The real openai SDK runs over an in-memory transport; google-auth credentials
are fakes. No network, no real credentials."""

import datetime
import json
import sys
from unittest.mock import AsyncMock, patch

import httpx2
import openai as openai_sdk
import pytest
from pydantic import BaseModel

from llm_gateway import (
    GatewayConfig,
    LLMError,
    PolicyViolationError,
    ProviderAuthError,
    ProviderPolicy,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.capabilities import UNKNOWN, capabilities_for, missing_capabilities
from llm_gateway.errors import ErrorKind, classify
from llm_gateway.pricing import DEFAULT_PRICES, lookup_price
from llm_gateway.providers import (
    CALLS,
    CHAT_CALLS,
    CONFIGURED,
    DEFAULT_MODEL,
    STREAM_CALLS,
    gemini,
    mistral,
    vertex,
)
from llm_gateway.providers.base import ProviderResult
from llm_gateway.structured import OutputSchema
from tests.test_sdk_compat import _sse

pytest.importorskip("google.auth", reason="needs the vertex extra")

PROJECT = "interviewer-eu"
MODEL = "google/gemini-3.5-flash"

_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"ok": true}'},
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


class _Wire:
    """Records every request (headers, URL and JSON body) and replies."""

    def __init__(self, respond):
        self.respond = respond
        self.requests: list[httpx2.Request] = []
        self.bodies: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content or b"{}"))
        return self.respond(request)


def _ok(body: dict = _COMPLETION):
    return lambda req: httpx2.Response(200, request=req, json=body)


def _fake_google_credentials(*, error: Exception | None = None):
    from google.auth import credentials

    class FakeCredentials(credentials.Credentials):
        def __init__(self):
            super().__init__()
            self.refreshes = 0

        def refresh(self, request):
            self.refreshes += 1
            if error is not None:
                raise error
            self.token = f"google-access-token-{self.refreshes}"
            self.expiry = datetime.datetime.now(datetime.UTC).replace(
                tzinfo=None
            ) + datetime.timedelta(hours=1)

    return FakeCredentials()


def _config(**overrides) -> GatewayConfig:
    settings = dict(
        gemini_project_id=PROJECT,
        provider_order="gemini",
        retry_attempts=1,
        retry_base_delay_seconds=0,
        _env_file=None,
    )
    settings.update(overrides)
    return GatewayConfig(**settings)


def _sdk_client(config: GatewayConfig, wire: _Wire) -> openai_sdk.AsyncOpenAI:
    """The real SDK client `gemini._client` would build, over a mock transport."""
    return openai_sdk.AsyncOpenAI(
        api_key=gemini._token_provider(config),
        base_url=gemini.base_url(config),
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(wire)),
    )


def _seed(config: GatewayConfig, credentials) -> None:
    """Make the vertex credential cache hand out `credentials` (no ADC lookup)."""
    vertex._credentials[vertex._auth_key(config)] = (credentials, None)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for name in ("GEMINI_PROJECT_ID", "GEMINI_LOCATION", "GEMINI_MODEL", "VERTEX_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    for module in (gemini, vertex, mistral):
        module._clients.clear()
    vertex._credentials.clear()
    vertex._missing_extra_logged.clear()
    reset_circuit_breakers()
    yield
    for module in (gemini, vertex, mistral):
        module._clients.clear()
    vertex._credentials.clear()
    vertex._missing_extra_logged.clear()
    reset_circuit_breakers()


# ─── Registry and configuration ────────────────────────────────────────────


@pytest.mark.parametrize("table", [CALLS, CHAT_CALLS, STREAM_CALLS, CONFIGURED, DEFAULT_MODEL])
def test_gemini_is_registered_everywhere(table):
    assert "gemini" in table


def test_gemini_is_off_unless_its_own_project_is_set():
    assert not gemini.configured(GatewayConfig(_env_file=None))
    # Configuring the vertex provider must not switch gemini on.
    assert not gemini.configured(GatewayConfig(_env_file=None, vertex_project_id=PROJECT))
    assert gemini.configured(_config())
    assert not gemini.configured(_config(gemini_model=""))


def test_defaults_env_names_and_normalisation(monkeypatch):
    config = GatewayConfig(_env_file=None)
    assert (config.gemini_location, config.gemini_model) == ("eu", MODEL)
    monkeypatch.setenv("GEMINI_PROJECT_ID", " interviewer-eu ")
    monkeypatch.setenv("GEMINI_LOCATION", " EUROPE-WEST4 ")
    monkeypatch.setenv("GEMINI_MODEL", "google/gemini-3.8-flash")
    config = GatewayConfig(_env_file=None)
    assert config.gemini_project_id == PROJECT
    assert config.gemini_location == "europe-west4"
    assert gemini.default_model(config) == "google/gemini-3.8-flash"
    assert DEFAULT_MODEL["gemini"](config) == "google/gemini-3.8-flash"


@pytest.mark.parametrize("location", ["global", "us", "eu", "europe-west4", "us-central1"])
def test_valid_locations(location):
    assert _config(gemini_location=location).gemini_location == location


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gemini_location", "Europe"),
        ("gemini_location", "eu/../x"),
        ("gemini_project_id", "Not A Project"),
    ],
)
def test_invalid_values_fail_at_startup(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (
            "europe-west4",
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/interviewer-eu"
            "/locations/europe-west4/endpoints/openapi",
        ),
        (
            "global",
            "https://aiplatform.googleapis.com/v1/projects/interviewer-eu"
            "/locations/global/endpoints/openapi",
        ),
        (
            "eu",
            "https://aiplatform.eu.rep.googleapis.com/v1/projects/interviewer-eu"
            "/locations/eu/endpoints/openapi",
        ),
    ],
)
def test_base_url_per_location(location, expected):
    assert gemini.base_url(_config(gemini_location=location)) == expected


def test_provider_order_policy_and_metadata_accept_gemini():
    config = _config(
        provider_order="gemini,anthropic",
        policy_only="gemini",
        provider_metadata={"gemini": {"region": "eu", "retention": "zero", "dpa": True}},
    )
    assert config.provider_order_list == ["gemini", "anthropic"]
    assert "gemini/google/gemini-3.5-flash" not in DEFAULT_PRICES


# ─── Client, request shape and token refresh ───────────────────────────────


def test_client_targets_the_project_location_and_is_cached():
    config = _config(gemini_location="europe-west4", request_timeout_seconds=12)
    client = gemini._client(config)
    assert client is gemini._client(config)
    assert client is not gemini._client(_config(gemini_location="global"))
    assert str(client.base_url) == (
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/interviewer-eu"
        "/locations/europe-west4/endpoints/openapi/"
    )
    assert client.max_retries == 0 and client.timeout.read == 12


async def test_adc_credentials_are_loaded_once_and_no_token_is_fetched_up_front():
    credentials = _fake_google_credentials()
    config = _config()
    with patch("google.auth.default", return_value=(credentials, "p")) as default:
        gemini._client(config)
        assert credentials.refreshes == 0
        assert await gemini._token_provider(config)() == "google-access-token-1"
        assert await gemini._token_provider(config)() == "google-access-token-1"
    default.assert_called_once_with(scopes=[vertex.CLOUD_PLATFORM_SCOPE])
    assert credentials.refreshes == 1


async def test_request_is_a_bearer_call_to_the_openapi_endpoint_with_the_model():
    credentials = _fake_google_credentials()
    config = _config(gemini_location="europe-west4")
    _seed(config, credentials)
    wire = _Wire(_ok())
    with patch.object(gemini, "_client", return_value=_sdk_client(config, wire)):
        result = await gemini.chat(
            config, [{"role": "user", "content": "hi"}], tools=None, max_tokens=20
        )
    (request,) = wire.requests
    assert request.headers["authorization"] == "Bearer google-access-token-1"
    assert "x-api-key" not in request.headers
    assert str(request.url) == (
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/interviewer-eu"
        "/locations/europe-west4/endpoints/openapi/chat/completions"
    )
    assert wire.bodies[-1]["model"] == MODEL and wire.bodies[-1]["max_tokens"] == 20
    assert (result.input_tokens, result.output_tokens) == (11, 3)


async def test_expired_tokens_are_refreshed_between_requests_not_baked_in():
    credentials = _fake_google_credentials()
    config = _config()
    _seed(config, credentials)
    wire = _Wire(_ok())
    messages = [{"role": "user", "content": "hi"}]
    with patch.object(gemini, "_client", return_value=_sdk_client(config, wire)):
        await gemini.chat(config, messages, tools=None, max_tokens=5)
        await gemini.chat(config, messages, tools=None, max_tokens=5)
        assert credentials.refreshes == 1  # a valid token is reused
        # An hour later: google-auth reports the token expired.
        credentials.expiry = datetime.datetime.now(datetime.UTC).replace(
            tzinfo=None
        ) - datetime.timedelta(minutes=1)
        await gemini.chat(config, messages, tools=None, max_tokens=5)
    assert [r.headers["authorization"] for r in wire.requests] == [
        "Bearer google-access-token-1",
        "Bearer google-access-token-1",
        "Bearer google-access-token-2",
    ]


async def test_tools_and_tool_choice_pass_through():
    config = _config()
    _seed(config, _fake_google_credentials())
    wire = _Wire(_ok(_TOOL_COMPLETION))
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with patch.object(gemini, "_client", return_value=_sdk_client(config, wire)):
        result = await gemini.chat(
            config,
            [{"role": "user", "content": "hi"}],
            tools=tools,
            max_tokens=20,
            tool_choice="auto",
        )
    assert wire.bodies[-1]["tools"] == tools and wire.bodies[-1]["tool_choice"] == "auto"
    assert [c.name for c in result.tool_calls] == ["lookup"]


class _Answer(BaseModel):
    ok: bool


async def test_structured_output_is_strict_json_schema():
    config = _config()
    _seed(config, _fake_google_credentials())
    wire = _Wire(_ok())
    with patch.object(gemini, "_client", return_value=_sdk_client(config, wire)):
        result = await gemini.call(
            config, "sys", "hi", 20, output_schema=OutputSchema.from_spec(_Answer)
        )
    assert wire.bodies[-1]["response_format"]["type"] == "json_schema"
    assert result.text == '{"ok": true}'


async def test_streaming():
    chunks = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL,
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
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL,
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    ]
    wire = _Wire(
        lambda req: httpx2.Response(
            200,
            request=req,
            content=_sse(chunks, named=False),
            headers={"content-type": "text/event-stream"},
        )
    )
    config = _config()
    _seed(config, _fake_google_credentials())
    with patch.object(gemini, "_client", return_value=_sdk_client(config, wire)):
        text = "".join(
            [
                d.content or ""
                async for d in gemini.stream_chat(
                    config, [{"role": "user", "content": "hi"}], tools=None, max_tokens=5
                )
            ]
        )
    assert text == "Hello"
    assert wire.bodies[-1]["stream"] is True
    assert wire.requests[-1].headers["authorization"] == "Bearer google-access-token-1"


# ─── Errors and failover ───────────────────────────────────────────────────


async def test_google_auth_failures_are_provider_auth_errors_that_fail_over():
    from google.auth.exceptions import RefreshError

    config = _config(mistral_api_key="k", provider_order="gemini,mistral")
    _seed(config, _fake_google_credentials(error=RefreshError("invalid_grant secret-detail")))
    ok = _Wire(_ok({**_COMPLETION, "model": "m"}))
    down = _Wire(_ok())
    mistral_client = openai_sdk.AsyncOpenAI(
        api_key="k",
        base_url=mistral.BASE_URL,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(ok)),
    )
    with (
        patch.object(gemini, "_client", return_value=_sdk_client(config, down)),
        patch.object(mistral, "_client", return_value=mistral_client),
    ):
        result = await chat(
            messages=[{"role": "user", "content": "hi"}], max_tokens=5, config=config
        )
    assert result.model == "mistral-small-latest" and down.requests == []

    with patch.object(gemini, "_client", return_value=_sdk_client(config, down)):
        with pytest.raises(ProviderAuthError) as excinfo:
            await gemini.chat(config, [{"role": "user", "content": "hi"}], None, 5)
    assert classify(excinfo.value) is ErrorKind.AUTH
    assert "RefreshError" in str(excinfo.value)
    assert "secret-detail" not in str(excinfo.value)


async def test_missing_google_auth_extra_is_an_auth_error():
    config = _config()
    _seed(config, _fake_google_credentials())
    with patch.dict(sys.modules, {"google.auth.transport.requests": None}):
        with pytest.raises(ProviderAuthError, match="vertex"):
            await gemini._token_provider(config)()


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (429, ErrorKind.RATE_LIMITED),
        (503, ErrorKind.TRANSIENT),
        (400, ErrorKind.INVALID_REQUEST),
        (403, ErrorKind.AUTH),
    ],
)
async def test_http_errors_are_classified_like_other_openai_style_providers(status, kind):
    config = _config()
    _seed(config, _fake_google_credentials())
    wire = _Wire(
        lambda req: httpx2.Response(status, request=req, json={"error": {"message": "nope"}})
    )
    with patch.object(gemini, "_client", return_value=_sdk_client(config, wire)):
        with pytest.raises(openai_sdk.APIStatusError) as excinfo:
            await gemini.chat(config, [{"role": "user", "content": "hi"}], None, 5)
    assert classify(excinfo.value) is kind


async def test_all_providers_failing_raises_llm_error():
    config = _config()
    _seed(config, _fake_google_credentials())
    wire = _Wire(lambda req: httpx2.Response(401, request=req, json={"error": {"message": "x"}}))
    with patch.object(gemini, "_client", return_value=_sdk_client(config, wire)):
        with pytest.raises(LLMError):
            await chat(messages=[{"role": "user", "content": "hi"}], max_tokens=5, config=config)


# ─── Capabilities, pricing, policy ─────────────────────────────────────────


def test_capabilities():
    for model in (MODEL, "gemini-3.5-flash", "google/gemini-3.8-flash"):
        capabilities = capabilities_for("gemini", model)
        assert capabilities.structured_output == "strict"
        assert (capabilities.tools, capabilities.images, capabilities.streaming) == (True,) * 3
    assert capabilities_for("gemini", "google/gemma-3") == UNKNOWN
    assert (
        missing_capabilities(
            "gemini", MODEL, ["tools", "structured_output"], require_parameters=True
        )
        == []
    )
    assert missing_capabilities("gemini", "google/gemma-3", ["tools"], require_parameters=True)


def test_no_default_price_so_cost_is_unknown_not_an_error():
    assert not any(key.startswith("gemini/") for key in DEFAULT_PRICES)
    assert lookup_price("gemini", MODEL) is None
    price = {"gemini/google/gemini-3.5-flash": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}
    config = _config(model_prices=price)
    assert lookup_price("gemini", MODEL, config.model_prices).input_per_mtok == 1.0


async def test_result_has_no_cost_without_a_price_and_a_cost_with_one():
    calls = {"gemini": AsyncMock(return_value=ProviderResult("ok", MODEL, 1_000_000, 1_000_000))}
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(system="s", prompt="p", config=_config())
        assert result.provider == "gemini" and result.cost_usd is None
        priced = _config(
            model_prices={
                "gemini/google/gemini-3.5-flash": {"input_per_mtok": 1, "output_per_mtok": 2}
            }
        )
        result = await complete_with_usage(system="s", prompt="p", config=priced)
    assert result.cost_usd == pytest.approx(3.0)


async def test_policy_fails_closed_without_metadata_and_routes_with_it():
    calls = {"gemini": AsyncMock(return_value=ProviderResult("ok", MODEL, 1, 1))}
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(PolicyViolationError) as info:
            await complete_with_usage(system="s", prompt="p", config=_config(policy_residency="eu"))
        assert "gemini" in info.value.exclusions
        calls["gemini"].assert_not_called()

        asserted = _config(
            policy_residency="eu",
            policy_require_dpa=True,
            provider_metadata={"gemini": {"region": "eu", "retention": "zero", "dpa": True}},
        )
        result = await complete_with_usage(
            system="s", prompt="p", config=asserted, policy=ProviderPolicy(residency="eu")
        )
    assert result.provider == "gemini"


async def test_pre_check_skips_gemini_for_an_unsupported_feature_but_not_a_supported_one():
    # `google/gemma-*` served through the same endpoint is unknown, so it only
    # counts under require_parameters.
    calls = {"gemini": AsyncMock(return_value=ProviderResult("ok", "google/gemma-3", 1, 1))}
    config = _config(gemini_model="google/gemma-3")
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(system="s", prompt="p", config=config)
        assert result.provider == "gemini"
        with pytest.raises(LLMError):
            await complete_with_usage(
                system="s",
                prompt="p",
                config=config,
                policy=ProviderPolicy(require_parameters=True),
                output_schema=_Answer,
            )


async def test_stream_chat_through_the_gateway_uses_the_registry():
    from llm_gateway.providers.base import StreamDelta

    async def fake_stream(config, messages, tools, max_tokens, **kwargs):
        yield StreamDelta(content="hi")

    with patch("llm_gateway.streaming.STREAM_CALLS", {"gemini": fake_stream}):
        text = "".join(
            [
                d.content or ""
                async for d in stream_chat(
                    messages=[{"role": "user", "content": "x"}], max_tokens=5, config=_config()
                )
            ]
        )
    assert text == "hi"


# ─── Shutdown ──────────────────────────────────────────────────────────────


async def test_aclose_closes_clients_and_the_shared_credentials():
    config = _config()
    _seed(config, _fake_google_credentials())
    gemini._client(config)
    await gemini.aclose()
    assert gemini._clients == {} and vertex._credentials == {}
