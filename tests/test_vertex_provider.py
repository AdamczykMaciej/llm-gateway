"""0.6.0: the vertex provider (Anthropic Claude on Google Cloud Vertex AI),
through the *real* anthropic SDK's `AsyncAnthropicVertex` over an in-memory
httpx2 transport.

No network and no real credentials: request/response tests give the Vertex
client a fixed access token, and credential tests build google-auth objects
without fetching a token. Tests that need google-auth are skipped when the
`vertex` extra isn't installed; everything else runs either way.
"""

import contextlib
import importlib.util
import json
import logging
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic as anthropic_sdk
import httpx2
import pytest
from pydantic import BaseModel, ValidationError

from llm_gateway import (
    EmptyCompletionError,
    GatewayConfig,
    LLMError,
    ModelPrice,
    PolicyViolationError,
    ProviderAuthError,
    UnsupportedCapabilityError,
    Usage,
    breaker,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.capabilities import UNKNOWN, ModelCapabilities, capabilities_for
from llm_gateway.errors import ErrorKind, classify
from llm_gateway.policy import KNOWN_PROVIDERS, ProviderPolicy
from llm_gateway.pricing import VERTEX_REGIONAL_PREMIUM, lookup_price
from llm_gateway.providers import CONFIGURED, anthropic, groq, vertex
from llm_gateway.providers._anthropic_translate import STRUCTURED_OUTPUT_TOOL_NAME
from tests.test_sdk_compat import _anthropic_client, _openai_client, _sse, _status


def _has_google_auth() -> bool:
    try:
        return importlib.util.find_spec("google.auth") is not None
    except ModuleNotFoundError:
        return False


needs_google_auth = pytest.mark.skipif(not _has_google_auth(), reason="needs the vertex extra")

PROJECT = "interviewer-eu"
LOCATION = "europe-west1"
MODEL = "claude-haiku-4-5@20251001"
TOKEN = "vertex-test-token"
SERVICE_ACCOUNT = "llm-gateway@interviewer-eu.iam.gserviceaccount.com"
APP_ID_URI = "api://gcp-workload-identity"
AUDIENCE = (
    "//iam.googleapis.com/projects/123456789012/locations/global/"
    "workloadIdentityPools/azure-pool/providers/azure-provider"
)
# The shape `gcloud iam workload-identity-pools create-cred-config --azure`
# writes: no key, only where to get the Azure token and where to exchange it.
EXTERNAL_ACCOUNT = {
    "type": "external_account",
    "audience": AUDIENCE,
    "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
    "token_url": "https://sts.googleapis.com/v1/token",
    "credential_source": {
        "url": "http://169.254.169.254/metadata/identity/oauth2/token"
        f"?api-version=2018-02-01&resource={APP_ID_URI}",
        "headers": {"Metadata": "True"},
        "format": {"type": "json", "subject_token_field_name": "access_token"},
    },
}
AUTHORIZED_USER = {
    "type": "authorized_user",
    "client_id": "fake-client.apps.googleusercontent.com",
    "client_secret": "fake-client-secret",
    "refresh_token": "fake-refresh-token",
}
SERVICE_ACCOUNT_KEY = {
    "type": "service_account",
    "client_email": SERVICE_ACCOUNT,
    "private_key": "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----\n",
}
MESSAGES = [{"role": "user", "content": "hi"}]
VERTEX_TEXT = "Vertex says hi"
GROQ_TEXT = "from groq"

_ENV = (
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "PROVIDER_ORDER",
    "MODEL_PRICES",
    "PROVIDER_METADATA",
    "POLICY_RESIDENCY",
    "VERTEX_PROJECT_ID",
    "VERTEX_LOCATION",
    "VERTEX_MODEL",
    "VERTEX_CREDENTIALS_FILE",
    "VERTEX_IMPERSONATE_SERVICE_ACCOUNT",
    "VERTEX_AZURE_APP_ID_URI",
    "VERTEX_AZURE_MANAGED_IDENTITY_CLIENT_ID",
    "VERTEX_STRUCTURED_OUTPUTS",
    # Read by google-auth and by the SDK's Vertex client when not given.
    "GOOGLE_APPLICATION_CREDENTIALS",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLOUD_ML_REGION",
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    reset_circuit_breakers()
    vertex._clients.clear()
    vertex._credentials.clear()
    vertex._missing_extra_logged.clear()
    yield
    reset_circuit_breakers()
    vertex._clients.clear()
    vertex._credentials.clear()
    vertex._missing_extra_logged.clear()


def _config(**overrides) -> GatewayConfig:
    settings = dict(
        vertex_project_id=PROJECT,
        provider_order="vertex",
        retry_base_delay_seconds=0,
        _env_file=None,
    )
    settings.update(overrides)
    return GatewayConfig(**settings)


def _write_json(tmp_path, name: str, info: dict):
    path = tmp_path / name
    path.write_text(json.dumps(info), encoding="utf-8")
    return str(path)


# ─── Wire fakes ────────────────────────────────────────────────────────────


class _Wire:
    """httpx2 MockTransport handler that records requests and answers with
    the current responder."""

    def __init__(self, respond):
        self.respond = respond
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.respond(request)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content or b"{}") for r in self.requests]


def _message(text: str = VERTEX_TEXT, *, usage: dict | None = None, content=None) -> dict:
    return {
        "id": "msg_vrtx_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": text}] if content is None else content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 12, "output_tokens": 4},
    }


def _stream_events(text: str | None = VERTEX_TEXT) -> list[dict]:
    events: list[dict] = [
        {
            "type": "message_start",
            "message": {
                **_message(),
                "content": [],
                "usage": {"input_tokens": 9, "output_tokens": 0},
            },
        }
    ]
    if text is not None:
        events += [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
            {"type": "content_block_stop", "index": 0},
        ]
    events += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn" if text else "max_tokens", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    return events


def _sse_response(request: httpx2.Request, events: list[dict], *, named: bool) -> httpx2.Response:
    return httpx2.Response(
        200,
        request=request,
        headers={"content-type": "text/event-stream"},
        content=_sse(events, named=named),
    )


def _vertex_ok(request: httpx2.Request) -> httpx2.Response:
    if json.loads(request.content).get("stream"):
        return _sse_response(request, _stream_events(), named=True)
    return httpx2.Response(200, request=request, json=_message())


def _json(body: dict):
    return lambda request: httpx2.Response(200, request=request, json=body)


def _google_error(status: int, rpc_status: str):
    body = {"error": {"code": status, "message": rpc_status.lower(), "status": rpc_status}}
    return lambda request: httpx2.Response(status, request=request, json=body)


def _groq_ok(request: httpx2.Request) -> httpx2.Response:
    if json.loads(request.content).get("stream"):
        chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m"}
        events = [
            {**chunk, "choices": [{"index": 0, "delta": {"content": GROQ_TEXT}}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {**chunk, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ]
        return _sse_response(request, events, named=False)
    completion = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": GROQ_TEXT},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return httpx2.Response(200, request=request, json=completion)


def _vertex_client(wire, *, location: str = LOCATION, credentials=None):
    auth = {"credentials": credentials} if credentials is not None else {"access_token": TOKEN}
    return anthropic_sdk.AsyncAnthropicVertex(
        region=location,
        project_id=PROJECT,
        base_url=vertex.base_url(location),
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(wire)),
        **auth,
    )


@contextlib.contextmanager
def _providers(*, vertex_wire=None, anthropic_wire=None, groq_wire=None, location=LOCATION):
    with contextlib.ExitStack() as stack:
        if vertex_wire is not None:
            client = _vertex_client(vertex_wire, location=location)
            stack.enter_context(patch.object(vertex, "_client", new=AsyncMock(return_value=client)))
        if anthropic_wire is not None:
            stack.enter_context(
                patch.object(anthropic, "_client", return_value=_anthropic_client(anthropic_wire))
            )
        if groq_wire is not None:
            stack.enter_context(
                patch.object(groq, "_client", return_value=_openai_client(groq_wire))
            )
        yield


async def _served_text(engine: str, config: GatewayConfig) -> str:
    if engine == "complete":
        return (await complete_with_usage(system="s", prompt="p", config=config)).text
    if engine == "chat":
        return (await chat(messages=MESSAGES, config=config)).content
    return "".join([d.content or "" async for d in stream_chat(messages=MESSAGES, config=config)])


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
            self.token = "google-access-token"

    return FakeCredentials()


# ─── Configuration ─────────────────────────────────────────────────────────


def test_defaults_env_names_and_normalisation(monkeypatch, tmp_path):
    defaults = GatewayConfig(_env_file=None)
    assert defaults.vertex_project_id == ""
    assert defaults.vertex_location == "europe-west1"
    assert defaults.vertex_model == "claude-haiku-4-5@20251001"
    assert defaults.vertex_credentials_file == ""
    assert defaults.vertex_impersonate_service_account == ""
    assert defaults.vertex_azure_app_id_uri == ""
    assert defaults.vertex_azure_managed_identity_client_id == ""
    assert defaults.vertex_structured_outputs is False
    assert not vertex.configured(defaults)
    assert "vertex" not in defaults.provider_order_list

    path = _write_json(tmp_path, "wif.json", EXTERNAL_ACCOUNT)
    monkeypatch.setenv("VERTEX_PROJECT_ID", f" {PROJECT} ")
    monkeypatch.setenv("VERTEX_LOCATION", " EU ")
    monkeypatch.setenv("VERTEX_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("VERTEX_CREDENTIALS_FILE", path)
    monkeypatch.setenv("VERTEX_IMPERSONATE_SERVICE_ACCOUNT", SERVICE_ACCOUNT)
    monkeypatch.setenv("VERTEX_AZURE_APP_ID_URI", APP_ID_URI)
    monkeypatch.setenv("VERTEX_AZURE_MANAGED_IDENTITY_CLIENT_ID", "mi-client-id")
    monkeypatch.setenv("VERTEX_STRUCTURED_OUTPUTS", "true")
    config = GatewayConfig(_env_file=None)
    assert config.vertex_structured_outputs is True
    assert config.vertex_project_id == PROJECT
    assert config.vertex_location == "eu"
    assert config.vertex_model == "claude-haiku-4-5"
    assert config.vertex_credentials_file == path
    assert config.vertex_impersonate_service_account == SERVICE_ACCOUNT
    assert config.vertex_azure_app_id_uri == APP_ID_URI
    assert config.vertex_azure_managed_identity_client_id == "mi-client-id"
    assert vertex.configured(config)


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, False),
        ({"vertex_project_id": PROJECT}, True),
        ({"vertex_project_id": PROJECT, "vertex_model": " "}, False),
        ({"vertex_model": MODEL}, False),
    ],
)
def test_configured_truth_table(settings, expected):
    config = GatewayConfig(_env_file=None, **settings)
    assert vertex.configured(config) is expected
    assert CONFIGURED["vertex"](config) is expected


@pytest.mark.parametrize("location", ["global", "eu", "us", "europe-west1", "europe-west4"])
def test_valid_locations(location):
    assert _config(vertex_location=location.upper()).vertex_location == location


@pytest.mark.parametrize(
    "field, value",
    [
        ("vertex_location", "europe west1"),
        ("vertex_location", "europe-west"),
        ("vertex_location", "mars"),
        ("vertex_location", ""),
        ("vertex_project_id", "My Project"),
        ("vertex_project_id", "abc"),
        ("vertex_impersonate_service_account", "not-an-email"),
        ("vertex_impersonate_service_account", "someone@gmail.com"),
    ],
)
def test_invalid_values_fail_at_startup(field, value):
    with pytest.raises(ValidationError):
        _config(**{field: value})


def test_credentials_file_is_validated_without_echoing_its_content(tmp_path):
    with pytest.raises(ValidationError, match="can't be read"):
        _config(vertex_credentials_file=str(tmp_path / "missing.json"))

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValidationError, match="not valid JSON"):
        _config(vertex_credentials_file=str(broken))

    key = _write_json(tmp_path, "key.json", SERVICE_ACCOUNT_KEY)
    with pytest.raises(ValidationError) as excinfo:
        _config(vertex_credentials_file=key)
    message = str(excinfo.value)
    assert "service account key" in message and "Workload Identity Federation" in message
    assert "PRIVATE KEY" not in message and "not-a-real-key" not in message

    nested_key = {"type": "impersonated_service_account", "source_credentials": SERVICE_ACCOUNT_KEY}
    with pytest.raises(ValidationError, match="service account key"):
        _config(vertex_credentials_file=_write_json(tmp_path, "nested.json", nested_key))

    other = _write_json(tmp_path, "other.json", {"type": "gdch_service_account"})
    with pytest.raises(ValidationError, match="type must be one of"):
        _config(vertex_credentials_file=other)

    ok = _write_json(tmp_path, "wif.json", EXTERNAL_ACCOUNT)
    assert _config(vertex_credentials_file=ok).vertex_credentials_file == ok


def test_azure_and_impersonation_settings_are_cross_checked(tmp_path):
    wif = _write_json(tmp_path, "wif.json", EXTERNAL_ACCOUNT)
    user = _write_json(tmp_path, "user.json", AUTHORIZED_USER)
    with pytest.raises(ValidationError, match="external_account"):
        _config(vertex_azure_app_id_uri=APP_ID_URI)
    with pytest.raises(ValidationError, match="external_account"):
        _config(vertex_azure_app_id_uri=APP_ID_URI, vertex_credentials_file=user)
    with pytest.raises(ValidationError, match="needs vertex_azure_app_id_uri"):
        _config(vertex_credentials_file=wif, vertex_azure_managed_identity_client_id="mi")

    other_sa = {
        **EXTERNAL_ACCOUNT,
        "service_account_impersonation_url": vertex.service_account_impersonation_url(
            "other@interviewer-eu.iam.gserviceaccount.com"
        ),
    }
    with pytest.raises(ValidationError, match="differs"):
        _config(
            vertex_credentials_file=_write_json(tmp_path, "other.json", other_sa),
            vertex_impersonate_service_account=SERVICE_ACCOUNT,
        )
    impersonated = {"type": "impersonated_service_account", "source_credentials": AUTHORIZED_USER}
    with pytest.raises(ValidationError, match="already impersonates"):
        _config(
            vertex_credentials_file=_write_json(tmp_path, "imp.json", impersonated),
            vertex_impersonate_service_account=SERVICE_ACCOUNT,
        )


def test_vertex_is_a_known_provider_for_policy_and_prices():
    assert "vertex" in KNOWN_PROVIDERS
    assert ProviderPolicy(only="vertex").only == ("vertex",)
    config = _config(
        model_prices={f"vertex/{MODEL}": {"input_per_mtok": 2, "output_per_mtok": 3}},
        provider_metadata={"vertex": {"region": "eu"}},
    )
    assert config.provider_metadata["vertex"].region == "eu"


# ─── Endpoints, imports and the missing extra ─────────────────────────────


@pytest.mark.parametrize(
    "location, expected",
    [
        ("global", "https://aiplatform.googleapis.com/v1"),
        ("eu", "https://aiplatform.eu.rep.googleapis.com/v1"),
        ("us", "https://aiplatform.us.rep.googleapis.com/v1"),
        ("europe-west1", "https://europe-west1-aiplatform.googleapis.com/v1"),
    ],
)
def test_base_url_matches_the_sdk_endpoint_for_each_location(location, expected):
    assert vertex.base_url(location) == expected
    sdk = anthropic_sdk.AsyncAnthropicVertex(region=location, project_id=PROJECT, access_token="t")
    assert str(sdk.base_url) == f"{expected}/"


def test_gateway_imports_and_serves_without_google_auth():
    code = "\n".join(
        [
            "import sys",
            "for name in ('google', 'google.auth', 'google.oauth2', 'azure', 'azure.identity'):",
            "    sys.modules[name] = None",
            "import llm_gateway",
            "from llm_gateway.providers import CONFIGURED, vertex",
            "from llm_gateway.service.app import create_app",
            "config = llm_gateway.GatewayConfig(_env_file=None, vertex_project_id='my-project')",
            "create_app(config)",
            "assert CONFIGURED['vertex'](config)",
            "print('ok')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


async def test_missing_extra_is_a_logged_auth_failure_that_fails_over(caplog):
    blocked = dict.fromkeys(
        (
            "google",
            "google.auth",
            "google.auth.exceptions",
            "google.auth.transport",
            "google.auth.transport.requests",
            "google.oauth2",
        )
    )
    groq_wire = _Wire(_groq_ok)
    config = _config(provider_order="vertex,groq", groq_api_key="k", breaker_failure_threshold=5)
    with (
        patch.dict(sys.modules, blocked),
        _providers(groq_wire=groq_wire),
        caplog.at_level(logging.ERROR, logger="llm_gateway"),
    ):
        first = await complete_with_usage(system="s", prompt="p", config=config)
        second = await complete_with_usage(system="s", prompt="p", config=config)
        with pytest.raises(LLMError) as excinfo:
            await complete_with_usage(system="s", prompt="p", config=_config())

    assert (first.provider, second.provider) == ("groq", "groq")
    assert vertex._clients == {} and vertex._credentials == {}
    errors = [r for r in caplog.records if "google-auth" in r.getMessage()]
    assert len(errors) == 1  # logged once, not per call
    cause = excinfo.value.__cause__
    assert isinstance(cause, ProviderAuthError)
    assert classify(cause) is ErrorKind.AUTH
    assert "llm-gateway[vertex]" in str(cause)


# ─── Client construction and credentials (google-auth) ────────────────────


@needs_google_auth
async def test_client_uses_adc_location_project_timeouts_and_is_cached():
    credentials = _fake_google_credentials()
    config = _config(request_timeout_seconds=12)
    with patch("google.auth.default", return_value=(credentials, "adc-project")) as default:
        client = await vertex._client(config)
        again = await vertex._client(config)
        other_location = await vertex._client(_config(vertex_location="global"))

    assert client is again and other_location is not client
    default.assert_called_once_with(scopes=[vertex.CLOUD_PLATFORM_SCOPE])
    assert isinstance(client, anthropic_sdk.AsyncAnthropicVertex)
    assert (client.region, client.project_id) == (LOCATION, PROJECT)
    assert client.credentials is credentials and other_location.credentials is credentials
    assert str(client.base_url) == "https://europe-west1-aiplatform.googleapis.com/v1/"
    assert str(other_location.base_url) == "https://aiplatform.googleapis.com/v1/"
    assert client.max_retries == 0
    assert client.timeout.read == 12
    assert credentials.refreshes == 0  # no token is fetched until a request needs one

    await vertex.aclose()
    assert vertex._clients == {} and vertex._credentials == {}


@needs_google_auth
async def test_sdk_refreshes_the_google_token_and_sends_it_as_a_bearer():
    credentials = _fake_google_credentials()
    wire = _Wire(_vertex_ok)
    client = _vertex_client(wire, credentials=credentials)
    config = _config()
    with patch.object(vertex, "_client", new=AsyncMock(return_value=client)):
        await vertex.call(config, "s", "p", 10)
        await vertex.call(config, "s", "p", 10)
    assert credentials.refreshes == 1  # reused until google-auth reports it expired
    assert [r.headers["authorization"] for r in wire.requests] == ["Bearer google-access-token"] * 2
    assert all("x-api-key" not in r.headers for r in wire.requests)


@needs_google_auth
def test_external_account_file_with_impersonation_uses_the_sts_impersonation_url(tmp_path):
    from google.auth import external_account, identity_pool

    config = _config(
        vertex_credentials_file=_write_json(tmp_path, "wif.json", EXTERNAL_ACCOUNT),
        vertex_impersonate_service_account=SERVICE_ACCOUNT,
    )
    # Loading must not resolve a project id (an STS exchange over the network).
    no_lookup = AssertionError("get_project_id makes network calls")
    with patch.object(external_account.Credentials, "get_project_id", side_effect=no_lookup):
        credentials, closeable = vertex.load_credentials(config)
    assert isinstance(credentials, identity_pool.Credentials)
    assert credentials.service_account_email == SERVICE_ACCOUNT
    assert credentials.token is None and closeable is None
    assert vertex.CLOUD_PLATFORM_SCOPE in credentials.scopes


@needs_google_auth
def test_non_external_source_is_wrapped_in_impersonated_credentials(tmp_path):
    from google.auth import impersonated_credentials

    config = _config(
        vertex_credentials_file=_write_json(tmp_path, "user.json", AUTHORIZED_USER),
        vertex_impersonate_service_account=SERVICE_ACCOUNT,
    )
    credentials, _ = vertex.load_credentials(config)
    assert isinstance(credentials, impersonated_credentials.Credentials)
    assert credentials.service_account_email == SERVICE_ACCOUNT


@needs_google_auth
def test_adc_resolving_to_a_service_account_key_is_refused():
    from google.oauth2 import service_account

    key_credentials = MagicMock(spec=service_account.Credentials)
    with (
        patch("google.auth.default", return_value=(key_credentials, PROJECT)),
        pytest.raises(ProviderAuthError, match="service account key"),
    ):
        vertex.load_credentials(_config())


def _rsa_private_key_pem() -> str:
    """A throwaway key generated in-process, so google-auth can build real
    service-account credentials offline."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


@needs_google_auth
def test_adc_impersonating_from_a_service_account_key_is_refused(tmp_path, monkeypatch):
    import google.auth
    from google.auth import impersonated_credentials

    adc = {
        "type": "impersonated_service_account",
        "service_account_impersonation_url": vertex.service_account_impersonation_url(
            SERVICE_ACCOUNT
        ),
        "source_credentials": {
            "type": "service_account",
            "project_id": PROJECT,
            "private_key_id": "0",
            "private_key": _rsa_private_key_pem(),
            "client_email": "source@interviewer-eu.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
        },
    }
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", _write_json(tmp_path, "adc.json", adc))
    # google-auth itself accepts the file: impersonated credentials over a key.
    resolved, _ = google.auth.default(scopes=[vertex.CLOUD_PLATFORM_SCOPE])
    assert isinstance(resolved, impersonated_credentials.Credentials)

    with pytest.raises(ProviderAuthError, match="service account key") as excinfo:
        vertex.load_credentials(_config())
    assert "PRIVATE KEY" not in str(excinfo.value)
    with pytest.raises(ProviderAuthError, match="service account key"):
        vertex.load_credentials(_config(vertex_impersonate_service_account=SERVICE_ACCOUNT))


@needs_google_auth
def test_adc_resolving_to_a_gdch_service_account_is_refused():
    from google.auth import impersonated_credentials
    from google.oauth2 import gdch_credentials

    gdch = MagicMock(spec=gdch_credentials.ServiceAccountCredentials)
    with (
        patch("google.auth.default", return_value=(gdch, PROJECT)),
        pytest.raises(ProviderAuthError, match="service account key"),
    ):
        vertex.load_credentials(_config())

    over_gdch = MagicMock(spec=impersonated_credentials.Credentials)
    over_gdch._source_credentials = gdch
    with (
        patch("google.auth.default", return_value=(over_gdch, PROJECT)),
        pytest.raises(ProviderAuthError, match="service account key"),
    ):
        vertex.load_credentials(_config())


@needs_google_auth
def test_unusable_credentials_are_an_auth_error():
    from google.auth.exceptions import DefaultCredentialsError

    with (
        patch("google.auth.default", side_effect=DefaultCredentialsError("no ADC found")),
        pytest.raises(ProviderAuthError) as excinfo,
    ):
        vertex.load_credentials(_config())
    assert classify(excinfo.value) is ErrorKind.AUTH
    assert "DefaultCredentialsError" in str(excinfo.value)


class _FakeAzureError(Exception):
    pass


class _FakeClientAuthenticationError(_FakeAzureError):
    pass


class _FakeServiceRequestError(_FakeAzureError):
    """What azure-core raises when the identity endpoint can't be reached."""


class _FakeManagedIdentityCredential:
    instances: list["_FakeManagedIdentityCredential"] = []

    def __init__(self, client_id=None):
        self.client_id = client_id
        self.scopes: list[tuple] = []
        self.error: Exception | None = None
        self.closed = False
        _FakeManagedIdentityCredential.instances.append(self)

    def get_token(self, *scopes):
        self.scopes.append(scopes)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(token="azure-managed-identity-jwt", expires_on=0)

    def close(self):
        self.closed = True


def _fake_azure_modules() -> dict:
    identity = ModuleType("azure.identity")
    identity.ManagedIdentityCredential = _FakeManagedIdentityCredential
    exceptions = ModuleType("azure.core.exceptions")
    exceptions.AzureError = _FakeAzureError
    exceptions.ClientAuthenticationError = _FakeClientAuthenticationError
    exceptions.ServiceRequestError = _FakeServiceRequestError
    return {
        "azure": ModuleType("azure"),
        "azure.identity": identity,
        "azure.core": ModuleType("azure.core"),
        "azure.core.exceptions": exceptions,
    }


@needs_google_auth
def test_azure_managed_identity_supplies_the_subject_token(tmp_path):
    from google.auth import identity_pool
    from google.auth.exceptions import RefreshError

    _FakeManagedIdentityCredential.instances.clear()
    config = _config(
        vertex_credentials_file=_write_json(tmp_path, "wif.json", EXTERNAL_ACCOUNT),
        vertex_impersonate_service_account=SERVICE_ACCOUNT,
        vertex_azure_app_id_uri=APP_ID_URI,
        vertex_azure_managed_identity_client_id="mi-client-id",
    )
    with patch.dict(sys.modules, _fake_azure_modules()):
        credentials, closeable = vertex.load_credentials(config)
        assert isinstance(credentials, identity_pool.Credentials)
        assert credentials.service_account_email == SERVICE_ACCOUNT
        # The IMDS credential_source is replaced: the token comes from azure-identity.
        assert credentials.retrieve_subject_token(None) == "azure-managed-identity-jwt"

        (managed_identity,) = _FakeManagedIdentityCredential.instances
        assert closeable is managed_identity
        assert managed_identity.client_id == "mi-client-id"
        assert managed_identity.scopes == [(f"{APP_ID_URI}/.default",)]

        managed_identity.error = _FakeClientAuthenticationError("AADSTS700016: app not found")
        with pytest.raises(RefreshError) as excinfo:
            credentials.retrieve_subject_token(None)
    assert "_FakeClientAuthenticationError" in str(excinfo.value)
    assert "AADSTS" not in str(excinfo.value)


@needs_google_auth
async def test_unreachable_identity_endpoint_is_an_auth_error(tmp_path):
    from google.auth.exceptions import RefreshError

    _FakeManagedIdentityCredential.instances.clear()
    config = _config(
        vertex_credentials_file=_write_json(tmp_path, "wif.json", EXTERNAL_ACCOUNT),
        vertex_azure_app_id_uri=APP_ID_URI,
    )
    wire = _Wire(_vertex_ok)
    with patch.dict(sys.modules, _fake_azure_modules()):
        credentials, _ = vertex.load_credentials(config)
        (managed_identity,) = _FakeManagedIdentityCredential.instances
        managed_identity.error = _FakeServiceRequestError("connection refused to IDENTITY_ENDPOINT")
        with pytest.raises(RefreshError, match="_FakeServiceRequestError"):
            credentials.retrieve_subject_token(None)

        # Through the SDK: the refresh fails before any request is sent.
        client = _vertex_client(wire, credentials=credentials)
        with (
            patch.object(vertex, "_client", new=AsyncMock(return_value=client)),
            pytest.raises(ProviderAuthError) as excinfo,
        ):
            await vertex.call(config, "s", "p", 10)
    assert classify(excinfo.value) is ErrorKind.AUTH
    assert isinstance(excinfo.value.__cause__, RefreshError)
    assert wire.requests == []


async def test_credential_losing_a_load_race_is_closed():
    winner = (object(), MagicMock())
    loser = MagicMock()
    config = _config()

    def load(config):
        # Another first call finished loading while this one was in its thread.
        vertex._credentials[vertex._auth_key(config)] = winner
        return (object(), loser)

    with patch.object(vertex, "load_credentials", side_effect=load):
        assert await vertex._google_credentials(config) is winner[0]
    loser.close.assert_called_once()
    winner[1].close.assert_not_called()


@needs_google_auth
def test_azure_supplier_without_azure_identity_names_the_azure_extra(tmp_path):
    config = _config(
        vertex_credentials_file=_write_json(tmp_path, "wif.json", EXTERNAL_ACCOUNT),
        vertex_azure_app_id_uri=APP_ID_URI,
    )
    blocked = dict.fromkeys(("azure", "azure.identity", "azure.core", "azure.core.exceptions"))
    with patch.dict(sys.modules, blocked), pytest.raises(ProviderAuthError) as excinfo:
        vertex.load_credentials(config)
    assert "llm-gateway[azure]" in str(excinfo.value)


@needs_google_auth
@pytest.mark.parametrize("engine", ["complete", "chat", "stream"])
async def test_google_token_refresh_error_is_auth_fails_over_and_counts_for_breaker(engine):
    from google.auth.exceptions import RefreshError

    credentials = _fake_google_credentials(
        error=RefreshError("invalid_grant: audience does not match")
    )
    vertex_wire, groq_wire = _Wire(_vertex_ok), _Wire(_groq_ok)
    client = _vertex_client(vertex_wire, credentials=credentials)
    config = _config(provider_order="vertex,groq", groq_api_key="k", breaker_failure_threshold=1)
    with (
        patch.object(vertex, "_client", new=AsyncMock(return_value=client)),
        _providers(groq_wire=groq_wire),
    ):
        assert await _served_text(engine, config) == GROQ_TEXT
        with pytest.raises(ProviderAuthError) as excinfo:
            await vertex.call(_config(), "s", "p", 10)

    assert vertex_wire.requests == []
    assert credentials.refreshes == 2  # once per attempt: never retried
    assert breaker.is_open("vertex")
    assert classify(excinfo.value) is ErrorKind.AUTH
    assert isinstance(excinfo.value.__cause__, RefreshError)
    assert "invalid_grant" not in str(excinfo.value)


# ─── Requests: parity with the anthropic provider ─────────────────────────


async def test_chat_request_matches_the_anthropic_provider_and_targets_the_model_url():
    messages = [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    ]
    sampling = {"temperature": 0.2, "top_p": 0.9, "stop": ["END"], "seed": 7}
    anthropic_wire, vertex_wire = _Wire(_json(_message())), _Wire(_json(_message()))
    config = _config(anthropic_api_key="k")
    with _providers(vertex_wire=vertex_wire, anthropic_wire=anthropic_wire):
        direct = await anthropic.chat(
            config, messages, tools, 100, model=MODEL, tool_choice="auto", sampling=sampling
        )
        served = await vertex.chat(
            config, messages, tools, 100, tool_choice="auto", sampling=sampling
        )

    assert served == direct
    anthropic_body, vertex_body = anthropic_wire.bodies[0], vertex_wire.bodies[0]
    assert vertex_body.pop("anthropic_version") == "vertex-2023-10-16"
    assert anthropic_body.pop("model") == MODEL
    assert "model" not in vertex_body
    assert vertex_body == anthropic_body
    request = vertex_wire.requests[0]
    assert str(request.url) == (
        f"https://europe-west1-aiplatform.googleapis.com/v1/projects/{PROJECT}"
        f"/locations/europe-west1/publishers/anthropic/models/{MODEL}:rawPredict"
    )
    assert request.headers["authorization"] == f"Bearer {TOKEN}"


class _Score(BaseModel):
    score: int
    reason: str


async def test_structured_output_and_prompt_caching_pass_through_like_anthropic():
    long_system = "Rubric. " * 1600  # above Haiku 4.5's 4096-token caching minimum
    reply = _message('{"score": 4, "reason": "clear"}')
    anthropic_wire, vertex_wire = _Wire(_json(reply)), _Wire(_json(reply))
    with _providers(vertex_wire=vertex_wire, anthropic_wire=anthropic_wire):
        result = await complete_with_usage(
            system=long_system,
            prompt="Rate it",
            config=_config(vertex_structured_outputs=True),
            output_schema=_Score,
            cache_system=True,
        )
        await complete_with_usage(
            system=long_system,
            prompt="Rate it",
            config=_config(anthropic_api_key="k", provider_order="anthropic", claude_model=MODEL),
            output_schema=_Score,
            cache_system=True,
        )

    assert result.provider == "vertex"
    assert result.parsed == _Score(score=4, reason="clear")
    vertex_body, anthropic_body = vertex_wire.bodies[0], anthropic_wire.bodies[0]
    assert vertex_body["system"] == [
        {"type": "text", "text": long_system, "cache_control": {"type": "ephemeral"}}
    ]
    assert vertex_body["output_config"]["format"]["type"] == "json_schema"
    assert set(vertex_body["output_config"]["format"]["schema"]["properties"]) == {
        "score",
        "reason",
    }
    vertex_body.pop("anthropic_version")
    anthropic_body.pop("model")
    assert vertex_body == anthropic_body


@pytest.mark.parametrize(
    "location, expected_cost",
    [("global", 0.0087), ("europe-west1", 0.00957), ("eu", 0.00957)],
)
async def test_usage_and_cost_apply_the_regional_premium(location, expected_cost):
    usage = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 4000,
    }
    wire = _Wire(_json(_message(usage=usage)))
    with _providers(vertex_wire=wire, location=location):
        result = await complete_with_usage(
            system="s", prompt="p", config=_config(vertex_location=location)
        )

    assert result.provider == "vertex" and result.model == MODEL
    assert result.usage == Usage(7000, 500, 2000, 4000, 0)
    assert result.cost_usd == pytest.approx(expected_cost)
    assert str(wire.requests[0].url).startswith(vertex.base_url(location))


def test_default_prices_regional_premium_and_overrides():
    global_price = lookup_price("vertex", MODEL)
    regional = lookup_price("vertex", MODEL, vertex_location="europe-west1")
    assert VERTEX_REGIONAL_PREMIUM == 1.1
    assert global_price == ModelPrice(
        input_per_mtok=1.0,
        output_per_mtok=5.0,
        cached_input_per_mtok=0.1,
        cache_write_per_mtok=1.25,
    )
    assert regional == ModelPrice(
        input_per_mtok=1.1,
        output_per_mtok=5.5,
        cached_input_per_mtok=0.11,
        cache_write_per_mtok=1.375,
    )
    assert lookup_price("vertex", "claude-haiku-4-5", vertex_location="eu") == regional
    override = ModelPrice(input_per_mtok=2.0, output_per_mtok=3.0)
    overrides = {f"vertex/{MODEL}": override}
    assert lookup_price("vertex", MODEL, overrides, vertex_location="europe-west1") == override
    # The premium is Vertex-only.
    assert (
        lookup_price("anthropic", "claude-haiku-4-5", vertex_location="europe-west1").input_per_mtok
        == 1.0
    )
    assert lookup_price("vertex", "claude-unknown@1", vertex_location="eu") is None


async def test_empty_reply_is_guarded_under_provider_vertex():
    wire = _Wire(_json(_message(content=[])))
    with _providers(vertex_wire=wire), pytest.raises(EmptyCompletionError) as excinfo:
        await vertex.call(_config(), "s", "p", 10)
    assert (excinfo.value.provider, excinfo.value.model) == ("vertex", MODEL)


# ─── Errors, retries, breaker and failover ─────────────────────────────────


@pytest.mark.parametrize(
    "status, rpc_status, kind, attempts, counted",
    [
        (429, "RESOURCE_EXHAUSTED", ErrorKind.RATE_LIMITED, 1, True),
        (500, "INTERNAL", ErrorKind.TRANSIENT, 2, True),
        (503, "UNAVAILABLE", ErrorKind.TRANSIENT, 2, True),
        (400, "INVALID_ARGUMENT", ErrorKind.INVALID_REQUEST, 1, False),
        (404, "NOT_FOUND", ErrorKind.INVALID_REQUEST, 1, False),
        (401, "UNAUTHENTICATED", ErrorKind.AUTH, 1, True),
        (403, "PERMISSION_DENIED", ErrorKind.AUTH, 1, True),
    ],
)
async def test_http_errors_fail_over_as_classified(status, rpc_status, kind, attempts, counted):
    vertex_wire, groq_wire = _Wire(_google_error(status, rpc_status)), _Wire(_groq_ok)
    config = _config(
        provider_order="vertex,groq",
        groq_api_key="k",
        breaker_failure_threshold=1,
        retry_attempts=2,
    )
    with _providers(vertex_wire=vertex_wire, groq_wire=groq_wire):
        result = await complete_with_usage(system="s", prompt="p", config=config)
        assert len(vertex_wire.requests) == attempts
        assert breaker.is_open("vertex") is counted
        with pytest.raises(anthropic_sdk.APIStatusError) as excinfo:
            await vertex.call(_config(), "s", "p", 10)

    assert result.provider == "groq"
    assert classify(excinfo.value) is kind


@pytest.mark.parametrize(
    "body, kind",
    [
        ({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}, ErrorKind.RATE_LIMITED),
        ({"error": {"code": 503, "status": "UNAVAILABLE"}}, ErrorKind.TRANSIENT),
        ({"error": {"code": 403, "status": "PERMISSION_DENIED"}}, ErrorKind.AUTH),
        ({"error": {"code": 400, "status": "INVALID_ARGUMENT"}}, ErrorKind.INVALID_REQUEST),
        ({"error": {"code": 500, "status": "SOMETHING_NEW"}}, ErrorKind.UNKNOWN),
        # An Anthropic error type still decides when both are present.
        (
            {"error": {"type": "overloaded_error", "status": "INVALID_ARGUMENT"}},
            ErrorKind.TRANSIENT,
        ),
    ],
)
def test_google_error_body_inside_a_200_stream_is_classified_by_status(body, kind):
    request = httpx2.Request("POST", "https://europe-west1-aiplatform.googleapis.com/v1/x")
    error = anthropic_sdk.APIStatusError(
        "stream error", response=httpx2.Response(200, request=request), body=body
    )
    assert classify(error) is kind


@pytest.mark.parametrize("engine", ["complete", "chat", "stream"])
async def test_chain_anthropic_vertex_groq_fails_over_and_breakers_skip(engine):
    anthropic_wire = _Wire(_status(429))
    vertex_wire = _Wire(_vertex_ok)
    groq_wire = _Wire(_groq_ok)
    config = _config(
        anthropic_api_key="k",
        groq_api_key="k",
        provider_order="anthropic,vertex,groq",
        breaker_failure_threshold=1,
        retry_attempts=2,
    )
    with _providers(vertex_wire=vertex_wire, anthropic_wire=anthropic_wire, groq_wire=groq_wire):
        # anthropic is rate limited (spend limit, quota): Vertex serves.
        first = await _served_text(engine, config)
        # anthropic's breaker is open; Vertex is unavailable (retried once), groq serves.
        vertex_wire.respond = _google_error(503, "UNAVAILABLE")
        second = await _served_text(engine, config)
        # Both breakers are open: straight to groq.
        third = await _served_text(engine, config)

    assert (first, second, third) == (VERTEX_TEXT, GROQ_TEXT, GROQ_TEXT)
    assert len(anthropic_wire.requests) == 1
    assert len(vertex_wire.requests) == 3
    assert len(groq_wire.requests) == 2
    assert breaker.is_open("anthropic") and breaker.is_open("vertex")


# ─── Streaming ─────────────────────────────────────────────────────────────


async def test_stream_text_tool_calls_usage_and_request_options():
    events = [
        {
            "type": "message_start",
            "message": {
                **_message(),
                "content": [],
                "usage": {"input_tokens": 9, "output_tokens": 0, "cache_read_input_tokens": 3},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"q": "x"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 6},
        },
        {"type": "message_stop"},
    ]
    wire = _Wire(lambda request: _sse_response(request, events, named=True))
    config = _config(stream_idle_timeout_seconds=7)
    with _providers(vertex_wire=wire):
        deltas = [
            d
            async for d in stream_chat(
                messages=MESSAGES,
                tools=[{"type": "function", "function": {"name": "lookup"}}],
                config=config,
            )
        ]

    request = wire.requests[0]
    assert str(request.url).endswith(f"/models/{MODEL}:streamRawPredict")
    body = json.loads(request.content)
    assert body["stream"] is True and body["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in body
    assert request.extensions["timeout"]["read"] == 7
    assert [d.content for d in deltas if d.content] == ["Hi"]
    tool_deltas = [d.tool_call_deltas[0] for d in deltas if d.tool_call_deltas]
    assert (tool_deltas[0]["id"], tool_deltas[0]["function"]["name"]) == ("toolu_1", "lookup")
    final = deltas[-1]
    assert (final.finish_reason, final.provider, final.model) == ("tool_calls", "vertex", MODEL)
    assert final.usage_details == Usage(12, 6, 3, 0, 0)


async def test_empty_stream_is_guarded_and_fails_over():
    empty = _stream_events(text=None)
    vertex_wire = _Wire(lambda request: _sse_response(request, empty, named=True))
    groq_wire = _Wire(_groq_ok)
    config = _config(provider_order="vertex,groq", groq_api_key="k", breaker_failure_threshold=1)
    with _providers(vertex_wire=vertex_wire, groq_wire=groq_wire):
        text = "".join(
            [d.content or "" async for d in stream_chat(messages=MESSAGES, config=config)]
        )
        with pytest.raises(EmptyCompletionError) as excinfo:
            [d async for d in vertex.stream_chat(_config(), MESSAGES, None, 10)]

    assert text == GROQ_TEXT
    assert len(vertex_wire.requests) == 2  # one per stream: not retried
    assert breaker.is_open("vertex")
    assert (excinfo.value.provider, excinfo.value.finish_reason) == ("vertex", "stop")


# ─── Policy and capabilities ───────────────────────────────────────────────


async def test_eu_residency_includes_vertex_only_when_its_metadata_asserts_eu():
    anthropic_wire, vertex_wire = _Wire(_json(_message())), _Wire(_vertex_ok)
    settings = dict(anthropic_api_key="k", provider_order="anthropic,vertex", policy_residency="eu")
    eu = _config(
        **settings,
        provider_metadata={"vertex": {"region": "eu", "trains_on_data": False, "dpa": True}},
    )
    with _providers(vertex_wire=vertex_wire, anthropic_wire=anthropic_wire):
        served = await complete_with_usage(system="s", prompt="p", config=eu)
        with pytest.raises(PolicyViolationError) as unasserted:
            await complete_with_usage(system="s", prompt="p", config=_config(**settings))
        with pytest.raises(PolicyViolationError) as global_endpoint:
            await complete_with_usage(
                system="s",
                prompt="p",
                config=_config(**settings, provider_metadata={"vertex": {"region": "global"}}),
            )

    assert served.provider == "vertex"
    assert anthropic_wire.requests == []
    assert len(vertex_wire.requests) == 1
    assert "region=unknown does not match residency=eu" in unasserted.value.exclusions["vertex"]
    assert "region=global does not match residency=eu" in global_endpoint.value.exclusions["vertex"]


@pytest.mark.parametrize(
    "model", [MODEL, "claude-haiku-4-5", "claude-sonnet-4-5@20250929", "claude-opus-4-1@20250805"]
)
def test_capabilities_mirror_the_anthropic_model_family(model):
    anthropic_caps = capabilities_for("anthropic", model.split("@")[0])
    assert capabilities_for("vertex", model, vertex_structured_outputs=True) == anthropic_caps
    assert capabilities_for("vertex", model) == ModelCapabilities(
        structured_output="unsupported",
        tools=anthropic_caps.tools,
        images=anthropic_caps.images,
        streaming=anthropic_caps.streaming,
    )


def test_haiku_capabilities_and_unknown_models():
    assert capabilities_for("vertex", MODEL, vertex_structured_outputs=True) == ModelCapabilities(
        structured_output="strict", tools=True, images=True, streaming=True
    )
    assert capabilities_for("vertex", MODEL) == ModelCapabilities(
        structured_output="unsupported", tools=True, images=True, streaming=True
    )
    assert capabilities_for("vertex", "gemini-2.5-flash", vertex_structured_outputs=True) == UNKNOWN


async def test_structured_outputs_off_skips_vertex_for_output_schema_without_a_request():
    anthropic_reply = _json(_message('{"score": 4, "reason": "clear"}'))
    anthropic_wire, vertex_wire = _Wire(anthropic_reply), _Wire(_vertex_ok)
    # Default settings: no POLICY_REQUIRE_PARAMETERS needed.
    config = _config(anthropic_api_key="k", provider_order="vertex,anthropic")
    with _providers(vertex_wire=vertex_wire, anthropic_wire=anthropic_wire):
        served = await complete_with_usage(
            system="s", prompt="p", config=config, output_schema=_Score
        )
        with pytest.raises(UnsupportedCapabilityError) as excinfo:
            await complete_with_usage(
                system="s", prompt="p", config=_config(), output_schema=_Score
            )
        # A plain completion still goes to Vertex.
        plain = await complete_with_usage(system="s", prompt="p", config=config)

    assert (served.provider, served.parsed) == ("anthropic", _Score(score=4, reason="clear"))
    assert plain.provider == "vertex"
    assert len(vertex_wire.requests) == 1  # only the plain completion
    assert not breaker.is_open("vertex")
    assert excinfo.value.exclusions["vertex"] == (
        f"model {MODEL} does not support structured output",
    )


async def test_structured_outputs_off_still_serves_chat_response_format_on_vertex():
    # chat()'s response_format is a forced tool call, not Vertex structured outputs.
    tool_reply = _message(
        content=[
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": STRUCTURED_OUTPUT_TOOL_NAME,
                "input": {"score": 4, "reason": "clear"},
            }
        ]
    )
    wire = _Wire(_json(tool_reply))
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "score", "schema": _Score.model_json_schema()},
    }
    with _providers(vertex_wire=wire):
        result = await chat(messages=MESSAGES, config=_config(), response_format=response_format)
    assert json.loads(result.content) == {"score": 4, "reason": "clear"}
    assert "output_config" not in wire.bodies[0]


async def test_structured_outputs_on_sends_output_config_to_vertex():
    wire = _Wire(_json(_message('{"score": 5, "reason": "great"}')))
    config = _config(anthropic_api_key="k", provider_order="vertex,anthropic")
    with _providers(vertex_wire=wire, anthropic_wire=_Wire(_status(500))):
        result = await complete_with_usage(
            system="s",
            prompt="p",
            config=config.model_copy(update={"vertex_structured_outputs": True}),
            output_schema=_Score,
        )
    assert (result.provider, result.parsed) == ("vertex", _Score(score=5, reason="great"))
    assert wire.bodies[0]["output_config"]["format"]["type"] == "json_schema"


# ─── HTTP service and shutdown ─────────────────────────────────────────────


def test_models_endpoint_lists_vertex_when_configured():
    from fastapi.testclient import TestClient

    from llm_gateway.service.app import create_app

    headers = {"Authorization": "Bearer secret-key"}
    configured = TestClient(create_app(_config(gateway_api_keys="secret-key")))
    by_id = {m["id"]: m for m in configured.get("/v1/models", headers=headers).json()["data"]}
    assert by_id[f"vertex/{MODEL}"] == {
        "id": f"vertex/{MODEL}",
        "object": "model",
        "configured": True,
        "available": True,
    }

    unconfigured = TestClient(
        create_app(GatewayConfig(_env_file=None, gateway_api_keys="secret-key"))
    )
    by_id = {m["id"]: m for m in unconfigured.get("/v1/models", headers=headers).json()["data"]}
    assert by_id[f"vertex/{MODEL}"]["configured"] is False


async def test_aclose_closes_clients_and_azure_credentials():
    client = MagicMock()
    client.close = AsyncMock()
    managed_identity = MagicMock()
    vertex._clients[("key",)] = client
    vertex._credentials[("a",)] = (object(), managed_identity)
    vertex._credentials[("b",)] = (object(), None)

    await vertex.aclose()

    client.close.assert_awaited_once()
    managed_identity.close.assert_called_once()
    assert vertex._clients == {} and vertex._credentials == {}
