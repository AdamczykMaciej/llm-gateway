"""Anthropic Claude on Google Cloud Vertex AI (Agent Platform).

The same model family as the `anthropic` provider, billed through Google
Cloud. Requests go through the Anthropic SDK's `AsyncAnthropicVertex` and the
anthropic provider's own request/response code (`messages_call()`,
`messages_chat()`, `messages_stream()`), so translation, prompt caching,
structured output, usage reporting and the empty-response guard are shared.
The Vertex client moves `model` into the URL
(`.../locations/<location>/publishers/anthropic/models/<model>:rawPredict`)
and sends `anthropic_version: vertex-2023-10-16` in the body.

Auth is Google OAuth access tokens, never an API key:

- no `vertex_credentials_file`: Application Default Credentials
  (`google.auth.default()`), e.g. an attached service account on Google Cloud
  or `gcloud auth application-default login` locally;
- `vertex_credentials_file`: a credential configuration file, typically an
  `external_account` config for Workload Identity Federation. It holds no key.
  Service-account key files are rejected at startup, and so is ADC that
  resolves to one;
- `vertex_azure_app_id_uri`: Workload Identity Federation from an Azure
  managed identity. The external_account config's `credential_source` is
  replaced by a subject-token supplier that asks azure-identity's
  `ManagedIdentityCredential` for a token for that application ID URI, which
  works on Azure Container Apps (its identity endpoint is `IDENTITY_ENDPOINT`
  with a rotating `IDENTITY_HEADER`, not the VM metadata address gcloud's
  `--azure` config reads);
- `vertex_impersonate_service_account`: the Google service account to act as.
  For an external_account config it becomes the config's
  `service_account_impersonation_url` (the STS flow gcloud's
  `--service-account` writes); otherwise the source credentials are wrapped
  in `impersonated_credentials.Credentials`.

Credentials are built on first use in a worker thread (loading ADC can probe
the metadata server), and the SDK refreshes the token in a worker thread
whenever google-auth reports it expired. google-auth, and azure-identity for
the Azure supplier, are imported lazily: install the `vertex` extra (and the
`azure` extra for `vertex_azure_app_id_uri`).

Missing extras and every google-auth error (`RefreshError`,
`DefaultCredentialsError`, `TransportError` from the token exchange) raise
`ProviderAuthError` (AUTH: fails over, counts for the breaker). HTTP failures
are ordinary anthropic SDK errors and are classified like Anthropic's own:
429 `RESOURCE_EXHAUSTED` is RATE_LIMITED, 5xx TRANSIENT, 400/404
INVALID_REQUEST.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from ..config import (
    GatewayConfig,
    azure_app_id_uri_scope,
    read_vertex_credentials_file,
    service_account_impersonation_url,
)
from ..errors import ProviderAuthError
from ..structured import OutputSchema
from . import anthropic as anthropic_messages
from .base import (
    ChatResult,
    ProviderResult,
    StreamDelta,
    guard_empty_stream,
    sdk_client_cache_key,
    sdk_client_options,
    stream_request_options,
)

logger = logging.getLogger("llm_gateway")

PROVIDER = "vertex"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

_clients: dict[tuple, Any] = {}
# Auth settings -> (google credentials, azure-identity credential or None).
# Shared by every client that uses the same credentials.
_credentials: dict[tuple, tuple[Any, Any]] = {}
_missing_extra_logged: set[str] = set()


def base_url(location: str) -> str:
    """The Vertex AI endpoint for a location, as the SDK derives it: the global
    endpoint, a multi-region (`us`, `eu`) `rep` endpoint, or a regional one."""
    if location == "global":
        return "https://aiplatform.googleapis.com/v1"
    if location in ("us", "eu"):
        return f"https://aiplatform.{location}.rep.googleapis.com/v1"
    return f"https://{location}-aiplatform.googleapis.com/v1"


def configured(config: GatewayConfig) -> bool:
    return bool(config.vertex_project_id and config.vertex_model)


def default_model(config: GatewayConfig) -> str:
    return config.vertex_model


def _missing_extra(error: ImportError, *, extra: str, library: str) -> ProviderAuthError:
    if extra not in _missing_extra_logged:
        _missing_extra_logged.add(extra)
        logger.error(
            "llm_gateway vertex: needs %s (pip install 'llm-gateway[%s]'); failing over (%s)",
            library,
            extra,
            type(error).__name__,
        )
    return ProviderAuthError(
        PROVIDER,
        f"requires {library}: install the '{extra}' extra (pip install 'llm-gateway[{extra}]').",
    )


def _auth_error_types() -> tuple[type[BaseException], ...]:
    try:
        from google.auth.exceptions import GoogleAuthError
    except ImportError:
        return ()
    return (GoogleAuthError,)


def _auth_failure(error: BaseException) -> ProviderAuthError:
    return ProviderAuthError(
        PROVIDER,
        f"could not obtain a Google access token ({type(error).__name__}). Check the "
        "credential configuration, the workload identity pool provider and the "
        "'Vertex AI User' (roles/aiplatform.user) grant.",
    )


def _azure_subject_token_supplier(config: GatewayConfig) -> tuple[Any, Any]:
    """A google-auth subject-token supplier backed by an Azure managed
    identity, and the azure-identity credential (to close on shutdown)."""
    from google.auth import exceptions as google_exceptions
    from google.auth import identity_pool

    try:
        from azure.core.exceptions import AzureError
        from azure.identity import ManagedIdentityCredential
    except ImportError as e:
        raise _missing_extra(e, extra="azure", library="azure-identity") from None

    client_id = config.vertex_azure_managed_identity_client_id
    credential = (
        ManagedIdentityCredential(client_id=client_id) if client_id else ManagedIdentityCredential()
    )
    scope = azure_app_id_uri_scope(config.vertex_azure_app_id_uri)

    class ManagedIdentitySupplier(identity_pool.SubjectTokenSupplier):
        def get_subject_token(self, context: Any, request: Any) -> str:
            try:
                return credential.get_token(scope).token
            # AzureError also covers an unreachable identity endpoint
            # (ServiceRequestError), not only ClientAuthenticationError.
            except AzureError as e:
                raise google_exceptions.RefreshError(
                    f"could not acquire an Azure managed identity token for {scope} "
                    f"({type(e).__name__})"
                ) from e

    return ManagedIdentitySupplier(), credential


_AWS_SUBJECT_TOKEN_TYPE = "urn:ietf:params:aws:token-type:aws4_request"


def _credentials_from_info(info: dict, **kwargs: Any) -> Any:
    """Credentials for a parsed credential configuration, with the loader for
    its type. Unlike `google.auth.load_credentials_from_dict()` (deprecated
    upstream), this never resolves a project id, which for external_account
    configs means an STS exchange and a Resource Manager call at load time."""
    kind = info.get("type")
    scopes = [CLOUD_PLATFORM_SCOPE]
    if kind == "external_account":
        if info.get("subject_token_type") == _AWS_SUBJECT_TOKEN_TYPE:
            from google.auth import aws

            return aws.Credentials.from_info(info, scopes=scopes)
        if (info.get("credential_source") or {}).get("executable") is not None:
            from google.auth import pluggable

            return pluggable.Credentials.from_info(info, scopes=scopes)
        from google.auth import identity_pool

        return identity_pool.Credentials.from_info(info, scopes=scopes, **kwargs)
    if kind == "external_account_authorized_user":
        from google.auth import external_account_authorized_user

        return external_account_authorized_user.Credentials.from_info(info, scopes=scopes)
    if kind == "impersonated_service_account":
        from google.auth import impersonated_credentials

        return impersonated_credentials.Credentials.from_impersonated_service_account_info(
            info, scopes=scopes
        )
    from google.oauth2 import credentials as user_credentials

    return user_credentials.Credentials.from_authorized_user_info(info, scopes=scopes)


def _key_based(credentials: Any) -> bool:
    """True when credentials are backed by a long-lived key: a service account
    key or a GDCH service account, directly or as the source (at any depth)
    of impersonated credentials. Inspects types only, never the network."""
    from google.auth import impersonated_credentials
    from google.oauth2 import gdch_credentials, service_account

    for _ in range(10):  # impersonation chains are short; never loop forever
        if not isinstance(credentials, impersonated_credentials.Credentials):
            break
        credentials = getattr(credentials, "_source_credentials", None)
    return isinstance(
        credentials, service_account.Credentials | gdch_credentials.ServiceAccountCredentials
    )


def load_credentials(config: GatewayConfig) -> tuple[Any, Any]:
    """(google credentials, closeable azure credential or None) for the
    config. Blocking (file reads, and ADC may probe the metadata server), but
    it fetches no token: the SDK refreshes on the first request."""
    try:
        import google.auth
        import google.auth.transport.requests  # noqa: F401 (the SDK refreshes with it)
        from google.auth import exceptions as google_exceptions
        from google.auth import impersonated_credentials
        from google.oauth2 import gdch_credentials, service_account  # noqa: F401 (_key_based)
    except ImportError as e:
        raise _missing_extra(e, extra="vertex", library="google-auth") from None

    impersonate = config.vertex_impersonate_service_account
    closeable = None
    try:
        if not config.vertex_credentials_file:
            credentials, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
        else:
            info = read_vertex_credentials_file(config.vertex_credentials_file)
            if impersonate and info.get("type") == "external_account":
                info["service_account_impersonation_url"] = service_account_impersonation_url(
                    impersonate
                )
                impersonate = ""
            if config.vertex_azure_app_id_uri:
                supplier, closeable = _azure_subject_token_supplier(config)
                info.pop("credential_source", None)
                credentials = _credentials_from_info(info, subject_token_supplier=supplier)
            else:
                credentials = _credentials_from_info(info)
    except (google_exceptions.GoogleAuthError, ValueError) as e:
        raise ProviderAuthError(
            PROVIDER,
            f"could not load Google credentials ({type(e).__name__}). Check "
            "vertex_credentials_file, or Application Default Credentials when it is unset.",
        ) from e

    if _key_based(credentials):
        raise ProviderAuthError(
            PROVIDER,
            "the credentials resolve to a service account key (directly, as the source of "
            "impersonated credentials, or a GDCH service account); keys are not accepted. "
            "Use Workload Identity Federation (an external_account credential "
            "configuration) or an attached service account.",
        )
    if impersonate:
        credentials = impersonated_credentials.Credentials(
            source_credentials=credentials,
            target_principal=impersonate,
            target_scopes=[CLOUD_PLATFORM_SCOPE],
        )
    return credentials, closeable


def _auth_key(config: GatewayConfig) -> tuple:
    return (
        config.vertex_credentials_file,
        config.vertex_impersonate_service_account,
        config.vertex_azure_app_id_uri,
        config.vertex_azure_managed_identity_client_id,
    )


async def _google_credentials(config: GatewayConfig) -> Any:
    key = _auth_key(config)
    cached = _credentials.get(key)
    if cached is None:
        loaded = await asyncio.to_thread(load_credentials, config)
        cached = _credentials.setdefault(key, loaded)
        if cached is not loaded and loaded[1] is not None:
            loaded[1].close()  # a concurrent first call won the race; keep one credential
    return cached[0]


def _load_sdk() -> tuple[Any, Any]:
    try:
        import google.auth  # noqa: F401
        from anthropic import AsyncAnthropicVertex, DefaultAsyncHttpxClient
    except ImportError as e:
        raise _missing_extra(e, extra="vertex", library="google-auth") from None
    return AsyncAnthropicVertex, DefaultAsyncHttpxClient


async def _client(config: GatewayConfig) -> Any:
    key = (
        config.vertex_project_id,
        config.vertex_location,
        *_auth_key(config),
        *sdk_client_cache_key("", config),
    )
    client = _clients.get(key)
    if client is not None:
        return client
    client_class, http_client_class = _load_sdk()
    credentials = await _google_credentials(config)
    client = client_class(
        region=config.vertex_location,
        project_id=config.vertex_project_id,
        credentials=credentials,
        # Explicit, so ANTHROPIC_VERTEX_BASE_URL in the environment can't redirect it.
        base_url=base_url(config.vertex_location),
        http_client=http_client_class(verify=False) if not config.ssl_verify else None,
        **sdk_client_options(config),
    )
    return _clients.setdefault(key, client)


async def aclose() -> None:
    """Close every cached SDK client and Azure credential (e.g. on shutdown)."""
    clients = list(_clients.values())
    closeables = [closeable for _, closeable in _credentials.values() if closeable is not None]
    _clients.clear()
    _credentials.clear()
    for client in clients:
        await client.close()
    for closeable in closeables:
        closeable.close()


async def call(
    config: GatewayConfig,
    system: str,
    prompt: str,
    max_tokens: int,
    model: str | None = None,
    *,
    output_schema: OutputSchema | None = None,
    cache_system: bool = False,
) -> ProviderResult:
    auth_errors = _auth_error_types()
    try:
        return await anthropic_messages.messages_call(
            lambda: _client(config),
            provider=PROVIDER,
            model=model or default_model(config),
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
            output_schema=output_schema,
            cache_system=cache_system,
        )
    except auth_errors as e:
        raise _auth_failure(e) from e


async def chat(
    config: GatewayConfig,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    *,
    model: str | None = None,
    tool_choice: object = None,
    sampling: dict | None = None,
    response_format: dict | None = None,
) -> ChatResult:
    auth_errors = _auth_error_types()
    try:
        return await anthropic_messages.messages_chat(
            lambda: _client(config),
            provider=PROVIDER,
            model=model or default_model(config),
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            sampling=sampling,
            response_format=response_format,
        )
    except auth_errors as e:
        raise _auth_failure(e) from e


async def _stream_chat(
    config: GatewayConfig,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    *,
    model: str | None = None,
    tool_choice: object = None,
    sampling: dict | None = None,
    response_format: dict | None = None,
) -> AsyncIterator[StreamDelta]:
    deltas = anthropic_messages.messages_stream(
        lambda: _client(config),
        model=model or default_model(config),
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        tool_choice=tool_choice,
        sampling=sampling,
        response_format=response_format,
        request_options=stream_request_options(config),
    )
    auth_errors = _auth_error_types()
    async with aclosing(deltas):
        try:
            async for delta in deltas:
                yield delta
        except auth_errors as e:
            raise _auth_failure(e) from e


async def stream_chat(
    config: GatewayConfig,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    *,
    model: str | None = None,
    tool_choice: object = None,
    sampling: dict | None = None,
    response_format: dict | None = None,
) -> AsyncIterator[StreamDelta]:
    """`_stream_chat()` behind `guard_empty_stream()`: a stream with no text
    and no tool calls raises `EmptyCompletionError` before its first chunk."""
    deltas = _stream_chat(
        config,
        messages,
        tools,
        max_tokens,
        model=model,
        tool_choice=tool_choice,
        sampling=sampling,
        response_format=response_format,
    )
    guarded = guard_empty_stream(deltas, provider=PROVIDER, model=model or default_model(config))
    async with aclosing(guarded):
        async for delta in guarded:
            yield delta
