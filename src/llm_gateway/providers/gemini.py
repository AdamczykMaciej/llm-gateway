"""Google Gemini on Vertex AI, through Vertex's OpenAI-compatible endpoint.

Requests go to
`<vertex endpoint>/projects/<project>/locations/<location>/endpoints/openapi`
(https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library,
checked 2026-09-29) with the shared OpenAI-compatible request code, so
Chat Completions, streaming, tool calling and `json_schema` output follow the
OpenAI shapes. Model ids are `google/<model>`, e.g. `google/gemini-3.5-flash`.

Auth is a Google OAuth access token, never an API key, and it reuses the
`vertex` provider's credential machinery and settings unchanged:
Application Default Credentials, a `vertex_credentials_file` (Workload
Identity Federation), `vertex_azure_app_id_uri`,
`vertex_impersonate_service_account`; service-account keys are refused. Access
tokens live one hour, so the token is not baked into the cached client: the
SDK awaits `_token_provider()` before every request, which returns the
current token and refreshes the credentials (in a worker thread) once google-auth
says they are no longer valid. Needs the `vertex` extra (and `azure` for the
Azure supplier).

Gemini has its own project (`gemini_project_id`, which switches it on),
location (`gemini_location`, default `eu`) and model (`gemini_model`). Which
models are served in which location is Google's to say, not this module's; see
the README. Residency, retention, training and DPA facts come from the
operator's PROVIDER_METADATA only.

Missing extras and every google-auth failure raise `ProviderAuthError`
(AUTH: fails over, counts for the breaker). HTTP failures are ordinary OpenAI
SDK errors.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from ..config import GatewayConfig
from ..errors import ProviderAuthError
from . import vertex
from .openai_compatible import OpenAICompatProvider, OpenAICompatSpec

PROVIDER = "gemini"


def base_url(config: GatewayConfig) -> str:
    """The OpenAI-compatible endpoint for the configured project and location."""
    location = config.gemini_location
    return (
        f"{vertex.base_url(location)}/projects/{config.gemini_project_id}"
        f"/locations/{location}/endpoints/openapi"
    )


def _fresh_token(credentials: Any) -> str:
    """The credentials' access token, refreshed first when it is missing or
    about to expire. Blocking: run it in a worker thread."""
    from google.auth.transport.requests import Request

    if not credentials.valid:
        credentials.refresh(Request())
    token = credentials.token
    if not token:
        raise ProviderAuthError(PROVIDER, "the Google credentials returned no access token.")
    return token


def _token_provider(config: GatewayConfig) -> Callable[[], Awaitable[str]]:
    """The callable the OpenAI SDK awaits before every request."""

    async def token() -> str:
        auth_errors = vertex._auth_error_types()
        try:
            credentials = await vertex._google_credentials(config)
            return await asyncio.to_thread(_fresh_token, credentials)
        except auth_errors as e:
            raise ProviderAuthError(
                PROVIDER,
                f"could not obtain a Google access token ({type(e).__name__}). Check the "
                "credential configuration, the workload identity pool provider and the "
                "'Vertex AI User' (roles/aiplatform.user) grant.",
            ) from e
        except ImportError as e:
            raise vertex._missing_extra(e, extra="vertex", library="google-auth") from None

    return token


_provider = OpenAICompatProvider(
    OpenAICompatSpec(
        provider=PROVIDER,
        api_key=lambda config: "",
        base_url=base_url,
        default_model=lambda config: config.gemini_model,
        strict_json_schema=lambda config: True,
        also_configured=lambda config: bool(config.gemini_project_id and config.gemini_model),
        token_provider=_token_provider,
        client_key=lambda config: vertex._auth_key(config),
    )
)

_clients = _provider._clients


def _client(config: GatewayConfig) -> AsyncOpenAI:
    return _provider._client(config)


async def aclose() -> None:
    """Close every cached SDK client, then the credentials shared with the
    `vertex` provider (e.g. on shutdown)."""
    clients = list(_clients.values())
    _clients.clear()
    for client in clients:
        await client.close()
    await vertex.aclose()


# Looked up by name at call time, so `patch.object(<module>, "_client", ...)`
# works exactly as for the other providers.
_provider.client_factory = lambda config: _client(config)
configured = _provider.configured
default_model = _provider.default_model
call = _provider.call
chat = _provider.chat
stream_chat = _provider.stream_chat
