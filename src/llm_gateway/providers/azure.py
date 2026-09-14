"""Azure AI Foundry / Azure OpenAI, through the v1 API.

The v1 API speaks the OpenAI chat-completions wire format at
`<endpoint>/openai/v1/`, with no `api-version` parameter, so this provider is
the plain openai SDK with Azure's base URL and auth
(https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle).
`model` is the deployment name, which is arbitrary, so nothing here keys off
the model id: `GatewayConfig.azure_reasoning_effort` applies to every request.

Auth (`GatewayConfig.azure_auth`):
- "api_key": the resource key, sent by the SDK as `Authorization: Bearer <key>`,
  exactly as Microsoft's own `OpenAI(api_key=..., base_url=...)` example does.
- "entra": Microsoft Entra ID tokens for scope `https://ai.azure.com/.default`.
  `AsyncOpenAI` awaits a callable `api_key` before every request
  (`api_key: str | Callable[[], Awaitable[str]]`), so the token provider is
  azure-identity's *async* `get_bearer_token_provider`, which caches the token
  and refreshes it before expiry. azure-identity (and aiohttp, which its async
  credentials need) are imported lazily: install the `azure` extra.

Token-acquisition failures, and a missing azure-identity, raise
`ProviderAuthError` (AUTH: fails over, counts for the breaker). Every other
failure is an ordinary openai SDK error, classified like OpenAI's own.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any

import httpx2
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from ..config import GatewayConfig
from ..errors import ProviderAuthError
from ..structured import OutputSchema
from .base import (
    ChatResult,
    ProviderResult,
    StreamDelta,
    guard_empty_stream,
    openai_sampling_kwargs,
    parse_openai_style_chunk,
    parse_openai_style_response,
    provider_result_from_openai_style,
    sdk_client_cache_key,
    sdk_client_options,
    stream_request_options,
)

logger = logging.getLogger("llm_gateway")

PROVIDER = "azure"
# Token scope for the v1 API with Microsoft Entra ID (Cognitive Services
# OpenAI User role at runtime).
TOKEN_SCOPE = "https://ai.azure.com/.default"

_clients: dict[tuple, AsyncOpenAI] = {}
# Entra credential + token provider per managed-identity client id ("" means
# DefaultAzureCredential), shared by every client that uses that identity.
_credentials: dict[str, tuple[Any, Callable[[], Awaitable[str]]]] = {}
_missing_identity_logged = False


def base_url(endpoint: str) -> str:
    """The v1 base URL for an endpoint given with or without `/openai/v1`
    and trailing slashes: `https://r.openai.azure.com/openai/v1/`."""
    url = endpoint.strip().rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if url.lower().endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    return f"{url}/openai/v1/"


def configured(config: GatewayConfig) -> bool:
    if not (config.azure_endpoint.strip() and config.azure_model.strip()):
        return False
    return config.azure_auth != "api_key" or bool(config.azure_api_key)


def default_model(config: GatewayConfig) -> str:
    return config.azure_model.strip()


def reasoning_kwargs(config: GatewayConfig) -> dict:
    """`{"reasoning_effort": ...}` unless `azure_reasoning_effort` is ""."""
    effort = config.azure_reasoning_effort
    return {"reasoning_effort": effort} if effort else {}


def _http_client(config: GatewayConfig) -> httpx2.AsyncClient | None:
    # openai>=3.0 is httpx2-based; plain httpx clients are only a temporary
    # legacy escape hatch there, so build an httpx2 one.
    return DefaultAsyncHttpx2Client(verify=False) if not config.ssl_verify else None


def _missing_identity(error: ImportError) -> ProviderAuthError:
    global _missing_identity_logged
    if not _missing_identity_logged:
        _missing_identity_logged = True
        logger.error(
            "llm_gateway azure: azure_auth=entra needs azure-identity and aiohttp "
            "(pip install 'llm-gateway[azure]'); failing over (%s)",
            type(error).__name__,
        )
    return ProviderAuthError(
        PROVIDER,
        "azure_auth=entra requires azure-identity and aiohttp: "
        "install the 'azure' extra (pip install 'llm-gateway[azure]').",
    )


def _load_identity() -> tuple[Any, type[Exception]]:
    """(azure.identity.aio, ClientAuthenticationError), imported on first use."""
    try:
        from azure.core.exceptions import ClientAuthenticationError
        from azure.identity import aio as identity
    except ImportError as e:
        raise _missing_identity(e) from None
    return identity, ClientAuthenticationError


def _credential(config: GatewayConfig, identity: Any) -> Any:
    """A user-assigned managed identity when its client id is set, else the
    DefaultAzureCredential chain."""
    kwargs: dict = {} if config.ssl_verify else {"connection_verify": False}
    client_id = config.azure_managed_identity_client_id.strip()
    if client_id:
        return identity.ManagedIdentityCredential(client_id=client_id, **kwargs)
    return identity.DefaultAzureCredential(**kwargs)


def _token_provider(config: GatewayConfig) -> Callable[[], Awaitable[str]]:
    client_id = config.azure_managed_identity_client_id.strip()
    cached = _credentials.get(client_id)
    if cached is not None:
        return cached[1]
    identity, auth_error = _load_identity()
    try:
        credential = _credential(config, identity)
    except ImportError as e:  # the async transport needs aiohttp
        raise _missing_identity(e) from None
    get_token = identity.get_bearer_token_provider(credential, TOKEN_SCOPE)

    async def token() -> str:
        try:
            return await get_token()
        except auth_error as e:
            raise ProviderAuthError(
                PROVIDER,
                f"could not acquire a Microsoft Entra token for {TOKEN_SCOPE} "
                f"({type(e).__name__}). Check the identity and its "
                "'Cognitive Services OpenAI User' role assignment.",
            ) from e
        except ImportError as e:
            raise _missing_identity(e) from None

    _credentials[client_id] = (credential, token)
    return token


def _client(config: GatewayConfig) -> AsyncOpenAI:
    auth_secret = (
        config.azure_api_key
        if config.azure_auth == "api_key"
        else config.azure_managed_identity_client_id.strip()
    )
    key = (
        base_url(config.azure_endpoint),
        config.azure_auth,
        *sdk_client_cache_key(auth_secret, config),
    )
    client = _clients.get(key)
    if client is None:
        api_key: str | Callable[[], Awaitable[str]] = (
            config.azure_api_key if config.azure_auth == "api_key" else _token_provider(config)
        )
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url(config.azure_endpoint),
            http_client=_http_client(config),
            **sdk_client_options(config),
        )
        _clients[key] = client
    return client


async def aclose() -> None:
    """Close every cached SDK client and Entra credential (e.g. on shutdown)."""
    clients = list(_clients.values())
    credentials = [credential for credential, _ in _credentials.values()]
    _clients.clear()
    _credentials.clear()
    for client in clients:
        await client.close()
    for credential in credentials:
        await credential.close()


def _request_kwargs(
    config: GatewayConfig,
    tools: list[dict] | None,
    tool_choice: object,
    sampling: dict | None,
    response_format: dict | None,
) -> dict:
    kwargs: dict = {**openai_sampling_kwargs(sampling), **reasoning_kwargs(config)}
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    return kwargs


def token_budget_kwargs(config: GatewayConfig, max_tokens: int) -> dict:
    """The token budget under `GatewayConfig.azure_max_tokens_param`.

    The default, `max_completion_tokens`, suits reasoning deployments: gpt-oss
    and the o-series need it, and it covers reasoning plus visible tokens.
    Set `max_tokens` for a deployment that rejects it (a rejection is a 400,
    which fails over without tripping the breaker, so it would otherwise cost
    a wasted round trip on every call). Which field each deployment accepts
    isn't verified here."""
    return {config.azure_max_tokens_param: max_tokens}


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
    # `cache_system` needs nothing here: Azure caches long prompt prefixes
    # automatically and reports hits as usage.prompt_tokens_details.cached_tokens.
    model = model or default_model(config)
    kwargs: dict = reasoning_kwargs(config)
    if output_schema is not None:
        kwargs["response_format"] = output_schema.openai_response_format()
    resp = await _client(config).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        **token_budget_kwargs(config, max_tokens),
        **kwargs,
    )
    return provider_result_from_openai_style(resp, model, provider=PROVIDER)


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
    model = model or default_model(config)
    kwargs = _request_kwargs(config, tools, tool_choice, sampling, response_format)
    resp = await _client(config).chat.completions.create(
        model=model,
        messages=messages,
        **token_budget_kwargs(config, max_tokens),
        **kwargs,
    )
    return parse_openai_style_response(resp, model, provider=PROVIDER)


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
    model = model or default_model(config)
    kwargs = _request_kwargs(config, tools, tool_choice, sampling, response_format)
    stream = await _client(config).chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **token_budget_kwargs(config, max_tokens),
        **kwargs,
        **stream_request_options(config),
    )
    async for chunk in stream:
        yield parse_openai_style_chunk(chunk, model)


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
    and no tool calls (e.g. only `finish_reason="content_filter"`) raises
    `EmptyCompletionError` before its first chunk."""
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
