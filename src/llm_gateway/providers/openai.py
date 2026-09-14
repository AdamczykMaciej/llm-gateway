from collections.abc import AsyncIterator

from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from ..config import GatewayConfig
from ..structured import OutputSchema
from .base import (
    ChatResult,
    ProviderResult,
    StreamDelta,
    openai_sampling_kwargs,
    parse_openai_style_chunk,
    parse_openai_style_response,
    provider_result_from_openai_style,
    sdk_client_cache_key,
    sdk_client_options,
    stream_request_options,
)

_clients: dict[tuple, AsyncOpenAI] = {}


def _client(config: GatewayConfig) -> AsyncOpenAI:
    key = sdk_client_cache_key(config.openai_api_key, config)
    client = _clients.get(key)
    if client is None:
        # openai>=3.0 is httpx2-based; plain httpx clients are only a
        # temporary legacy escape hatch there, so build an httpx2 one.
        http_client = DefaultAsyncHttpx2Client(verify=False) if not config.ssl_verify else None
        client = AsyncOpenAI(
            api_key=config.openai_api_key,
            http_client=http_client,
            **sdk_client_options(config),
        )
        _clients[key] = client
    return client


def configured(config: GatewayConfig) -> bool:
    return bool(config.openai_api_key)


def default_model(config: GatewayConfig) -> str:
    return config.openai_model


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
    # `cache_system` needs nothing here: OpenAI caches long prompt prefixes
    # automatically and reports hits as usage.prompt_tokens_details.cached_tokens.
    model = model or default_model(config)
    kwargs: dict = {}
    if output_schema is not None:
        kwargs["response_format"] = output_schema.openai_response_format()
    resp = await _client(config).chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    return provider_result_from_openai_style(resp, model)


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
    kwargs: dict = openai_sampling_kwargs(sampling)
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = await _client(config).chat.completions.create(
        model=model, max_tokens=max_tokens, messages=messages, **kwargs
    )
    return parse_openai_style_response(resp, model)


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
    model = model or default_model(config)
    kwargs: dict = openai_sampling_kwargs(sampling)
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    stream = await _client(config).chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
        **stream_request_options(config),
    )
    async for chunk in stream:
        yield parse_openai_style_chunk(chunk, model)
