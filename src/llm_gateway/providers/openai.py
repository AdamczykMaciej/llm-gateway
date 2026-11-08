import re
from collections.abc import AsyncGenerator
from contextlib import aclosing

from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from ..config import GatewayConfig
from ..structured import OutputSchema
from .base import (
    REASONING_REJECTED_SAMPLING,
    ChatResult,
    ProviderResult,
    StreamDelta,
    close_clients,
    drop_keys,
    guard_empty_stream,
    openai_sampling_kwargs,
    parse_openai_style_chunk,
    parse_openai_style_response,
    provider_result_from_openai_style,
    remember_client,
    sdk_client_cache_key,
    sdk_client_options,
    sdk_messages,
    stream_request_options,
)

_clients: dict[tuple, AsyncOpenAI] = {}

# Reasoning models (o-series, gpt-5 family; `gpt-5-chat*` is not one): the API
# rejects `max_tokens` (it wants `max_completion_tokens`, which also covers
# the reasoning tokens) and any non-default sampling, each a 400 on every call.
_REASONING_MODEL = re.compile(r"^(?:o\d|gpt-5(?!-chat))")


def _request_params(model: str, max_tokens: int, kwargs: dict) -> dict:
    """`max_tokens` and sampling kwargs in the form `model` accepts."""
    if _REASONING_MODEL.match(model):
        return {
            "max_completion_tokens": max_tokens,
            **drop_keys(kwargs, REASONING_REJECTED_SAMPLING),
        }
    return {"max_tokens": max_tokens, **kwargs}


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
        remember_client(_clients, key, client)
    return client


async def aclose() -> None:
    """Close every cached SDK client (e.g. on shutdown)."""
    await close_clients(_clients)


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
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        **_request_params(model, max_tokens, kwargs),
    )
    return provider_result_from_openai_style(resp, model, provider="openai")


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
    parallel_tool_calls: bool | None = None,
) -> ChatResult:
    model = model or default_model(config)
    kwargs: dict = openai_sampling_kwargs(sampling)
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = await _client(config).chat.completions.create(
        model=model, messages=sdk_messages(messages), **_request_params(model, max_tokens, kwargs)
    )
    return parse_openai_style_response(resp, model, provider="openai")


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
    parallel_tool_calls: bool | None = None,
) -> AsyncGenerator[StreamDelta]:
    model = model or default_model(config)
    kwargs: dict = openai_sampling_kwargs(sampling)
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
    if response_format is not None:
        kwargs["response_format"] = response_format
    stream = await _client(config).chat.completions.create(
        model=model,
        messages=sdk_messages(messages),
        stream=True,
        stream_options={"include_usage": True},
        **_request_params(model, max_tokens, kwargs),
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
    parallel_tool_calls: bool | None = None,
    cache_system: bool = False,
) -> AsyncGenerator[StreamDelta]:
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
        parallel_tool_calls=parallel_tool_calls,
    )
    guarded = guard_empty_stream(
        deltas, provider="openai", model=model or default_model(config), strip_think=True
    )
    async with aclosing(guarded):
        async for delta in guarded:
            yield delta


def residency(config: GatewayConfig) -> str:
    """Where prompts are processed: 'global'. See residency.py."""
    return "global"
