from collections.abc import AsyncIterator

from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from ..config import GatewayConfig
from .base import (
    ChatResult,
    ProviderResult,
    StreamDelta,
    check_openai_style_completion,
    openai_sampling_kwargs,
    parse_openai_style_chunk,
    parse_openai_style_response,
    sdk_client_cache_key,
    sdk_client_options,
    stream_request_options,
)

_clients: dict[tuple, AsyncOpenAI] = {}


def _client(config: GatewayConfig) -> AsyncOpenAI:
    key = sdk_client_cache_key(config.groq_api_key, config)
    client = _clients.get(key)
    if client is None:
        # openai>=3.0 is httpx2-based; plain httpx clients are only a
        # temporary legacy escape hatch there, so build an httpx2 one.
        http_client = DefaultAsyncHttpx2Client(verify=False) if not config.ssl_verify else None
        client = AsyncOpenAI(
            api_key=config.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            http_client=http_client,
            **sdk_client_options(config),
        )
        _clients[key] = client
    return client


# Groq models that accept `reasoning_effort`, per
# https://console.groq.com/docs/reasoning (checked 2026-09-14): GPT-OSS 20B and
# 120B take low|medium|high; Qwen 3.6 27B and Qwen 3.8 27B take
# none|default|low|medium|high. No other model gets `reasoning_effort` at all,
# because Groq rejects it for models that do not support it. "low" keeps the
# hidden reasoning from spending the whole `max_tokens` budget and leaving the
# visible reply empty (see EmptyCompletionError).
_REASONING_EFFORT_PREFIXES = ("openai/gpt-oss-", "gpt-oss-")
_REASONING_EFFORT_MODELS = frozenset({"qwen/qwen3.6-27b", "qwen/qwen3.8-27b"})
REASONING_EFFORT = "low"


def reasoning_kwargs(model: str) -> dict:
    """`{"reasoning_effort": "low"}` for a Groq reasoning model, else `{}`."""
    model_id = model.strip().lower()
    if model_id.startswith(_REASONING_EFFORT_PREFIXES) or model_id in _REASONING_EFFORT_MODELS:
        return {"reasoning_effort": REASONING_EFFORT}
    return {}


def configured(config: GatewayConfig) -> bool:
    return bool(config.groq_api_key)


def default_model(config: GatewayConfig) -> str:
    return config.groq_model


async def call(
    config: GatewayConfig,
    system: str,
    prompt: str,
    max_tokens: int,
    model: str | None = None,
) -> ProviderResult:
    model = model or default_model(config)
    resp = await _client(config).chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        **reasoning_kwargs(model),
    )
    check_openai_style_completion(resp, provider="groq", model=model)
    input_tokens = resp.usage.prompt_tokens if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0
    return ProviderResult(
        text=(resp.choices[0].message.content or "").strip(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


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
    kwargs.update(reasoning_kwargs(model))
    resp = await _client(config).chat.completions.create(
        model=model, max_tokens=max_tokens, messages=messages, **kwargs
    )
    check_openai_style_completion(resp, provider="groq", model=model)
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
    kwargs.update(reasoning_kwargs(model))
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
