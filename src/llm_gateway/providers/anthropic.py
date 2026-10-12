"""Anthropic's Messages API.

The request/response logic lives in `messages_call()`, `messages_chat()` and
`messages_stream()`, which take the SDK client through an async factory and
the provider id to report. `call()`, `chat()` and `stream_chat()` bind them to
the first-party `AsyncAnthropic` client; providers/vertex.py binds the same
functions to `AsyncAnthropicVertex`, so both serve identical request bodies.
"""

import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any

from anthropic import (
    AsyncAnthropic,
    BadRequestError,
    DefaultAsyncHttpxClient,
    NotFoundError,
)

from ..config import GatewayConfig
from ..errors import InvalidOutputError
from ..structured import OutputSchema
from ._anthropic_caching import system_param
from ._anthropic_translate import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    from_anthropic_response,
    text_blocks,
    thinking_tokens,
    to_anthropic_messages,
    to_anthropic_sampling,
    to_anthropic_structured_output_tool,
    to_anthropic_tool_choice,
    to_anthropic_tools,
    usage_from_anthropic,
    usage_from_anthropic_response,
)
from .base import (
    ChatResult,
    ProviderResult,
    StreamDelta,
    as_int,
    ensure_not_empty,
    guard_empty_stream,
    sdk_client_cache_key,
    sdk_client_options,
    stream_request_options,
)

PROVIDER = "anthropic"

logger = logging.getLogger(__name__)

# Models that return a 400 for any non-default `temperature`/`top_p`/`top_k`
# (docs: platform.claude.com/docs/en/models/haiku-5-5/migration-guide, checked
# 2026-10-12): the 5.x generation and Opus 4.7/4.8. For these the sampling
# keys are dropped before the request, so a caller written for older models
# keeps working instead of failing every call.
_REJECTS_SAMPLING = re.compile(
    r"^claude-(?:(?:haiku|sonnet|opus|fable|mythos)-5(?:-\d+)?|opus-4-[78])(?:$|[-@])"
)
# Models that take `output_config.effort` and, for Haiku, `thinking: disabled`.
_TAKES_EFFORT = re.compile(r"^claude-(?:haiku|sonnet|opus|fable|mythos)-5(?:-\d+)?(?:$|[-@])")
_CAN_DISABLE_THINKING = re.compile(r"^claude-haiku-5(?:-\d+)?(?:$|[-@])")
# "disabled" is a 400 at these effort levels on Haiku 5.5.
_EFFORTS_WITHOUT_DISABLED_THINKING = frozenset({"xhigh", "max"})

# Words in a 400 body that mean "this model does not accept this request",
# as opposed to a bad prompt (too long, malformed). Only these trigger the
# model fallback.
_MODEL_REJECTION_HINTS = (
    "model",
    "temperature",
    "top_p",
    "top_k",
    "thinking",
    "effort",
    "prefill",
    "output_config",
    "not supported",
    "unsupported",
)

# Returns the SDK client to send one request with: `AsyncAnthropic` or
# `AsyncAnthropicVertex`, which share the `messages` resource.
ClientFactory = Callable[[], Awaitable[Any]]

_clients: dict[tuple, AsyncAnthropic] = {}


def _client(config: GatewayConfig) -> AsyncAnthropic:
    key = sdk_client_cache_key(config.anthropic_api_key, config)
    client = _clients.get(key)
    if client is None:
        # anthropic>=1.0 only accepts httpx2 clients (a plain httpx.AsyncClient
        # raises TypeError). The SDK's own factory is httpx2-backed and keeps
        # its default connection limits/redirect handling.
        http_client = DefaultAsyncHttpxClient(verify=False) if not config.ssl_verify else None
        client = AsyncAnthropic(
            api_key=config.anthropic_api_key,
            http_client=http_client,
            **sdk_client_options(config),
        )
        _clients[key] = client
    return client


def _client_factory(config: GatewayConfig) -> ClientFactory:
    async def get() -> AsyncAnthropic:
        return _client(config)

    return get


def configured(config: GatewayConfig) -> bool:
    return bool(config.anthropic_api_key)


def default_model(config: GatewayConfig) -> str:
    return config.claude_model


# anthropic 1.0 removed these from messages.create()/messages.stream()'s
# named keyword arguments (passing them raises TypeError), but the Messages
# API still accepts them in the request body — so they travel via
# `extra_body`. `stop_sequences` is still a named argument.
_BODY_ONLY_SAMPLING_KEYS = ("temperature", "top_p")


def _sampling_kwargs(sampling: dict | None, model: str | None = None) -> dict:
    """SDK-call kwargs for translated sampling params. Shared by chat() and
    stream_chat(). `temperature`/`top_p` are dropped for a model that rejects
    them (see _REJECTS_SAMPLING); `stop_sequences` always passes."""
    params = to_anthropic_sampling(sampling)
    if model and _REJECTS_SAMPLING.match(model):
        params.pop("temperature", None)
        params.pop("top_p", None)
    body = {k: params.pop(k) for k in _BODY_ONLY_SAMPLING_KEYS if k in params}
    if body:
        params["extra_body"] = body
    return params


def _apply_thinking_effort(kwargs: dict, model: str, thinking: str, effort: str) -> None:
    """Add `output_config.effort` and `thinking` to the request for the models
    that take them; a no-op for any other model (so the same config is safe
    after a fallback to an older model). Haiku 5.5 thinks by default and bills
    thinking as output tokens: `thinking="disabled"` / a low effort keep that
    spend predictable. "disabled" is not sent together with effort xhigh/max
    (a 400 on Haiku 5.5)."""
    if not _TAKES_EFFORT.match(model):
        return
    if effort:
        kwargs["output_config"] = {**kwargs.get("output_config", {}), "effort": effort}
    body = None
    if thinking == "disabled":
        if _CAN_DISABLE_THINKING.match(model) and effort not in _EFFORTS_WITHOUT_DISABLED_THINKING:
            body = {"type": "disabled"}
    elif thinking == "adaptive":
        body = {"type": "adaptive"}
    if body:
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "thinking": body}


def is_model_rejection(exc: BaseException) -> bool:
    """True for a 404 (model not found) or a 400 whose message says the model
    does not accept the request (unknown model, unsupported parameter). Other
    400s, such as an over-long prompt, are not model problems."""
    if isinstance(exc, NotFoundError):
        return True
    if not isinstance(exc, BadRequestError):
        return False
    message = str(getattr(exc, "message", "") or exc).lower()
    return any(hint in message for hint in _MODEL_REJECTION_HINTS)


def _fallback_model(config: GatewayConfig, requested: str | None, exc: BaseException) -> str | None:
    """The model to retry once with, or None. Only for the configured default
    model (an explicit per-call model is the caller's choice). Logs one
    WARNING with the status and model names only, never request text."""
    fallback = config.claude_model_fallback
    if requested or not fallback or fallback == config.claude_model:
        return None
    if not is_model_rejection(exc):
        return None
    logger.warning(
        "anthropic model %s rejected the request (status %s); retrying once with %s",
        config.claude_model,
        getattr(exc, "status_code", "?"),
        fallback,
    )
    return fallback


def _resolve_tool_kwargs(
    tools: list[dict] | None,
    tool_choice: object,
    response_format: dict | None,
) -> tuple[dict, bool]:
    """Shared by chat() and stream_chat(). Returns (kwargs, emulating_structured_output)."""
    structured_tool = to_anthropic_structured_output_tool(response_format)
    if structured_tool:
        return (
            {
                "tools": [structured_tool],
                "tool_choice": {"type": "tool", "name": STRUCTURED_OUTPUT_TOOL_NAME},
            },
            True,
        )
    if tool_choice == "none":
        return {}, False  # omit tools entirely — the only way to hard-block tool use
    anthropic_tools = to_anthropic_tools(tools)
    if not anthropic_tools:
        return {}, False
    kwargs: dict = {"tools": anthropic_tools}
    choice = to_anthropic_tool_choice(tool_choice)
    if choice:
        kwargs["tool_choice"] = choice
    return kwargs, False


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
    async def run(chosen: str) -> ProviderResult:
        return await messages_call(
            _client_factory(config),
            provider=PROVIDER,
            model=chosen,
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
            output_schema=output_schema,
            cache_system=cache_system,
            thinking=config.anthropic_thinking,
            effort=config.anthropic_effort,
        )

    try:
        return await run(model or default_model(config))
    except (NotFoundError, BadRequestError) as exc:
        fallback = _fallback_model(config, model, exc)
        if fallback is None:
            raise
        return await run(fallback)


async def messages_call(
    get_client: ClientFactory,
    *,
    provider: str,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
    output_schema: OutputSchema | None,
    cache_system: bool,
    thinking: str = "",
    effort: str = "",
) -> ProviderResult:
    """Plain completion. The answer is every `text` block joined in order:
    a model that thinks returns `thinking`/`redacted_thinking` blocks before
    its text, so the first block is not necessarily the answer."""
    kwargs: dict = {}
    if output_schema is not None:
        # Native structured outputs (constrained decoding), which Anthropic's
        # docs recommend over forcing a tool call. Forced tool use is also
        # rejected together with manual extended thinking. The JSON arrives
        # as an ordinary text block.
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": output_schema.strict_schema(require_all_properties=False),
            }
        }
    _apply_thinking_effort(kwargs, model, thinking, effort)
    resp = await (await get_client()).messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_param(system, model, cache_system),
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    usage = usage_from_anthropic_response(resp.usage)
    text = "".join(text_blocks(resp.content)).strip()
    ensure_not_empty(
        text,
        provider=provider,
        model=model,
        finish_reason=resp.stop_reason,
        reasoning_tokens=usage.reasoning_tokens,
    )
    if output_schema is not None and resp.stop_reason == "refusal":
        raise InvalidOutputError("The model refused the request (stop_reason=refusal).")
    return ProviderResult(
        text=text,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        stop_reason=resp.stop_reason,
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
    async def run(chosen: str) -> ChatResult:
        return await messages_chat(
            _client_factory(config),
            provider=PROVIDER,
            model=chosen,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            sampling=sampling,
            response_format=response_format,
            thinking=config.anthropic_thinking,
            effort=config.anthropic_effort,
        )

    try:
        return await run(model or default_model(config))
    except (NotFoundError, BadRequestError) as exc:
        fallback = _fallback_model(config, model, exc)
        if fallback is None:
            raise
        return await run(fallback)


async def messages_chat(
    get_client: ClientFactory,
    *,
    provider: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    tool_choice: object,
    sampling: dict | None,
    response_format: dict | None,
    thinking: str = "",
    effort: str = "",
) -> ChatResult:
    system, anthropic_messages = to_anthropic_messages(messages)
    tool_kwargs, emulating_structured_output = _resolve_tool_kwargs(
        tools, tool_choice, response_format
    )
    kwargs: dict = {**_sampling_kwargs(sampling, model), **tool_kwargs}
    _apply_thinking_effort(kwargs, model, thinking, effort)

    resp = await (await get_client()).messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=anthropic_messages,
        **kwargs,
    )
    result = from_anthropic_response(resp, model)

    if emulating_structured_output and result.tool_calls:
        call = result.tool_calls[0]
        return ChatResult(
            content=json.dumps(call.arguments),
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            tool_calls=[],
            finish_reason="stop",
            cache_read_input_tokens=result.cache_read_input_tokens,
            cache_creation_input_tokens=result.cache_creation_input_tokens,
            reasoning_tokens=result.reasoning_tokens,
        )
    ensure_not_empty(
        result.content,
        has_tool_calls=bool(result.tool_calls),
        provider=provider,
        model=model,
        finish_reason=resp.stop_reason,
        reasoning_tokens=result.reasoning_tokens,
    )
    return result


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
    chosen = model or default_model(config)
    started = False
    while True:
        deltas = messages_stream(
            _client_factory(config),
            model=chosen,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            sampling=sampling,
            response_format=response_format,
            request_options=stream_request_options(config),
            thinking=config.anthropic_thinking,
            effort=config.anthropic_effort,
        )
        try:
            async with aclosing(deltas):
                async for delta in deltas:
                    started = True
                    yield delta
            return
        except (NotFoundError, BadRequestError) as exc:
            # Retry once, and only before the first chunk reached the caller.
            fallback = None if started else _fallback_model(config, model, exc)
            if fallback is None:
                raise
            chosen = fallback
            model = fallback  # now explicit: no second fallback


async def messages_stream(
    get_client: ClientFactory,
    *,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    tool_choice: object,
    sampling: dict | None,
    response_format: dict | None,
    request_options: dict,
    thinking: str = "",
    effort: str = "",
) -> AsyncIterator[StreamDelta]:
    """Translate Anthropic's raw SSE event stream into normalized
    StreamDelta chunks. Anthropic's tool-call streaming already arrives
    fragment-by-fragment (content_block_start announces id+name,
    subsequent input_json_delta events carry only argument fragments) —
    the same shape OpenAI's own streaming uses, so no re-chunking is needed,
    just a translation of event names/fields.
    """
    system, anthropic_messages = to_anthropic_messages(messages)
    tool_kwargs, emulating_structured_output = _resolve_tool_kwargs(
        tools, tool_choice, response_format
    )
    kwargs: dict = {**_sampling_kwargs(sampling, model), **tool_kwargs}
    _apply_thinking_effort(kwargs, model, thinking, effort)

    # Raw Anthropic usage fields. message_start carries the input side;
    # message_delta carries cumulative counts, where any field it includes
    # supersedes the earlier value.
    raw_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    reasoning_tokens: object = None
    stop_reason = "end_turn"
    structured_tool_index: int | None = None
    structured_output_parts: list[str] = []

    async with (await get_client()).messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=anthropic_messages,
        **kwargs,
        **request_options,
    ) as stream:
        async for event in stream:
            if event.type == "message_start":
                _update_usage(raw_usage, event.message.usage)

            elif event.type == "content_block_start":
                block = event.content_block
                if block.type != "tool_use":
                    continue
                if emulating_structured_output and block.name == STRUCTURED_OUTPUT_TOOL_NAME:
                    structured_tool_index = event.index
                    continue  # synthetic tool — never surfaced to the caller
                yield StreamDelta(
                    tool_call_deltas=[
                        {
                            "index": event.index,
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": ""},
                        }
                    ],
                    model=model,
                )

            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    yield StreamDelta(content=delta.text, model=model)
                elif delta.type == "input_json_delta":
                    if event.index == structured_tool_index:
                        structured_output_parts.append(delta.partial_json)
                        continue
                    yield StreamDelta(
                        tool_call_deltas=[
                            {"index": event.index, "function": {"arguments": delta.partial_json}}
                        ],
                        model=model,
                    )

            elif event.type == "message_delta":
                _update_usage(raw_usage, event.usage)
                reasoning_tokens = thinking_tokens(event.usage) or reasoning_tokens
                stop_reason = event.delta.stop_reason

    if structured_output_parts:
        yield StreamDelta(content="".join(structured_output_parts), model=model)

    finish_reason = (
        "tool_calls" if (stop_reason == "tool_use" and not emulating_structured_output) else "stop"
    )
    usage = usage_from_anthropic(**raw_usage, reasoning_tokens=reasoning_tokens)
    yield StreamDelta(
        finish_reason=finish_reason,
        model=model,
        usage=(usage.input_tokens, usage.output_tokens),
        usage_details=usage,
    )


def _update_usage(raw_usage: dict[str, int], sdk_usage: object) -> None:
    for name in raw_usage:
        value = getattr(sdk_usage, name, None)
        if as_int(value) or value == 0:
            raw_usage[name] = as_int(value)


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
