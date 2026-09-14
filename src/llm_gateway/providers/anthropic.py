import json
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from ..config import GatewayConfig
from ..errors import InvalidOutputError
from ..structured import OutputSchema
from ._anthropic_caching import system_param
from ._anthropic_translate import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    from_anthropic_response,
    text_blocks,
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
    sdk_client_cache_key,
    sdk_client_options,
    stream_request_options,
)

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


def configured(config: GatewayConfig) -> bool:
    return bool(config.anthropic_api_key)


def default_model(config: GatewayConfig) -> str:
    return config.claude_model


# anthropic 1.0 removed these from messages.create()/messages.stream()'s
# named keyword arguments (passing them raises TypeError), but the Messages
# API still accepts them in the request body — so they travel via
# `extra_body`. `stop_sequences` is still a named argument.
_BODY_ONLY_SAMPLING_KEYS = ("temperature", "top_p")


def _sampling_kwargs(sampling: dict | None) -> dict:
    """SDK-call kwargs for translated sampling params. Shared by chat() and
    stream_chat()."""
    params = to_anthropic_sampling(sampling)
    body = {k: params.pop(k) for k in _BODY_ONLY_SAMPLING_KEYS if k in params}
    if body:
        params["extra_body"] = body
    return params


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
    """Plain completion. The answer is every `text` block joined in order:
    a model that thinks returns `thinking`/`redacted_thinking` blocks before
    its text, so the first block is not necessarily the answer."""
    model = model or default_model(config)
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
    resp = await _client(config).messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_param(system, model, cache_system),
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    texts = text_blocks(resp.content)
    if not texts:
        raise InvalidOutputError(
            f"Anthropic returned no text content (stop_reason={resp.stop_reason})."
        )
    if output_schema is not None and resp.stop_reason == "refusal":
        raise InvalidOutputError("The model refused the request (stop_reason=refusal).")
    usage = usage_from_anthropic_response(resp.usage)
    return ProviderResult(
        text="".join(texts).strip(),
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
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
    model = model or default_model(config)
    system, anthropic_messages = to_anthropic_messages(messages)
    tool_kwargs, emulating_structured_output = _resolve_tool_kwargs(
        tools, tool_choice, response_format
    )
    kwargs: dict = {**_sampling_kwargs(sampling), **tool_kwargs}

    resp = await _client(config).messages.create(
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
        )
    return result


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
    """Translate Anthropic's raw SSE event stream into normalized
    StreamDelta chunks. Anthropic's tool-call streaming already arrives
    fragment-by-fragment (content_block_start announces id+name,
    subsequent input_json_delta events carry only argument fragments) —
    the same shape OpenAI's own streaming uses, so no re-chunking is needed,
    just a translation of event names/fields.
    """
    model = model or default_model(config)
    system, anthropic_messages = to_anthropic_messages(messages)
    tool_kwargs, emulating_structured_output = _resolve_tool_kwargs(
        tools, tool_choice, response_format
    )
    kwargs: dict = {**_sampling_kwargs(sampling), **tool_kwargs}

    # Raw Anthropic usage fields. message_start carries the input side;
    # message_delta carries cumulative counts, where any field it includes
    # supersedes the earlier value.
    raw_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    stop_reason = "end_turn"
    structured_tool_index: int | None = None
    structured_output_parts: list[str] = []

    async with _client(config).messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=anthropic_messages,
        **kwargs,
        **stream_request_options(config),
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
                stop_reason = event.delta.stop_reason

    if structured_output_parts:
        yield StreamDelta(content="".join(structured_output_parts), model=model)

    finish_reason = (
        "tool_calls" if (stop_reason == "tool_use" and not emulating_structured_output) else "stop"
    )
    usage = usage_from_anthropic(**raw_usage)
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
