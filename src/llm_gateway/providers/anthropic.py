import json
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from ..config import GatewayConfig
from ._anthropic_translate import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    from_anthropic_response,
    to_anthropic_messages,
    to_anthropic_sampling,
    to_anthropic_structured_output_tool,
    to_anthropic_tool_choice,
    to_anthropic_tools,
)
from .base import ChatResult, ProviderResult, StreamDelta

_clients: dict[tuple[str, bool], AsyncAnthropic] = {}


def _client(config: GatewayConfig) -> AsyncAnthropic:
    key = (config.anthropic_api_key, config.ssl_verify)
    client = _clients.get(key)
    if client is None:
        import httpx

        http_client = httpx.AsyncClient(verify=False) if not config.ssl_verify else None
        client = AsyncAnthropic(api_key=config.anthropic_api_key, http_client=http_client)
        _clients[key] = client
    return client


def configured(config: GatewayConfig) -> bool:
    return bool(config.anthropic_api_key)


def default_model(config: GatewayConfig) -> str:
    return config.claude_model


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
) -> ProviderResult:
    model = model or default_model(config)
    resp = await _client(config).messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    input_tokens = resp.usage.input_tokens if resp.usage else 0
    output_tokens = resp.usage.output_tokens if resp.usage else 0
    return ProviderResult(
        text=resp.content[0].text.strip(),
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
    system, anthropic_messages = to_anthropic_messages(messages)
    tool_kwargs, emulating_structured_output = _resolve_tool_kwargs(
        tools, tool_choice, response_format
    )
    kwargs: dict = {**to_anthropic_sampling(sampling), **tool_kwargs}

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
    kwargs: dict = {**to_anthropic_sampling(sampling), **tool_kwargs}

    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"
    structured_tool_index: int | None = None
    structured_output_parts: list[str] = []

    async with _client(config).messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=anthropic_messages,
        **kwargs,
    ) as stream:
        async for event in stream:
            if event.type == "message_start":
                input_tokens = event.message.usage.input_tokens

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
                output_tokens = event.usage.output_tokens
                stop_reason = event.delta.stop_reason

    if structured_output_parts:
        yield StreamDelta(content="".join(structured_output_parts), model=model)

    finish_reason = (
        "tool_calls" if (stop_reason == "tool_use" and not emulating_structured_output) else "stop"
    )
    yield StreamDelta(finish_reason=finish_reason, model=model, usage=(input_tokens, output_tokens))
