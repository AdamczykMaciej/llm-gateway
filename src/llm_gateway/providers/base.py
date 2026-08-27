"""Shared provider-call contracts.

Every provider module exposes:
- `async def call(config, system, prompt, max_tokens, model) -> ProviderResult`
  — plain text completion, used by `router.complete()`.
- `async def chat(config, messages, tools, max_tokens, model) -> ChatResult`
  — tool-calling-capable, used by `chat.chat()`. `messages` and `tools` are
  OpenAI-wire-shaped dicts; each provider translates to/from its own native
  format internally (Anthropic's tool format is nothing like OpenAI's —
  see providers/anthropic.py).

Providers own their own client caching internally.
"""

import json
from dataclasses import dataclass, field
from typing import Any

# OpenAI-named sampling params accepted on requests. OpenAI/Groq accept these
# verbatim (same param names); Anthropic needs translation — see
# _anthropic_translate.to_anthropic_sampling for which of these it actually
# supports (seed, presence_penalty, frequency_penalty have no equivalent
# there and are silently dropped, not errored).
OPENAI_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
)


def openai_sampling_kwargs(sampling: dict | None) -> dict:
    """Filter to only the set (non-None) sampling keys — safe to **-expand
    directly into an OpenAI-compatible SDK call (OpenAI, Groq)."""
    if not sampling:
        return {}
    return {k: v for k, v in sampling.items() if k in OPENAI_SAMPLING_KEYS and v is not None}


@dataclass(frozen=True)
class ProviderResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ChatResult:
    content: str | None
    model: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_calls"


@dataclass(frozen=True)
class StreamDelta:
    """One normalized chunk of a streamed response. `tool_call_deltas` are
    OpenAI-shaped fragments (`[{"index":, "id":, "type":"function",
    "function":{"name":, "arguments":}}]`) — only the first fragment for a
    given tool call carries id/name, matching OpenAI's own streaming
    semantics exactly (so client-side aggregation code written against real
    OpenAI streaming, e.g. LangChain's, works unmodified)."""

    content: str | None = None
    tool_call_deltas: list[dict] | None = None
    finish_reason: str | None = None
    model: str | None = None
    usage: tuple[int, int] | None = None  # (input_tokens, output_tokens) — final chunk only


def parse_openai_style_response(resp: Any, model: str) -> ChatResult:
    """Shared response parser for any provider speaking the OpenAI chat-
    completions wire format natively (OpenAI itself, Groq)."""
    choice = resp.choices[0]
    msg = choice.message
    tool_calls = [
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments or "{}"),
        )
        for tc in (msg.tool_calls or [])
    ]
    finish_reason = "tool_calls" if tool_calls else "stop"
    input_tokens = resp.usage.prompt_tokens if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0
    return ChatResult(
        content=msg.content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


def parse_openai_style_chunk(chunk: Any, model: str) -> StreamDelta:
    """Shared streaming-chunk parser for OpenAI/Groq — their chunks are
    already shaped almost exactly like our normalized StreamDelta."""
    if not chunk.choices:
        # The final chunk when stream_options={"include_usage": True} is set
        # carries only usage, no choices.
        usage = (chunk.usage.prompt_tokens, chunk.usage.completion_tokens) if chunk.usage else None
        return StreamDelta(model=model, usage=usage)

    choice = chunk.choices[0]
    delta = choice.delta
    tool_call_deltas = None
    if delta.tool_calls:
        tool_call_deltas = [
            {
                "index": tc.index,
                **({"id": tc.id, "type": "function"} if tc.id else {}),
                "function": {
                    **({"name": tc.function.name} if tc.function and tc.function.name else {}),
                    "arguments": (tc.function.arguments or "") if tc.function else "",
                },
            }
            for tc in delta.tool_calls
        ]
    return StreamDelta(
        content=delta.content,
        tool_call_deltas=tool_call_deltas,
        finish_reason=choice.finish_reason,
        model=model,
    )
