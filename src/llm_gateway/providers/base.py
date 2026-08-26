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
