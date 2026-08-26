"""Translate between the OpenAI wire format (what the gateway's HTTP API and
every other provider speak) and Anthropic's native Messages API tool format,
which is structurally different in every direction:

- System messages are a top-level `system` param, not a message role.
- A tool call is a `tool_use` content block on an assistant message, with
  `input` already a parsed dict (not a JSON string like OpenAI's
  `function.arguments`).
- A tool result is a `tool_result` content block *inside a user message*,
  not its own `role: "tool"` message.
- Tool definitions use `input_schema`, not `function.parameters`.

Scope: single-round-trip tool calling — enough for a ReAct-style agent loop.
Not handling: image/multi-modal content blocks, parallel-tool-call edge
cases beyond what falls out naturally, or every `tool_choice` variant.
"""

import json
from typing import Any

from .base import ChatResult, ToolCall


def to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return (system_prompt, anthropic_messages)."""
    system_parts = [m["content"] for m in messages if m["role"] == "system" and m.get("content")]
    system = "\n".join(system_parts)

    anthropic_messages: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue

        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m.get("content") or "",
            }
            # Consecutive tool results belong in one user turn, mirroring how
            # an assistant message can request several tool calls at once.
            if (
                anthropic_messages
                and anthropic_messages[-1]["role"] == "user"
                and all(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in anthropic_messages[-1]["content"]
                )
            ):
                anthropic_messages[-1]["content"].append(block)
            else:
                anthropic_messages.append({"role": "user", "content": [block]})
            continue

        if role == "assistant" and m.get("tool_calls"):
            content: list[dict] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"] or "{}"),
                    }
                )
            anthropic_messages.append({"role": "assistant", "content": content})
            continue

        anthropic_messages.append({"role": role, "content": m.get("content") or ""})

    return system, anthropic_messages


def to_anthropic_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


def to_anthropic_tool_choice(tool_choice: Any) -> dict | None:
    """Best-effort mapping — Anthropic has no direct equivalent of OpenAI's
    "none", so that case is left unmapped (falls back to the model's own
    default rather than a hard block)."""
    if tool_choice in (None, "none"):
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "tool", "name": tool_choice["function"]["name"]}
    return None


def from_anthropic_response(resp: Any, model: str) -> ChatResult:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

    input_tokens = resp.usage.input_tokens if resp.usage else 0
    output_tokens = resp.usage.output_tokens if resp.usage else 0
    return ChatResult(
        content="\n".join(text_parts).strip() or None,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
    )
