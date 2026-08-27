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
Not handling: parallel-tool-call edge cases beyond what falls out
naturally, or every `tool_choice` variant.
"""

import json
from typing import Any

from .base import ChatResult, ToolCall


def _translate_image_url(url: str) -> dict:
    """OpenAI's `image_url.url` is either a data: URI or an https:// URL —
    Anthropic wants a `source` block naming which kind explicitly."""
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header.removeprefix("data:").split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _translate_content(content: str | list[dict] | None) -> str | list[dict]:
    """Plain string content passes through unchanged. OpenAI's multi-part
    content list (text + image_url parts, for multi-modal messages) is
    translated block-by-block — everything but image_url is already
    Anthropic-shaped (`{"type": "text", "text": ...}` is identical in both)."""
    if not isinstance(content, list):
        return content or ""
    blocks = []
    for part in content:
        if part.get("type") == "image_url":
            blocks.append(_translate_image_url(part["image_url"]["url"]))
        else:
            blocks.append(part)
    return blocks


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
                translated = _translate_content(m["content"])
                if isinstance(translated, list):
                    content.extend(translated)
                else:
                    content.append({"type": "text", "text": translated})
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

        anthropic_messages.append({"role": role, "content": _translate_content(m.get("content"))})

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


def to_anthropic_sampling(sampling: dict | None) -> dict:
    """Best-effort mapping of OpenAI-named sampling params. `seed`,
    `presence_penalty`, and `frequency_penalty` have no Anthropic equivalent
    and are silently dropped, not errored."""
    if not sampling:
        return {}
    out: dict = {}
    if sampling.get("temperature") is not None:
        out["temperature"] = sampling["temperature"]
    if sampling.get("top_p") is not None:
        out["top_p"] = sampling["top_p"]
    stop = sampling.get("stop")
    if stop:
        out["stop_sequences"] = stop if isinstance(stop, list) else [stop]
    return out


# Anthropic has no native `response_format` (json_schema/json_object) the way
# OpenAI does — the only way to get schema-constrained output is to force a
# single tool call and treat its `input` as the structured result. This name
# is purely an internal implementation detail; from_anthropic_response()
# never surfaces it as a real tool call to the caller (see anthropic.py's
# chat(), which unwraps it back into plain `content` before returning).
STRUCTURED_OUTPUT_TOOL_NAME = "__llm_gateway_structured_output"


def to_anthropic_structured_output_tool(response_format: dict | None) -> dict | None:
    if not response_format or response_format.get("type") != "json_schema":
        return None
    schema = response_format["json_schema"]["schema"]
    return {
        "name": STRUCTURED_OUTPUT_TOOL_NAME,
        "description": "Provide the response matching the required schema.",
        "input_schema": schema,
    }


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
