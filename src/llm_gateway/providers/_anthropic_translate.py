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

from ..errors import InvalidOutputError, InvalidRequestError
from .base import ChatResult, ToolCall, Usage, as_int, normalize_finish_reason


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
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str) or not url:
                raise InvalidRequestError("An image_url content part has no url.")
            blocks.append(_translate_image_url(url))
        elif kind == "text":
            if str(part.get("text") or "").strip():  # Anthropic rejects empty text blocks
                blocks.append(part)
        else:
            raise InvalidRequestError(
                f"Content part type {kind!r} is not supported by Anthropic; "
                "only 'text' and 'image_url' parts are."
            )
    if not blocks:
        raise InvalidRequestError("A message has no non-empty text or image content.")
    return blocks


def _parse_tool_arguments(arguments: str | None) -> Any:
    """Parse a tool call's JSON arguments from the caller's message history.
    Hostile input (invalid JSON, or nesting deep enough to blow the parser's
    recursion limit) becomes a plain ValueError, never a RecursionError."""
    try:
        return json.loads(arguments or "{}")
    except (ValueError, RecursionError):
        raise InvalidRequestError("tool call arguments are not valid JSON") from None


def _system_text(content: str | list[dict]) -> str:
    """A system message's text: OpenAI also allows a list of text parts."""
    if isinstance(content, str):
        return content
    return "\n".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return (system_prompt, anthropic_messages)."""
    system_parts = [
        _system_text(m["content"]) for m in messages if m["role"] == "system" and m.get("content")
    ]
    system = "\n".join(part for part in system_parts if part)

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
                        "input": _parse_tool_arguments(tc["function"]["arguments"]),
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
    and are silently dropped, not errored. Returns Messages-API field names;
    providers/anthropic.py decides which of them go in `extra_body` (the
    1.x SDK no longer takes temperature/top_p as named arguments)."""
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


_JSON_OBJECT_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else: no prose, no code fences."
)


def json_object_messages(messages: list[dict], response_format: dict | None) -> list[dict]:
    """Anthropic has no JSON mode, so `response_format={"type": "json_object"}`
    would otherwise be dropped silently and prose could come back. Append a
    system instruction instead (last, so a cached system prefix is unchanged).
    Any other format returns `messages` itself."""
    if not response_format or response_format.get("type") != "json_object":
        return messages
    return [*messages, {"role": "system", "content": _JSON_OBJECT_INSTRUCTION}]


def to_anthropic_structured_output_tool(response_format: dict | None) -> dict | None:
    if not response_format or response_format.get("type") != "json_schema":
        return None
    schema = response_format["json_schema"]["schema"]
    return {
        "name": STRUCTURED_OUTPUT_TOOL_NAME,
        "description": "Provide the response matching the required schema.",
        "input_schema": schema,
    }


def usage_from_anthropic(
    input_tokens: object,
    output_tokens: object,
    cache_read_input_tokens: object = None,
    cache_creation_input_tokens: object = None,
    reasoning_tokens: object = None,
) -> Usage:
    """Normalized Usage from Anthropic's usage fields.

    Anthropic's `input_tokens` excludes cached tokens ("tokens after the last
    cache breakpoint"), so cache reads and writes are added back to make
    `Usage.input_tokens` the total prompt size, as it is for OpenAI."""
    cache_read = as_int(cache_read_input_tokens)
    cache_creation = as_int(cache_creation_input_tokens)
    return Usage(
        input_tokens=as_int(input_tokens) + cache_read + cache_creation,
        output_tokens=as_int(output_tokens),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        reasoning_tokens=as_int(reasoning_tokens),
    )


def usage_from_anthropic_response(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    return usage_from_anthropic(
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
        getattr(usage, "cache_read_input_tokens", None),
        getattr(usage, "cache_creation_input_tokens", None),
        thinking_tokens(usage),
    )


def thinking_tokens(usage: Any) -> object:
    """`output_tokens_details.thinking_tokens` from an Anthropic usage object,
    or None when absent."""
    return getattr(getattr(usage, "output_tokens_details", None), "thinking_tokens", None)


def text_blocks(content: Any) -> list[str]:
    """The text of every `text` block, in order. `thinking` and
    `redacted_thinking` blocks (which come first when the model thinks) and
    `tool_use` blocks carry no answer text and are skipped."""
    return [block.text for block in content if block.type == "text"]


def from_anthropic_response(resp: Any, model: str) -> ChatResult:
    tool_calls = [
        ToolCall(id=block.id, name=block.name, arguments=block.input)
        for block in resp.content
        if block.type == "tool_use"
    ]
    usage = usage_from_anthropic_response(resp.usage)
    stop_reason = getattr(resp, "stop_reason", None)
    if tool_calls and normalize_finish_reason(stop_reason) == "length":
        # The budget ran out inside a tool_use block: its input is incomplete.
        raise InvalidOutputError("The reply was cut off by max_tokens inside a tool call.")
    finish_reason = normalize_finish_reason(stop_reason)
    if tool_calls:
        finish_reason = "tool_calls"
    elif finish_reason == "tool_calls":
        finish_reason = "stop"
    return ChatResult(
        content="\n".join(text_blocks(resp.content)).strip() or None,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )
