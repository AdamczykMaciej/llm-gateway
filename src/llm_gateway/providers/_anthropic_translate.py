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
import re
from typing import Any, cast

from ..errors import InvalidRequestError, ToolCallTruncatedError
from ._wire import (
    AnthropicBlock,
    AnthropicImageBlock,
    AnthropicMessage,
    AnthropicTextBlock,
    AnthropicTool,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    as_dict,
)
from .base import ChatResult, ToolCall, Usage, as_int, normalize_finish_reason


def _translate_image_url(url: str) -> AnthropicImageBlock:
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


def _translate_content(content: str | list[dict[str, Any]] | None) -> str | list[AnthropicBlock]:
    """Plain string content passes through unchanged. OpenAI's multi-part
    content list (text + image_url parts, for multi-modal messages) is
    translated block-by-block — everything but image_url is already
    Anthropic-shaped (`{"type": "text", "text": ...}` is identical in both)."""
    if not isinstance(content, list):
        return content or ""
    blocks: list[AnthropicBlock] = []
    for part in content:
        # Caller data: a non-dict part is rejected below, not trusted by its type hint.
        kind = (as_dict(part) or {}).get("type")
        if kind == "image_url":
            image = part.get("image_url")
            image_dict = as_dict(image)
            url = image_dict.get("url") if image_dict is not None else image
            if not isinstance(url, str) or not url:
                raise InvalidRequestError("An image_url content part has no url.")
            blocks.append(_translate_image_url(url))
        elif kind == "text":
            text = str(part.get("text") or "")
            if text.strip():  # Anthropic rejects empty text blocks
                blocks.append(cast("AnthropicTextBlock", part))
        else:
            raise InvalidRequestError(
                f"Content part type {kind!r} is not supported by Anthropic; "
                "only 'text' and 'image_url' parts are."
            )
    if not blocks:
        raise InvalidRequestError("A message has no non-empty text or image content.")
    return blocks


def _parse_tool_arguments(arguments: str | None) -> dict[str, Any]:
    """Parse a tool call's JSON arguments from the caller's message history.
    Hostile input (invalid JSON, or nesting deep enough to blow the parser's
    recursion limit) becomes a plain ValueError, never a RecursionError."""
    try:
        parsed = json.loads(arguments or "{}")
    except (ValueError, RecursionError):
        raise InvalidRequestError("tool call arguments are not valid JSON") from None
    if not isinstance(parsed, dict):
        raise InvalidRequestError("tool call arguments are not a JSON object")
    return cast("dict[str, Any]", parsed)


def _tool_result_content(content: Any) -> str | list[AnthropicBlock]:
    """A tool message's content for a `tool_result` block: text stays text, a
    list of OpenAI parts becomes Anthropic text/image blocks (a result may
    carry an image, e.g. a screenshot)."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    return _translate_content(cast("list[dict[str, Any]]", content))


def _system_text(content: str | list[dict[str, Any]]) -> str:
    """A system message's text: OpenAI also allows a list of text parts."""
    if isinstance(content, str):
        return content
    parts = (as_dict(part) for part in content)
    return "\n".join(str(p.get("text", "")) for p in parts if p and p.get("type") == "text")


_ROLES = ("system", "user", "assistant", "tool")


def _check_shape(messages: list[dict[str, Any]]) -> None:
    """Reject structurally broken history (caller data) with a typed error
    naming the message index only, never a KeyError/TypeError from deep inside
    the translation."""
    for index, raw in enumerate(messages):
        m = as_dict(raw)
        problem = None
        if m is None or m.get("role") not in _ROLES:
            problem = "has no valid role"
        elif m["role"] == "tool" and not isinstance(m.get("tool_call_id"), str):
            problem = "is a tool message without a tool_call_id"
        elif (
            m["role"] == "system" and m.get("content") and not isinstance(m["content"], (str, list))
        ):
            problem = "has system content that is neither text nor a list of parts"
        elif m["role"] == "assistant" and m.get("tool_calls"):
            calls = m["tool_calls"]
            if not isinstance(calls, list) or not all(
                (call := as_dict(c)) is not None
                and isinstance(call.get("id"), str)
                and (fn := as_dict(call.get("function"))) is not None
                and isinstance(fn.get("name"), str)
                and isinstance(fn.get("arguments", ""), (str, type(None)))
                for c in cast("list[Any]", calls)
            ):
                problem = "has a malformed tool call"
        if problem:
            raise InvalidRequestError(f"messages[{index}] {problem}.")


_VALID_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_INVALID_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _anthropic_tool_ids(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Old id -> id Anthropic accepts, for the ids in `messages` that it would
    reject. Anthropic wants `[a-zA-Z0-9_-]{1,64}`, but OpenAI-compatible hosts
    mint ids like `functions.add:0`, which a failover to Claude would otherwise
    turn into a 400. The mapping is applied to tool calls and their results
    alike and never maps two ids to one."""
    ids: list[str] = []
    for m in messages:
        if m.get("role") == "tool":
            ids.append(m["tool_call_id"])
        calls: list[dict[str, Any]] = m.get("tool_calls") or []
        ids.extend(tc["id"] for tc in calls)
    used = {i for i in ids if _VALID_ID.fullmatch(i)}
    mapping: dict[str, str] = {}
    for old in dict.fromkeys(ids):
        if old in used:
            continue
        base = _INVALID_ID_CHARS.sub("_", old)[:56] or "call"
        new, n = base, 1
        while new in used:
            n += 1
            new = f"{base}_{n}"
        used.add(new)
        mapping[old] = new
    return mapping


def _mapped(ids: dict[str, str], tool_id: str) -> str:
    return ids.get(tool_id, tool_id)


def to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[AnthropicMessage]]:
    """Return (system_prompt, anthropic_messages)."""
    _check_shape(messages)
    ids = _anthropic_tool_ids(messages)
    system_parts = [
        _system_text(m["content"]) for m in messages if m["role"] == "system" and m.get("content")
    ]
    system = "\n".join(part for part in system_parts if part)

    anthropic_messages: list[AnthropicMessage] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue

        if role == "tool":
            block: AnthropicToolResultBlock = {
                "type": "tool_result",
                "tool_use_id": _mapped(ids, m["tool_call_id"]),
                "content": _tool_result_content(m.get("content")),
            }
            # Consecutive tool results belong in one user turn, mirroring how
            # an assistant message can request several tool calls at once.
            previous = anthropic_messages[-1] if anthropic_messages else None
            previous_content = previous["content"] if previous else None
            if (
                previous is not None
                and previous["role"] == "user"
                and isinstance(previous_content, list)
                and all(b["type"] == "tool_result" for b in previous_content)
            ):
                previous_content.append(block)
            else:
                anthropic_messages.append({"role": "user", "content": [block]})
            continue

        if role == "assistant" and m.get("tool_calls"):
            content: list[AnthropicBlock] = []
            if m.get("content"):
                translated = _translate_content(m["content"])
                if isinstance(translated, list):
                    content.extend(translated)
                else:
                    content.append({"type": "text", "text": translated})
            for tc in m["tool_calls"]:
                tool_use: AnthropicToolUseBlock = {
                    "type": "tool_use",
                    "id": _mapped(ids, tc["id"]),
                    "name": tc["function"]["name"],
                    "input": _parse_tool_arguments(tc["function"].get("arguments")),
                }
                content.append(tool_use)
            anthropic_messages.append({"role": "assistant", "content": content})
            continue

        anthropic_messages.append({"role": role, "content": _translate_content(m.get("content"))})

    return system, anthropic_messages


def to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[AnthropicTool] | None:
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


def to_anthropic_tool_choice(tool_choice: Any) -> dict[str, Any] | None:
    """Best-effort mapping — Anthropic has no direct equivalent of OpenAI's
    "none", so that case is left unmapped (falls back to the model's own
    default rather than a hard block)."""
    if tool_choice in (None, "none"):
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    choice = as_dict(tool_choice)
    if choice is not None and choice.get("type") == "function":
        return {"type": "tool", "name": choice["function"]["name"]}
    return None


def to_anthropic_sampling(sampling: dict[str, Any] | None) -> dict[str, Any]:
    """Best-effort mapping of OpenAI-named sampling params. `seed`,
    `presence_penalty`, and `frequency_penalty` have no Anthropic equivalent
    and are silently dropped, not errored. Returns Messages-API field names;
    providers/anthropic.py decides which of them go in `extra_body` (the
    1.x SDK no longer takes temperature/top_p as named arguments)."""
    if not sampling:
        return {}
    out: dict[str, Any] = {}
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


def json_object_messages(
    messages: list[dict[str, Any]], response_format: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Anthropic has no JSON mode, so `response_format={"type": "json_object"}`
    would otherwise be dropped silently and prose could come back. Append a
    system instruction instead (last, so a cached system prefix is unchanged).
    Any other format returns `messages` itself."""
    if not response_format or response_format.get("type") != "json_object":
        return messages
    return [*messages, {"role": "system", "content": _JSON_OBJECT_INSTRUCTION}]


def to_anthropic_structured_output_tool(
    response_format: dict[str, Any] | None,
) -> dict[str, Any] | None:
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
        raise ToolCallTruncatedError(
            "The reply was cut off by max_tokens inside a tool call; "
            "retry with a larger max_tokens.",
            tuple(call.name for call in tool_calls),
        )
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
