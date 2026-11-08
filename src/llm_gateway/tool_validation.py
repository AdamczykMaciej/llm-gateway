"""Pre-flight validation of a `chat()` / `stream_chat()` request's tool
definitions, `tool_choice` and tool-call/tool-result pairing.

A malformed request would otherwise reach a provider, come back as a 400 and
be counted as that provider's failure (failing over to the next provider,
which rejects it too). Raising `InvalidRequestError` first gives the caller
one clear error and costs no network call. Messages name the tool or message
index only, never message content.

The schema check is structural, not a full JSON Schema validator (the gateway
has no such dependency): it catches what providers reject in practice.
"""

import dataclasses
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .errors import InvalidRequestError
from .providers._anthropic_translate import STRUCTURED_OUTPUT_TOOL_NAME
from .providers.base import ToolCall
from .structured import _schema_errors

_TOOL_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_JSON_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})
_CHOICES = frozenset({"auto", "none", "required"})
_MAX_DEPTH = 64
# OpenAI's documented ceiling on tools per request; the other OpenAI-style
# hosts are at or below it, so a larger set fails fast instead of as a 400
# that every provider in the chain would repeat.
MAX_TOOLS = 128
MAX_TOOLS_JSON_CHARS = 1_000_000


def _check_schema(schema: Any, where: str, depth: int = 0) -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise InvalidRequestError(f"{where} must be a JSON Schema object.")
    if depth > _MAX_DEPTH:
        raise InvalidRequestError(f"{where} is nested too deeply.")
    type_ = schema.get("type")
    if type_ is not None:
        types = type_ if isinstance(type_, list) else [type_]
        if not types or any(not isinstance(t, str) or t not in _JSON_TYPES for t in types):
            raise InvalidRequestError(f"{where}: 'type' must be one of {sorted(_JSON_TYPES)}.")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise InvalidRequestError(f"{where}: 'enum' must be a non-empty list.")
    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list) or not all(isinstance(r, str) for r in required)
    ):
        raise InvalidRequestError(f"{where}: 'required' must be a list of property names.")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise InvalidRequestError(f"{where}: 'properties' must be an object.")
        for name, sub in properties.items():
            _check_schema(sub, f"{where}.properties.{name}", depth + 1)
    items = schema.get("items")
    if items is not None:
        for sub in items if isinstance(items, list) else [items]:
            _check_schema(sub, f"{where}.items", depth + 1)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            branches = schema[key]
            if not isinstance(branches, list) or not branches:
                raise InvalidRequestError(f"{where}: '{key}' must be a non-empty list.")
            for sub in branches:
                _check_schema(sub, f"{where}.{key}", depth + 1)
    for key in ("$defs", "definitions"):
        if key in schema:
            if not isinstance(schema[key], dict):
                raise InvalidRequestError(f"{where}: '{key}' must be an object.")
            for name, sub in schema[key].items():
                _check_schema(sub, f"{where}.{key}.{name}", depth + 1)


def _tool_names(tools: list[dict] | None) -> set[str]:
    names: set[str] = set()
    if tools and len(tools) > MAX_TOOLS:
        raise InvalidRequestError(
            f"{len(tools)} tools were given; at most {MAX_TOOLS} are supported per request."
        )
    for index, tool in enumerate(tools or []):
        label = f"tools[{index}]"
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            raise InvalidRequestError(
                f"{label} must be {{'type': 'function', 'function': {{...}}}}."
            )
        if tool.get("type", "function") != "function":
            raise InvalidRequestError(f"{label}: only tools of type 'function' are supported.")
        name = function.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise InvalidRequestError(f"{label}: 'name' must match [a-zA-Z0-9_-]{{1,64}}.")
        if name == STRUCTURED_OUTPUT_TOOL_NAME:
            raise InvalidRequestError(f"{label}: '{name}' is reserved by the gateway.")
        if name in names:
            raise InvalidRequestError(f"{label}: duplicate tool name '{name}'.")
        names.add(name)
        parameters = function.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, dict) or parameters.get("type", "object") != "object":
                raise InvalidRequestError(
                    f"tool '{name}': 'parameters' must be a JSON Schema with type 'object'."
                )
            _check_schema(parameters, f"tool '{name}' parameters")
            try:
                json.dumps(parameters)
            except (TypeError, ValueError):
                raise InvalidRequestError(f"tool '{name}': 'parameters' is not JSON.") from None
    if tools:
        try:
            size = len(json.dumps(tools))
        except (TypeError, ValueError):
            raise InvalidRequestError("tools are not JSON.") from None
        if size > MAX_TOOLS_JSON_CHARS:
            raise InvalidRequestError(
                f"The tool definitions are {size} characters as JSON; "
                f"the limit is {MAX_TOOLS_JSON_CHARS}."
            )
    return names


def _check_tool_choice(tool_choice: object, names: set[str]) -> None:
    if tool_choice is None:
        return
    if isinstance(tool_choice, str):
        if tool_choice not in _CHOICES:
            raise InvalidRequestError("tool_choice must be 'auto', 'none', 'required' or a tool.")
        if tool_choice == "required" and not names:
            raise InvalidRequestError("tool_choice 'required' needs at least one tool.")
        return
    choice = tool_choice if isinstance(tool_choice, dict) else {}
    function = choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or choice.get("type", "function") != "function":
        raise InvalidRequestError(
            "tool_choice must be 'auto', 'none', 'required' or "
            "{'type': 'function', 'function': {'name': ...}}."
        )
    if name not in names:
        raise InvalidRequestError(f"tool_choice names tool '{name}', which is not in tools.")


def _check_pairing(messages: list[dict]) -> None:
    seen: set[str] = set()
    pending: dict[str, int] = {}

    def no_unanswered(index: int) -> None:
        if pending:
            ids = ", ".join(sorted(pending))
            raise InvalidRequestError(
                f"messages[{index}]: the assistant tool call(s) {ids} have no tool result "
                "before the next message."
            )

    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise InvalidRequestError(f"messages[{index}]: a tool message needs tool_call_id.")
            if call_id not in pending:
                raise InvalidRequestError(
                    f"messages[{index}]: tool result for tool_call_id '{call_id}' has no "
                    "matching unanswered tool call in a preceding assistant message."
                )
            del pending[call_id]
            continue
        no_unanswered(index)
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                call_id = call.get("id") if isinstance(call, dict) else None
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
                    raise InvalidRequestError(
                        f"messages[{index}]: each tool call needs an id and a function."
                    )
                if call_id in seen:
                    raise InvalidRequestError(
                        f"messages[{index}]: duplicate tool call id '{call_id}'."
                    )
                seen.add(call_id)
                pending[call_id] = index
    no_unanswered(len(messages))


def validate_chat_request(
    messages: list[dict], tools: list[dict] | None, tool_choice: object
) -> None:
    """Raise `InvalidRequestError` for a request no provider could accept."""
    names = _tool_names(tools)
    _check_tool_choice(tool_choice, names)
    _check_pairing(messages)


@dataclass(frozen=True)
class ToolCallIssue:
    """One problem with a model's tool call, safe to log and to feed back.
    `kind` is "unknown_tool" or "invalid_arguments"; `paths` are the argument
    names/paths that failed (never values)."""

    call_id: str
    name: str
    kind: str
    paths: tuple[str, ...] = ()
    message: str = ""


def check_tool_calls(
    tools: list[dict] | None, tool_calls: Sequence[ToolCall]
) -> list[ToolCallIssue]:
    """Check each of a model's `tool_calls` against the `tools` it was given:
    the tool must exist and its arguments must satisfy the tool's `parameters`
    schema (the subset `complete_with_usage(validate=...)` documents: type,
    enum, const, required, properties, additionalProperties=false, items,
    anyOf/oneOf, local $ref). Returns the issues, empty when all is well; never
    raises, so an agent loop can answer each bad call with
    `tool_error_message(issue)` and let the model correct itself."""
    schemas: dict[str, Any] = {}
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            schemas[function["name"]] = function.get("parameters") or {}
    issues: list[ToolCallIssue] = []
    for call in tool_calls:
        if call.name not in schemas:
            issues.append(
                ToolCallIssue(
                    call.id,
                    call.name,
                    "unknown_tool",
                    (),
                    f"There is no tool named '{call.name}'. "
                    f"Available: {', '.join(sorted(schemas))}.",
                )
            )
            continue
        schema = schemas[call.name]
        found: list[str] = []
        _schema_errors(call.arguments, schema, schema, "", found)
        if found:
            paths = tuple(dict.fromkeys(found))
            issues.append(
                ToolCallIssue(
                    call.id,
                    call.name,
                    "invalid_arguments",
                    paths,
                    f"The arguments for '{call.name}' do not match its schema at: "
                    f"{', '.join(paths)}.",
                )
            )
    return issues


def tool_error_message(issue: ToolCallIssue) -> dict:
    """A `role: "tool"` message answering the failed call with its problem, to
    append to the history so the model can retry with corrected arguments."""
    return {"role": "tool", "tool_call_id": issue.call_id, "content": f"Error: {issue.message}"}


def unique_tool_call_ids(tool_calls: list[ToolCall]) -> list[ToolCall]:
    """`tool_calls` with every id non-empty and unique within the turn: a host
    that repeats an id or leaves it empty (Gemini's OpenAI endpoint does) would
    otherwise make the tool results impossible to pair and be rejected on the
    next request. Unchanged ids and the original list are kept when all is
    well; a repaired id is `<id>__2`, `call_<n>`, ...  The caller answers with
    the ids it sees, so the round trip stays consistent."""
    seen: set[str] = set()
    fixed: list[ToolCall] = []
    changed = False
    for position, call in enumerate(tool_calls):
        new_id = call.id
        if not new_id or new_id in seen:
            base = new_id or "call"
            n = 1 if not new_id else 2
            new_id = f"{base}_{position}" if not new_id else f"{base}__{n}"
            while new_id in seen:
                n += 1
                new_id = f"{base}__{n}"
            changed = True
            call = dataclasses.replace(call, id=new_id)
        seen.add(new_id)
        fixed.append(call)
    return fixed if changed else tool_calls
