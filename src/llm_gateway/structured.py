"""Structured output for `complete_with_usage(output_schema=...)`.

The caller passes a pydantic model class or a JSON Schema dict. Each provider
turns it into its native mechanism (see `call()` in providers/*.py):

- Anthropic: `output_config.format = {"type": "json_schema", "schema": ...}`
  (structured outputs: generally available, constrained decoding, the JSON
  arrives as a text block).
- OpenAI: `response_format = {"type": "json_schema", "json_schema":
  {"name": ..., "strict": true, "schema": ...}}`.
- Groq: the same `json_schema` shape on the models Groq documents as
  supporting strict mode, otherwise JSON mode (`json_object`) with the schema
  in the system prompt.

Whatever comes back is parsed and validated here, inside the provider
attempt, so an unusable reply raises `InvalidOutputError` and the engine fails
over to the next provider (errors.py explains why that never trips the
circuit breaker).
"""

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .errors import InvalidOutputError

# Keywords that Anthropic structured outputs and OpenAI strict mode reject.
# They are removed only from the schema *sent to the provider*: a pydantic
# model still enforces them when the reply is validated, so a violation fails
# over like any other invalid output.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "default",
        "examples",
    }
)
# String formats Anthropic structured outputs accept.
_SUPPORTED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)
# Keys whose values are data, not sub-schemas.
_LITERAL_KEYS = frozenset({"enum", "const", "required", "type", "description", "title", "$ref"})
# Keys whose values map names to sub-schemas.
_SCHEMA_MAP_KEYS = frozenset({"properties", "$defs", "definitions"})


def _is_object_schema(node: dict) -> bool:
    type_ = node.get("type")
    return (
        type_ == "object" or (isinstance(type_, list) and "object" in type_) or "properties" in node
    )


def _strict(node: Any, *, require_all: bool) -> Any:
    if isinstance(node, list):
        return [_strict(item, require_all=require_all) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_KEYWORDS:
            continue
        if key == "minItems" and value not in (0, 1):
            continue
        if key == "format" and value not in _SUPPORTED_FORMATS:
            continue
        if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            out[key] = {name: _strict(sub, require_all=require_all) for name, sub in value.items()}
        elif key == "oneOf":
            out["anyOf"] = _strict(value, require_all=require_all)
        elif key in _LITERAL_KEYS:
            out[key] = value
        else:
            out[key] = _strict(value, require_all=require_all)
    if _is_object_schema(out):
        out["additionalProperties"] = False
        if require_all:
            out["required"] = list(out.get("properties", {}))
    return out


def to_strict_schema(schema: dict, *, require_all_properties: bool) -> dict:
    """Return a copy of `schema` that strict/constrained-decoding modes
    accept: every object gets `additionalProperties: false`, unsupported
    keywords are dropped, and `oneOf` becomes `anyOf`.

    `require_all_properties` also marks every property required, as OpenAI
    strict mode demands. Pydantic fields with defaults then always come back
    explicitly, which still validates."""
    return _strict(copy.deepcopy(schema), require_all=require_all_properties)


_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def _unfence(text: str) -> str:
    """JSON mode without constrained decoding sometimes wraps the object in a
    markdown code fence. Strip one if it is the whole reply."""
    match = _FENCE.match(text)
    return match.group(1) if match else text


def _safe_name(name: str) -> str:
    # OpenAI requires ^[a-zA-Z0-9_-]{1,64}$ for json_schema.name.
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64] or "response"


@dataclass(frozen=True)
class OutputSchema:
    """A caller's schema, normalized once per call."""

    name: str
    json_schema: dict
    model: type[BaseModel] | None = None

    @classmethod
    def from_spec(cls, spec: dict | type[BaseModel], name: str | None = None) -> "OutputSchema":
        if isinstance(spec, type) and issubclass(spec, BaseModel):
            return cls(_safe_name(name or spec.__name__), spec.model_json_schema(), spec)
        if isinstance(spec, dict):
            return cls(_safe_name(name or str(spec.get("title") or "response")), spec)
        raise TypeError("output_schema must be a pydantic BaseModel subclass or a JSON Schema dict")

    def strict_schema(self, *, require_all_properties: bool) -> dict:
        return to_strict_schema(self.json_schema, require_all_properties=require_all_properties)

    def openai_response_format(self) -> dict:
        """OpenAI-wire `response_format` for strict JSON-schema output."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "strict": True,
                "schema": self.strict_schema(require_all_properties=True),
            },
        }

    def instructions(self) -> str:
        """Appended to the system prompt where the provider has no schema
        parameter (Groq JSON mode)."""
        return (
            "Respond with only a single JSON object, with no prose and no code fence, that "
            "conforms to this JSON Schema:\n" + json.dumps(self.json_schema, separators=(",", ":"))
        )

    def parse(self, text: str) -> Any:
        """Parse and validate a reply, raising `InvalidOutputError`.

        With a pydantic model the reply is fully validated, including the
        constraints `to_strict_schema` removed from the request. With a plain
        dict schema only the top-level type and the `required` keys are
        checked: the gateway deliberately has no JSON Schema validator
        dependency, so pass a pydantic model for full validation."""
        try:
            data = json.loads(_unfence(text))
        except json.JSONDecodeError as e:
            raise InvalidOutputError(
                f"Structured output is not valid JSON ({e.msg} at position {e.pos})."
            ) from None
        if self.model is not None:
            try:
                return self.model.model_validate(data)
            except ValidationError as e:
                locations = ", ".join(
                    ".".join(map(str, err["loc"])) or "<root>" for err in e.errors()
                )
                raise InvalidOutputError(
                    f"Structured output failed validation against {self.model.__name__} "
                    f"({e.error_count()} error(s) at: {locations})."
                ) from None
        self._check_shallow(data)
        return data

    def _check_shallow(self, data: Any) -> None:
        type_ = self.json_schema.get("type")
        if type_ == "object":
            if not isinstance(data, dict):
                raise InvalidOutputError("Structured output is not a JSON object.")
            missing = [key for key in self.json_schema.get("required", []) if key not in data]
            if missing:
                raise InvalidOutputError(
                    f"Structured output is missing required key(s): {', '.join(missing)}."
                )
        elif type_ == "array" and not isinstance(data, list):
            raise InvalidOutputError("Structured output is not a JSON array.")
