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


_ANY_FENCE = re.compile(r"```(?:json)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_MAX_EMBEDDED_CANDIDATES = 5


def _loads_lenient(text: str) -> Any:
    """`json.loads` of the reply, tolerating what JSON mode without
    constrained decoding adds around the value: a code fence, a sentence
    before it, or a closing remark after it. The strict parse always runs
    first; only when it fails is the first fenced block, then the first JSON
    object/array embedded in the text, tried. Raises the strict parse's
    `JSONDecodeError` when nothing usable is found."""
    try:
        return json.loads(_unfence(text))
    except json.JSONDecodeError as error:
        strict_error = error
    for fenced in _ANY_FENCE.findall(text)[:1]:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"[{\[]", text)][:_MAX_EMBEDDED_CANDIDATES]
    for start in starts:
        try:
            return decoder.raw_decode(text, start)[0]
        except json.JSONDecodeError:
            continue
    raise strict_error


def _pydantic_paths(error: ValidationError) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for err in error.errors():
        seen[".".join(map(str, err["loc"])) or "<root>"] = None
    return tuple(seen)


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
            data = _loads_lenient(text)
        except json.JSONDecodeError as e:
            raise InvalidOutputError(
                f"Structured output is not valid JSON ({e.msg} at position {e.pos})."
            ) from None
        except RecursionError:
            # Hostile model output nested past the parser's recursion limit.
            raise InvalidOutputError("Structured output is nested too deeply.") from None
        if self.model is not None:
            try:
                return self.model.model_validate(data)
            except ValidationError as e:
                paths = _pydantic_paths(e)
                raise InvalidOutputError(
                    f"Structured output failed validation against {self.model.__name__} "
                    f"({e.error_count()} error(s) at: {', '.join(paths)}).",
                    paths,
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


# ── Opt-in output validation: complete_with_usage(validate=...) ─────────────

_MAX_VALIDATION_ERRORS = 20
_JSON_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def _resolve_ref(root: dict, ref: str) -> Any:
    node: Any = root
    if not ref.startswith("#/"):
        return {}
    for part in ref[2:].split("/"):
        node = (
            node.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(node, dict) else {}
        )
    return node if isinstance(node, dict) else {}


def _json_equal(a: Any, b: Any) -> bool:
    """JSON equality: unlike Python's `==`, `true` is not `1` and `false` is
    not `0`, at any depth."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _schema_errors(
    data: Any, schema: Any, root: dict, path: str, out: list[str], depth: int = 0
) -> None:
    """Append the paths where `data` violates `schema`. A subset of JSON
    Schema: type, enum, const, required, properties, additionalProperties
    (false), items, anyOf/oneOf and local $ref. Other keywords are ignored."""
    if len(out) >= _MAX_VALIDATION_ERRORS or not isinstance(schema, dict) or depth > 64:
        return
    here = path or "<root>"
    if "$ref" in schema:
        _schema_errors(data, _resolve_ref(root, schema["$ref"]), root, path, out, depth + 1)
        return
    types = schema.get("type")
    if types is not None:
        names = types if isinstance(types, list) else [types]
        if not any(_JSON_TYPE_CHECKS.get(n, lambda v: True)(data) for n in names):
            out.append(here)
            return
    if "enum" in schema and not any(_json_equal(data, v) for v in schema["enum"] or []):
        out.append(here)
    if "const" in schema and not _json_equal(data, schema["const"]):
        out.append(here)
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches:
            attempts = []
            for branch in branches:
                found: list[str] = []
                _schema_errors(data, branch, root, path, found, depth + 1)
                attempts.append(found)
            if not any(not found for found in attempts):
                out.append(here)
    if isinstance(data, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in data:
                out.append(f"{path}.{name}" if path else name)
        for name, value in data.items():
            child = f"{path}.{name}" if path else name
            if name in properties:
                _schema_errors(value, properties[name], root, child, out, depth + 1)
            elif schema.get("additionalProperties") is False:
                out.append(child)
    elif isinstance(data, list) and isinstance(schema.get("items"), dict):
        for index, value in enumerate(data):
            child = f"{path}.{index}" if path else str(index)
            _schema_errors(value, schema["items"], root, child, out, depth + 1)


@dataclass(frozen=True)
class OutputValidator:
    """A caller's `validate=` spec: a pydantic model class (full validation)
    or a JSON Schema dict (the subset `_schema_errors` documents)."""

    spec: dict | type[BaseModel]

    @classmethod
    def from_spec(cls, spec: dict | type[BaseModel]) -> "OutputValidator":
        if (isinstance(spec, type) and issubclass(spec, BaseModel)) or isinstance(spec, dict):
            return cls(spec)
        raise TypeError("validate must be a pydantic BaseModel subclass or a JSON Schema dict")

    def check(self, text: str) -> Any:
        """The typed result (model instance, or parsed JSON for a dict
        schema), else `InvalidOutputError` carrying the failing paths."""
        schema = OutputSchema.from_spec(self.spec)
        if schema.model is not None:
            return schema.parse(text)
        try:
            data = _loads_lenient(text)
        except json.JSONDecodeError as e:
            raise InvalidOutputError(
                f"Structured output is not valid JSON ({e.msg} at position {e.pos}).", ("<root>",)
            ) from None
        except RecursionError:
            raise InvalidOutputError(
                "Structured output is nested too deeply.", ("<root>",)
            ) from None
        paths: list[str] = []
        _schema_errors(data, self.spec, self.spec, "", paths)
        if paths:
            unique = tuple(dict.fromkeys(paths))
            raise InvalidOutputError(
                f"Structured output failed validation ({len(unique)} error(s) at: "
                f"{', '.join(unique)}).",
                unique,
            )
        return data
