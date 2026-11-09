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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

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


def _dict(value: object) -> dict[str, Any] | None:
    """`value` as a JSON object, or None. Schemas and data are untrusted JSON,
    so every reader narrows with this instead of trusting a type hint."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _list(value: object) -> list[Any] | None:
    return cast("list[Any]", value) if isinstance(value, list) else None


def _is_object_schema(node: dict[str, Any]) -> bool:
    type_ = node.get("type")
    return (
        type_ == "object" or (isinstance(type_, list) and "object" in type_) or "properties" in node
    )


def _strict(node: Any, *, require_all: bool) -> Any:
    items = _list(node)
    if items is not None:
        return [_strict(item, require_all=require_all) for item in items]
    mapping = _dict(node)
    if mapping is None:
        return node
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in _UNSUPPORTED_KEYWORDS:
            continue
        if key == "minItems" and value not in (0, 1):
            continue
        if key == "format" and value not in _SUPPORTED_FORMATS:
            continue
        value_map = _dict(value)
        if key in _SCHEMA_MAP_KEYS and value_map is not None:
            out[key] = {
                name: _strict(sub, require_all=require_all) for name, sub in value_map.items()
            }
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


def to_strict_schema(schema: dict[str, Any], *, require_all_properties: bool) -> dict[str, Any]:
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
    json_schema: dict[str, Any]
    model: type[BaseModel] | None = None

    @classmethod
    def from_spec(
        cls, spec: dict[str, Any] | type[BaseModel], name: str | None = None
    ) -> "OutputSchema":
        if isinstance(spec, type) and issubclass(spec, BaseModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            return cls(_safe_name(name or spec.__name__), spec.model_json_schema(), spec)
        if isinstance(spec, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            return cls(_safe_name(name or str(spec.get("title") or "response")), spec)
        raise TypeError("output_schema must be a pydantic BaseModel subclass or a JSON Schema dict")

    def strict_schema(self, *, require_all_properties: bool) -> dict[str, Any]:
        return to_strict_schema(self.json_schema, require_all_properties=require_all_properties)

    def openai_response_format(self) -> dict[str, Any]:
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
        checked first, then the same JSON Schema subset `validate=` uses
        (type, enum, required, properties, constraints, ...; `pattern` and
        `format` are not checked). Pass a pydantic model for full validation."""
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
        found: list[str] = []
        _schema_errors(data, self.json_schema, self.json_schema, "", found)
        if found:
            unique = tuple(dict.fromkeys(found))
            raise InvalidOutputError(
                f"Structured output failed validation ({len(unique)} error(s) at: "
                f"{', '.join(unique)}).",
                unique,
            )
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
_JSON_TYPE_CHECKS: dict[str, Callable[[object], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def _resolve_ref(root: dict[str, Any], ref: str) -> Any:
    node: Any = root
    if not ref.startswith("#/"):
        return {}
    for part in ref[2:].split("/"):
        current = _dict(node)
        node = (
            current.get(part.replace("~1", "/").replace("~0", "~")) if current is not None else {}
        )
    return _dict(node) or {}


def _json_equal(a: Any, b: Any) -> bool:
    """JSON equality: unlike Python's `==`, `true` is not `1` and `false` is
    not `0`, at any depth."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    a_map, b_map = _dict(a), _dict(b)
    if a_map is not None and b_map is not None:
        return a_map.keys() == b_map.keys() and all(_json_equal(a_map[k], b_map[k]) for k in a_map)
    a_list, b_list = _list(a), _list(b)
    if a_list is not None and b_list is not None:
        return len(a_list) == len(b_list) and all(
            _json_equal(x, y) for x, y in zip(a_list, b_list, strict=True)
        )
    return a == b


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _constraint_errors(data: Any, schema: dict[str, Any], here: str, out: list[str]) -> None:
    """Size and range keywords: minimum/maximum (and exclusive), minLength/
    maxLength, minItems/maxItems/uniqueItems, minProperties/maxProperties.
    `pattern` and `format` are deliberately not checked (a regex from a
    schema run on model-written text is a ReDoS surface)."""

    def bound(key: str) -> Any:
        value = schema.get(key)
        return value if _is_number(value) else None

    def check(key: str, size: float, *, upper: bool) -> None:
        limit = bound(key)
        if limit is not None and (size > limit if upper else size < limit):
            out.append(here)

    if _is_number(data):
        check("minimum", data, upper=False)
        check("maximum", data, upper=True)
        low, high = bound("exclusiveMinimum"), bound("exclusiveMaximum")
        if (low is not None and data <= low) or (high is not None and data >= high):
            out.append(here)
    elif isinstance(data, str):
        check("minLength", len(data), upper=False)
        check("maxLength", len(data), upper=True)
    elif isinstance(data, list):
        items = cast("list[Any]", data)
        check("minItems", len(items), upper=False)
        check("maxItems", len(items), upper=True)
        if schema.get("uniqueItems") is True and any(
            _json_equal(a, b) for i, a in enumerate(items) for b in items[i + 1 :]
        ):
            out.append(here)
    elif isinstance(data, dict):
        check("minProperties", len(cast("dict[str, Any]", data)), upper=False)
        check("maxProperties", len(cast("dict[str, Any]", data)), upper=True)


def _schema_errors(
    data: Any, schema: Any, root: dict[str, Any], path: str, out: list[str], depth: int = 0
) -> None:
    """Append the paths where `data` violates `schema`. A subset of JSON
    Schema: type, enum, const, required, properties, additionalProperties
    (false or a schema), items, anyOf/oneOf (exactly one)/allOf/not, size and
    range limits (see `_constraint_errors`) and local $ref. Other keywords are
    ignored."""
    schema_map = _dict(schema)
    if len(out) >= _MAX_VALIDATION_ERRORS or schema_map is None or depth > 64:
        return
    schema = schema_map
    here = path or "<root>"
    if "$ref" in schema:
        _schema_errors(data, _resolve_ref(root, schema["$ref"]), root, path, out, depth + 1)
        return
    types = schema.get("type")
    if types is not None:
        names = cast("list[Any]", types) if isinstance(types, list) else [types]
        if not any(_JSON_TYPE_CHECKS.get(n, lambda _: True)(data) for n in names):
            out.append(here)
            return
    if "enum" in schema and not any(
        _json_equal(data, v) for v in cast("list[Any]", schema["enum"] or [])
    ):
        out.append(here)
    if "const" in schema and not _json_equal(data, schema["const"]):
        out.append(here)
    for key in ("anyOf", "oneOf"):
        branches = _list(schema.get(key))
        if branches:
            matches = 0
            for branch in branches:
                found: list[str] = []
                _schema_errors(data, branch, root, path, found, depth + 1)
                matches += not found
            if matches == 0 or (key == "oneOf" and matches > 1):
                out.append(here)
    for branch in _list(schema.get("allOf")) or []:
        _schema_errors(data, branch, root, path, out, depth + 1)
    if "not" in schema:
        rejected: list[str] = []
        _schema_errors(data, schema["not"], root, path, rejected, depth + 1)
        if not rejected:
            out.append(here)
    _constraint_errors(data, schema, here, out)
    data_map = _dict(data)
    data_list = _list(data)
    if data_map is not None:
        properties = cast("dict[str, Any]", schema.get("properties") or {})
        for name in cast("list[Any]", schema.get("required") or []):
            if name not in data_map:
                out.append(f"{path}.{name}" if path else name)
        for name, value in data_map.items():
            child = f"{path}.{name}" if path else name
            if name in properties:
                _schema_errors(value, properties[name], root, child, out, depth + 1)
            elif schema.get("additionalProperties") is False:
                out.append(child)
            elif isinstance(schema.get("additionalProperties"), dict):
                _schema_errors(value, schema["additionalProperties"], root, child, out, depth + 1)
    elif data_list is not None and isinstance(schema.get("items"), dict):
        for index, value in enumerate(data_list):
            child = f"{path}.{index}" if path else str(index)
            _schema_errors(value, schema["items"], root, child, out, depth + 1)


@dataclass(frozen=True)
class OutputValidator:
    """A caller's `validate=` spec: a pydantic model class (full validation)
    or a JSON Schema dict (the subset `_schema_errors` documents)."""

    spec: dict[str, Any] | type[BaseModel]

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | type[BaseModel]) -> "OutputValidator":
        if (isinstance(spec, type) and issubclass(spec, BaseModel)) or isinstance(spec, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
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
        assert isinstance(self.spec, dict)  # a model spec returned above
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
