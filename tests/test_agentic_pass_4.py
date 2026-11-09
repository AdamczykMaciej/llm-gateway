"""Agentic pass 4 (0.21.0): constraint keywords in tool-argument and output
validation, duplicate JSON keys / NaN in tool arguments, full validation of a
dict `output_schema`. Fakes only; no provider is called."""

from types import SimpleNamespace

import pytest

from llm_gateway import InvalidOutputError, ToolCall
from llm_gateway.providers.base import parse_openai_style_response
from llm_gateway.structured import OutputSchema
from llm_gateway.tool_validation import check_tool_calls


def _tool(parameters: dict) -> list[dict]:
    return [{"type": "function", "function": {"name": "t", "parameters": parameters}}]


def _bad(parameters: dict, arguments: dict) -> tuple[str, ...]:
    issues = check_tool_calls(_tool(parameters), [ToolCall("c1", "t", arguments)])
    return issues[0].paths if issues else ()


OBJ = {"type": "object"}


@pytest.mark.parametrize(
    ("prop", "good", "bad"),
    [
        ({"type": "integer", "minimum": 1}, 1, 0),
        ({"type": "integer", "maximum": 5}, 5, 6),
        ({"type": "number", "exclusiveMinimum": 0}, 0.1, 0),
        ({"type": "number", "exclusiveMaximum": 1}, 0.9, 1),
        ({"type": "string", "minLength": 2}, "ab", "a"),
        ({"type": "string", "maxLength": 2}, "ab", "abc"),
        ({"type": "array", "minItems": 1}, [1], []),
        ({"type": "array", "maxItems": 1}, [1], [1, 2]),
        ({"type": "array", "uniqueItems": True}, [1, 2], [1, 1]),
        ({"allOf": [{"type": "integer"}, {"minimum": 3}]}, 3, 2),
        ({"oneOf": [{"type": "integer"}, {"minimum": 0}]}, -1, 5),  # 5 matches both
        ({"not": {"type": "string"}}, 1, "x"),
        ({"type": "object", "additionalProperties": {"type": "integer"}}, {"a": 1}, {"a": "x"}),
    ],
)
def test_constraint_keywords_are_enforced(prop, good, bad):
    schema = {**OBJ, "properties": {"v": prop}}
    assert _bad(schema, {"v": good}) == ()
    assert _bad(schema, {"v": bad}) == ("v",) or _bad(schema, {"v": bad})[0].startswith("v")


def test_unique_items_treats_true_and_one_as_different():
    schema = {**OBJ, "properties": {"v": {"type": "array", "uniqueItems": True}}}
    assert _bad(schema, {"v": [True, 1]}) == ()


def _completion(arguments: str):
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="t", arguments=arguments))
    msg = SimpleNamespace(content=None, tool_calls=[call], refusal=None)
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    return SimpleNamespace(choices=[choice], usage=usage, model="m")


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1, "a": 2}',  # duplicate key: last-wins differs between parsers
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": -Infinity}',
    ],
)
def test_ambiguous_tool_arguments_are_rejected(raw):
    with pytest.raises(InvalidOutputError):
        parse_openai_style_response(_completion(raw), "m")


def test_plain_tool_arguments_still_parse():
    result = parse_openai_style_response(_completion('{"a": 1, "b": {"c": [1.5]}}'), "m")
    assert result.tool_calls[0].arguments == {"a": 1, "b": {"c": [1.5]}}


def test_dict_output_schema_is_fully_validated():
    schema = OutputSchema.from_spec(
        {
            "type": "object",
            "required": ["n"],
            "properties": {"n": {"type": "integer"}, "tags": {"type": "array", "maxItems": 1}},
            "additionalProperties": False,
        }
    )
    assert schema.parse('{"n": 1}') == {"n": 1}
    for bad in ('{"n": "x"}', '{"n": 1, "extra": 2}', '{"n": 1, "tags": [1, 2]}'):
        with pytest.raises(InvalidOutputError) as info:
            schema.parse(bad)
        assert info.value.paths
