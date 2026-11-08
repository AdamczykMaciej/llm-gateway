"""Pre-flight request validation: every rejection branch, plus properties."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_gateway import InvalidRequestError
from llm_gateway.tool_validation import validate_chat_request

U = [{"role": "user", "content": "hi"}]


def _tool(name="t", parameters=None, **extra):
    function = {"name": name}
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function, **extra}


def _call(call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": "t", "arguments": "{}"}}


def _bad(messages=U, tools=None, tool_choice=None, match=""):
    with pytest.raises(InvalidRequestError, match=match):
        validate_chat_request(messages, tools, tool_choice)


def test_valid_request_passes():
    schema = {
        "type": "object",
        "properties": {"a": {"type": ["string", "null"], "enum": ["x"]}},
        "required": ["a"],
        "items": [{"type": "string"}],
        "anyOf": [{"type": "object"}],
        "$defs": {"d": {"type": "integer"}},
    }
    messages = [
        *U,
        {"role": "assistant", "tool_calls": [_call("a"), _call("b")]},
        {"role": "tool", "tool_call_id": "b", "content": "1"},
        {"role": "tool", "tool_call_id": "a", "content": "2"},
    ]
    validate_chat_request(
        messages, [_tool(parameters=schema)], {"type": "function", "function": {"name": "t"}}
    )
    validate_chat_request(U, [_tool(parameters={"properties": {"b": True}})], "required")


@pytest.mark.parametrize(
    ("tools", "match"),
    [
        (["nope"], "must be"),
        ([{"type": "function"}], "must be"),
        ([{"type": "retrieval", "function": {"name": "t"}}], "only tools of type"),
        ([_tool(name="bad name")], "name"),
        ([_tool(name=3)], "name"),
        ([_tool(), _tool()], "duplicate"),
        ([_tool(parameters={"type": "array"})], "type 'object'"),
        ([_tool(parameters=[])], "type 'object'"),
        ([_tool(parameters={"properties": {"a": {"type": "wat"}}})], "'type' must be one of"),
        ([_tool(parameters={"properties": {"a": {"type": []}}})], "'type' must be one of"),
        ([_tool(parameters={"properties": {"a": {"type": {"x": 1}}}})], "'type' must be one of"),
        ([_tool(parameters={"properties": {"a": {"type": [["x"]]}}})], "'type' must be one of"),
        ([_tool(parameters={"enum": []})], "'enum'"),
        ([_tool(parameters={"enum": "x"})], "'enum'"),
        ([_tool(parameters={"required": "a"})], "'required'"),
        ([_tool(parameters={"required": [1]})], "'required'"),
        ([_tool(parameters={"properties": []})], "'properties'"),
        ([_tool(parameters={"properties": {"a": 5}})], "JSON Schema object"),
        ([_tool(parameters={"items": 5})], "JSON Schema object"),
        ([_tool(parameters={"anyOf": []})], "'anyOf'"),
        ([_tool(parameters={"oneOf": "x"})], "'oneOf'"),
        ([_tool(parameters={"allOf": [5]})], "JSON Schema object"),
        ([_tool(parameters={"$defs": []})], "'\\$defs'"),
        ([_tool(parameters={"definitions": {"a": 5}})], "JSON Schema object"),
        ([_tool(parameters={"type": "object", "default": {1, 2}})], "not JSON"),
    ],
)
def test_rejects_malformed_tools(tools, match):
    _bad(tools=tools, match=match)


def test_rejects_too_deep_schema():
    schema: dict = {"type": "object"}
    for _ in range(70):
        schema = {"type": "object", "properties": {"n": schema}}
    _bad(tools=[_tool(parameters=schema)], match="nested too deeply")


@pytest.mark.parametrize(
    ("choice", "tools", "match"),
    [
        ("sometimes", [_tool()], "tool_choice must be"),
        ("required", None, "at least one tool"),
        (5, [_tool()], "tool_choice must be"),
        ({"type": "function", "function": {}}, [_tool()], "tool_choice must be"),
        ({"type": "other", "function": {"name": "t"}}, [_tool()], "tool_choice must be"),
        ({"type": "function", "function": {"name": "zzz"}}, [_tool()], "not in tools"),
    ],
)
def test_rejects_bad_tool_choice(choice, tools, match):
    _bad(tools=tools, tool_choice=choice, match=match)


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([{"role": "tool", "content": "x"}], "needs tool_call_id"),
        ([{"role": "tool", "tool_call_id": "", "content": "x"}], "needs tool_call_id"),
        ([*U, {"role": "tool", "tool_call_id": "zz", "content": "x"}], "no "),
        ([{"role": "assistant", "tool_calls": ["x"]}], "needs an id"),
        ([{"role": "assistant", "tool_calls": [{"id": "a"}]}], "needs an id"),
        (
            [
                {"role": "assistant", "tool_calls": [_call("a")]},
                {"role": "tool", "tool_call_id": "a", "content": "x"},
                {"role": "assistant", "tool_calls": [_call("a")]},
            ],
            "duplicate",
        ),
        ([{"role": "assistant", "tool_calls": [_call("a")]}], "no tool result"),
        (
            [{"role": "assistant", "tool_calls": [_call("a")]}, {"role": "user", "content": "x"}],
            "no tool result",
        ),
        (
            [
                {"role": "assistant", "tool_calls": [_call("a")]},
                {"role": "tool", "tool_call_id": "a", "content": "x"},
                {"role": "tool", "tool_call_id": "a", "content": "x"},
            ],
            "no ",
        ),
    ],
)
def test_rejects_bad_pairing(messages, match):
    _bad(messages=messages, match=match)


@given(st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=6, unique=True))
def test_any_fully_answered_set_of_calls_is_valid(ids):
    messages = [
        *U,
        {"role": "assistant", "tool_calls": [_call(i) for i in ids]},
        *({"role": "tool", "tool_call_id": i, "content": "r"} for i in reversed(ids)),
    ]
    validate_chat_request(messages, None, None)


@given(st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=6, unique=True), st.data())
def test_dropping_any_result_is_rejected(ids, data):
    dropped = data.draw(st.sampled_from(ids))
    messages = [
        {"role": "assistant", "tool_calls": [_call(i) for i in ids]},
        *({"role": "tool", "tool_call_id": i, "content": "r"} for i in ids if i != dropped),
        {"role": "user", "content": "next"},
    ]
    with pytest.raises(InvalidRequestError):
        validate_chat_request(messages, None, None)


@given(st.text(max_size=80))
def test_tool_name_acceptance_matches_the_documented_pattern(name):
    import re

    expected = re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) is not None
    try:
        validate_chat_request(U, [_tool(name=name)], None)
        accepted = True
    except InvalidRequestError:
        accepted = False
    assert accepted == expected
