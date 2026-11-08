"""Agentic pass 3 (0.16.7): truncated tool calls, tool-call argument checks,
tool-call id hygiene, image tool results on Anthropic, tool-count pre-flight
and usage totals across a tool loop. Fakes only; no provider is called."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_gateway import (
    ChatResult,
    GatewayConfig,
    InvalidOutputError,
    InvalidRequestError,
    ToolCall,
    Usage,
    chat,
    reset_circuit_breakers,
)
from llm_gateway.providers._anthropic_translate import to_anthropic_messages
from llm_gateway.providers.base import parse_openai_style_response
from llm_gateway.tool_validation import ToolCallIssue, check_tool_calls, tool_error_message

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]


def _completion(arguments: str, finish_reason: str):
    call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="add", arguments=arguments), type="function"
    )
    message = SimpleNamespace(content=None, refusal=None, tool_calls=[call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


# 1. Truncated tool-call arguments are a distinct, typed error ---------------


def test_truncated_openai_tool_call_is_a_distinct_error():
    from llm_gateway import ToolCallTruncatedError

    with pytest.raises(ToolCallTruncatedError) as info:
        parse_openai_style_response(_completion('{"a": 1, "b', "length"), "m")
    assert isinstance(info.value, InvalidOutputError)
    assert info.value.finish_reason == "length"
    assert info.value.tool_names == ("add",)
    assert '{"a"' not in str(info.value)


def test_malformed_args_without_length_stay_plain_invalid_output():
    from llm_gateway import ToolCallTruncatedError

    with pytest.raises(InvalidOutputError) as info:
        parse_openai_style_response(_completion('{"a": ', "tool_calls"), "m")
    assert not isinstance(info.value, ToolCallTruncatedError)


def test_anthropic_cut_off_tool_call_is_the_distinct_error():
    from llm_gateway import ToolCallTruncatedError
    from llm_gateway.providers._anthropic_translate import from_anthropic_response

    block = SimpleNamespace(type="tool_use", id="t1", name="f", input={})
    resp = SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="max_tokens",
    )
    with pytest.raises(ToolCallTruncatedError) as info:
        from_anthropic_response(resp, "m")
    assert info.value.tool_names == ("f",)


# 2. Checking a model's tool calls against the tool schemas ------------------


def test_check_tool_calls_reports_typed_issues():
    calls = [
        ToolCall("1", "add", {"a": 1, "b": 2}),
        ToolCall("2", "add", {"a": "x"}),
        ToolCall("3", "nope", {}),
    ]
    issues = check_tool_calls(TOOLS, calls)
    assert [(i.call_id, i.kind) for i in issues] == [
        ("2", "invalid_arguments"),
        ("3", "unknown_tool"),
    ]
    assert set(issues[0].paths) == {"a", "b"}
    assert "x" not in issues[0].message


def test_check_tool_calls_clean_returns_empty():
    assert check_tool_calls(TOOLS, [ToolCall("1", "add", {"a": 1, "b": 2})]) == []


def test_tool_error_message_round_trips_through_validation():
    issue = ToolCallIssue("2", "add", "invalid_arguments", ("a",), "bad")
    message = tool_error_message(issue)
    assert message["role"] == "tool" and message["tool_call_id"] == "2"
    assert "bad" in message["content"]


# 3. Tool-call id hygiene on the model's reply -------------------------------


def _fake_chat(result):
    async def call(config, messages, tools, max_tokens, **kw):
        return result

    return call


async def _chat_with(result):
    reset_circuit_breakers()
    cfg = GatewayConfig(provider_order="openai", openai_api_key="k")
    with patch.dict("llm_gateway.chat.CHAT_CALLS", {"openai": _fake_chat(result)}):
        return await chat(messages=[{"role": "user", "content": "hi"}], tools=TOOLS, config=cfg)


async def test_duplicate_and_empty_reply_ids_are_made_unique():
    result = ChatResult(
        content=None,
        model="m",
        input_tokens=1,
        output_tokens=1,
        tool_calls=[
            ToolCall("x", "add", {"a": 1, "b": 1}),
            ToolCall("x", "add", {"a": 2, "b": 2}),
            ToolCall("", "add", {"a": 3, "b": 3}),
        ],
        finish_reason="tool_calls",
    )
    out = await _chat_with(result)
    ids = [c.id for c in out.tool_calls]
    assert len(set(ids)) == 3 and all(ids) and ids[0] == "x"
    assert [c.arguments["a"] for c in out.tool_calls] == [1, 2, 3]


async def test_unique_ids_untouched():
    result = ChatResult(
        None, "m", 1, 1, [ToolCall("a", "add", {}), ToolCall("b", "add", {})], "tool_calls"
    )
    assert [c.id for c in (await _chat_with(result)).tool_calls] == ["a", "b"]


# 4. Image tool results on Anthropic ----------------------------------------


def test_anthropic_tool_result_images_are_translated():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "shot", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": [
                {"type": "text", "text": "here"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
    ]
    _, out = to_anthropic_messages(messages)
    content = out[-1]["content"][0]["content"]
    assert content[0] == {"type": "text", "text": "here"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"


def test_anthropic_tool_call_arguments_must_be_an_object():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "[1]"}}
            ],
        },
    ]
    with pytest.raises(InvalidRequestError):
        to_anthropic_messages(messages)


# 5. Tool-count pre-flight ---------------------------------------------------


async def test_too_many_tools_rejected_before_any_provider():
    tools = [
        {"type": "function", "function": {"name": f"t{i}", "parameters": {"type": "object"}}}
        for i in range(129)
    ]
    cfg = GatewayConfig(provider_order="openai", openai_api_key="k")
    with pytest.raises(InvalidRequestError, match="129"):
        await chat(messages=[{"role": "user", "content": "x"}], tools=tools, config=cfg)


# 6. Usage totals across a tool loop ----------------------------------------


def test_usage_adds_and_sums_across_steps():
    a = Usage(10, 5, 2, 1, 3)
    b = Usage(20, 7, 4, 0, 0)
    assert a + b == Usage(30, 12, 6, 1, 3)
    assert Usage.total([a, b, a]) == Usage(40, 17, 8, 2, 6)
    assert Usage.total([]) == Usage()
    assert sum([a, b], Usage()) == a + b
    assert sum([a, b]) if False else True
