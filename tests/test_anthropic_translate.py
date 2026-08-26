"""Unit tests for the OpenAI <-> Anthropic tool-format translation layer —
the riskiest part of tool-calling support, since Anthropic's native format
is structurally different in every direction (see the module docstring)."""

import json
from unittest.mock import MagicMock

from llm_gateway.providers._anthropic_translate import (
    from_anthropic_response,
    to_anthropic_messages,
    to_anthropic_tool_choice,
    to_anthropic_tools,
)


class TestToAnthropicMessages:
    def test_system_messages_concatenated_and_removed_from_list(self):
        system, msgs = to_anthropic_messages(
            [
                {"role": "system", "content": "Be nice."},
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ]
        )
        assert system == "Be nice.\nBe brief."
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_plain_user_and_assistant_passthrough(self):
        _, msgs = to_anthropic_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        assert msgs == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_assistant_tool_call_becomes_tool_use_block(self):
        _, msgs = to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": '{"city": "Warsaw"}'},
                        }
                    ],
                }
            ]
        )
        assert msgs == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_weather",
                        "input": {"city": "Warsaw"},
                    }
                ],
            }
        ]

    def test_assistant_tool_call_with_text_includes_both_blocks(self):
        _, msgs = to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}],
                }
            ]
        )
        assert msgs[0]["content"][0] == {"type": "text", "text": "Let me check."}
        assert msgs[0]["content"][1]["type"] == "tool_use"

    def test_tool_result_becomes_user_message_with_tool_result_block(self):
        _, msgs = to_anthropic_messages(
            [{"role": "tool", "tool_call_id": "call_1", "content": "22 C"}]
        )
        assert msgs == [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "22 C"}],
            }
        ]

    def test_consecutive_tool_results_merge_into_one_user_message(self):
        _, msgs = to_anthropic_messages(
            [
                {"role": "tool", "tool_call_id": "call_1", "content": "a"},
                {"role": "tool", "tool_call_id": "call_2", "content": "b"},
            ]
        )
        assert len(msgs) == 1
        assert len(msgs[0]["content"]) == 2
        assert msgs[0]["content"][0]["tool_use_id"] == "call_1"
        assert msgs[0]["content"][1]["tool_use_id"] == "call_2"

    def test_full_react_round_trip_shape(self):
        # system -> user -> assistant(tool_call) -> tool(result) -> (next turn)
        system, msgs = to_anthropic_messages(
            [
                {"role": "system", "content": "You are an agent."},
                {"role": "user", "content": "What's the weather in Warsaw?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": '{"city": "Warsaw"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "22 C, sunny"},
            ]
        )
        assert system == "You are an agent."
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]


class TestToAnthropicTools:
    def test_none_when_no_tools(self):
        assert to_anthropic_tools(None) is None
        assert to_anthropic_tools([]) is None

    def test_converts_function_shape_to_input_schema(self):
        tools = to_anthropic_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ]
        )
        assert tools == [
            {
                "name": "get_weather",
                "description": "Get the weather.",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]


class TestToAnthropicToolChoice:
    def test_none_and_openai_none_both_unmapped(self):
        assert to_anthropic_tool_choice(None) is None
        assert to_anthropic_tool_choice("none") is None

    def test_auto(self):
        assert to_anthropic_tool_choice("auto") == {"type": "auto"}

    def test_required_maps_to_any(self):
        assert to_anthropic_tool_choice("required") == {"type": "any"}

    def test_specific_function_choice(self):
        choice = to_anthropic_tool_choice({"type": "function", "function": {"name": "get_weather"}})
        assert choice == {"type": "tool", "name": "get_weather"}


class TestFromAnthropicResponse:
    def _block(self, **kwargs):
        b = MagicMock()
        for k, v in kwargs.items():
            setattr(b, k, v)
        return b

    def test_text_only_response(self):
        resp = MagicMock()
        resp.content = [self._block(type="text", text="Hello!")]
        resp.usage = MagicMock(input_tokens=10, output_tokens=5)

        result = from_anthropic_response(resp, "claude-x")

        assert result.content == "Hello!"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_tool_use_response(self):
        resp = MagicMock()
        resp.content = [
            self._block(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Warsaw"})
        ]
        resp.usage = MagicMock(input_tokens=10, output_tokens=5)

        result = from_anthropic_response(resp, "claude-x")

        assert result.content is None
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_1"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Warsaw"}

    def test_mixed_text_and_tool_use(self):
        resp = MagicMock()
        resp.content = [
            self._block(type="text", text="Checking the weather..."),
            self._block(
                type="tool_use", id="toolu_1", name="get_weather", input={"city": "Warsaw"}
            ),
        ]
        resp.usage = MagicMock(input_tokens=10, output_tokens=5)

        result = from_anthropic_response(resp, "claude-x")

        assert result.content == "Checking the weather..."
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1


def test_json_roundtrip_sanity():
    # Sanity check that the arguments JSON string in a tool_calls dict
    # (OpenAI shape) round-trips through the translation the way the real
    # HTTP layer produces it (json.dumps on the way out, json.loads here).
    args = {"city": "Warsaw", "unit": "celsius"}
    _, msgs = to_anthropic_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "x", "function": {"name": "f", "arguments": json.dumps(args)}}
                ],
            }
        ]
    )
    assert msgs[0]["content"][0]["input"] == args
