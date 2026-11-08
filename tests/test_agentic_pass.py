"""Agentic pass: pre-flight validation of tools / tool_choice / tool-result
pairing, and Anthropic `tool_choice="none"` on a tool-using history. No
provider is ever called."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_gateway import GatewayConfig, InvalidRequestError, LLMError, chat, stream_chat
from llm_gateway.providers import anthropic
from llm_gateway.providers.base import ChatResult

pytestmark = pytest.mark.asyncio

_USER = {"role": "user", "content": "hi"}


def _tool(name="get_weather", parameters=None):
    function = {"name": name}
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def _call(call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }


def _result(call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": "sunny"}


def _config():
    return GatewayConfig(anthropic_api_key="k", provider_order="anthropic", retry_attempts=1)


async def _rejected(**kwargs) -> str:
    provider = AsyncMock(
        return_value=ChatResult(content="x", model="m", input_tokens=1, output_tokens=1)
    )
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": provider}):
        with pytest.raises(InvalidRequestError) as info:
            await chat(config=_config(), **kwargs)
    provider.assert_not_called()
    assert isinstance(info.value, LLMError) and isinstance(info.value, ValueError)
    return str(info.value)


@pytest.mark.parametrize(
    ("tools", "fragment"),
    [
        ([_tool("bad name!")], "name"),
        ([_tool(), _tool()], "duplicate"),
        ([_tool(parameters={"type": "banana"})], "type"),
        ([_tool(parameters={"type": "object", "required": "city"})], "required"),
        (
            [
                _tool(
                    parameters={
                        "type": "object",
                        "properties": {"u": {"type": "string", "enum": []}},
                    }
                )
            ],
            "enum",
        ),
        ([_tool(parameters={"type": "string"})], "object"),
        ([{"type": "function"}], "tools[0]"),
    ],
)
async def test_bad_tool_definitions_rejected_before_any_provider(tools, fragment):
    assert fragment in await _rejected(messages=[_USER], tools=tools)


@pytest.mark.parametrize(
    "tool_choice",
    [{"type": "function", "function": {"name": "nope"}}, "sometimes", "required"],
)
async def test_bad_tool_choice_rejected(tool_choice):
    tools = [] if tool_choice == "required" else [_tool()]
    assert "tool_choice" in await _rejected(messages=[_USER], tools=tools, tool_choice=tool_choice)


async def test_tool_result_without_call_rejected():
    message = await _rejected(messages=[_USER, _result("call_9")], tools=[_tool()])
    assert "call_9" in message and "messages[1]" in message


async def test_tool_result_without_id_rejected():
    message = await _rejected(messages=[_USER, {"role": "tool", "content": "x"}])
    assert "tool_call_id" in message


async def test_unanswered_parallel_call_rejected():
    assistant = {"role": "assistant", "content": None, "tool_calls": [_call("a"), _call("b")]}
    message = await _rejected(messages=[_USER, assistant, _result("a")], tools=[_tool()])
    assert "b" in message


async def test_duplicate_result_rejected():
    assistant = {"role": "assistant", "content": None, "tool_calls": [_call("a")]}
    await _rejected(messages=[_USER, assistant, _result("a"), _result("a")], tools=[_tool()])


async def test_valid_parallel_round_trip_passes():
    assistant = {"role": "assistant", "content": None, "tool_calls": [_call("a"), _call("b")]}
    messages = [_USER, assistant, _result("b"), _result("a")]
    ok = ChatResult(content="done", model="m", input_tokens=1, output_tokens=1)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(return_value=ok)}):
        out = await chat(
            messages=messages,
            tools=[_tool(parameters={"type": "object", "properties": {"c": {"type": "string"}}})],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            config=_config(),
        )
    assert out.content == "done"


async def test_stream_chat_validates_too():
    with pytest.raises(InvalidRequestError):
        async for _ in stream_chat(messages=[_USER, _result("x")], config=_config()):
            pass


async def test_anthropic_tool_choice_none_keeps_tools_when_history_has_tool_blocks():
    create = AsyncMock()
    resp = MagicMock()
    block = MagicMock(type="text", text="done")
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=1, output_tokens=1)
    create.return_value = resp
    client = MagicMock()
    client.messages.create = create
    assistant = {"role": "assistant", "content": None, "tool_calls": [_call("a")]}
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        await anthropic.chat(
            GatewayConfig(anthropic_api_key="k"),
            [_USER, assistant, _result("a")],
            tools=[_tool()],
            max_tokens=10,
            tool_choice="none",
        )
    assert create.call_args.kwargs["tools"][0]["name"] == "get_weather"
    assert create.call_args.kwargs["tool_choice"] == {"type": "none"}


async def test_service_returns_400_for_invalid_tool_request():
    from fastapi.testclient import TestClient

    from llm_gateway.service.app import create_app

    client = TestClient(
        create_app(GatewayConfig(anthropic_api_key="k", provider_order="anthropic"))
    )
    body = {"messages": [_USER, _result("ghost")]}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400
    assert "ghost" in resp.json()["error"]["message"]
