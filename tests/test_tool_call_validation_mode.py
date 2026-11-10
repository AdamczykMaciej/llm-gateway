"""`chat(validate_tool_calls=...)`: opt-in check of returned tool calls."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from llm_gateway import (
    AllProvidersExhaustedError,
    GatewayConfig,
    InvalidRequestError,
    ToolCallInvalidError,
    chat,
    reset_circuit_breakers,
)
from llm_gateway.providers.base import ChatResult, ToolCall

pytestmark = pytest.mark.asyncio

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "s": {"type": "string", "maxLength": 3}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
    }
]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw) -> GatewayConfig:
    return GatewayConfig(
        anthropic_api_key="k", openai_api_key="k", retry_attempts=1,
        provider_order="anthropic,openai", **kw,
    )  # fmt: skip


def _result(*calls: ToolCall) -> ChatResult:
    return ChatResult(
        content=None, model="m", input_tokens=1, output_tokens=1,
        tool_calls=list(calls), finish_reason="tool_calls",
    )  # fmt: skip


BAD = ToolCall(id="c1", name="add", arguments={"a": "x"})
GOOD = ToolCall(id="c2", name="add", arguments={"a": 1})
GHOST = ToolCall(id="c3", name="rm_rf", arguments={})


async def _run(first: ChatResult, second: ChatResult | None = None, **kw):
    a = AsyncMock(return_value=first)
    o = AsyncMock(return_value=second or first)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": a, "openai": o}):
        return await chat(messages=MESSAGES, tools=TOOLS, config=_cfg(), **kw), a, o


async def test_default_off_passes_invalid_calls_through():
    out, _, o = await _run(_result(BAD, GHOST))
    assert [c.id for c in out.tool_calls] == ["c1", "c3"]
    o.assert_not_awaited()


async def test_error_mode_fails_over_to_valid_reply():
    out, a, o = await _run(_result(BAD), _result(GOOD), validate_tool_calls="error")
    assert [c.id for c in out.tool_calls] == ["c2"]
    a.assert_awaited_once()
    o.assert_awaited_once()


async def test_error_mode_hallucinated_tool_and_chain_exhausted():
    a = AsyncMock(return_value=_result(GHOST))
    o = AsyncMock(return_value=_result(BAD))
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": a, "openai": o}):
        with pytest.raises(AllProvidersExhaustedError) as exc:
            await chat(messages=MESSAGES, tools=TOOLS, config=_cfg(), validate_tool_calls="error")
    cause = exc.value.__cause__
    assert isinstance(cause, ToolCallInvalidError)
    assert cause.issues[0].kind == "invalid_arguments"
    assert cause.issues[0].paths == ("a",)
    assert "'x'" not in str(cause)  # no argument values


async def test_error_mode_enforces_string_length_bound():
    long = ToolCall(id="c4", name="add", arguments={"a": 1, "s": "toolong"})
    with pytest.raises(AllProvidersExhaustedError):
        await _run(_result(long), validate_tool_calls="error")


async def test_error_mode_accepts_valid_calls_and_text_replies():
    out, _, _ = await _run(_result(GOOD), validate_tool_calls="error")
    assert out.tool_calls[0].id == "c2"
    text = ChatResult(content="hello", model="m", input_tokens=1, output_tokens=1)
    out, _, _ = await _run(text, validate_tool_calls="error")
    assert out.content == "hello"


async def test_warn_mode_logs_and_returns(caplog):
    with caplog.at_level(logging.WARNING, logger="llm_gateway"):
        out, _, o = await _run(_result(BAD), validate_tool_calls="warn")
    assert out.tool_calls[0].id == "c1"
    o.assert_not_awaited()
    assert any("invalid_arguments" in r.getMessage() for r in caplog.records)
    assert "'x'" not in caplog.text


async def test_oversized_arguments_rejected_in_error_mode():
    big = ToolCall(id="c5", name="echo", arguments={"text": "y" * 50})
    cfg = _cfg(stream_max_tool_arguments_chars=20)
    a = AsyncMock(return_value=_result(big))
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": a, "openai": a}):
        with pytest.raises(AllProvidersExhaustedError) as exc:
            await chat(
                messages=MESSAGES,
                tools=[{"type": "function", "function": {"name": "echo"}}],
                config=cfg,
                validate_tool_calls="error",
            )
    assert exc.value.__cause__.issues[0].kind == "arguments_too_large"


async def test_bad_mode_value_is_invalid_request():
    with pytest.raises(InvalidRequestError, match="validate_tool_calls"):
        await chat(messages=MESSAGES, config=_cfg(), validate_tool_calls="strict")
