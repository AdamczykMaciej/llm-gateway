"""`stream_chat(validate_tool_calls=...)`: opt-in check of streamed tool calls."""

import dataclasses
import logging
from unittest.mock import patch

import pytest

from llm_gateway import (
    AllProvidersExhaustedError,
    GatewayConfig,
    InvalidRequestError,
    ToolCallInvalidError,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers.base import StreamDelta

pytestmark = pytest.mark.asyncio

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
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


def _call(index: int, name: str, args: str) -> list[StreamDelta]:
    """A tool call streamed as a header fragment plus two argument halves."""
    half = len(args) // 2
    return [
        StreamDelta(tool_call_deltas=[{"index": index, "id": f"c{index}", "type": "function",
                                       "function": {"name": name, "arguments": ""}}]),
        StreamDelta(tool_call_deltas=[{"index": index, "function": {"arguments": args[:half]}}]),
        StreamDelta(tool_call_deltas=[{"index": index, "function": {"arguments": args[half:]}}]),
    ]  # fmt: skip


FINAL = StreamDelta(finish_reason="tool_calls", usage=(1, 1))
GOOD = [*_call(0, "add", '{"a": 1}'), FINAL]
BAD = [*_call(0, "add", '{"a": "x"}'), FINAL]
GHOST = [*_call(0, "rm_rf", "{}"), FINAL]
NOT_JSON = [*_call(0, "add", '{"a": '), FINAL]


def _fake(deltas):
    async def gen():
        for d in deltas:
            yield d

    return lambda *a, **k: gen()


async def _run(first, second=None, **kw):
    calls = {"anthropic": 0, "openai": 0}

    def wrap(name, deltas):
        inner = _fake(deltas)

        def fn(*a, **k):
            calls[name] += 1
            return inner()

        return fn

    registry = {"anthropic": wrap("anthropic", first), "openai": wrap("openai", second or first)}
    out = []
    with patch("llm_gateway.streaming.STREAM_CALLS", registry):
        async for d in stream_chat(messages=MESSAGES, tools=TOOLS, config=_cfg(), **kw):
            out.append(d)
    return out, calls


def _same(out):
    """Deltas without the provider attribution stream_chat adds to usage."""
    return [dataclasses.replace(d, provider=None, request_id=None) for d in out]


def _ids(out):
    return [f["id"] for d in out for f in (d.tool_call_deltas or []) if "id" in f]


async def test_default_off_passes_invalid_calls_through():
    out, calls = await _run(BAD)
    assert _ids(out) == ["c0"]
    assert calls == {"anthropic": 1, "openai": 0}


async def test_error_mode_fails_over_when_nothing_was_yielded():
    out, calls = await _run(BAD, GOOD, validate_tool_calls="error")
    assert calls == {"anthropic": 1, "openai": 1}
    assert _same(out) == GOOD  # nothing of the bad attempt leaked, order preserved


async def test_error_mode_hallucinated_tool_exhausts_chain():
    with pytest.raises(AllProvidersExhaustedError) as exc:
        await _run(GHOST, GHOST, validate_tool_calls="error")
    cause = exc.value.__cause__
    assert isinstance(cause, ToolCallInvalidError)
    assert cause.issues[0].kind == "unknown_tool"


async def test_error_mode_malformed_json_is_an_issue_without_values():
    with pytest.raises(AllProvidersExhaustedError) as exc:
        await _run(NOT_JSON, NOT_JSON, validate_tool_calls="error")
    assert exc.value.__cause__.issues[0].kind == "arguments_not_json"


async def test_error_mode_after_text_raises_without_failover():
    text = StreamDelta(content="Let me check. ")
    got = []
    registry = {"anthropic": _fake([text, *BAD]), "openai": _fake(GOOD)}
    with patch("llm_gateway.streaming.STREAM_CALLS", registry):
        with pytest.raises(ToolCallInvalidError):
            async for d in stream_chat(
                messages=MESSAGES, tools=TOOLS, config=_cfg(), validate_tool_calls="error"
            ):
                got.append(d)
    assert got == [text]  # text was delivered, tool fragments were withheld


async def test_error_mode_text_then_valid_calls_stream_in_order():
    text = StreamDelta(content="ok ")
    out, _ = await _run([text, *GOOD], validate_tool_calls="error")
    assert _same(out) == [text, *GOOD]


async def test_error_mode_text_only_reply_streams_untouched():
    deltas = [StreamDelta(content="a"), StreamDelta(content="b"), FINAL]
    out, _ = await _run(deltas, validate_tool_calls="error")
    assert _same(out) == deltas


async def test_warn_mode_streams_everything_and_logs(caplog):
    with caplog.at_level(logging.WARNING, logger="llm_gateway"):
        out, calls = await _run(BAD, validate_tool_calls="warn")
    assert _same(out) == BAD
    assert calls["openai"] == 0
    assert any("invalid_arguments" in r.getMessage() for r in caplog.records)
    assert "'x'" not in caplog.text


async def test_no_tools_means_nothing_to_validate():
    registry = {"anthropic": _fake(BAD)}
    with patch("llm_gateway.streaming.STREAM_CALLS", registry):
        out = [
            d
            async for d in stream_chat(
                messages=MESSAGES, config=_cfg(), force_provider="anthropic",
                validate_tool_calls="error",
            )
        ]  # fmt: skip
    assert _same(out) == BAD


async def test_bad_mode_value_is_invalid_request():
    with pytest.raises(InvalidRequestError, match="validate_tool_calls"):
        await _run(GOOD, validate_tool_calls="strict")


async def test_rejected_attempt_is_reported_failed_with_its_usage():
    reports = []
    await _run(BAD, GOOD, validate_tool_calls="error", on_attempt=reports.append)
    assert [(r.provider, r.outcome) for r in reports] == [
        ("anthropic", "failed"),
        ("openai", "succeeded"),
    ]
    assert reports[0].usage is not None  # billed, so the ledger sees it
    assert reports[0].error_type == "ToolCallInvalidError"
