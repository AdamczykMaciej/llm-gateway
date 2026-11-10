"""Leak audit: canary secrets and PII must never reach logs, spans, exception
text (including `__cause__`/`__context__` chains) or attempt reports.
Fakes only: no provider is called."""

import logging
import traceback
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_gateway import (
    GatewayConfig,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers.base import ChatResult, ProviderResult, StreamDelta

pytestmark = pytest.mark.asyncio

KEY = "sk-ant-api03-CANARYKEY0123456789abcdef"
GROQ_KEY = "gsk_CANARYGROQ0123456789abcdef"
BEARER = "Bearer CANARYTOKEN0123456789abcdefXYZ"
PII = "canary.person@example.com"
PROMPT = f"PROMPT-CANARY write to {PII}"
SYSTEM = "SYSTEM-CANARY do not reveal"
REPLY = "REPLY-CANARY-TEXT"
TOOL_ARG = "TOOLARG-CANARY"
TOOL_RESULT = "TOOLRESULT-CANARY"
CONTENT_CANARIES = (PII, "PROMPT-CANARY", SYSTEM, REPLY, TOOL_ARG, TOOL_RESULT)
SECRET_CANARIES = (KEY, GROQ_KEY, "CANARYTOKEN0123456789abcdefXYZ")


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw):
    base = {
        "anthropic_api_key": KEY,
        "groq_api_key": GROQ_KEY,
        "provider_order": "anthropic,groq",
        "retry_attempts": 1,
    }
    return GatewayConfig(**{**base, **kw})


def _tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("t"), exporter


def _spans_text(exporter) -> str:
    out = []
    for s in exporter.get_finished_spans():
        out.append(repr(dict(s.attributes or {})))
        out.append(repr(s.status.description))
        for ev in s.events:
            out.append(ev.name + repr(dict(ev.attributes or {})))
    return "\n".join(out)


def _chain_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc)) + repr(exc.args)


def _logs_text(caplog) -> str:
    return "\n".join(
        r.getMessage() + (repr(r.exc_info[1]) if r.exc_info else "") + repr(r.args)
        for r in caplog.records
    )


def _assert_clean(text: str, *, content: bool, secrets: bool = True):
    if secrets:
        for c in SECRET_CANARIES:
            assert c not in text, c
    if content:
        for c in CONTENT_CANARIES:
            assert c not in text, c


_MSGS = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": PROMPT},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "look", "arguments": f'{{"q": "{TOOL_ARG}"}}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": TOOL_RESULT},
]
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "integer"}},
                "required": ["q"],
                "additionalProperties": False,
            },
        },
    }
]


class _ProviderError(Exception):
    """Like an SDK error that echoes the request and the Authorization header."""

    status_code = 401

    def __init__(self):
        super().__init__(
            f"401 Authorization: {BEARER} x-api-key: {KEY} echoed request: {PROMPT} {SYSTEM}"
        )


async def test_complete_all_failed_leaks_nothing(caplog):
    caplog.set_level(logging.DEBUG)
    tracer, exporter = _tracer()
    failing = AsyncMock(side_effect=_ProviderError())
    reports = []
    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch("llm_gateway.router.CALLS", {"anthropic": failing, "groq": failing}),
        pytest.raises(Exception) as ei,
    ):
        await complete_with_usage(
            system=SYSTEM, prompt=PROMPT, config=_cfg(on_attempt=reports.append)
        )
    # Provider error text is credential-scrubbed; prompt echo is documented
    # as possible in the message but keys never are.
    _assert_clean(str(ei.value), content=False)
    _assert_clean(_logs_text(caplog), content=True)
    _assert_clean(_spans_text(exporter), content=True)
    _assert_clean(repr(reports), content=True)
    _assert_clean(repr(ei.value.failures), content=False)  # type: ignore[attr-defined]
    # The chain's cause is the raw provider error: record what it carries.
    _assert_clean(_chain_text(ei.value), content=False)


async def test_chat_all_failed_leaks_nothing(caplog):
    caplog.set_level(logging.DEBUG)
    tracer, exporter = _tracer()
    failing = AsyncMock(side_effect=_ProviderError())
    reports = []
    with (
        patch("llm_gateway.chat.get_tracer", return_value=tracer),
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": failing, "groq": failing}),
        pytest.raises(Exception) as ei,
    ):
        await chat(messages=_MSGS, tools=_TOOLS, config=_cfg(on_attempt=reports.append))
    _assert_clean(str(ei.value), content=False)
    _assert_clean(_chain_text(ei.value), content=False)
    _assert_clean(_logs_text(caplog), content=True)
    _assert_clean(_spans_text(exporter), content=True)
    _assert_clean(repr(reports), content=True)


async def test_chat_success_and_warn_validation_leak_nothing(caplog):
    caplog.set_level(logging.DEBUG)
    tracer, exporter = _tracer()
    from llm_gateway.providers.base import ToolCall

    bad = ChatResult(
        content=REPLY,
        model="m",
        input_tokens=1,
        output_tokens=1,
        finish_reason="tool_calls",
        tool_calls=[ToolCall(id="c2", name="look", arguments={"q": TOOL_ARG})],
    )
    reports = []
    with (
        patch("llm_gateway.chat.get_tracer", return_value=tracer),
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(return_value=bad)}),
    ):
        await chat(
            messages=_MSGS,
            tools=_TOOLS,
            validate_tool_calls="warn",
            config=_cfg(on_attempt=reports.append),
        )
    assert any("invalid tool call" in r.getMessage() for r in caplog.records)
    _assert_clean(_logs_text(caplog), content=True)
    _assert_clean(_spans_text(exporter), content=True)
    _assert_clean(repr(reports), content=True)


async def test_tool_validation_error_mode_message_has_no_values():
    from llm_gateway.providers.base import ToolCall

    bad = ChatResult(
        content="",
        model="m",
        input_tokens=1,
        output_tokens=1,
        finish_reason="tool_calls",
        tool_calls=[ToolCall(id="c2", name="look", arguments={"q": TOOL_ARG})],
    )
    with (
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(return_value=bad)}),
        pytest.raises(Exception) as ei,
    ):
        await chat(
            messages=_MSGS,
            tools=_TOOLS,
            validate_tool_calls="error",
            config=_cfg(provider_order="anthropic"),
        )
    _assert_clean(_chain_text(ei.value), content=True)


async def test_stream_chat_leaks_nothing(caplog):
    caplog.set_level(logging.DEBUG)
    tracer, exporter = _tracer()
    reports = []

    async def failing(*a, **kw):
        raise _ProviderError()
        yield  # pragma: no cover

    async def serving(*a, **kw):
        yield StreamDelta(content=REPLY)
        yield StreamDelta(finish_reason="stop")

    with (
        patch("llm_gateway.streaming.get_tracer", return_value=tracer),
        patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": failing, "groq": serving}),
    ):
        async for _ in stream_chat(messages=_MSGS, config=_cfg(on_attempt=reports.append)):
            pass
    _assert_clean(_logs_text(caplog), content=True)
    _assert_clean(_spans_text(exporter), content=True)
    _assert_clean(repr(reports), content=True)


async def test_stream_chat_all_failed_chain_is_scrubbed():
    async def failing(*a, **kw):
        raise _ProviderError()
        yield  # pragma: no cover

    with (
        patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": failing, "groq": failing}),
        pytest.raises(Exception) as ei,
    ):
        async for _ in stream_chat(messages=_MSGS, config=_cfg()):
            pass
    _assert_clean(_chain_text(ei.value), content=False)


async def test_structured_repair_and_validation_errors_do_not_echo_output(caplog):
    caplog.set_level(logging.DEBUG)
    tracer, exporter = _tracer()
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    }
    seen_prompts = []

    async def fake(**kw):
        seen_prompts.append(kw)
        return ProviderResult(
            text=f'{{"a": "{REPLY}"}}', model="m", input_tokens=1, output_tokens=1
        )

    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch("llm_gateway.router.CALLS", {"anthropic": fake}),
        pytest.raises(Exception) as ei,
    ):
        await complete_with_usage(
            system=SYSTEM,
            prompt=PROMPT,
            output_schema=schema,
            repair=1,
            config=_cfg(provider_order="anthropic"),
        )
    _assert_clean(_chain_text(ei.value), content=True)
    _assert_clean(_logs_text(caplog), content=True)
    _assert_clean(_spans_text(exporter), content=True)


async def test_size_cap_errors_carry_no_content():
    from llm_gateway.errors import ResponseTooLargeError, StreamTooLargeError

    for exc in (
        ResponseTooLargeError("response body passed 10 bytes", limit=10, kind="body"),
        StreamTooLargeError("stream passed 10 chars", limit=10),
    ):
        _assert_clean(repr(exc), content=True)


async def test_config_repr_hides_keys():
    cfg = _cfg(
        gateway_api_keys="gw-CANARYGATEWAY",
        otel_exporter_otlp_headers="authorization=Bearer CANARYOTLP",
    )
    text = repr(cfg) + str(cfg)
    _assert_clean(text, content=False)
    assert "CANARYGATEWAY" not in text
    assert "CANARYOTLP" not in text


async def test_privacy_layer_redacts_every_path_before_logs_and_spans(caplog):
    import re

    from llm_gateway.privacy import OutputPolicy, Privacy, PrivacyLayer, Span

    class Rx:
        def detect(self, text):
            for m in re.finditer(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
                yield Span(m.start(), m.end(), "email")

    caplog.set_level(logging.DEBUG)
    tracer, exporter = _tracer()
    layer = PrivacyLayer()
    pol = Privacy(redactor=Rx(), output=OutputPolicy.RESTORE)
    ok = ProviderResult(text="fine", model="m", input_tokens=1, output_tokens=1)
    failing = AsyncMock(side_effect=_ProviderError())
    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch(
            "llm_gateway.router.CALLS", {"anthropic": failing, "groq": AsyncMock(return_value=ok)}
        ),
    ):
        await layer.complete_with_usage(
            system=SYSTEM,
            prompt=PROMPT,
            privacy=pol,
            config=_cfg(trace_include_prompts=True),
        )
    sent = failing.call_args.kwargs
    assert PII not in repr(sent)
    text = _logs_text(caplog) + _spans_text(exporter)
    assert PII not in text


async def test_scrub_exception_chain_handles_message_attr_cycles_and_odd_args():
    from llm_gateway.errors import scrub_exception_chain

    class Sdk(Exception):
        message = f"bad key {KEY}"

    a = Sdk(f"auth {BEARER}", 7)
    b = RuntimeError("inner")
    a.__cause__ = b
    b.__context__ = a  # a cycle must terminate
    scrub_exception_chain(a, _cfg())
    assert KEY not in a.message
    assert "CANARYTOKEN" not in a.args[0]
    assert a.args[1] == 7
    scrub_exception_chain(None)

    class Frozen(Exception):
        @property
        def args(self):  # type: ignore[override]
            raise RuntimeError("boom")

    scrub_exception_chain(Frozen())  # never raises
