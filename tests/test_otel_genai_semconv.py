"""OpenTelemetry GenAI semantic-convention attributes. In-memory exporter, fake providers:
no network and no real provider call."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from llm_gateway import GatewayConfig, chat, complete_with_usage, reset_circuit_breakers
from llm_gateway.providers.base import ChatResult, ProviderResult

_USER = {"role": "user", "content": "hi"}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**kw):
    base = {
        "anthropic_api_key": "sk-ant-x",
        "groq_api_key": "gsk-x",
        "provider_order": "anthropic,groq",
        "retry_attempts": 1,
    }
    return GatewayConfig(**{**base, **kw})


def _tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("t"), exporter


def _by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


async def _chat_with_fallback():
    tracer, exporter = _tracer()
    failing = AsyncMock(side_effect=TimeoutError("boom"))
    serving = AsyncMock(
        return_value=ChatResult(
            content="RESPONSE-SECRET",
            model="served-v2",
            input_tokens=7,
            output_tokens=3,
            finish_reason="stop",
        )
    )
    with (
        patch("llm_gateway.chat.get_tracer", return_value=tracer),
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": failing, "groq": serving}),
    ):
        await chat(messages=[_USER], config=_config())
    return exporter


@pytest.mark.asyncio
async def test_chat_span_has_genai_attributes_and_deprecated_aliases():
    exporter = await _chat_with_fallback()
    (root,) = _by_name(exporter, "llm_gateway.chat")
    a = root.attributes
    assert root.kind == SpanKind.CLIENT
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "groq"
    assert a["gen_ai.system"] == "groq"  # deprecated alias
    assert a["gen_ai.request.model"] != "served-v2"
    assert a["gen_ai.response.model"] == "served-v2"
    assert a["gen_ai.usage.input_tokens"] == 7
    assert a["gen_ai.usage.output_tokens"] == 3
    assert tuple(a["gen_ai.response.finish_reasons"]) == ("stop",)
    assert a["llm_gateway.finish_reason"] == "stop"  # deprecated alias
    assert a["llm_gateway.fallback"] is True


@pytest.mark.asyncio
async def test_attempt_spans_error_type_and_fallback_event():
    exporter = await _chat_with_fallback()
    attempts = sorted(
        _by_name(exporter, "llm_gateway.attempt"), key=lambda s: s.attributes["llm_gateway.attempt"]
    )
    assert attempts[0].attributes["error.type"] == "TimeoutError"
    assert "error.type" not in attempts[1].attributes
    assert all(s.attributes["gen_ai.operation.name"] == "chat" for s in attempts)
    assert all(
        s.attributes["gen_ai.provider.name"] == s.attributes["gen_ai.system"] for s in attempts
    )
    (root,) = _by_name(exporter, "llm_gateway.chat")
    events = [e for e in root.events if e.name == "llm_gateway.fallback"]
    assert len(events) == 1
    assert events[0].attributes["gen_ai.provider.name"] == "groq"
    assert root.attributes["llm_gateway.error_code"] == "TimeoutError"
    assert root.attributes["error.type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_failed_call_sets_error_type_on_call_span():
    tracer, exporter = _tracer()
    failing = AsyncMock(side_effect=TimeoutError("x"))
    with (
        patch("llm_gateway.chat.get_tracer", return_value=tracer),
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": failing, "groq": failing}),
        pytest.raises(Exception),  # noqa: B017 — the gateway's exhausted-chain error
    ):
        await chat(messages=[_USER], config=_config())
    (root,) = _by_name(exporter, "llm_gateway.chat")
    assert root.attributes["error.type"]


@pytest.mark.asyncio
async def test_content_is_not_recorded_by_default_and_responses_never():
    exporter = await _chat_with_fallback()
    blob = repr([dict(s.attributes) for s in exporter.get_finished_spans()])
    assert "RESPONSE-SECRET" not in blob
    assert "gen_ai.input.messages" not in blob and "gen_ai.prompt" not in blob


async def _complete(config):
    tracer, exporter = _tracer()
    fake = AsyncMock(return_value=ProviderResult("RESPONSE-SECRET", "m-served", 4, 2))
    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch("llm_gateway.router.CALLS", {"anthropic": fake}),
    ):
        await complete_with_usage(system="s", prompt="PROMPT-TEXT", config=config)
    return exporter


@pytest.mark.asyncio
async def test_complete_default_records_no_prompt():
    exporter = await _complete(_config())
    (root,) = _by_name(exporter, "llm_gateway.complete")
    assert root.attributes["gen_ai.response.model"] == "m-served"
    assert root.attributes["gen_ai.provider.name"] == "anthropic"
    assert "PROMPT-TEXT" not in repr(dict(root.attributes))


@pytest.mark.asyncio
async def test_complete_opt_in_records_masked_prompt_only():
    exporter = await _complete(_config(trace_include_prompts=True))
    (root,) = _by_name(exporter, "llm_gateway.complete")
    msgs = json.loads(root.attributes["gen_ai.input.messages"])
    assert msgs[0]["parts"][0]["content"] == "PROMPT-TEXT"
    assert root.attributes["gen_ai.prompt"] == "PROMPT-TEXT"
    assert "RESPONSE-SECRET" not in repr(
        [dict(s.attributes) for s in exporter.get_finished_spans()]
    )


@pytest.mark.asyncio
async def test_request_max_tokens_on_call_spans():
    exporter = await _chat_with_fallback()
    (root,) = _by_name(exporter, "llm_gateway.chat")
    assert root.attributes["gen_ai.request.max_tokens"] == 2000
    exporter = await _complete(_config())
    (root,) = _by_name(exporter, "llm_gateway.complete")
    assert root.attributes["gen_ai.request.max_tokens"] == 2000


@pytest.mark.asyncio
async def test_complete_span_and_attempt_carry_finish_reason_and_response_model():
    tracer, exporter = _tracer()
    fake = AsyncMock(return_value=ProviderResult("x", "m-served", 4, 2, stop_reason="end_turn"))
    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch("llm_gateway.router.CALLS", {"anthropic": fake}),
    ):
        await complete_with_usage(system="s", prompt="p", config=_config())
    (root,) = _by_name(exporter, "llm_gateway.complete")
    assert tuple(root.attributes["gen_ai.response.finish_reasons"]) == ("end_turn",)
    (att,) = _by_name(exporter, "llm_gateway.attempt")
    assert tuple(att.attributes["gen_ai.response.finish_reasons"]) == ("end_turn",)
    assert att.attributes["gen_ai.response.model"] == "m-served"


@pytest.mark.asyncio
async def test_chat_attempt_span_has_finish_reason_only_on_success():
    exporter = await _chat_with_fallback()
    attempts = sorted(
        _by_name(exporter, "llm_gateway.attempt"), key=lambda s: s.attributes["llm_gateway.attempt"]
    )
    assert "gen_ai.response.finish_reasons" not in attempts[0].attributes
    assert tuple(attempts[1].attributes["gen_ai.response.finish_reasons"]) == ("stop",)
    assert attempts[1].attributes["gen_ai.response.model"] == "served-v2"
