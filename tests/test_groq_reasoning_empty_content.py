"""0.4.1: Groq reasoning models and empty completions, through the *real*
openai SDK pointed at Groq's base URL over an in-memory transport.

Groq retired llama-3.3-70b-versatile; its replacement `openai/gpt-oss-120b` is
a reasoning model. At Groq's default (medium) reasoning effort a small
`max_tokens` can be spent entirely on reasoning, which Groq returns as empty
`content`, a `reasoning` field and `finish_reason="length"`. 0.4.0 returned
that as "" from `complete()`.
"""

import contextlib
import json
import logging
from unittest.mock import patch

import httpx2
import openai as openai_sdk
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from llm_gateway import (
    EmptyCompletionError,
    GatewayConfig,
    LLMError,
    breaker,
    chat,
    complete,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.errors import POLICIES, ErrorKind, classify
from llm_gateway.providers import anthropic, groq, openai
from tests.test_sdk_compat import (
    _ANTHROPIC_MESSAGE,
    _anthropic_client,
    _openai_client,
    _Recorder,
    _sse,
)

GPT_OSS = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
REASONING = "Let me weigh each competency against the answer before scoring ..."


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _groq_client(recorder: _Recorder) -> openai_sdk.AsyncOpenAI:
    return openai_sdk.AsyncOpenAI(
        api_key="test-key",
        base_url=GROQ_BASE_URL,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)),
    )


def _completion(message: dict, *, finish_reason="stop", usage=None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": GPT_OSS,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": usage
        or {
            "prompt_tokens": 120,
            "completion_tokens": 260,
            "total_tokens": 380,
            "completion_tokens_details": {"reasoning_tokens": 200},
        },
    }


def _reply(message: dict, **kwargs):
    body = _completion(message, **kwargs)
    return lambda req: httpx2.Response(200, request=req, json=body)


_BUDGET_SPENT = {
    "prompt_tokens": 900,
    "completion_tokens": 300,
    "total_tokens": 1200,
    "completion_tokens_details": {"reasoning_tokens": 300},
}


def _reasoning_only(content=""):
    return _reply(
        {"role": "assistant", "content": content, "reasoning": REASONING},
        finish_reason="length",
        usage=_BUDGET_SPENT,
    )


def _openai_fallback():
    return _reply({"role": "assistant", "content": "from openai"})


def _chain(**overrides) -> GatewayConfig:
    settings = dict(
        groq_api_key="k",
        openai_api_key="k",
        groq_model=GPT_OSS,
        provider_order="groq,openai",
        retry_attempts=2,
        breaker_failure_threshold=1,
    )
    settings.update(overrides)
    return GatewayConfig(**settings)


@contextlib.contextmanager
def _with(groq_recorder: _Recorder, openai_recorder: _Recorder | None = None):
    """Groq through a real SDK client at Groq's base URL; OpenAI as fallback."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(groq, "_client", return_value=_groq_client(groq_recorder)))
        if openai_recorder is not None:
            stack.enter_context(
                patch.object(openai, "_client", return_value=_openai_client(openai_recorder))
            )
        yield


class Verdict(BaseModel):
    score: int
    summary: str


# ─── Empty content is a counted, non-retried provider failure ─────────────


@pytest.mark.parametrize("content", ["", None, "  \n\t "])
async def test_empty_content_fails_over_counts_for_breaker_and_is_not_retried(content, caplog):
    primary = _Recorder(_reasoning_only(content))
    fallback = _Recorder(_openai_fallback())
    with _with(primary, fallback), caplog.at_level(logging.WARNING, logger="llm_gateway"):
        result = await complete_with_usage(system="s", prompt="p", max_tokens=300, config=_chain())

    assert (result.provider, result.text) == ("openai", "from openai")
    assert len(primary.bodies) == 1  # retry_attempts=2, but not retried
    assert breaker.is_open("groq")  # threshold 1: the failure was counted
    warning = next(r for r in caplog.records if "empty completion" in r.getMessage())
    assert warning.levelno == logging.WARNING
    message = warning.getMessage()
    assert "provider=groq" in message and f"model={GPT_OSS}" in message
    assert "finish_reason=length" in message and "reasoning_tokens=300" in message
    assert REASONING not in message


async def test_empty_content_everywhere_raises_llm_error_with_diagnostics_only():
    primary = _Recorder(_reasoning_only())
    config = _chain(provider_order="groq")
    with _with(primary), pytest.raises(LLMError) as excinfo:
        await complete(system="s", prompt="p", max_tokens=300, config=config)

    cause = excinfo.value.__cause__
    assert isinstance(cause, EmptyCompletionError)
    assert classify(cause) is ErrorKind.EMPTY_RESPONSE
    assert POLICIES[ErrorKind.EMPTY_RESPONSE].retry is False
    assert POLICIES[ErrorKind.EMPTY_RESPONSE].trips_breaker is True
    assert (cause.provider, cause.model, cause.finish_reason, cause.reasoning_tokens) == (
        "groq",
        GPT_OSS,
        "length",
        300,
    )
    assert REASONING not in str(excinfo.value)


async def test_chat_empty_content_fails_over():
    primary = _Recorder(_reasoning_only())
    fallback = _Recorder(_openai_fallback())
    with _with(primary, fallback):
        result = await chat(messages=[{"role": "user", "content": "hi"}], config=_chain())
    assert result.content == "from openai"
    assert breaker.is_open("groq")


async def test_chat_tool_calls_with_null_content_are_fine():
    tool_reply = _reply(
        {
            "role": "assistant",
            "content": None,
            "reasoning": REASONING,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                }
            ],
        },
        finish_reason="tool_calls",
    )
    primary = _Recorder(tool_reply)
    with _with(primary):
        result = await chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            config=_chain(provider_order="groq"),
        )

    assert result.content is None
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("lookup", {"q": "x"})]
    assert primary.bodies[-1]["reasoning_effort"] == "low"
    assert not breaker.is_open("groq")


def _groq_stream(chunks: list[dict]):
    return lambda req: httpx2.Response(
        200,
        request=req,
        headers={"content-type": "text/event-stream"},
        content=_sse(chunks, named=False),
    )


_CHUNK = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": GPT_OSS}


async def test_stream_with_only_reasoning_fails_over_before_the_first_chunk():
    reasoning_only = [
        {**_CHUNK, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {**_CHUNK, "choices": [{"index": 0, "delta": {"reasoning": "Thinking ..."}}]},
        {**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
        {**_CHUNK, "choices": [], "usage": _BUDGET_SPENT},
    ]
    fallback_chunks = [
        {**_CHUNK, "choices": [{"index": 0, "delta": {"content": "from openai"}}]},
        {**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    primary = _Recorder(_groq_stream(reasoning_only))
    fallback = _Recorder(_groq_stream(fallback_chunks))
    with _with(primary, fallback):
        deltas = [
            d
            async for d in stream_chat(
                messages=[{"role": "user", "content": "hi"}], config=_chain()
            )
        ]

    assert "".join(d.content or "" for d in deltas) == "from openai"
    assert primary.bodies[-1]["stream"] is True
    assert primary.bodies[-1]["reasoning_effort"] == "low"
    assert len(primary.bodies) == 1
    assert breaker.is_open("groq")


async def test_stream_keeps_leading_whitespace_chunks_once_text_arrives():
    chunks = [
        {**_CHUNK, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {**_CHUNK, "choices": [{"index": 0, "delta": {"content": "\n"}}]},
        {**_CHUNK, "choices": [{"index": 0, "delta": {"content": "Hello"}}]},
        {**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    with _with(_Recorder(_groq_stream(chunks))):
        deltas = [
            d
            async for d in stream_chat(
                messages=[{"role": "user", "content": "hi"}],
                config=_chain(provider_order="groq"),
            )
        ]
    assert "".join(d.content or "" for d in deltas) == "\nHello"


async def test_anthropic_thinking_only_reply_is_an_empty_completion():
    body = {
        **_ANTHROPIC_MESSAGE,
        "content": [{"type": "thinking", "thinking": "Plan.", "signature": "sig"}],
        "stop_reason": "max_tokens",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 50,
            "output_tokens_details": {"thinking_tokens": 50},
        },
    }
    recorder = _Recorder(lambda req: httpx2.Response(200, request=req, json=body))
    config = GatewayConfig(anthropic_api_key="k")
    with (
        patch.object(anthropic, "_client", return_value=_anthropic_client(recorder)),
        pytest.raises(EmptyCompletionError) as excinfo,
    ):
        await anthropic.call(config, "s", "p", 50)
    assert (excinfo.value.finish_reason, excinfo.value.reasoning_tokens) == ("max_tokens", 50)


# ─── reasoning_effort ─────────────────────────────────────────────────────


async def test_gpt_oss_json_reply_sends_low_reasoning_effort_and_reports_reasoning_tokens():
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reply = _reply(
        {
            "role": "assistant",
            "content": json.dumps({"score": 80, "summary": "Solid."}),
            "reasoning": REASONING,
        }
    )
    recorder = _Recorder(reply)
    with (
        _with(recorder),
        patch("llm_gateway.router.get_tracer", return_value=tracer_provider.get_tracer("t")),
    ):
        result = await complete_with_usage(
            system="s",
            prompt="p",
            max_tokens=300,
            config=_chain(provider_order="groq"),
            output_schema=Verdict,
        )

    assert result.parsed == Verdict(score=80, summary="Solid.")
    body = recorder.bodies[-1]
    assert body["reasoning_effort"] == "low"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    # Reasoning tokens are part of output_tokens, and reported separately.
    assert (result.usage.output_tokens, result.usage.reasoning_tokens) == (260, 200)
    (span,) = exporter.get_finished_spans()
    assert span.attributes["gen_ai.usage.output_tokens"] == 260
    assert span.attributes["llm_gateway.usage.reasoning_tokens"] == 200


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("openai/gpt-oss-120b", "low"),
        ("openai/gpt-oss-20b", "low"),
        ("gpt-oss-120b", "low"),
        ("qwen/qwen3.8-27b", "low"),
        ("qwen/qwen3.6-27b", None),  # accepts only none/default
        ("llama-3.1-8b-instant", None),
        ("moonshotai/kimi-k2-instruct", None),
    ],
)
async def test_reasoning_effort_is_sent_only_to_groq_reasoning_models(model, effort, caplog):
    recorder = _Recorder(_reply({"role": "assistant", "content": "ok"}))
    config = _chain(provider_order="groq", groq_model=model)
    with _with(recorder), caplog.at_level(logging.DEBUG, logger="llm_gateway"):
        await complete(system="s", prompt="p", config=config)

    assert recorder.bodies[-1].get("reasoning_effort") == effort
    served = next(r for r in caplog.records if "llm_gateway served" in r.getMessage())
    assert served.levelno == logging.DEBUG
    assert "reasoning_tokens=200" in served.getMessage()


async def test_openai_provider_never_gets_reasoning_effort():
    recorder = _Recorder(_reply({"role": "assistant", "content": "ok"}))
    config = GatewayConfig(openai_api_key="k", provider_order="openai", retry_attempts=1)
    with patch.object(openai, "_client", return_value=_openai_client(recorder)):
        await complete(system="s", prompt="p", config=config)
    assert "reasoning_effort" not in recorder.bodies[-1]


def test_strict_json_schema_match_covers_groqs_gpt_oss_ids():
    assert GPT_OSS.startswith(groq.STRICT_JSON_SCHEMA_MODELS)
    assert "openai/gpt-oss-20b".startswith(groq.STRICT_JSON_SCHEMA_MODELS)


# ─── 0.4.2: the default Groq model ────────────────────────────────────────


async def test_groq_default_model_is_not_the_retired_llama_and_gets_low_effort(monkeypatch):
    # Groq retired llama-3.3-70b-versatile on 2026-08-16.
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    defaults = GatewayConfig(_env_file=None)
    assert defaults.groq_model != "llama-3.3-70b-versatile"
    assert defaults.groq_model == GPT_OSS

    recorder = _Recorder(_reply({"role": "assistant", "content": "ok"}))
    config = GatewayConfig(
        _env_file=None, groq_api_key="k", provider_order="groq", retry_attempts=1
    )
    with _with(recorder):
        await complete(system="s", prompt="p", config=config)
    assert recorder.bodies[-1]["model"] == GPT_OSS
    assert recorder.bodies[-1]["reasoning_effort"] == "low"
