"""Generic OpenTelemetry tracing — no vendor SDK dependency.

Any OTLP-compatible collector (Langfuse, Grafana Cloud, Honeycomb, a local
collector, ...) can ingest these spans by setting the standard
OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_HEADERS env vars on the
process. If those aren't set, spans are simply not exported anywhere — the
SDK's default no-op behavior — so tracing never has to be "enabled" for the
gateway to work.
"""

import asyncio
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace

from .config import GatewayConfig
from .errors import ErrorKind, InvalidRequestError, classify
from .pii import mask_pii

_tracer: trace.Tracer | None = None


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("llm_gateway")
    return _tracer


@contextmanager
def call_span(tracer: trace.Tracer, name: str) -> Iterator[trace.Span]:
    """The current span for one gateway call. OpenTelemetry's default would
    record an escaping exception on the span (message, stack trace and the
    chained provider error, which can echo request content) and use its text
    as the status description. Here the status carries the exception type
    only and no exception event is recorded."""
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(trace.StatusCode.ERROR, type(exc).__name__)
            raise


def set_call_attributes(
    span: trace.Span,
    *,
    config: GatewayConfig,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    fallback: bool,
    system: str,
    prompt: str = "",
    error_code: str | None = None,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> None:
    """Set attributes on an in-progress span. Never raises — a tracing bug
    must not break the actual LLM call."""
    try:
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        # The reasoning share of output_tokens (already included in it).
        span.set_attribute("llm_gateway.usage.reasoning_tokens", reasoning_tokens)
        # Subsets of input_tokens (see providers.base.Usage). Counts only,
        # never prompt text.
        span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read_input_tokens)
        span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation_input_tokens)
        span.set_attribute("llm_gateway.latency_ms", round(latency_ms, 1))
        span.set_attribute("llm_gateway.fallback", fallback)
        if error_code:
            span.set_attribute("llm_gateway.error_code", error_code)
        if config.trace_include_prompts and prompt:
            span.set_attribute("gen_ai.prompt", mask_pii(prompt[:2000]))
    except Exception:  # noqa: BLE001 — tracing must never break the real request
        pass


def set_chat_attributes(
    span: trace.Span,
    *,
    config: GatewayConfig,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    fallback: bool,
    tool_call_count: int = 0,
    finish_reason: str = "",
    error_code: str | None = None,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> None:
    """Same as set_call_attributes, for the tool-calling chat() path — which
    has no single "prompt" string to (optionally) trace, just a message list."""
    try:
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        # The reasoning share of output_tokens (already included in it).
        span.set_attribute("llm_gateway.usage.reasoning_tokens", reasoning_tokens)
        # Subsets of input_tokens (see providers.base.Usage). Counts only,
        # never prompt text.
        span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read_input_tokens)
        span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation_input_tokens)
        span.set_attribute("llm_gateway.latency_ms", round(latency_ms, 1))
        span.set_attribute("llm_gateway.fallback", fallback)
        span.set_attribute("llm_gateway.tool_call_count", tool_call_count)
        if finish_reason:
            span.set_attribute("llm_gateway.finish_reason", finish_reason)
        if error_code:
            span.set_attribute("llm_gateway.error_code", error_code)
    except Exception:  # noqa: BLE001 — tracing must never break the real request
        pass


_REQUEST_ID = re.compile(r"[A-Za-z0-9._:\-]{1,128}")


def resolve_request_id(request_id: object = None) -> str:
    """The caller's `request_id` if it is a short token (letters, digits and
    `. _ : -`, at most 128 characters), else a fresh random one. A bad value
    raises `InvalidRequestError` rather than being silently replaced, so a
    caller's correlation id never quietly goes missing."""
    if request_id is None:
        return uuid.uuid4().hex
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        raise InvalidRequestError(
            "request_id must be a string of 1-128 letters, digits or . _ : - characters."
        )
    return request_id


def outcome_of(error: BaseException | None) -> str:
    """ "ok", "cancelled" (the caller cancelled or closed the stream: not a
    provider failure), or the `ErrorKind` value of the failure (e.g. "timeout")."""
    if isinstance(error, asyncio.CancelledError | GeneratorExit):
        return "cancelled"
    return "ok" if error is None else str(classify(error) or ErrorKind.UNKNOWN)


def emit_attempt_span(
    tracer: trace.Tracer,
    *,
    request_id: str,
    attempt: int,
    provider: str,
    model: str,
    started_ns: int,
    error: BaseException | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    restart: int = 0,
) -> None:
    """Record one provider attempt as a child span of the current call span.

    Carries identifiers, the outcome class and counts only: never message
    content, error text or keys. Never raises."""
    try:
        span = tracer.start_span("llm_gateway.attempt", start_time=started_ns)
        span.set_attribute("llm_gateway.request_id", request_id)
        span.set_attribute("llm_gateway.attempt", attempt)
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("llm_gateway.outcome", outcome_of(error))
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        if restart:
            span.set_attribute("llm_gateway.restart", restart)
        now = time.time_ns()
        span.set_attribute("llm_gateway.latency_ms", round((now - started_ns) / 1e6, 1))
        span.end(end_time=now)
    except Exception:  # noqa: BLE001 — tracing must never break the real request
        pass
