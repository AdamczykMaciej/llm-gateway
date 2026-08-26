"""Generic OpenTelemetry tracing — no vendor SDK dependency.

Any OTLP-compatible collector (Langfuse, Grafana Cloud, Honeycomb, a local
collector, ...) can ingest these spans by setting the standard
OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_HEADERS env vars on the
process. If those aren't set, spans are simply not exported anywhere — the
SDK's default no-op behavior — so tracing never has to be "enabled" for the
gateway to work.
"""

from opentelemetry import trace

from .config import GatewayConfig
from .pii import mask_pii

_tracer: trace.Tracer | None = None


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("llm_gateway")
    return _tracer


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
) -> None:
    """Set attributes on an in-progress span. Never raises — a tracing bug
    must not break the actual LLM call."""
    try:
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("llm_gateway.latency_ms", round(latency_ms, 1))
        span.set_attribute("llm_gateway.fallback", fallback)
        if error_code:
            span.set_attribute("llm_gateway.error_code", error_code)
        if config.trace_include_prompts and prompt:
            span.set_attribute("gen_ai.prompt", mask_pii(prompt[:2000]))
    except Exception:  # noqa: BLE001 — tracing must never break the real request
        pass
