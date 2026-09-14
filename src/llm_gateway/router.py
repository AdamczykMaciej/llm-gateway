"""The gateway's core: a configurable multi-provider fallback chain with a
per-provider circuit breaker and OTel tracing.

This is the engine — `llm_gateway.service` is a thin HTTP wrapper around it.
"""

import time

from opentelemetry.trace import StatusCode

from . import breaker
from .config import GatewayConfig
from .errors import LLMDeadlineExceeded, LLMError, deadline_exceeded
from .providers import CALLS, CONFIGURED, DEFAULT_MODEL
from .retry import Deadline, call_with_retry
from .tracing import get_tracer, set_call_attributes

__all__ = ["LLMError", "LLMDeadlineExceeded", "complete", "record_provider_failure"]


def record_provider_failure(
    provider: str, error: Exception, *, config: GatewayConfig, deadline: Deadline
) -> None:
    """Feed a provider's definitive failure (retries exhausted or not
    applicable) to its circuit breaker, per errors.POLICIES. Shared by all
    three engines. A timeout that only fired because the call's deadline ran
    out is not counted: it says more about earlier providers using up the
    budget than about this one."""
    if deadline.expired and isinstance(error, TimeoutError):
        return
    breaker.record_error(
        provider,
        error,
        threshold=config.breaker_failure_threshold,
        cooldown_seconds=config.breaker_cooldown_seconds,
    )


async def complete(
    *,
    system: str,
    prompt: str,
    max_tokens: int = 2000,
    config: GatewayConfig | None = None,
    force_provider: str | None = None,
    model: str | None = None,
) -> str:
    """Return a text completion, trying providers in `config.provider_order`.

    Pass `force_provider` (e.g. "anthropic") to call exactly that provider
    with no fallback — used by the HTTP service when a caller explicitly
    requests `model="<provider>/<model>"`. `model` only has an effect
    together with `force_provider`: a model name only means something within
    one provider's namespace, so it's meaningless across a multi-provider
    fallback chain — each provider in the chain always uses its own
    configured default model there.

    Raises `LLMError` when no provider succeeds, or its subclass
    `LLMDeadlineExceeded` when `config.call_deadline_seconds` runs out first.
    """
    config = config or GatewayConfig()
    order = [force_provider] if force_provider else config.provider_order_list
    model_override = model if force_provider else None

    tracer = get_tracer()
    start = time.monotonic() * 1000
    deadline = Deadline(config.call_deadline_seconds)
    attempted_any = False
    last_error: Exception | None = None

    with tracer.start_as_current_span("llm_gateway.complete") as span:
        span.set_attribute("llm_gateway.fallback", False)

        for index, provider in enumerate(order):
            call = CALLS.get(provider)
            if call is None:
                continue
            if not CONFIGURED[provider](config):
                continue
            if not force_provider and breaker.is_open(provider):
                continue
            if deadline.expired:
                break

            attempted_any = True
            is_fallback = index > 0
            resolved_model = model_override or DEFAULT_MODEL[provider](config)

            try:
                result = await call_with_retry(
                    lambda call=call: call(config, system, prompt, max_tokens, model_override),
                    attempts=config.retry_attempts,
                    base_delay_seconds=config.retry_base_delay_seconds,
                    attempt_timeout_seconds=config.request_timeout_seconds,
                    deadline=deadline,
                )
            except Exception as e:  # noqa: BLE001 — any provider failure (after its retries) tries the next
                record_provider_failure(provider, e, config=config, deadline=deadline)
                last_error = e
                latency = time.monotonic() * 1000 - start
                set_call_attributes(
                    span,
                    config=config,
                    provider=provider,
                    model=resolved_model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency,
                    fallback=is_fallback,
                    system=system,
                    prompt=prompt,
                    error_code=type(e).__name__,
                )
                continue

            breaker.record_success(provider)
            latency = time.monotonic() * 1000 - start
            set_call_attributes(
                span,
                config=config,
                provider=provider,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=latency,
                fallback=is_fallback,
                system=system,
                prompt=prompt,
            )
            return result.text

        if deadline.expired:
            span.set_status(StatusCode.ERROR, "call deadline exceeded")
            raise deadline_exceeded(config.call_deadline_seconds, last_error) from last_error
        span.set_status(
            StatusCode.ERROR,
            str(last_error) if last_error else "No provider available",
        )
        if not attempted_any:
            raise LLMError(
                "No LLM provider available. Set ANTHROPIC_API_KEY, GROQ_API_KEY, or "
                "OPENAI_API_KEY, matching provider_order."
            )
        raise LLMError(f"All configured providers failed. Last error: {last_error}") from last_error
