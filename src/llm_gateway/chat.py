"""Tool-calling-capable completion — the engine behind ReAct-style agents.

Structurally parallel to router.complete() (same provider-chain + circuit
breaker shape), kept as a separate function rather than unified with it:
`complete()` returns plain text for simple callers, `chat()` returns a
`ChatResult` that may carry tool_calls, and the two have genuinely
different provider-call shapes (`system, prompt` vs a full `messages`
list). Both share the same breaker state (`breaker.py`) and provider
registries (`CONFIGURED`, `DEFAULT_MODEL`), so a provider tripped by one
path is correctly skipped by the other too.
"""

import time

from opentelemetry.trace import StatusCode

from . import breaker
from .config import GatewayConfig
from .providers import CHAT_CALLS, CONFIGURED, DEFAULT_MODEL
from .providers.base import ChatResult
from .retry import call_with_retry
from .router import LLMError
from .tracing import get_tracer, set_chat_attributes


async def chat(
    *,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: object = None,
    max_tokens: int = 2000,
    config: GatewayConfig | None = None,
    force_provider: str | None = None,
    model: str | None = None,
    sampling: dict | None = None,
    response_format: dict | None = None,
) -> ChatResult:
    """Return a ChatResult (text and/or tool_calls), trying providers in
    `config.provider_order` — same fallback/circuit-breaker semantics as
    `router.complete()`. `messages` and `tools` are OpenAI-wire-shaped
    dicts; each provider translates internally (see providers/anthropic.py
    for the one that actually needs translating).
    """
    config = config or GatewayConfig()
    order = [force_provider] if force_provider else config.provider_order_list
    model_override = model if force_provider else None

    tracer = get_tracer()
    start = time.monotonic() * 1000
    attempted_any = False
    last_error: Exception | None = None

    with tracer.start_as_current_span("llm_gateway.chat") as span:
        span.set_attribute("llm_gateway.fallback", False)

        for index, provider in enumerate(order):
            call = CHAT_CALLS.get(provider)
            if call is None:
                continue
            if not CONFIGURED[provider](config):
                continue
            if not force_provider and breaker.is_open(provider):
                continue

            attempted_any = True
            is_fallback = index > 0
            resolved_model = model_override or DEFAULT_MODEL[provider](config)

            try:
                result = await call_with_retry(
                    lambda call=call: call(
                        config,
                        messages,
                        tools,
                        max_tokens,
                        model=model_override,
                        tool_choice=tool_choice,
                        sampling=sampling,
                        response_format=response_format,
                    ),
                    attempts=config.retry_attempts,
                    base_delay_seconds=config.retry_base_delay_seconds,
                )
            except Exception as e:  # noqa: BLE001 — any provider failure (after its retries) tries the next
                breaker.record_failure(
                    provider,
                    threshold=config.breaker_failure_threshold,
                    cooldown_seconds=config.breaker_cooldown_seconds,
                )
                last_error = e
                latency = time.monotonic() * 1000 - start
                set_chat_attributes(
                    span,
                    config=config,
                    provider=provider,
                    model=resolved_model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency,
                    fallback=is_fallback,
                    error_code=type(e).__name__,
                )
                continue

            breaker.record_success(provider)
            latency = time.monotonic() * 1000 - start
            set_chat_attributes(
                span,
                config=config,
                provider=provider,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=latency,
                fallback=is_fallback,
                tool_call_count=len(result.tool_calls),
                finish_reason=result.finish_reason,
            )
            return result

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
