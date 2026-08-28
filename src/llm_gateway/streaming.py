"""Streaming completion — same provider-chain + circuit breaker shape as
chat.py, with one fundamental difference: fallback can only happen *before*
the first chunk is yielded to the caller. Once a byte has been forwarded
over the wire, we're committed to that provider — there's no way to un-send
a partial response, so a mid-stream failure ends the stream rather than
silently retrying on another provider. This mirrors how every real
streaming proxy handles this (there's no other option); the pre-flight
here (pulling one item from the async generator before yielding anything)
is what makes normal "provider N is down" fallback still work for the
common case where a failure happens immediately (auth, connection, rate
limit) rather than mid-generation.

Unlike router.py/chat.py, this module doesn't use retry.call_with_retry on
the pre-flight pull: a retry that itself fails partway through opening the
stream would need the same "can we still cleanly restart" reasoning as
fallback does, and pre-flight failures are already the case the plain
provider-order loop below handles. Not worth the added complexity for a
"short" retry layer — revisit if pre-flight failures turn out to be common
enough in practice to matter.
"""

import time

from opentelemetry.trace import StatusCode

from . import breaker
from .config import GatewayConfig
from .providers import CONFIGURED, DEFAULT_MODEL, STREAM_CALLS
from .providers.base import StreamDelta
from .router import LLMError
from .tracing import get_tracer, set_chat_attributes


async def stream_chat(
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
):
    config = config or GatewayConfig()
    order = [force_provider] if force_provider else config.provider_order_list
    model_override = model if force_provider else None

    tracer = get_tracer()
    start = time.monotonic() * 1000
    attempted_any = False
    last_error: Exception | None = None

    with tracer.start_as_current_span("llm_gateway.stream_chat") as span:
        span.set_attribute("llm_gateway.fallback", False)

        for index, provider in enumerate(order):
            stream_fn = STREAM_CALLS.get(provider)
            if stream_fn is None:
                continue
            if not CONFIGURED[provider](config):
                continue
            if not force_provider and breaker.is_open(provider):
                continue

            attempted_any = True
            is_fallback = index > 0
            resolved_model = model_override or DEFAULT_MODEL[provider](config)
            generator = stream_fn(
                config,
                messages,
                tools,
                max_tokens,
                model=model_override,
                tool_choice=tool_choice,
                sampling=sampling,
                response_format=response_format,
            )

            try:
                first_delta = await generator.__anext__()
            except StopAsyncIteration:
                breaker.record_success(provider)
                return
            except Exception as e:  # noqa: BLE001 — pre-flight failure tries the next provider
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

            # Committed to this provider — no more silent fallback past this point.
            breaker.record_success(provider)
            input_tokens = output_tokens = 0
            finish_reason = "stop"
            tool_call_count = 0
            delta: StreamDelta | None = first_delta
            while delta is not None:
                if delta.tool_call_deltas:
                    tool_call_count += sum(1 for d in delta.tool_call_deltas if "id" in d)
                if delta.usage:
                    input_tokens, output_tokens = delta.usage
                if delta.finish_reason:
                    finish_reason = delta.finish_reason
                yield delta
                try:
                    delta = await generator.__anext__()
                except StopAsyncIteration:
                    delta = None

            latency = time.monotonic() * 1000 - start
            set_chat_attributes(
                span,
                config=config,
                provider=provider,
                model=resolved_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                fallback=is_fallback,
                tool_call_count=tool_call_count,
                finish_reason=finish_reason,
            )
            return

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
