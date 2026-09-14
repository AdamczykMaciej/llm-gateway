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

The pre-flight pull goes through retry.call_with_retry like the other
engines: nothing has reached the caller yet, so re-opening the stream on a
transient error is exactly as safe as falling over to the next provider.
That matters now that the SDKs' own retries are off (sdk_max_retries=0).

Time bounds: the SDK's read timeout (stream_idle_timeout_seconds) caps the
wait for the stream to start and every stall inside it; the gateway itself
adds no per-attempt bound, because a provider may legitimately withhold the
first normalized chunk until generation ends (Anthropic's structured-output
emulation does). call_deadline_seconds bounds the whole call — including a
stream that has already started, which then ends with LLMDeadlineExceeded.
"""

import asyncio
import dataclasses
import logging
import time

from opentelemetry.trace import StatusCode

from . import breaker
from .config import GatewayConfig
from .errors import LLMError, deadline_exceeded
from .providers import CONFIGURED, DEFAULT_MODEL, STREAM_CALLS
from .providers.base import StreamDelta
from .retry import Deadline, call_with_retry
from .router import record_provider_failure
from .tracing import get_tracer, set_chat_attributes

logger = logging.getLogger("llm_gateway")


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
    deadline = Deadline(config.call_deadline_seconds)
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
            if deadline.expired:
                break

            attempted_any = True
            is_fallback = index > 0
            resolved_model = model_override or DEFAULT_MODEL[provider](config)

            async def open_stream(stream_fn=stream_fn):
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
                    return generator, await generator.__anext__()
                except StopAsyncIteration:
                    return generator, None

            try:
                generator, first_delta = await call_with_retry(
                    open_stream,
                    attempts=config.retry_attempts,
                    base_delay_seconds=config.retry_base_delay_seconds,
                    deadline=deadline,
                )
            except Exception as e:  # noqa: BLE001 — pre-flight failure tries the next provider
                record_provider_failure(provider, e, config=config, deadline=deadline)
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
            if first_delta is None:
                return
            input_tokens = output_tokens = 0
            cache_read_input_tokens = cache_creation_input_tokens = reasoning_tokens = 0
            finish_reason = "stop"
            tool_call_count = 0
            delta: StreamDelta | None = first_delta
            while delta is not None:
                if delta.tool_call_deltas:
                    tool_call_count += sum(1 for d in delta.tool_call_deltas if "id" in d)
                if delta.usage:
                    input_tokens, output_tokens = delta.usage
                    if delta.usage_details:
                        cache_read_input_tokens = delta.usage_details.cache_read_input_tokens
                        cache_creation_input_tokens = (
                            delta.usage_details.cache_creation_input_tokens
                        )
                        reasoning_tokens = delta.usage_details.reasoning_tokens
                    # Attribute the usage to the provider that actually served.
                    delta = dataclasses.replace(delta, provider=provider)
                if delta.finish_reason:
                    finish_reason = delta.finish_reason
                yield delta
                try:
                    async with asyncio.timeout(deadline.remaining()):
                        delta = await generator.__anext__()
                except StopAsyncIteration:
                    delta = None
                except TimeoutError as e:
                    if not deadline.expired:
                        raise
                    await generator.aclose()
                    span.set_status(StatusCode.ERROR, "call deadline exceeded")
                    raise deadline_exceeded(config.call_deadline_seconds, e) from e

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
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                reasoning_tokens=reasoning_tokens,
            )
            logger.debug(
                "llm_gateway served stream: provider=%s model=%s input_tokens=%s "
                "output_tokens=%s reasoning_tokens=%s",
                provider,
                resolved_model,
                input_tokens,
                output_tokens,
                reasoning_tokens,
            )
            return

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
