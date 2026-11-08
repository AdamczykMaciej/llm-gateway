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

After the first chunk the provider's generator is always closed when the
engine generator ends — normally, on error, on consumer close or cancellation
— so an abandoned stream never keeps its upstream request open. A mid-stream
provider failure is recorded against the circuit breaker, then re-raised.
"""

import asyncio
import dataclasses
import logging
import time
from collections.abc import AsyncGenerator

from opentelemetry.trace import StatusCode

from . import breaker
from .config import GatewayConfig
from .errors import (
    AllProvidersExhaustedError,
    GatewayNotConfiguredError,
    LLMError,
    deadline_exceeded,
    redact_secrets,
)
from .policy import ProviderPolicy
from .providers import CONFIGURED, DEFAULT_MODEL, STREAM_CALLS
from .providers.base import StreamDelta
from .retry import Deadline, call_with_retry
from .router import record_provider_failure
from .routing import RequestProfile, plan_chain
from .tool_results import cap_tool_results, resolve_cap
from .tool_validation import validate_chat_request
from .tracing import emit_attempt_span, get_tracer, resolve_request_id, set_chat_attributes

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
    policy: ProviderPolicy | None = None,
    cache_system: bool = False,
    request_id: str | None = None,
    parallel_tool_calls: bool | None = None,
    max_tool_result_chars: int | None = None,
) -> AsyncGenerator[StreamDelta, None]:
    """Stream a chat completion. `policy` narrows `config.policy` as in
    `complete_with_usage()`; the chain is filtered before the first provider
    is opened, so pre-flight failover never reaches an excluded provider, and
    `PolicyViolationError` is raised before any chunk when none is left.

    `cache_system`: prompt caching, as in `complete_with_usage()`. On
    Anthropic and Vertex Claude the **first** system message is the cacheable
    static prefix (it gets `cache_control`); pass the per-request part as a
    second system message. Other providers ignore it. Cache hits and writes
    arrive in the final chunk's `usage_details`.

    `request_id`, `parallel_tool_calls` and `max_tool_result_chars` behave as
    in `chat()`; the request id arrives on the final (usage) chunk and is on
    every span."""
    rid = resolve_request_id(request_id)
    validate_chat_request(messages, tools, tool_choice)
    config = config or GatewayConfig()
    messages = cap_tool_results(
        messages, resolve_cap(max_tool_result_chars, config.tool_result_max_chars)
    )
    # Only passed when used, so provider callables written against the older
    # signature keep working.
    extra: dict = {"cache_system": True} if cache_system else {}
    if parallel_tool_calls is not None:
        extra["parallel_tool_calls"] = parallel_tool_calls
    order = [force_provider] if force_provider else config.provider_order_list
    model_override = model if force_provider else None

    tracer = get_tracer()
    start = time.monotonic() * 1000
    deadline = Deadline(config.call_deadline_seconds)
    attempted_any = False
    last_error: Exception | None = None

    with tracer.start_as_current_span("llm_gateway.stream_chat") as span:
        span.set_attribute("llm_gateway.fallback", False)
        span.set_attribute("llm_gateway.request_id", rid)
        attempt_no = 0
        plan = plan_chain(
            order,
            config=config,
            policy=policy,
            profile=RequestProfile.for_chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                response_format=response_format,
                streaming=True,
            ),
            registry=STREAM_CALLS,
            model_override=model_override,
            span=span,
        )

        for index, provider in enumerate(plan.providers):
            stream_fn = STREAM_CALLS.get(provider)
            if stream_fn is None:
                continue
            if not CONFIGURED[provider](config):
                continue
            resolved_model = model_override or DEFAULT_MODEL[provider](config)
            if not force_provider and breaker.is_open(provider, resolved_model):
                continue
            if deadline.expired:
                break
            if not await plan.allows(provider, span):
                continue
            if not force_provider and not breaker.allow(provider, resolved_model):
                continue

            attempted_any = True
            is_fallback = index > 0
            attempt_no += 1
            attempt_started_ns = time.time_ns()

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
                    **extra,
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
                    attempt_timeout_seconds=config.stream_idle_timeout_seconds,
                    deadline=deadline,
                )
            except Exception as e:  # noqa: BLE001 — pre-flight failure tries the next provider
                record_provider_failure(
                    provider, e, config=config, deadline=deadline, model=resolved_model
                )
                last_error = e
                emit_attempt_span(
                    tracer,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    started_ns=attempt_started_ns,
                    error=e,
                )
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
            breaker.record_success(provider, resolved_model)
            input_tokens = output_tokens = 0
            stream_error: BaseException | None = None
            if first_delta is None:
                emit_attempt_span(
                    tracer,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    started_ns=attempt_started_ns,
                )
                return
            cache_read_input_tokens = cache_creation_input_tokens = reasoning_tokens = 0
            finish_reason = "stop"
            tool_call_count = 0
            try:
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
                        delta = dataclasses.replace(delta, provider=provider, request_id=rid)
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
            except LLMError as e:
                stream_error = e
                raise
            except Exception as e:  # noqa: BLE001 — a mid-stream failure: counted, then propagated
                stream_error = e
                record_provider_failure(
                    provider, e, config=config, deadline=deadline, model=resolved_model
                )
                raise
            finally:
                # The attempt span is written however the stream ends: done,
                # failed, or abandoned by the consumer.
                emit_attempt_span(
                    tracer,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    started_ns=attempt_started_ns,
                    error=stream_error if isinstance(stream_error, Exception) else None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                # Consumer stopped reading (close, cancel, disconnect) or the
                # stream failed: close the provider's generator so its HTTP
                # response is released and the upstream request stops.
                await generator.aclose()

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
            raise deadline_exceeded(
                config.call_deadline_seconds, last_error, config
            ) from last_error
        span.set_status(
            StatusCode.ERROR,
            redact_secrets(str(last_error), config) if last_error else "No provider available",
        )
        if not attempted_any:
            if plan.denied:
                raise plan.violation()
            if plan.any_configured(config):
                raise AllProvidersExhaustedError(
                    "Every configured provider is currently unavailable (circuit breaker "
                    "open, or the call deadline ran out before any attempt). This is very "
                    "likely temporary; retrying after BREAKER_COOLDOWN_SECONDS is reasonable."
                )
            raise GatewayNotConfiguredError(
                "No LLM provider available. Set ANTHROPIC_API_KEY, GROQ_API_KEY, "
                "OPENAI_API_KEY, AZURE_ENDPOINT and AZURE_MODEL, or VERTEX_PROJECT_ID, "
                "matching provider_order."
            )
        raise AllProvidersExhaustedError(
            "All configured providers failed. Last error: "
            f"{redact_secrets(str(last_error), config)}{plan.failure_note()}"
        ) from last_error
