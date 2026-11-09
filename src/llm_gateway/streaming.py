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
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from typing import Any, cast

from opentelemetry.trace import StatusCode

from . import breaker
from .attempts import AttemptHook, report_attempt, usage_of
from .config import GatewayConfig
from .errors import (
    AllProvidersExhaustedError,
    ErrorKind,
    FirstTokenTimeoutError,
    GatewayNotConfiguredError,
    LLMError,
    ProviderFailure,
    StreamTooLargeError,
    all_failed,
    classify,
    deadline_exceeded,
)
from .idempotency import IDEMPOTENCY_HEADER_PROVIDERS, stream_slot
from .policy import ProviderPolicy
from .providers import CONFIGURED, DEFAULT_MODEL, STREAM_CALLS
from .providers.base import StreamDelta, Usage
from .retry import Deadline, call_with_retry
from .router import record_provider_failure
from .routing import RequestProfile, plan_chain
from .tool_results import cap_tool_results, resolve_cap
from .tool_validation import validate_chat_request
from .tracing import (
    call_span,
    emit_attempt_span,
    fallback_reason_of,
    get_tracer,
    resolve_idempotency_key,
    resolve_request_id,
    set_chat_attributes,
)
from .types import Message, ToolChoice, ToolSpec

logger = logging.getLogger("llm_gateway")


def _check_stream_size(
    delta: StreamDelta, config: GatewayConfig, total: int, arguments: dict[int, int]
) -> int:
    """Add `delta`'s size to the running totals; raise `StreamTooLargeError`
    past a cap. Returns the new stream total."""
    per_call = config.stream_max_tool_arguments_chars
    overall = config.stream_max_total_chars
    total += len(delta.content or "")
    for fragment in delta.tool_call_deltas or ():
        text = (fragment.get("function") or {}).get("arguments") or ""
        total += len(text)
        index = fragment.get("index", 0)
        arguments[index] = arguments.get(index, 0) + len(text)
        if per_call and arguments[index] > per_call:
            raise StreamTooLargeError(
                f"A streamed tool call's arguments exceeded {per_call} characters "
                "(stream_max_tool_arguments_chars).",
                limit=per_call,
            )
    if overall and total > overall:
        raise StreamTooLargeError(
            f"The streamed reply exceeded {overall} characters (stream_max_total_chars).",
            limit=overall,
        )
    return total


# Failure kinds that must never restart a stream: the next provider would be
# given the same request or would produce the same filtered/invalid output.
_NO_RESTART_KINDS = frozenset(
    {ErrorKind.INVALID_REQUEST, ErrorKind.INVALID_OUTPUT, ErrorKind.POLICY_VIOLATION}
)


def _restartable(error: BaseException) -> bool:
    """A mid-stream provider failure another provider could recover from.
    Gateway errors (`LLMError`: deadline, ...), cancellation and invalid
    request/output (content filter, oversize) are excluded."""
    if isinstance(error, LLMError):
        return False
    return classify(error) not in _NO_RESTART_KINDS


def _positive(seconds: float | None) -> float | None:
    return seconds if seconds is not None and seconds > 0 else None


async def stream_chat(
    *,
    messages: Sequence[Message | dict[str, Any]],
    tools: Sequence[ToolSpec | dict[str, Any]] | None = None,
    tool_choice: ToolChoice | dict[str, Any] | str | None = None,
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
    on_attempt: AttemptHook | None = None,
    first_token_timeout_seconds: float | None = None,
    restart_on_failure: bool | None = None,
    idempotency_key: str | None = None,
) -> AsyncGenerator[StreamDelta, None]:
    """Stream a chat completion.

    Yields `StreamDelta` chunks (text, tool-call fragments, a final chunk with
    usage and finish reason, and restart markers when `restart_on_failure`).

    `policy` narrows `config.policy` as in
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
    every span.

    `on_attempt`: a sync or async callable receiving an `AttemptReport` once
    per provider attempt; replaces `config.on_attempt` for this call. A stream
    that fails after its usage chunk, or is closed early, reports the usage
    seen so far (else `None`), and an early close is reported as "cancelled".

    `first_token_timeout_seconds`: fail an attempt that yields no first chunk
    within this many seconds with a `FirstTokenTimeoutError` (a builtin
    `TimeoutError`, so a provider timeout: counts for the breaker, no retry)
    and fail over to the next provider, which is safe because nothing was
    yielded. Overrides `config.first_token_timeout_seconds`; `None` there
    too means no such bound. It only tightens the wait for the first chunk,
    which `stream_idle_timeout_seconds` and `call_deadline_seconds` already
    cap: the smallest of the three applies, so a value above the idle bound
    changes nothing. After the first chunk only the idle bound (the
    SDK's read timeout) and the deadline apply.

    `restart_on_failure` (default: `config.stream_restart_on_failure`, off):
    when a provider fails after chunks were yielded (transport error, idle
    timeout, provider 5xx; never cancellation, a content filter, an oversize
    stream, an invalid request or the call deadline), continue on the next
    provider of the chain, at most `config.stream_max_restarts` (default 1)
    times. The new attempt starts with one `StreamDelta(restart=True)` marker
    (no content, no tool-call deltas; `provider`/`model` name the new
    provider, `restart_attempt`, `failed_provider`, `failed_error`), then
    streams from the beginning. The consumer must discard everything received
    before the marker, text and tool-call fragments alike. If no provider can
    take over, the original mid-stream error is raised. Every attempt is
    reported to `on_attempt` and gets its own span.

    `idempotency_key`: as in `chat()`: validated (`InvalidRequestError` before
        any chunk), put on the call span, every attempt span and every
        `AttemptReport`, and sent as the `Idempotency-Key` header to OpenAI
        only (also on a restart attempt, which streams the same request). A
        stream cannot be shared between consumers, so unlike `chat()` there is
        no de-duplication: a second `stream_chat()` with the same key while the
        first is still open (not finished, failed or closed) raises
        `InvalidRequestError` on first iteration, before any provider call.
        The key is free again once the first stream ends. Without a key,
        behaviour is unchanged.

    Raises:
        InvalidRequestError: malformed request (see `chat()`), before any chunk;
            or `idempotency_key` already in use by an open stream.
        PolicyViolationError: no provider satisfies the routing policy, before
            any chunk.
        LLMDeadlineExceeded: the call deadline ran out.
        GatewayNotConfiguredError, AllProvidersExhaustedError: no provider could
            open the stream. Once a chunk was yielded there is no failover
            (unless `restart_on_failure`), so a later failure propagates to the
            consumer.
    """
    ikey = resolve_idempotency_key(idempotency_key)
    impl = _stream_chat(
        messages=cast("list[dict]", messages),
        tools=cast("list[dict] | None", tools),
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        config=config,
        force_provider=force_provider,
        model=model,
        sampling=sampling,
        response_format=response_format,
        policy=policy,
        cache_system=cache_system,
        request_id=request_id,
        parallel_tool_calls=parallel_tool_calls,
        max_tool_result_chars=max_tool_result_chars,
        on_attempt=on_attempt,
        first_token_timeout_seconds=first_token_timeout_seconds,
        restart_on_failure=restart_on_failure,
        idempotency_key=ikey,
    )
    async with aclosing(impl) as chunks:
        if ikey is None:
            async for delta in chunks:
                yield delta
        else:
            with stream_slot(ikey):
                async for delta in chunks:
                    yield delta


async def _stream_chat(
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
    on_attempt: AttemptHook | None = None,
    first_token_timeout_seconds: float | None = None,
    restart_on_failure: bool | None = None,
    idempotency_key: str | None = None,
) -> AsyncGenerator[StreamDelta, None]:
    """The body of `stream_chat()`; see there for the contract."""
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
    ikey = idempotency_key
    order = [force_provider] if force_provider else config.provider_order_list
    model_override = model if force_provider else None
    first_token_limit = _positive(
        first_token_timeout_seconds
        if first_token_timeout_seconds is not None
        else config.first_token_timeout_seconds
    )

    restart_enabled = (
        restart_on_failure if restart_on_failure is not None else config.stream_restart_on_failure
    )
    max_restarts = max(config.stream_max_restarts, 0) if restart_enabled else 0
    restarts = 0
    # Set when an attempt died mid-stream and the next provider is to take over:
    # (failed provider, error). Cleared by the marker once a replacement opens.
    pending_restart: tuple[str, BaseException] | None = None
    midstream_error: BaseException | None = None

    tracer = get_tracer()
    start = time.monotonic() * 1000
    deadline = Deadline(config.call_deadline_seconds)
    attempted_any = False
    last_error: Exception | None = None
    failures: list[ProviderFailure] = []

    with call_span(tracer, "llm_gateway.stream_chat") as span:
        span.set_attribute("llm_gateway.fallback", False)
        span.set_attribute("llm_gateway.request_id", rid)
        if idempotency_key:
            span.set_attribute("llm_gateway.idempotency_key", idempotency_key)
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
            breaker_state = breaker.state(provider, resolved_model)
            if not force_provider and not breaker.allow(provider, resolved_model):
                continue

            attempted_any = True
            is_fallback = index > 0
            attempt_no += 1
            attempt_started_ns = time.time_ns()
            span_extra: dict = {
                "idempotency_key": ikey,
                "breaker_state": breaker_state,
                "fallback_reason": fallback_reason_of(last_error)
                if is_fallback or attempt_no > 1
                else None,
            }
            attempt_extra = (
                {**extra, "idempotency_key": ikey}
                if ikey and provider in IDEMPOTENCY_HEADER_PROVIDERS
                else extra
            )

            async def open_stream(stream_fn=stream_fn, attempt_extra=attempt_extra):
                generator = stream_fn(
                    config,
                    messages,
                    tools,
                    max_tokens,
                    model=model_override,
                    tool_choice=tool_choice,
                    sampling=sampling,
                    response_format=response_format,
                    **attempt_extra,
                )
                try:
                    # Its own timeout scope, so a miss is told apart from the
                    # idle bound and the deadline (which wrap this whole call).
                    limit = asyncio.timeout(first_token_limit)
                    try:
                        async with limit:
                            return generator, await generator.__anext__()
                    except TimeoutError:
                        if first_token_limit is not None and limit.expired():
                            await generator.aclose()
                            raise FirstTokenTimeoutError(
                                "No first chunk within first_token_timeout_seconds "
                                f"({first_token_limit:g} s)."
                            ) from None
                        raise
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
            except asyncio.CancelledError as e:
                # Cancelled while opening the stream (the provider generator is
                # closed by the cancellation itself): not a provider failure.
                breaker.release_probe(provider, resolved_model)
                emit_attempt_span(
                    tracer,
                    **span_extra,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    started_ns=attempt_started_ns,
                    error=e,
                )
                await report_attempt(
                    on_attempt,
                    config,
                    idempotency_key=ikey,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    outcome="cancelled",
                    usage=None,
                    error=e,
                    is_fallback=is_fallback,
                    latency_ms=(time.time_ns() - attempt_started_ns) / 1e6,
                )
                raise
            except Exception as e:  # noqa: BLE001 — pre-flight failure tries the next provider
                record_provider_failure(
                    provider,
                    e,
                    config=config,
                    deadline=deadline,
                    model=resolved_model,
                    failures=failures,
                )
                last_error = e
                emit_attempt_span(
                    tracer,
                    **span_extra,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    started_ns=attempt_started_ns,
                    error=e,
                )
                await report_attempt(
                    on_attempt,
                    config,
                    idempotency_key=ikey,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    outcome="failed",
                    usage=None,
                    error=e,
                    is_fallback=is_fallback,
                    latency_ms=(time.time_ns() - attempt_started_ns) / 1e6,
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
            restart_marker: StreamDelta | None = None
            if pending_restart is not None:
                restart_marker = StreamDelta(
                    restart=True,
                    restart_attempt=restarts,
                    provider=provider,
                    model=resolved_model,
                    request_id=rid,
                    failed_provider=pending_restart[0],
                    failed_error=type(pending_restart[1]).__name__,
                )
                failed_error_name = type(pending_restart[1]).__name__
                pending_restart = midstream_error = None
                span.add_event(
                    "llm_gateway.stream_restart",
                    {
                        "llm_gateway.restart": restarts,
                        "gen_ai.provider.name": provider,
                        "error.type": failed_error_name,
                    },
                )
            if first_delta is None:
                emit_attempt_span(
                    tracer,
                    **span_extra,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    started_ns=attempt_started_ns,
                )
                await report_attempt(
                    on_attempt,
                    config,
                    idempotency_key=ikey,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    outcome="succeeded",
                    usage=None,
                    error=None,
                    is_fallback=is_fallback,
                    latency_ms=(time.time_ns() - attempt_started_ns) / 1e6,
                )
                if restart_marker is not None:
                    yield restart_marker
                return
            cache_read_input_tokens = cache_creation_input_tokens = reasoning_tokens = 0
            usage_seen: Usage | None = None
            finish_reason = "stop"
            tool_call_count = 0
            total_chars = 0
            argument_chars: dict[int, int] = {}
            try:
                if restart_marker is not None:
                    yield restart_marker
                delta: StreamDelta | None = first_delta
                while delta is not None:
                    total_chars = _check_stream_size(delta, config, total_chars, argument_chars)
                    if delta.tool_call_deltas:
                        tool_call_count += sum(1 for d in delta.tool_call_deltas if "id" in d)
                    if delta.usage:
                        input_tokens, output_tokens = delta.usage
                        usage_seen = usage_of(input_tokens, output_tokens, delta.usage_details)
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
            except (asyncio.CancelledError, GeneratorExit) as e:
                # Cancelled task or consumer `aclose()`: a terminal "cancelled"
                # attempt span, never a breaker failure.
                stream_error = e
                raise
            except Exception as e:  # noqa: BLE001 — a mid-stream failure: counted, then propagated
                stream_error = e
                record_provider_failure(
                    provider,
                    e,
                    config=config,
                    deadline=deadline,
                    model=resolved_model,
                    failures=failures,
                )
                if restarts >= max_restarts or deadline.expired or not _restartable(e):
                    raise
                # Opt-in: hand over to the next provider (after the attempt is
                # reported in `finally`); if none can open, `midstream_error`
                # is raised after the loop.
                restarts += 1
                pending_restart = (provider, e)
                midstream_error = last_error = e
            finally:
                # The attempt span is written however the stream ends: done,
                # failed, or abandoned by the consumer.
                emit_attempt_span(
                    tracer,
                    **span_extra,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    started_ns=attempt_started_ns,
                    error=stream_error,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    restart=restarts if restart_marker is not None else 0,
                )
                # Consumer stopped reading (close, cancel, disconnect) or the
                # stream failed: close the provider's generator so its HTTP
                # response is released and the upstream request stops. Done
                # before the metering hook, which may be slow or be cancelled
                # itself, so the upstream request never waits on it.
                try:
                    await generator.aclose()
                except Exception:  # noqa: BLE001 — a failing close must not mask the real exit (e.g. a cancellation)
                    logger.debug("llm_gateway: closing a provider stream failed", exc_info=True)
                await report_attempt(
                    on_attempt,
                    config,
                    idempotency_key=ikey,
                    request_id=rid,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    outcome=(
                        "succeeded"
                        if stream_error is None
                        else "cancelled"
                        if isinstance(stream_error, asyncio.CancelledError | GeneratorExit)
                        else "failed"
                    ),
                    usage=usage_seen,
                    error=stream_error,
                    is_fallback=is_fallback,
                    latency_ms=(time.time_ns() - attempt_started_ns) / 1e6,
                    partial=stream_error is not None,
                    streamed_chars=total_chars,
                )

            if pending_restart is not None:
                # Died mid-stream; the loop tries the next provider.
                span.set_attribute("llm_gateway.restarts", restarts)
                continue

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
                config.call_deadline_seconds, last_error, config, failures
            ) from last_error
        span.set_status(
            StatusCode.ERROR,
            type(last_error).__name__ if last_error else "No provider available",
        )
        if midstream_error is not None:
            # A restart was wanted but no provider could take over: the caller
            # sees the original mid-stream failure.
            raise midstream_error
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
        raise all_failed(last_error, config, failures, plan.failure_note()) from last_error
