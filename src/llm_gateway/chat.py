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

import asyncio
import dataclasses
import logging
import time
from collections.abc import Sequence
from typing import Any, cast

from opentelemetry.trace import StatusCode

from . import breaker
from .attempts import AttemptHook, report_attempt
from .config import GatewayConfig
from .errors import (
    AllProvidersExhaustedError,
    GatewayNotConfiguredError,
    ProviderFailure,
    all_failed,
    deadline_exceeded,
)
from .idempotency import IDEMPOTENCY_HEADER_PROVIDERS, single_flight
from .policy import ProviderPolicy
from .providers import CHAT_CALLS, CONFIGURED, DEFAULT_MODEL
from .providers.base import ChatResult, Usage
from .retry import Deadline, call_with_retry, resolve_deadline_seconds
from .router import record_provider_failure, retry_hooks
from .routing import RequestProfile, plan_chain
from .tool_results import cap_tool_results, resolve_cap
from .tool_validation import (
    enforce_tool_calls,
    resolve_validate_mode,
    unique_tool_call_ids,
    validate_chat_request,
)
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


async def _chat(
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
    request_id: str | None = None,
    parallel_tool_calls: bool | None = None,
    max_tool_result_chars: int | None = None,
    on_attempt: AttemptHook | None = None,
    idempotency_key: str | None = None,
    deadline_seconds: float | None = None,
    validate_tool_calls: str | None = None,
) -> ChatResult:
    """Return a ChatResult (text and/or tool_calls), trying providers in
    `config.provider_order` — same fallback/circuit-breaker semantics as
    `router.complete()`. `messages` and `tools` are OpenAI-wire-shaped
    dicts; each provider translates internally (see providers/anthropic.py
    for the one that actually needs translating). Raises `LLMError`, or its
    subclass `LLMDeadlineExceeded` when `config.call_deadline_seconds` runs
    out first. `policy` narrows `config.policy` exactly as in
    `complete_with_usage()`; `PolicyViolationError` when no provider is left.

    `request_id`: a correlation id put on every span of the call and returned
    as `ChatResult.request_id`; generated when omitted (see tracing.py).

    `on_attempt`: a sync or async callable receiving an `AttemptReport` once
    per provider attempt (served, failed or cancelled); replaces
    `config.on_attempt` for this call. Exceptions inside it are logged and
    ignored (see attempts.py). A failed `chat()` attempt reports no usage.

    `parallel_tool_calls`: `False` asks the model for at most one tool call
    per turn, where the provider has such a switch (OpenAI-style
    `parallel_tool_calls`, Anthropic `disable_parallel_tool_use`; see
    `providers.PARALLEL_TOOL_CALLS_SUPPORTED`). Providers without one ignore
    it, so a caller that needs exactly one call must still handle several.
    `None` (default) sends nothing and keeps each provider's default.

    `max_tool_result_chars`: truncate `role: "tool"` message text longer than
    this (with a marker) before sending; overrides
    `config.tool_result_max_chars`, 0 disables (see tool_results.py).

    `validate_tool_calls`: "off" (default, `None`), "warn" or "error". Checks
    each reply's tool calls with `check_tool_calls` (a tool the request never
    offered, arguments failing the tool's JSON Schema, arguments over
    `config.stream_max_tool_arguments_chars`). "warn" logs the issues (names
    and paths, never values) and returns the reply; "error" raises
    `ToolCallInvalidError` inside the attempt, so the engine fails over and an
    exhausted chain chains from it. Anything else is `InvalidRequestError`.

    Raises:
        InvalidRequestError: malformed messages, tools, `tool_choice`,
            `request_id` or `max_tool_result_chars`; nothing was sent.
        PolicyViolationError: no provider satisfies the routing policy
            (`ResidencyViolationError`, `UnsupportedCapabilityError` are
            subclasses).
        LLMDeadlineExceeded: `config.call_deadline_seconds` ran out.
        GatewayNotConfiguredError: no provider in the chain is configured.
        AllProvidersExhaustedError: every configured provider failed.
        PrivacyPolicyMissingError: only via `PrivacyLayer(strict=True)`.
    """
    rid = resolve_request_id(request_id)
    ikey = resolve_idempotency_key(idempotency_key)
    validate_chat_request(messages, tools, tool_choice)
    validate_mode = resolve_validate_mode(validate_tool_calls)
    config = config or GatewayConfig()
    messages = cap_tool_results(
        messages, resolve_cap(max_tool_result_chars, config.tool_result_max_chars)
    )
    history_tool_call_ids = {
        call["id"]
        for message in messages
        for call in message.get("tool_calls") or []
        if isinstance(call.get("id"), str)
    }
    # Only passed when used, so provider callables written against the older
    # signature keep working.
    extra: dict = {}
    if parallel_tool_calls is not None:
        extra["parallel_tool_calls"] = parallel_tool_calls
    order = [force_provider] if force_provider else config.provider_order_list
    model_override = model if force_provider else None

    tracer = get_tracer()
    start = time.monotonic() * 1000
    deadline = Deadline(
        resolve_deadline_seconds(deadline_seconds, config.call_deadline_seconds),
        config.max_attempts_per_call,
    )
    attempted_any = False
    last_error: Exception | None = None
    failures: list[ProviderFailure] = []

    with call_span(tracer, "llm_gateway.chat") as span:
        span.set_attribute("gen_ai.request.max_tokens", max_tokens)
        span.set_attribute("llm_gateway.fallback", False)
        span.set_attribute("llm_gateway.request_id", rid)
        if ikey:
            span.set_attribute("llm_gateway.idempotency_key", ikey)
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
            ),
            registry=CHAT_CALLS,
            model_override=model_override,
            span=span,
        )

        for index, provider in enumerate(plan.providers):
            call = CHAT_CALLS.get(provider)
            if call is None:
                continue
            if not CONFIGURED[provider](config):
                continue
            resolved_model = model_override or DEFAULT_MODEL[provider](config)
            if not force_provider and breaker.is_open(provider, resolved_model):
                continue
            if deadline.expired or deadline.attempts_exhausted:
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

            async def _attempt(call=call, attempt_extra=attempt_extra) -> ChatResult:
                reply = await call(
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
                enforce_tool_calls(
                    tools,
                    reply.tool_calls,
                    validate_mode,
                    config.stream_max_tool_arguments_chars,
                )
                return reply

            try:
                result = await call_with_retry(
                    _attempt,
                    attempts=config.retry_attempts,
                    **retry_hooks(
                        provider, resolved_model, config=config, force_provider=force_provider
                    ),
                    base_delay_seconds=config.retry_base_delay_seconds,
                    attempt_timeout_seconds=config.request_timeout_seconds,
                    deadline=deadline,
                )
            except asyncio.CancelledError as e:
                # The caller gave up: not a provider failure (no breaker count,
                # no failover). Hand back a half-open probe lease and leave a
                # terminal attempt span.
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
            except Exception as e:  # noqa: BLE001 — any provider failure (after its retries) tries the next
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

            breaker.record_success(provider, resolved_model)
            emit_attempt_span(
                tracer,
                **span_extra,
                request_id=rid,
                attempt=attempt_no,
                provider=provider,
                model=result.model,
                started_ns=attempt_started_ns,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                response_model=result.model,
                finish_reason=result.finish_reason,
            )
            await report_attempt(
                on_attempt,
                config,
                idempotency_key=ikey,
                request_id=rid,
                attempt=attempt_no,
                provider=provider,
                model=result.model,
                outcome="succeeded",
                usage=Usage(
                    result.input_tokens,
                    result.output_tokens,
                    result.cache_read_input_tokens,
                    result.cache_creation_input_tokens,
                    result.reasoning_tokens,
                ),
                is_fallback=is_fallback,
                latency_ms=(time.time_ns() - attempt_started_ns) / 1e6,
            )
            latency = time.monotonic() * 1000 - start
            set_chat_attributes(
                span,
                config=config,
                provider=provider,
                model=resolved_model,
                response_model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=latency,
                fallback=is_fallback,
                tool_call_count=len(result.tool_calls),
                finish_reason=result.finish_reason,
                cache_read_input_tokens=result.cache_read_input_tokens,
                cache_creation_input_tokens=result.cache_creation_input_tokens,
                reasoning_tokens=result.reasoning_tokens,
            )
            logger.debug(
                "llm_gateway served: provider=%s model=%s input_tokens=%s output_tokens=%s "
                "reasoning_tokens=%s",
                provider,
                result.model,
                result.input_tokens,
                result.output_tokens,
                result.reasoning_tokens,
            )
            return dataclasses.replace(
                result,
                request_id=rid,
                tool_calls=unique_tool_call_ids(result.tool_calls, history_tool_call_ids),
            )

        if deadline.expired:
            span.set_status(StatusCode.ERROR, "call deadline exceeded")
            raise deadline_exceeded(
                deadline.seconds or 0.0, last_error, config, failures
            ) from last_error
        span.set_status(
            StatusCode.ERROR,
            type(last_error).__name__ if last_error else "No provider available",
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
        raise all_failed(last_error, config, failures, plan.failure_note()) from last_error


async def chat(
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
    request_id: str | None = None,
    parallel_tool_calls: bool | None = None,
    max_tool_result_chars: int | None = None,
    on_attempt: AttemptHook | None = None,
    idempotency_key: str | None = None,
    deadline_seconds: float | None = None,
    validate_tool_calls: str | None = None,
) -> ChatResult:
    """Tool-calling completion with provider fallback; see `_chat` for the
    full contract.

    `deadline_seconds`: one total time budget for this call, replacing
    `config.call_deadline_seconds` (positive; `InvalidRequestError` otherwise):
    attempts, backoff, `Retry-After` waits and failovers all draw from it, and
    `LLMDeadlineExceeded` is raised when it runs out. `None` uses the config.

    `idempotency_key`: the caller's key for this LOGICAL call (1-128 letters,
    digits or `. _ : -`; `InvalidRequestError` otherwise). It is put on the
    call span, every attempt span and every `AttemptReport`, so retries and
    fallbacks of one call (and a caller's own re-submissions) correlate. It is
    sent as the `Idempotency-Key` header to OpenAI only; no other provider
    documents one, so none receives it. Concurrent `chat()` calls in one
    process with the same key and the same request share ONE provider call and
    one result (the followers' `on_attempt` never fires); the same key with a
    different request raises `InvalidRequestError`. Nothing is cached after
    the call finishes. Without a key, behaviour is unchanged."""
    impl_kwargs = {
        "messages": cast("list[dict]", messages),
        "tools": cast("list[dict] | None", tools),
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "config": config,
        "force_provider": force_provider,
        "model": model,
        "sampling": sampling,
        "response_format": response_format,
        "policy": policy,
        "request_id": request_id,
        "parallel_tool_calls": parallel_tool_calls,
        "max_tool_result_chars": max_tool_result_chars,
        "on_attempt": on_attempt,
        "idempotency_key": idempotency_key,
        "deadline_seconds": deadline_seconds,
        "validate_tool_calls": validate_tool_calls,
    }
    if idempotency_key is None:
        return await _chat(**impl_kwargs)
    ikey = resolve_idempotency_key(idempotency_key)
    assert ikey is not None
    fingerprint = {
        k: v
        for k, v in impl_kwargs.items()
        if k
        not in (
            "config",
            "policy",
            "request_id",
            "on_attempt",
            "idempotency_key",
            "deadline_seconds",
        )
    }
    return await single_flight(ikey, fingerprint, lambda: _chat(**impl_kwargs))
