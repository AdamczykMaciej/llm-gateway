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

import dataclasses
import logging
import time

from opentelemetry.trace import StatusCode

from . import breaker
from .config import GatewayConfig
from .errors import (
    AllProvidersExhaustedError,
    GatewayNotConfiguredError,
    deadline_exceeded,
    redact_secrets,
)
from .policy import ProviderPolicy
from .providers import CHAT_CALLS, CONFIGURED, DEFAULT_MODEL
from .providers.base import ChatResult
from .retry import Deadline, call_with_retry
from .router import record_provider_failure
from .routing import RequestProfile, plan_chain
from .tool_results import cap_tool_results, resolve_cap
from .tool_validation import validate_chat_request
from .tracing import emit_attempt_span, get_tracer, resolve_request_id, set_chat_attributes

logger = logging.getLogger("llm_gateway")


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
    policy: ProviderPolicy | None = None,
    request_id: str | None = None,
    parallel_tool_calls: bool | None = None,
    max_tool_result_chars: int | None = None,
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

    `parallel_tool_calls`: `False` asks the model for at most one tool call
    per turn, where the provider has such a switch (OpenAI-style
    `parallel_tool_calls`, Anthropic `disable_parallel_tool_use`; see
    `providers.PARALLEL_TOOL_CALLS_SUPPORTED`). Providers without one ignore
    it, so a caller that needs exactly one call must still handle several.
    `None` (default) sends nothing and keeps each provider's default.

    `max_tool_result_chars`: truncate `role: "tool"` message text longer than
    this (with a marker) before sending; overrides
    `config.tool_result_max_chars`, 0 disables (see tool_results.py).

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
    validate_chat_request(messages, tools, tool_choice)
    config = config or GatewayConfig()
    messages = cap_tool_results(
        messages, resolve_cap(max_tool_result_chars, config.tool_result_max_chars)
    )
    # Only passed when used, so provider callables written against the older
    # signature keep working.
    extra: dict = {}
    if parallel_tool_calls is not None:
        extra["parallel_tool_calls"] = parallel_tool_calls
    order = [force_provider] if force_provider else config.provider_order_list
    model_override = model if force_provider else None

    tracer = get_tracer()
    start = time.monotonic() * 1000
    deadline = Deadline(config.call_deadline_seconds)
    attempted_any = False
    last_error: Exception | None = None

    with tracer.start_as_current_span("llm_gateway.chat") as span:
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
                        **extra,
                    ),
                    attempts=config.retry_attempts,
                    base_delay_seconds=config.retry_base_delay_seconds,
                    attempt_timeout_seconds=config.request_timeout_seconds,
                    deadline=deadline,
                )
            except Exception as e:  # noqa: BLE001 — any provider failure (after its retries) tries the next
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

            breaker.record_success(provider, resolved_model)
            emit_attempt_span(
                tracer,
                request_id=rid,
                attempt=attempt_no,
                provider=provider,
                model=result.model,
                started_ns=attempt_started_ns,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
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
            return dataclasses.replace(result, request_id=rid)

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
