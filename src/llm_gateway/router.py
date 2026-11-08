"""The gateway's core: a configurable multi-provider fallback chain with a
per-provider circuit breaker and OTel tracing.

This is the engine — `llm_gateway.service` is a thin HTTP wrapper around it.
"""

import asyncio
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from . import breaker
from .config import GatewayConfig
from .errors import (
    AllProvidersExhaustedError,
    ErrorKind,
    GatewayNotConfiguredError,
    InvalidOutputError,
    InvalidRequestError,
    LLMDeadlineExceeded,
    LLMError,
    classify,
    deadline_exceeded,
    error_detail,
)
from .policy import ProviderPolicy
from .pricing import cost_usd, lookup_price
from .providers import CALLS, CONFIGURED, DEFAULT_MODEL
from .providers.base import Usage
from .retry import Deadline, call_with_retry
from .routing import RequestProfile, plan_chain, record_cost
from .structured import OutputSchema, OutputValidator
from .tracing import (
    call_span,
    emit_attempt_span,
    get_tracer,
    resolve_request_id,
    set_call_attributes,
)

__all__ = [
    "Completion",
    "LLMError",
    "LLMDeadlineExceeded",
    "complete",
    "complete_with_usage",
    "record_provider_failure",
]

logger = logging.getLogger("llm_gateway")


def record_provider_failure(
    provider: str,
    error: Exception,
    *,
    config: GatewayConfig,
    deadline: Deadline,
    model: str | None = None,
) -> None:
    """Feed a provider's definitive failure (retries exhausted or not
    applicable) to its circuit breaker, per errors.POLICIES. Shared by all
    three engines. A timeout that only fired because the call's deadline ran
    out is not counted: it says more about earlier providers using up the
    budget than about this one. With `model`, a model-specific failure opens
    only that (provider, model) (see breaker.py)."""
    if deadline.expired and isinstance(error, TimeoutError):
        return
    if classify(error) is ErrorKind.BILLING:
        # Operator-actionable and distinguishable in logs; no message body.
        logger.warning(
            "llm_gateway billing: provider=%s kind=billing billing=True "
            "(out of credit, spend limit or quota); breaker opened for %gs",
            provider,
            config.breaker_cooldown_seconds,
        )
    breaker.record_error(
        provider,
        error,
        threshold=config.breaker_failure_threshold,
        cooldown_seconds=config.breaker_cooldown_seconds,
        model=model,
    )


@dataclass(frozen=True)
class Completion:
    """What `complete_with_usage()` returns: the text plus who served it and
    what it cost.

    `provider` and `model` name the provider that actually answered, after
    any failover, and `usage` holds that provider's token counts (see
    `Usage` for how cached tokens are counted). `stop_reason` is the
    provider's own value ("end_turn", "stop", "max_tokens", ...). `parsed` is
    set only when `output_schema` was given: the validated pydantic instance,
    or the parsed JSON for a dict schema. `cost_usd` is the cost of `usage`
    at the model's price (pricing.py), or `None` when the model has no price;
    it is derived data, so it is excluded from equality. `request_id` is the
    call's correlation id (the caller's `request_id`, or a generated one): it
    is on every span of the call, so a result can be matched to its traces. It
    identifies the call, not the provider's own request."""

    text: str
    provider: str
    model: str
    usage: Usage
    stop_reason: str | None = None
    parsed: Any = None
    cost_usd: float | None = field(default=None, compare=False)
    request_id: str | None = field(default=None, compare=False)


async def complete(
    *,
    system: str,
    prompt: str,
    max_tokens: int = 2000,
    config: GatewayConfig | None = None,
    force_provider: str | None = None,
    model: str | None = None,
    cache_system: bool = False,
    policy: ProviderPolicy | None = None,
    request_id: str | None = None,
) -> str:
    """Return a text completion, trying providers in `config.provider_order`.

    Pass `force_provider` (e.g. "anthropic") to call exactly that provider
    with no fallback — used by the HTTP service when a caller explicitly
    requests `model="<provider>/<model>"`. `model` only has an effect
    together with `force_provider`: a model name only means something within
    one provider's namespace, so it's meaningless across a multi-provider
    fallback chain — each provider in the chain always uses its own
    configured default model there.

    Same as `(await complete_with_usage(...)).text`; use that to also get
    the serving provider, token usage, or structured output.

    Raises `LLMError` when no provider succeeds, its subclass
    `LLMDeadlineExceeded` when `config.call_deadline_seconds` runs out first,
    or its subclass `PolicyViolationError` when no provider satisfies the
    routing policy (see `complete_with_usage`). Also raises
    `GatewayNotConfiguredError` / `AllProvidersExhaustedError` (subclasses of
    `LLMError`) and `InvalidRequestError` for a malformed `request_id`.
    """
    completion = await complete_with_usage(
        system=system,
        prompt=prompt,
        max_tokens=max_tokens,
        config=config,
        force_provider=force_provider,
        model=model,
        cache_system=cache_system,
        policy=policy,
        request_id=request_id,
    )
    return completion.text


async def complete_with_usage(
    *,
    system: str,
    prompt: str,
    max_tokens: int = 2000,
    config: GatewayConfig | None = None,
    force_provider: str | None = None,
    model: str | None = None,
    output_schema: dict | type[BaseModel] | None = None,
    schema_name: str | None = None,
    cache_system: bool = False,
    policy: ProviderPolicy | None = None,
    request_id: str | None = None,
    validate: dict | type[BaseModel] | None = None,
    repair_attempts: int = 0,
) -> Completion:
    """`complete()`, returning a `Completion` instead of a string.

    `request_id`: a correlation id (1-128 of letters, digits, `. _ : -`) put
    on every span of the call and returned as `Completion.request_id`; one is
    generated when omitted.

    `validate`: opt-in check of the served text against a pydantic model
    class (full validation) or a JSON Schema dict (a documented subset: type,
    enum, const, required, properties, additionalProperties false, items,
    anyOf/oneOf, local $ref). It does not change the request; combine it with
    `output_schema` to also ask the provider for JSON. On success
    `Completion.parsed` holds the typed result. On failure the call raises
    `InvalidOutputError` whose `.paths` name the failing fields (never their
    values), after the provider has already been paid for: there is **no
    automatic retry** and no failover on a failed `validate`.

    `repair_attempts` (default 0): how many extra, paid calls to make when
    `validate` fails. Each re-asks the provider that answered, with the
    failing paths and its previous reply appended to the prompt; usage and
    cost are summed over all calls. The last `InvalidOutputError` is raised
    when they run out. Ignored without `validate`.

    `output_schema`: a pydantic model class or a JSON Schema dict. Each
    provider is asked for schema-conforming JSON through its native
    mechanism (see structured.py), and the reply is parsed and validated
    before the call counts as served. A reply that isn't valid JSON or fails
    validation fails over to the next provider, without a retry and without
    counting toward that provider's circuit breaker. `schema_name` names the
    schema where a provider wants a name (OpenAI); it defaults to the model's
    class name or the schema's `title`.

    `cache_system`: mark `system` as a cacheable prefix. On Anthropic it is
    sent with `cache_control: {"type": "ephemeral"}` when it is long enough
    for the model's minimum cacheable length (providers/_anthropic_caching.py).
    OpenAI and Groq cache automatically, so there it changes nothing. Cache
    hits and writes are reported in `Completion.usage`.

    `policy`: a `ProviderPolicy` merged with `config.policy`; it can only
    narrow it. Providers that fail it, can't serve the request, or exceed
    `max_cost_usd` are skipped before any call, and `PolicyViolationError` is
    raised when none is left (routing.py). This applies to `force_provider`
    too.

    Raises:
        InvalidRequestError: bad `request_id`, `repair_attempts`, `validate`
            spec; nothing was sent.
        PolicyViolationError: no provider satisfies the routing policy
            (`ResidencyViolationError`, `UnsupportedCapabilityError` are
            subclasses).
        LLMDeadlineExceeded: `config.call_deadline_seconds` ran out.
        GatewayNotConfiguredError: no provider in the chain is configured.
        AllProvidersExhaustedError: every configured provider failed.
        InvalidOutputError: `validate=` failed on the served text (`.paths`
            names the fields), or privacy post-processing left `parsed` invalid.
    """
    rid = resolve_request_id(request_id)
    if (
        isinstance(repair_attempts, bool)
        or not isinstance(repair_attempts, int)
        or repair_attempts < 0
    ):
        raise InvalidRequestError("repair_attempts must be a non-negative integer.")
    validator = OutputValidator.from_spec(validate) if validate is not None else None
    config = config or GatewayConfig()
    options = {
        "system": system,
        "max_tokens": max_tokens,
        "config": config,
        "force_provider": force_provider,
        "model": model,
        "output_schema": output_schema,
        "schema_name": schema_name,
        "cache_system": cache_system,
        "policy": policy,
        "request_id": rid,
    }
    completion = await _complete_once(prompt=prompt, **options)
    if validator is None:
        return completion
    repairs_left = repair_attempts
    while True:
        try:
            parsed = validator.check(completion.text)
        except InvalidOutputError as error:
            if repairs_left <= 0:
                raise
            repairs_left -= 1
            repaired = await _complete_once(
                prompt=_repair_prompt(prompt, completion.text, error),
                **{**options, "force_provider": completion.provider, "model": completion.model},
            )
            completion = _merge_attempts(completion, repaired)
            continue
        return dataclasses.replace(completion, parsed=parsed)


def _repair_prompt(prompt: str, reply: str, error: InvalidOutputError) -> str:
    where = ", ".join(error.paths) if error.paths else "the reply as a whole"
    return (
        f"{prompt}\n\nYour previous reply failed validation at: {where}.\n"
        f"Previous reply:\n{reply}\n\nReply again with only the corrected output."
    )


def _merge_attempts(first: Completion, latest: Completion) -> Completion:
    """`latest`, with the usage and cost of `first` added."""
    a, b = first.usage, latest.usage
    usage = Usage(
        a.input_tokens + b.input_tokens,
        a.output_tokens + b.output_tokens,
        a.cache_read_input_tokens + b.cache_read_input_tokens,
        a.cache_creation_input_tokens + b.cache_creation_input_tokens,
        a.reasoning_tokens + b.reasoning_tokens,
    )
    cost = (
        first.cost_usd + latest.cost_usd
        if first.cost_usd is not None and latest.cost_usd is not None
        else None
    )
    return dataclasses.replace(latest, usage=usage, cost_usd=cost)


async def _complete_once(
    *,
    system: str,
    prompt: str,
    max_tokens: int,
    config: GatewayConfig,
    force_provider: str | None,
    model: str | None,
    output_schema: dict | type[BaseModel] | None,
    schema_name: str | None,
    cache_system: bool,
    policy: ProviderPolicy | None,
    request_id: str,
) -> Completion:
    schema = (
        OutputSchema.from_spec(output_schema, schema_name) if output_schema is not None else None
    )
    order = [force_provider] if force_provider else config.provider_order_list
    model_override = model if force_provider else None
    # Only passed when used, so provider callables written against the 0.3
    # positional signature keep working.
    extra: dict = {}
    if schema is not None:
        extra["output_schema"] = schema
    if cache_system:
        extra["cache_system"] = True

    tracer = get_tracer()
    start = time.monotonic() * 1000
    deadline = Deadline(config.call_deadline_seconds)
    attempted_any = False
    last_error: Exception | None = None

    with call_span(tracer, "llm_gateway.complete") as span:
        span.set_attribute("llm_gateway.fallback", False)
        span.set_attribute("llm_gateway.request_id", request_id)
        attempt_no = 0
        plan = plan_chain(
            order,
            config=config,
            policy=policy,
            profile=RequestProfile.for_completion(
                system=system,
                prompt=prompt,
                max_tokens=max_tokens,
                output_schema=schema.json_schema if schema is not None else None,
            ),
            registry=CALLS,
            model_override=model_override,
            span=span,
        )

        for index, provider in enumerate(plan.providers):
            call = CALLS.get(provider)
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

            async def attempt(call=call):
                result = await call(config, system, prompt, max_tokens, model_override, **extra)
                # Inside the attempt, so invalid structured output is a
                # failure of this provider and the chain moves on.
                parsed = schema.parse(result.text) if schema is not None else None
                return result, parsed

            try:
                result, parsed = await call_with_retry(
                    attempt,
                    attempts=config.retry_attempts,
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
                    request_id=request_id,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    started_ns=attempt_started_ns,
                    error=e,
                )
                raise
            except Exception as e:  # noqa: BLE001 — any provider failure (after its retries) tries the next
                record_provider_failure(
                    provider, e, config=config, deadline=deadline, model=resolved_model
                )
                last_error = e
                emit_attempt_span(
                    tracer,
                    request_id=request_id,
                    attempt=attempt_no,
                    provider=provider,
                    model=resolved_model,
                    started_ns=attempt_started_ns,
                    error=e,
                )
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

            breaker.record_success(provider, resolved_model)
            latency = time.monotonic() * 1000 - start
            usage = result.usage
            emit_attempt_span(
                tracer,
                request_id=request_id,
                attempt=attempt_no,
                provider=provider,
                model=result.model,
                started_ns=attempt_started_ns,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
            set_call_attributes(
                span,
                config=config,
                provider=provider,
                model=result.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                latency_ms=latency,
                fallback=is_fallback,
                system=system,
                prompt=prompt,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
            logger.debug(
                "llm_gateway served: provider=%s model=%s input_tokens=%s output_tokens=%s "
                "reasoning_tokens=%s",
                provider,
                result.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.reasoning_tokens,
            )
            cost = cost_usd(
                lookup_price(
                    provider,
                    result.model,
                    config.model_prices,
                    vertex_location=config.vertex_location,
                ),
                usage,
            )
            record_cost(span, cost)
            return Completion(
                text=result.text,
                provider=provider,
                model=result.model,
                usage=usage,
                stop_reason=result.stop_reason,
                parsed=parsed,
                cost_usd=cost,
                request_id=request_id,
            )

        if deadline.expired:
            span.set_status(StatusCode.ERROR, "call deadline exceeded")
            raise deadline_exceeded(
                config.call_deadline_seconds, last_error, config
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
        raise AllProvidersExhaustedError(
            "All configured providers failed. Last error: "
            f"{error_detail(last_error, config)}{plan.failure_note()}"
        ) from last_error
