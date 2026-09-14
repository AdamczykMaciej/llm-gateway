"""The gateway's core: a configurable multi-provider fallback chain with a
per-provider circuit breaker and OTel tracing.

This is the engine — `llm_gateway.service` is a thin HTTP wrapper around it.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from . import breaker
from .config import GatewayConfig
from .errors import LLMDeadlineExceeded, LLMError, deadline_exceeded
from .providers import CALLS, CONFIGURED, DEFAULT_MODEL
from .providers.base import Usage
from .retry import Deadline, call_with_retry
from .structured import OutputSchema
from .tracing import get_tracer, set_call_attributes

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


@dataclass(frozen=True)
class Completion:
    """What `complete_with_usage()` returns: the text plus who served it and
    what it cost.

    `provider` and `model` name the provider that actually answered, after
    any failover, and `usage` holds that provider's token counts (see
    `Usage` for how cached tokens are counted). `stop_reason` is the
    provider's own value ("end_turn", "stop", "max_tokens", ...). `parsed` is
    set only when `output_schema` was given: the validated pydantic instance,
    or the parsed JSON for a dict schema."""

    text: str
    provider: str
    model: str
    usage: Usage
    stop_reason: str | None = None
    parsed: Any = None


async def complete(
    *,
    system: str,
    prompt: str,
    max_tokens: int = 2000,
    config: GatewayConfig | None = None,
    force_provider: str | None = None,
    model: str | None = None,
    cache_system: bool = False,
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

    Raises `LLMError` when no provider succeeds, or its subclass
    `LLMDeadlineExceeded` when `config.call_deadline_seconds` runs out first.
    """
    completion = await complete_with_usage(
        system=system,
        prompt=prompt,
        max_tokens=max_tokens,
        config=config,
        force_provider=force_provider,
        model=model,
        cache_system=cache_system,
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
) -> Completion:
    """`complete()`, returning a `Completion` instead of a string.

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
    """
    config = config or GatewayConfig()
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
            usage = result.usage
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
            return Completion(
                text=result.text,
                provider=provider,
                model=result.model,
                usage=usage,
                stop_reason=result.stop_reason,
                parsed=parsed,
            )

        if deadline.expired:
            span.set_status(StatusCode.ERROR, "call deadline exceeded")
            raise deadline_exceeded(config.call_deadline_seconds, last_error) from last_error
        span.set_status(
            StatusCode.ERROR,
            str(last_error) if last_error else "No provider available",
        )
        if not attempted_any:
            raise LLMError(
                "No LLM provider available. Set ANTHROPIC_API_KEY, GROQ_API_KEY, "
                "OPENAI_API_KEY, or AZURE_ENDPOINT and AZURE_MODEL, matching provider_order."
            )
        raise LLMError(f"All configured providers failed. Last error: {last_error}") from last_error
