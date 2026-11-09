"""Per-attempt usage and cost reporting (`on_attempt=`).

One `AttemptReport` per provider ATTEMPT (after that provider's own retries):
served, failed (and the chain moved on, or ended), or cancelled by the caller.
A failed attempt may still have been billed, so a host that keeps a spend
ledger sees it here and not only in the winning `Completion.usage`.

The report carries identifiers, numbers and the *type name* of an error. It
never carries prompt or response text, nor an error message (those can echo
content). A hook that raises is logged by exception type only and ignored: it
can never fail, delay beyond its own run, or alter the call.
"""

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .pricing import cost_usd, lookup_price
from .providers.base import Usage

if TYPE_CHECKING:
    from .config import GatewayConfig

__all__ = ["AttemptHook", "AttemptOutcome", "AttemptReport"]

logger = logging.getLogger("llm_gateway")

AttemptOutcome = Literal["succeeded", "failed", "cancelled"]


@dataclass(frozen=True)
class AttemptReport:
    """What `on_attempt` receives.

    `attempt` counts provider attempts within the call from 1 (a retry inside
    one provider is not a new attempt). `outcome`: "succeeded"; "failed"
    (`error_type` is the exception class name); or "cancelled" (the caller
    gave up, or closed a stream early). `usage` is what the provider reported
    for this attempt (also for a reply that then failed validation, or a
    stream that failed after its usage chunk), else `None`. `cost_usd` is
    that usage at the price table's rate, `None` without usage or price.
    `is_fallback` is true when the attempt is not the first provider of the
    chain. `request_id` is the call's correlation id.

    `partial` is true for a stream that was cut short after it started (the
    caller closed or cancelled it, or it failed mid-stream): `usage` may then
    be `None` or lag what the provider billed, so a ledger should treat the
    cost as a lower bound. `streamed_chars` is the text and tool-argument
    characters forwarded to the caller (0 outside streams); it lets a host
    estimate the unreported output. `idempotency_key` is the caller's key for
    the logical call (`chat(idempotency_key=)`), else `None`."""

    request_id: str
    attempt: int
    provider: str
    model: str
    outcome: AttemptOutcome
    usage: Usage | None = None
    cost_usd: float | None = None
    error_type: str | None = None
    is_fallback: bool = False
    latency_ms: float = field(default=0.0, compare=False)
    partial: bool = False
    streamed_chars: int = 0
    idempotency_key: str | None = None


AttemptHook = Callable[[AttemptReport], Awaitable[None] | None]


async def report_attempt(
    hook: AttemptHook | None,
    config: "GatewayConfig",
    *,
    request_id: str,
    attempt: int,
    provider: str,
    model: str,
    outcome: AttemptOutcome,
    usage: Usage | None = None,
    error: BaseException | None = None,
    is_fallback: bool = False,
    latency_ms: float = 0.0,
    partial: bool = False,
    streamed_chars: int = 0,
    idempotency_key: str | None = None,
) -> None:
    """Call the per-call `hook`, else `config.on_attempt`, once. Never raises
    an `Exception`."""
    hook = hook or config.on_attempt
    if hook is None:
        return
    if usage is None:
        # A reply that was billed and then rejected (empty, refusal, cut-off
        # tool call) carries its usage on the error.
        carried = getattr(error, "usage", None)
        usage = carried if isinstance(carried, Usage) else None
    try:
        cost = None
        if usage is not None:
            cost = cost_usd(
                lookup_price(
                    provider,
                    model,
                    config.model_prices,
                    vertex_location=config.vertex_location,
                ),
                usage,
            )
        outcome_value = hook(
            AttemptReport(
                request_id=request_id,
                attempt=attempt,
                provider=provider,
                model=model,
                outcome=outcome,
                usage=usage,
                cost_usd=cost,
                error_type=type(error).__name__ if error is not None else None,
                is_fallback=is_fallback,
                latency_ms=latency_ms,
                partial=partial,
                streamed_chars=streamed_chars,
                idempotency_key=idempotency_key,
            )
        )
        if inspect.isawaitable(outcome_value):
            await outcome_value
    except Exception as e:  # noqa: BLE001 — a metering hook must never break the call
        logger.warning("llm_gateway on_attempt hook failed: %s", type(e).__name__)


def usage_of(input_tokens: int, output_tokens: int, details: Any = None) -> Usage:
    """A `Usage` from a stream's counters (`details` is a `UsageDetails`-like)."""
    return Usage(
        input_tokens,
        output_tokens,
        getattr(details, "cache_read_input_tokens", 0),
        getattr(details, "cache_creation_input_tokens", 0),
        getattr(details, "reasoning_tokens", 0),
    )
