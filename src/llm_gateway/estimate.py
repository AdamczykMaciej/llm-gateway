"""Pre-call budget estimates over the models a call can actually route to.

`estimate_max_cost` reuses `routing.plan_chain` (the router's own planner), so
the models it prices are exactly those a real call could try: the configured
chain, minus providers without credentials, retired models, and anything the
config or per-call policy excludes. Prices of models outside that set never
matter, so adding a price row for an unused model cannot change an estimate.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .config import GatewayConfig
from .errors import PolicyViolationError
from .policy import ProviderPolicy
from .pricing import CHARS_PER_TOKEN, lookup_price, worst_case_cost_usd
from .providers import CALLS, CHAT_CALLS, CONFIGURED
from .routing import RequestProfile, plan_chain


@dataclass(frozen=True)
class RoutableModel:
    """One provider/model a call may be sent to, with its worst-case cost
    (`None` when the model has no price)."""

    provider: str
    model: str
    cost_usd: float | None

    @property
    def key(self) -> str:
        """The price-table key, `provider/model`."""
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class CostEstimate:
    """Worst-case cost of one call over its routable models.

    `max_cost_usd` is the largest per-model worst case, or `None` when any
    routable model has no price (listed in `unpriced`) or nothing is routable:
    an unknown price is never counted as 0. `priced_max_cost_usd` is the
    maximum over the priced models only, a lower bound when `complete` is
    False. `excluded` maps provider id to why the call cannot use it."""

    models: tuple[RoutableModel, ...]
    max_cost_usd: float | None
    unpriced: tuple[str, ...] = ()
    excluded: Mapping[str, tuple[str, ...]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.excluded is None:
            object.__setattr__(self, "excluded", {})

    @property
    def complete(self) -> bool:
        """True when something is routable and every routable model is priced."""
        return bool(self.models) and not self.unpriced

    @property
    def priced_max_cost_usd(self) -> float | None:
        """Largest cost among priced models; a lower bound unless `complete`."""
        costs = [m.cost_usd for m in self.models if m.cost_usd is not None]
        return max(costs) if costs else None


def _plan(
    config: GatewayConfig,
    *,
    input_tokens: int,
    max_output_tokens: int,
    policy: ProviderPolicy | None,
    provider: str | None,
    model: str | None,
    chat: bool,
) -> tuple[tuple[RoutableModel, ...], dict[str, tuple[str, ...]]]:
    if input_tokens < 0 or max_output_tokens < 0:
        raise ValueError("input_tokens and max_output_tokens must be >= 0")
    registry = CHAT_CALLS if chat else CALLS
    order = [provider] if provider else config.provider_order_list
    profile = RequestProfile(
        features=(),
        input_chars=input_tokens * CHARS_PER_TOKEN,
        max_output_tokens=max_output_tokens,
    )
    try:
        plan = plan_chain(
            order,
            config=config,
            policy=policy,
            profile=profile,
            registry=registry,
            model_override=model if provider else None,
        )
    except PolicyViolationError as exc:
        return (), dict(exc.exclusions)
    seen: set[str] = set()
    models: list[RoutableModel] = []
    for p in plan.providers:
        configured = CONFIGURED.get(p)
        if p in seen or p not in registry or (configured is not None and not configured(config)):
            continue
        seen.add(p)
        name = plan.models[p]
        price = lookup_price(p, name, config.model_prices, vertex_location=config.vertex_location)
        cost = worst_case_cost_usd(
            price, input_chars=profile.input_chars, max_output_tokens=max_output_tokens
        )
        models.append(RoutableModel(p, name, cost))
        fallback = plan.fallback_models.get(p)
        if fallback:
            fb_price = lookup_price(
                p, fallback, config.model_prices, vertex_location=config.vertex_location
            )
            fb_cost = worst_case_cost_usd(
                fb_price, input_chars=profile.input_chars, max_output_tokens=max_output_tokens
            )
            models.append(RoutableModel(p, fallback, fb_cost))
    return tuple(models), plan.exclusion_reasons()


def routable_models(
    config: GatewayConfig,
    *,
    policy: ProviderPolicy | None = None,
    provider: str | None = None,
    model: str | None = None,
    chat: bool = False,
) -> tuple[RoutableModel, ...]:
    """The provider/model pairs a call could be sent to, in chain order:
    `config.provider_order` (or just `provider`), minus providers without
    credentials, retired models (when `block_retired_models`) and whatever
    `config.policy` / `policy` exclude. `model` overrides the model only when
    `provider` is given, as for `complete(force_provider=..., model=...)`.
    The anthropic provider's one-shot `claude_model_fallback` is listed right
    after its primary model (when it differs, is not retired and no explicit
    `model` is given), as a call may be served by it.
    `chat=True` uses the `chat()` provider registry. Costs are for a zero-size
    call; use `estimate_max_cost` for real numbers."""
    return _plan(
        config,
        input_tokens=0,
        max_output_tokens=0,
        policy=policy,
        provider=provider,
        model=model,
        chat=chat,
    )[0]


def estimate_max_cost(
    config: GatewayConfig,
    *,
    input_tokens: int,
    max_output_tokens: int,
    policy: ProviderPolicy | None = None,
    provider: str | None = None,
    model: str | None = None,
    chat: bool = False,
) -> CostEstimate:
    """Worst-case cost of a call before it is made, over the models it can
    route to (see `routable_models`). Use it to reserve budget: reserve
    `max_cost_usd`, then settle with the real `Completion.cost_usd`. Uses the
    same upper-bound formula as `ProviderPolicy.max_cost_usd`. Never raises
    for an unroutable call: `models` is empty and `excluded` says why."""
    models, excluded = _plan(
        config,
        input_tokens=input_tokens,
        max_output_tokens=max_output_tokens,
        policy=policy,
        provider=provider,
        model=model,
        chat=chat,
    )
    unpriced = tuple(m.key for m in models if m.cost_usd is None)
    priced = [m.cost_usd for m in models if m.cost_usd is not None]
    top = max(priced) if priced and not unpriced else None
    return CostEstimate(models, top, unpriced, excluded)
