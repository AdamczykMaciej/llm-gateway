"""Pre-flight routing: the provider chain one call is allowed to use.

`plan_chain()` runs before any network call. It merges the call's `policy=`
into the global policy (the result is only ever stricter), then excludes
every provider in `provider_order` (or the forced provider) that:

- fails a compliance requirement against its operator-asserted metadata
  (policy.py);
- can't serve a feature the request uses (capabilities.py);
- could cost more than `max_cost_usd`, or has no price to check (pricing.py).

If no configured provider is left, it raises `PolicyViolationError`, or
`UnsupportedCapabilityError` when capabilities were the only reason, naming
every excluded provider and why. The engines iterate only the plan's
chain, so failover can never reach an excluded provider, in streams too.

`ChainPlan.allows()` then asks the host app's budget hook right before each
attempt. A denial skips that provider like any other exclusion and is never
reported to the circuit breaker.

Exclusions are logged at DEBUG on the `llm_gateway` logger and recorded on
the call's span. Neither contains request content.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from opentelemetry import trace

from .capabilities import missing_capabilities
from .config import GatewayConfig
from .errors import PolicyViolationError, UnsupportedCapabilityError
from .policy import ProviderPolicy, call_budget_check, compliance_reasons, metadata_for
from .pricing import json_chars, lookup_price, message_chars, worst_case_cost_usd
from .providers import CONFIGURED, DEFAULT_MODEL

logger = logging.getLogger("llm_gateway")

_POLICY = "policy"
_CAPABILITY = "capability"
_COST = "cost"
_BUDGET = "budget"


def _has_images(messages: list[dict] | None) -> bool:
    for message in messages or ():
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in ("image_url", "image")
            for part in content
        ):
            return True
    return False


@dataclass(frozen=True)
class RequestProfile:
    """What a request needs from a provider: the features it uses and its
    size for the cost estimate. Holds no request content."""

    features: tuple[str, ...] = ()
    input_chars: int = 0
    max_output_tokens: int = 0

    @classmethod
    def for_completion(
        cls, *, system: str, prompt: str, max_tokens: int, output_schema: dict | None = None
    ) -> "RequestProfile":
        return cls(
            features=("structured_output",) if output_schema is not None else (),
            input_chars=len(system) + len(prompt) + json_chars(output_schema),
            max_output_tokens=max_tokens,
        )

    @classmethod
    def for_chat(
        cls,
        *,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: object,
        max_tokens: int,
        response_format: dict | None,
        streaming: bool = False,
    ) -> "RequestProfile":
        features = []
        if tools and tool_choice != "none":
            features.append("tools")
        if _has_images(messages):
            features.append("images")
        if streaming:
            features.append("streaming")
        return cls(
            features=tuple(features),
            input_chars=message_chars(messages) + json_chars(tools) + json_chars(response_format),
            max_output_tokens=max_tokens,
        )


@dataclass
class ChainPlan:
    policy: ProviderPolicy
    """The effective (merged) policy."""
    providers: list[str] = field(default_factory=list)
    """The chain the engine may try, in order."""
    models: dict[str, str] = field(default_factory=dict)
    estimated_costs: dict[str, float | None] = field(default_factory=dict)
    exclusions: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    denied: list[str] = field(default_factory=list)
    """Providers the budget hook denied during the call."""

    def exclusion_reasons(self) -> dict[str, tuple[str, ...]]:
        return {p: tuple(reason for _, reason in items) for p, items in self.exclusions.items()}

    def _exclude(self, provider: str, kind: str, reason: str) -> None:
        self.exclusions.setdefault(provider, []).append((kind, reason))

    async def allows(self, provider: str, span: trace.Span | None = None) -> bool:
        """Ask the budget hook whether `provider` may be attempted now."""
        check = self.policy.budget_check
        if check is None or provider not in self.models:
            return True
        model = self.models[provider]
        if await call_budget_check(check, provider, model, self.estimated_costs[provider]):
            return True
        self._exclude(provider, _BUDGET, "denied by budget_check")
        self.denied.append(provider)
        logger.debug("llm_gateway budget_check denied provider=%s model=%s", provider, model)
        record_plan(span, self)
        return False

    def violation(self) -> PolicyViolationError:
        reasons = self.exclusion_reasons()
        detail = "; ".join(
            f"{provider} ({self.models.get(provider) or 'no model'}): {', '.join(items)}"
            for provider, items in reasons.items()
        )
        kinds = {kind for items in self.exclusions.values() for kind, _ in items}
        if kinds == {_CAPABILITY}:
            return UnsupportedCapabilityError(
                f"No provider can serve this request's features. Excluded: {detail}.", reasons
            )
        return PolicyViolationError(
            f"No provider satisfies the routing policy ({self.policy.describe()}). "
            f"Excluded: {detail}.",
            reasons,
        )

    def failure_note(self) -> str:
        """Appended to "All configured providers failed" when the budget hook
        also denied providers."""
        if not self.denied:
            return ""
        return f" Not attempted (denied by budget_check): {', '.join(self.denied)}."


def _available(provider: str, config: GatewayConfig, registry: Mapping) -> bool:
    if provider not in registry:
        return False
    configured = CONFIGURED.get(provider)
    return configured(config) if configured else True


def plan_chain(
    order: list[str],
    *,
    config: GatewayConfig,
    policy: ProviderPolicy | None,
    profile: RequestProfile,
    registry: Mapping,
    model_override: str | None,
    span: trace.Span | None = None,
) -> ChainPlan:
    """The chain a call may use. Raises `PolicyViolationError` when policy
    exclusions leave no configured provider. With no policy configured and a
    request every provider can serve, `plan.providers == order`."""
    effective = config.policy.narrowed_by(policy)
    plan = ChainPlan(policy=effective)
    for provider in order:
        if provider not in registry:
            # Not callable by this engine; it skips the provider as before.
            plan.providers.append(provider)
            continue
        if provider in plan.models:  # listed twice in provider_order
            if provider not in plan.exclusions:
                plan.providers.append(provider)
            continue
        default_model = DEFAULT_MODEL.get(provider)
        model = model_override or (default_model(config) if default_model else "")
        estimate = worst_case_cost_usd(
            lookup_price(
                provider, model, config.model_prices, vertex_location=config.vertex_location
            ),
            input_chars=profile.input_chars,
            max_output_tokens=profile.max_output_tokens,
        )
        plan.models[provider] = model
        plan.estimated_costs[provider] = estimate

        metadata = metadata_for(config.provider_metadata, provider)
        for reason in compliance_reasons(effective, provider, metadata):
            plan._exclude(provider, _POLICY, reason)
        for reason in missing_capabilities(
            provider,
            model,
            profile.features,
            require_parameters=effective.require_parameters,
            vertex_structured_outputs=config.vertex_structured_outputs,
        ):
            plan._exclude(provider, _CAPABILITY, reason)
        cap = effective.max_cost_usd
        if cap is not None:
            if estimate is None:
                plan._exclude(
                    provider, _COST, f"no price for {provider}/{model} to check max_cost_usd"
                )
            elif estimate > cap:
                plan._exclude(
                    provider,
                    _COST,
                    f"estimated worst-case cost ${estimate:.6f} exceeds max_cost_usd ${cap:g}",
                )
        if provider not in plan.exclusions:
            plan.providers.append(provider)

    if effective.sort == "price":
        # Stable: equal estimates keep provider_order; unpriced providers last.
        plan.providers.sort(
            key=lambda p: (plan.estimated_costs.get(p) is None, plan.estimated_costs.get(p) or 0.0)
        )

    for provider, reasons in plan.exclusion_reasons().items():
        logger.debug(
            "llm_gateway policy excluded provider=%s reasons=%s", provider, "; ".join(reasons)
        )
    record_plan(span, plan)

    if (
        plan.exclusions
        and not any(_available(p, config, registry) for p in plan.providers)
        and any(_available(p, config, registry) for p in plan.exclusions)
    ):
        raise plan.violation()
    return plan


def record_plan(span: trace.Span | None, plan: ChainPlan) -> None:
    """Span attributes for the plan. Never raises."""
    if span is None:
        return
    try:
        span.set_attribute("llm_gateway.policy", plan.policy.describe())
        span.set_attribute("llm_gateway.policy.eligible_providers", list(plan.providers))
        if plan.exclusions:
            span.set_attribute("llm_gateway.policy.excluded_providers", list(plan.exclusions))
            span.set_attribute(
                "llm_gateway.policy.exclusion_reasons",
                [f"{p}: {'; '.join(r)}" for p, r in plan.exclusion_reasons().items()],
            )
    except Exception:  # noqa: BLE001 — tracing must never break the real request
        pass


def record_cost(span: trace.Span, cost_usd: float | None) -> None:
    if cost_usd is None:
        return
    try:
        span.set_attribute("llm_gateway.cost_usd", cost_usd)
    except Exception:  # noqa: BLE001 — tracing must never break the real request
        pass
