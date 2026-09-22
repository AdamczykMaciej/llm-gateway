"""Provider compliance metadata and the routing policy over the provider chain.

Two pieces, both driven by configuration:

- `ProviderMetadata`: what the *operator* asserts about a provider id (where
  prompts are processed, how long they are retained, whether they are used
  for training, whether a DPA is signed). The library never asserts these
  facts on a vendor's behalf: a provider with no metadata gets the
  conservative defaults (region and retention "unknown", trains on data, no
  DPA), which fail every compliance requirement.
- `ProviderPolicy`: the requirements one call must meet. `GatewayConfig`
  holds the global policy (`POLICY_*` settings); a per-call `policy=` is
  merged with it by `narrowed_by()`, which only ever makes it stricter.

routing.py applies both to `provider_order` before any network call.
"""

import inspect
import math
import re
from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from .errors import PolicyViolationError

# Provider ids the policy understands (the providers/ registry's ids).
KNOWN_PROVIDERS: tuple[str, ...] = (
    "anthropic",
    "groq",
    "openai",
    "azure",
    "vertex",
    "mistral",
    "openrouter",
    "openai_compat",
)

Retention = Literal["zero", "abuse_monitoring_30d", "unknown"]
SortOrder = Literal["order", "price"]

_REGION = re.compile(r"[a-z][a-z0-9_-]{0,31}")


def normalize_region(value: str, *, field_name: str) -> str:
    region = str(value).strip().lower()
    if not _REGION.fullmatch(region):
        raise ValueError(
            f"{field_name} must be a short lowercase region code such as 'eu', 'us' or "
            f"'global', got {value!r}"
        )
    return region


class ProviderMetadata(BaseModel):
    """Operator-asserted compliance profile of one provider id.

    Every default is the conservative answer, so a provider only passes a
    requirement once the deploying app has asserted the fact explicitly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    region: str = "unknown"
    """Where prompts are processed, e.g. "eu", "us", "global"."""
    retention: Retention = "unknown"
    trains_on_data: bool = True
    dpa: bool = False
    notes: str = ""

    @field_validator("region")
    @classmethod
    def _check_region(cls, value: str) -> str:
        return normalize_region(value, field_name="region")


CONSERVATIVE_METADATA = ProviderMetadata()


def provider_ids(value: Iterable[str] | str, *, field_name: str) -> tuple[str, ...]:
    """Validated, lowercased, de-duplicated provider ids. A string is split
    on commas, like `PROVIDER_ORDER`."""
    items = value.split(",") if isinstance(value, str) else list(value)
    ids = tuple(dict.fromkeys(str(item).strip().lower() for item in items if str(item).strip()))
    unknown = [i for i in ids if i not in KNOWN_PROVIDERS]
    if unknown:
        raise ValueError(
            f"{field_name}: unknown provider id(s) {', '.join(map(repr, unknown))}; "
            f"known ids are {', '.join(KNOWN_PROVIDERS)}"
        )
    return ids


def validate_provider_metadata(
    metadata: Mapping[str, ProviderMetadata],
) -> dict[str, ProviderMetadata]:
    """Reject metadata for an unknown provider id: a typo would otherwise
    leave the real provider on the conservative defaults without a hint."""
    ids = provider_ids(metadata.keys(), field_name="provider_metadata")
    return dict(zip(ids, metadata.values(), strict=True))


def metadata_for(metadata: Mapping[str, ProviderMetadata], provider: str) -> ProviderMetadata:
    return metadata.get(provider, CONSERVATIVE_METADATA)


class BudgetCheck(Protocol):
    """Host-app hook called before each provider attempt with the provider
    id, the model and the call's estimated worst-case cost in USD (`None`
    when the model has no price). Return a truthy value to allow the attempt;
    may be sync or async. A denial excludes that provider for this call and
    never counts toward its circuit breaker. An exception propagates."""

    def __call__(
        self, provider: str, model: str, estimated_cost_usd: float | None, /
    ) -> bool | Awaitable[bool]: ...


async def call_budget_check(
    check: BudgetCheck, provider: str, model: str, estimated_cost_usd: float | None
) -> bool:
    result = check(provider, model, estimated_cost_usd)
    if inspect.isawaitable(result):
        result = await result
    return bool(result)


def _all_budget_checks(first: BudgetCheck | None, second: BudgetCheck | None) -> BudgetCheck | None:
    if first is None:
        return second
    if second is None:
        return first

    async def both(provider: str, model: str, estimated_cost_usd: float | None, /) -> bool:
        return await call_budget_check(
            first, provider, model, estimated_cost_usd
        ) and await call_budget_check(second, provider, model, estimated_cost_usd)

    return both


@dataclass(frozen=True)
class ProviderPolicy:
    """Requirements a provider must meet to serve a call.

    - `residency`: only providers whose asserted `region` equals this.
    - `require_zero_retention`: only providers with `retention="zero"`.
    - `forbid_training`: only providers asserted `trains_on_data=False`.
    - `require_dpa`: only providers asserted `dpa=True`.
    - `only` / `ignore`: provider ids (a list or a comma-separated string).
      `only=None` means no restriction; an empty `only` allows nothing.
    - `require_parameters`: also skip models whose support for a requested
      feature is unknown, and Groq models that only have JSON mode for
      `output_schema` (see capabilities.py).
    - `max_cost_usd`: skip providers whose estimated worst-case cost for the
      call exceeds this, or whose model has no price.
    - `sort`: "order" keeps `provider_order`; "price" tries the cheapest
      estimated provider first. `None` inherits.
    - `budget_check`: see `BudgetCheck`.
    """

    residency: str | None = None
    require_zero_retention: bool = False
    forbid_training: bool = False
    require_dpa: bool = False
    only: tuple[str, ...] | None = None
    ignore: tuple[str, ...] = ()
    require_parameters: bool = False
    max_cost_usd: float | None = None
    sort: SortOrder | None = None
    budget_check: BudgetCheck | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.residency is not None:
            residency = normalize_region(self.residency, field_name="residency")
            if residency == "unknown":
                raise ValueError("residency cannot be 'unknown'")
            object.__setattr__(self, "residency", residency)
        if self.only is not None:
            object.__setattr__(self, "only", provider_ids(self.only, field_name="only"))
        object.__setattr__(self, "ignore", provider_ids(self.ignore, field_name="ignore"))
        if self.max_cost_usd is not None:
            cap = self.max_cost_usd
            if (
                isinstance(cap, bool)
                or not isinstance(cap, int | float)
                or math.isnan(cap)
                or cap < 0
            ):
                raise ValueError(f"max_cost_usd must be a non-negative number, got {cap!r}")
            object.__setattr__(self, "max_cost_usd", float(cap))
        if self.sort not in (None, "order", "price"):
            raise ValueError(f"sort must be 'order' or 'price', got {self.sort!r}")
        if self.budget_check is not None and not callable(self.budget_check):
            raise TypeError("budget_check must be callable")

    def describe(self) -> str:
        """The active requirements, for error messages and traces."""
        parts = []
        if self.residency:
            parts.append(f"residency={self.residency}")
        for name in (
            "require_zero_retention",
            "forbid_training",
            "require_dpa",
            "require_parameters",
        ):
            if getattr(self, name):
                parts.append(f"{name}=true")
        if self.only is not None:
            parts.append(f"only={','.join(self.only) or '<none>'}")
        if self.ignore:
            parts.append(f"ignore={','.join(self.ignore)}")
        if self.max_cost_usd is not None:
            parts.append(f"max_cost_usd={self.max_cost_usd:g}")
        if self.sort == "price":
            parts.append("sort=price")
        if self.budget_check is not None:
            parts.append("budget_check")
        return ", ".join(parts) or "none"

    def narrowed_by(self, other: "ProviderPolicy | None") -> "ProviderPolicy":
        """This policy combined with `other` so that both hold: flags are
        OR-ed, `only` lists intersect, `ignore` lists union, the lower cost
        cap wins and both budget checks must allow. `other` can never loosen
        this policy. `sort` only orders providers that already qualify, so
        `other.sort` wins when set.

        Raises `PolicyViolationError` when the two residencies differ, since
        no provider can be in two regions."""
        if other is None:
            return self
        if self.residency and other.residency and self.residency != other.residency:
            raise PolicyViolationError(
                f"Per-call residency={other.residency} conflicts with the global "
                f"residency={self.residency}; a per-call policy can only narrow the global one."
            )
        if self.only is None:
            only = other.only
        elif other.only is None:
            only = self.only
        else:
            only = tuple(p for p in self.only if p in other.only)
        caps = [c for c in (self.max_cost_usd, other.max_cost_usd) if c is not None]
        return ProviderPolicy(
            residency=self.residency or other.residency,
            require_zero_retention=self.require_zero_retention or other.require_zero_retention,
            forbid_training=self.forbid_training or other.forbid_training,
            require_dpa=self.require_dpa or other.require_dpa,
            only=only,
            ignore=self.ignore + other.ignore,
            require_parameters=self.require_parameters or other.require_parameters,
            max_cost_usd=min(caps) if caps else None,
            sort=other.sort or self.sort,
            budget_check=_all_budget_checks(self.budget_check, other.budget_check),
        )


def compliance_reasons(
    policy: ProviderPolicy, provider: str, metadata: ProviderMetadata
) -> list[str]:
    """Why `provider` fails `policy`'s compliance requirements (empty when it
    passes). Mentions only config values, never request content."""
    reasons = []
    if provider in policy.ignore:
        reasons.append("listed in ignore")
    if policy.only is not None and provider not in policy.only:
        reasons.append("not listed in only")
    if policy.residency and metadata.region != policy.residency:
        reasons.append(f"region={metadata.region} does not match residency={policy.residency}")
    if policy.require_zero_retention and metadata.retention != "zero":
        reasons.append(f"retention={metadata.retention}, zero retention required")
    if policy.forbid_training and metadata.trains_on_data:
        reasons.append("trains_on_data is not asserted false")
    if policy.require_dpa and not metadata.dpa:
        reasons.append("no DPA asserted")
    return reasons
