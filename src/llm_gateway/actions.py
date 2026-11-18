"""Action policy: a code-made decision for every tool call an agent proposes.

`ActionPolicy` authorises *actions* (which tool, for whom, with what authority). It is unrelated to
`ProviderPolicy` (policy.py), which only routes a call across providers.

Opt-in and engine-neutral: nothing here is wired into `complete()`. A caller builds an
`ActionGuard` and asks it before running each tool. Rules of the layer:

- default deny: an unknown tool or a missing rule is `block`;
- fail closed: an engine error, a bad answer, an approval timeout, handler error or a denial is
  `block`; silence never means yes;
- arguments are never stored raw: the log keeps a digest and a masked preview (actionlog.py).

`PolicyEngine` is shaped like the OpenID AuthZEN Authorization API 1.0 Access Evaluation request
(`subject`, `action`, `resource`, `context`; each of subject and resource has `type`, `id` and
optional `properties`; action has `name` and optional `properties`) and its response (`decision`
boolean plus optional `context`). A response context may carry `outcome` ("require_approval" or
"halt") and `reason_code` to refine a boolean decision. No engine is a dependency: Cedar, Cerbos or
OPA are reached by writing an adapter that maps these mappings to their own API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .actionlog import ActionLog

Outcome = Literal["allow", "require_approval", "block", "halt"]
OUTCOMES: tuple[str, ...] = ("allow", "require_approval", "block", "halt")


def arguments_digest(arguments: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the arguments (sorted keys, non-JSON values by repr)."""
    blob = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionRequest:
    """One proposed tool call. `delegation` is the authority chain, for example (user, app)."""

    principal: str
    agent: str
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    delegation: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical arguments; this, not the values, is what gets logged."""
        return arguments_digest(self.arguments)

    def summary(self) -> str:
        """Approver-facing text built by code from the argument names, never from model text."""
        names = ", ".join(sorted(str(k) for k in self.arguments)) or "no arguments"
        return f"Agent {self.agent!r} wants to run {self.tool!r} for {self.principal!r} ({names})"

    def to_access_request(self) -> dict[str, Any]:
        """The AuthZEN 1.0 Access Evaluation request. Raw argument values are not included."""
        return {
            "subject": {
                "type": "agent",
                "id": self.agent,
                "properties": {"principal": self.principal, "delegation": list(self.delegation)},
            },
            "action": {"name": "invoke", "properties": {"arguments_digest": self.digest}},
            "resource": {"type": "tool", "id": self.tool},
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class Decision:
    """The code-made verdict: outcome, reason code, and the policy id and version that made it."""

    outcome: Outcome
    reason: str
    policy_id: str = ""
    policy_version: str = ""

    @property
    def allowed(self) -> bool:
        """True only for an `allow` outcome."""
        return self.outcome == "allow"


@runtime_checkable
class PolicyEngine(Protocol):
    """AuthZEN-shaped engine: takes the Access Evaluation request, returns `{decision, context}`."""

    policy_id: str
    policy_version: str

    def evaluate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Evaluate an AuthZEN-shaped request; return `{decision: bool, context: {...}}`."""
        ...


@runtime_checkable
class ApprovalHandler(Protocol):
    """Asks a human. Return True to approve; anything else, an exception or a timeout is a block."""

    async def request_approval(
        self, request: ActionRequest, summary: str, expires_at: datetime
    ) -> bool:
        """Ask a person; return True to approve. Anything else, or an error, blocks."""
        ...


class Rule(BaseModel):
    """One declarative rule. Empty `agents` / `principals` match any."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(min_length=1)
    effect: Outcome
    agents: tuple[str, ...] = ()
    principals: tuple[str, ...] = ()
    reason: str = ""


class ActionPolicy(BaseModel):
    """Versioned rules as data. First matching rule wins; no match is a block (default deny)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    rules: tuple[Rule, ...] = ()

    @classmethod
    def load(cls, data: Mapping[str, Any]) -> ActionPolicy:
        """Build from plain data (for example parsed JSON). Unknown fields raise ValidationError."""
        return cls.model_validate(dict(data))

    def missing_rules(self, tools: Iterable[str]) -> list[str]:
        """Registered tools that no rule names: run it in a test so rules cannot drift."""
        named = {r.tool for r in self.rules}
        return sorted(t for t in tools if t not in named and "*" not in named)


class DeclarativeEngine:
    """Built-in engine over an `ActionPolicy`, speaking the AuthZEN-shaped protocol."""

    def __init__(self, policy: ActionPolicy) -> None:
        self.policy = policy
        self.policy_id = policy.id
        self.policy_version = policy.version

    def evaluate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """First matching rule wins; no match is a `block` with reason `no_rule`."""
        tool = request["resource"]["id"]
        agent = request["subject"]["id"]
        principal = request["subject"].get("properties", {}).get("principal")
        for rule in self.policy.rules:
            if rule.tool not in (tool, "*"):
                continue
            if rule.agents and agent not in rule.agents:
                continue
            if rule.principals and principal not in rule.principals:
                continue
            return {
                "decision": rule.effect == "allow",
                "context": {
                    "outcome": rule.effect,
                    "reason_code": rule.reason or f"rule:{rule.tool}",
                },
            }
        return {"decision": False, "context": {"outcome": "block", "reason_code": "no_rule"}}


def _interpret(response: Mapping[str, Any]) -> tuple[Outcome, str]:
    decision = response.get("decision")
    if not isinstance(decision, bool):
        return "block", "engine_bad_response"
    ctx = response.get("context")
    ctx = ctx if isinstance(ctx, Mapping) else {}
    reason = str(ctx.get("reason_code") or ("allowed" if decision else "denied"))[:100]
    outcome = ctx.get("outcome")
    if outcome == "halt":
        return "halt", reason
    if outcome == "require_approval":
        return "require_approval", reason
    # A permit is only honoured when nothing in the context refines it away.
    if decision and outcome in (None, "allow"):
        return "allow", reason
    return "block", reason


class ActionGuard:
    """Decides, optionally asks a human, logs, and never raises for a policy outcome."""

    def __init__(
        self,
        engine: PolicyEngine,
        *,
        log: ActionLog | None = None,
        approval: ApprovalHandler | None = None,
        approval_timeout: float = 300.0,
        validators: Mapping[str, Callable[[Mapping[str, Any]], None]] | None = None,
    ) -> None:
        self.engine = engine
        self.log = log
        self.approval = approval
        self.approval_timeout = approval_timeout
        self.validators = dict(validators or {})

    def _record(self, request: ActionRequest, decision: Decision) -> Decision:
        if self.log is not None:
            self.log.append(request, decision)
        return decision

    def _decision(self, outcome: Outcome, reason: str) -> Decision:
        return Decision(
            outcome,
            reason,
            str(getattr(self.engine, "policy_id", "")),
            str(getattr(self.engine, "policy_version", "")),
        )

    async def authorize(self, request: ActionRequest) -> Decision:
        """Validate, evaluate, ask for approval if required, log, and return the decision."""
        validator = self.validators.get(request.tool)
        if validator is not None:
            try:
                validator(request.arguments)
            except Exception:
                return self._record(request, self._decision("block", "arguments_invalid"))
        try:
            outcome, reason = _interpret(self.engine.evaluate(request.to_access_request()))
        except Exception as exc:
            # Only the exception type is kept: its message could contain arguments.
            return self._record(
                request, self._decision("block", f"engine_error:{type(exc).__name__}")
            )
        if outcome != "require_approval":
            return self._record(request, self._decision(outcome, reason))
        return self._record(request, await self._ask(request, reason))

    async def _ask(self, request: ActionRequest, reason: str) -> Decision:
        if self.approval is None:
            return self._decision("block", "approval_unavailable")
        expires_at = datetime.now(UTC) + timedelta(seconds=self.approval_timeout)
        try:
            approved = await asyncio.wait_for(
                self.approval.request_approval(request, request.summary(), expires_at),
                timeout=self.approval_timeout,
            )
        except TimeoutError:
            return self._decision("block", "approval_expired")
        except Exception:
            return self._decision("block", "approval_error")
        if datetime.now(UTC) > expires_at:
            return self._decision("block", "approval_expired")
        if approved is True:
            return self._decision("allow", "approved")
        return self._decision("block", "approval_denied")


__all__ = [
    "OUTCOMES",
    "ActionGuard",
    "ActionPolicy",
    "ActionRequest",
    "ApprovalHandler",
    "Decision",
    "DeclarativeEngine",
    "Outcome",
    "PolicyEngine",
    "Rule",
    "arguments_digest",
]
