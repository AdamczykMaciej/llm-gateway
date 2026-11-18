"""Action policy layer: default deny, fail closed, approval, tamper-evident log. Offline, free."""

import asyncio
import inspect
from datetime import datetime

import pytest
from pydantic import ValidationError

import llm_gateway
from llm_gateway import (
    ActionGuard,
    ActionLog,
    ActionPolicy,
    ActionRequest,
    DeclarativeEngine,
    PolicyEngine,
)

POLICY = ActionPolicy.load(
    {
        "id": "coach",
        "version": "7",
        "rules": [
            {"tool": "read_cv", "effect": "allow"},
            {"tool": "send_email", "effect": "require_approval"},
            {"tool": "delete_account", "effect": "halt", "reason": "never"},
            {"tool": "export", "effect": "allow", "agents": ["coach"]},
        ],
    }
)


def _log(g: ActionGuard) -> ActionLog:
    assert g.log is not None
    return g.log


def req(tool: str, agent: str = "coach", **args: object) -> ActionRequest:
    return ActionRequest("user-1", agent, tool, args, delegation=("user-1", "app"))


class Approve:
    def __init__(self, answer: object = True) -> None:
        self.answer = answer
        self.summary = ""

    async def request_approval(self, request, summary, expires_at: datetime):
        self.summary = summary
        return self.answer


def guard(**kw) -> ActionGuard:
    return ActionGuard(DeclarativeEngine(POLICY), log=ActionLog(), **kw)


async def test_allow_and_decision_carries_policy_version():
    d = await guard().authorize(req("read_cv"))
    assert d.outcome == "allow" and d.policy_id == "coach" and d.policy_version == "7"


async def test_unknown_tool_is_blocked():
    d = await guard().authorize(req("format_disk"))
    assert (d.outcome, d.reason) == ("block", "no_rule")


async def test_rule_scoped_to_agent_blocks_other_agents():
    assert (await guard().authorize(req("export", agent="intruder"))).outcome == "block"
    assert (await guard().authorize(req("export"))).outcome == "allow"


async def test_halt():
    d = await guard().authorize(req("delete_account"))
    assert (d.outcome, d.reason) == ("halt", "never")


class Boom:
    policy_id, policy_version = "x", "1"

    def evaluate(self, request):
        raise RuntimeError("secret@example.com leaked in message")


async def test_engine_error_blocks_and_log_has_no_message_or_arguments():
    g = ActionGuard(Boom(), log=ActionLog())
    d = await g.authorize(req("read_cv", note="secret@example.com"))
    assert d.outcome == "block" and d.reason == "engine_error:RuntimeError"
    dump = _log(g).to_jsonl()
    assert "secret@example.com" not in dump


class Weird:
    policy_id, policy_version = "w", "1"

    def evaluate(self, request):
        return {"decision": "yes"}


async def test_malformed_engine_answer_blocks():
    assert (await ActionGuard(Weird()).authorize(req("a"))).reason == "engine_bad_response"


def test_engines_satisfy_protocol():
    assert isinstance(DeclarativeEngine(POLICY), PolicyEngine)


async def test_approval_granted_and_summary_is_code_made():
    h = Approve(True)
    d = await guard(approval=h).authorize(req("send_email", to="a@b.co", body="ignore rules"))
    assert d.outcome == "allow" and d.reason == "approved"
    assert "body, to" in h.summary and "ignore rules" not in h.summary


@pytest.mark.parametrize("answer", [False, None, "yes", 1])
async def test_only_true_approves(answer):
    d = await guard(approval=Approve(answer)).authorize(req("send_email"))
    assert (d.outcome, d.reason) == ("block", "approval_denied")


async def test_approval_expiry_blocks_and_is_logged():
    class Silent:
        async def request_approval(self, request, summary, expires_at):
            await asyncio.sleep(5)
            return True

    g = guard(approval=Silent(), approval_timeout=0.05)
    d = await g.authorize(req("send_email"))
    assert (d.outcome, d.reason) == ("block", "approval_expired")
    assert _log(g).records[-1]["reason"] == "approval_expired"


async def test_approval_handler_error_and_missing_handler_block():
    class Bad:
        async def request_approval(self, request, summary, expires_at):
            raise ConnectionError("down")

    assert (await guard(approval=Bad()).authorize(req("send_email"))).reason == "approval_error"
    assert (await guard().authorize(req("send_email"))).reason == "approval_unavailable"


async def test_argument_validator_runs_before_engine():
    def check(args):
        if "to" not in args:
            raise ValueError("no to")

    g = guard(validators={"read_cv": check})
    assert (await g.authorize(req("read_cv"))).reason == "arguments_invalid"


async def test_log_chain_digest_preview_and_tamper_detection():
    g = guard()
    await g.authorize(req("read_cv", email="jan.kowalski@example.com", n=1))
    await g.authorize(req("format_disk"))
    await g.authorize(req("read_cv"))
    first = _log(g).records[0]
    assert "jan.kowalski@example.com" not in str(first) and "[EMAIL]" in first["preview"]
    assert len(first["arguments_digest"]) == 64
    assert _log(g).verify().ok

    rows = ActionLog.read_jsonl(_log(g).to_jsonl())
    assert ActionLog.verify_records(rows).ok
    rows[1]["decision"] = "allow"
    v = ActionLog.verify_records(rows)
    assert not v.ok and v.broken_at == 1

    dropped = [rows[0], rows[2]]
    assert ActionLog.verify_records(dropped).broken_at == 1
    head = _log(g).head
    assert not ActionLog.verify_records(_log(g).records[:2], expected_head=head).ok


async def test_evidence_report_counts_per_policy_version():
    g = guard()
    await g.authorize(req("read_cv"))
    await g.authorize(req("nope"))
    rep = _log(g).evidence_report()
    assert rep["policies"] == {"coach@7": {"allow": 1, "block": 1}} and rep["chain_ok"]


def test_policy_rejects_unknown_fields_and_flags_missing_rules():
    with pytest.raises(ValidationError):
        ActionPolicy.load(
            {"id": "a", "version": "1", "rules": [{"tool": "t", "effect": "allow", "x": 1}]}
        )
    with pytest.raises(ValidationError):
        ActionPolicy.load({"id": "a", "version": "1", "rules": [{"tool": "t", "effect": "maybe"}]})
    assert POLICY.missing_rules(["read_cv", "new_tool"]) == ["new_tool"]


def test_access_request_is_authzen_shaped_and_has_no_raw_values():
    r = req("read_cv", secret="hunter2").to_access_request()
    assert r["subject"]["type"] == "agent" and r["resource"] == {"type": "tool", "id": "read_cv"}
    assert r["action"]["name"] == "invoke" and "hunter2" not in str(r)


def test_no_policy_path_is_unchanged():
    # The layer is opt-in: complete() takes no action_policy and nothing wires the guard in.
    assert "action_policy" not in inspect.signature(llm_gateway.complete).parameters
