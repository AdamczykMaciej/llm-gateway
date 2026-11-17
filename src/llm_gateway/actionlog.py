"""Append-only, hash-chained decision log for the action policy layer.

Each record holds who asked, which tool, the argument *digest*, a short PII-masked preview (reusing
`mask_pii`), the decision, reason and policy version, chained to the previous record by SHA-256.
Raw arguments are never stored. Exports: JSONL, OpenTelemetry span events, and an evidence report
(decisions per policy version) for record-keeping such as EU AI Act / GDPR. Not legal advice.

The chain detects edits, reordering and removal in the middle; truncation of the newest records is
only detected if the caller keeps the head hash (`head`) elsewhere and passes it as `expected_head`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

from .actions import ActionRequest, Decision
from .pii import mask_pii

GENESIS = "0" * 64
PREVIEW_CHARS = 120
_FIELDS = (
    "seq",
    "ts",
    "principal",
    "agent",
    "tool",
    "arguments_digest",
    "preview",
    "decision",
    "reason",
    "policy_id",
    "policy_version",
    "prev_hash",
)


def _hash(record: Mapping[str, Any]) -> str:
    body = {k: record.get(k) for k in _FIELDS}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _preview(arguments: Mapping[str, Any]) -> str:
    blob = json.dumps(arguments, sort_keys=True, default=repr, ensure_ascii=False)
    return (mask_pii(blob) or "")[:PREVIEW_CHARS]


@dataclass(frozen=True)
class Verification:
    """Result of a chain check: `broken_at` is the index of the first bad record."""

    ok: bool
    records: int
    broken_at: int | None = None  # index of the first bad record
    reason: str = ""


class ActionLog:
    """In-memory append-only log; persist it with `to_jsonl()` (and keep `head` elsewhere)."""

    def __init__(self, *, emit_otel: bool = True) -> None:
        self._records: list[dict[str, Any]] = []
        self._emit_otel = emit_otel

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Copies of all records."""
        return tuple(dict(r) for r in self._records)

    @property
    def head(self) -> str:
        """Hash of the newest record (all zeros when empty); store it to detect truncation."""
        return self._records[-1]["hash"] if self._records else GENESIS

    def append(self, request: ActionRequest, decision: Decision) -> dict[str, Any]:
        """Chain one decision onto the log; arguments are stored as digest and masked preview."""
        record: dict[str, Any] = {
            "seq": len(self._records),
            "ts": datetime.now(UTC).isoformat(),
            "principal": request.principal,
            "agent": request.agent,
            "tool": request.tool,
            "arguments_digest": request.digest,
            "preview": _preview(request.arguments),
            "decision": decision.outcome,
            "reason": decision.reason,
            "policy_id": decision.policy_id,
            "policy_version": decision.policy_version,
            "prev_hash": self.head,
        }
        record["hash"] = _hash(record)
        self._records.append(record)
        if self._emit_otel:
            span = trace.get_current_span()
            if span.is_recording():
                span.add_event(
                    "action_policy.decision",
                    {k: str(record[k]) for k in ("tool", "agent", "decision", "reason", "seq")}
                    | {"policy_version": record["policy_version"], "hash": record["hash"]},
                )
        return dict(record)

    @staticmethod
    def verify_records(
        records: Iterable[Mapping[str, Any]], *, expected_head: str | None = None
    ) -> Verification:
        """Check any record sequence (for example read back from JSONL)."""
        prev = GENESIS
        count = 0
        for i, rec in enumerate(records):
            count += 1
            if rec.get("seq") != i:
                return Verification(False, count, i, "sequence")
            if rec.get("prev_hash") != prev:
                return Verification(False, count, i, "prev_hash")
            if rec.get("hash") != _hash(rec):
                return Verification(False, count, i, "hash")
            prev = rec["hash"]
        if expected_head is not None and expected_head != prev:
            return Verification(False, count, count, "head")
        return Verification(True, count)

    def verify(self, *, expected_head: str | None = None) -> Verification:
        """Check this log's chain."""
        return self.verify_records(self._records, expected_head=expected_head)

    def to_jsonl(self) -> str:
        """One JSON record per line."""
        return "".join(json.dumps(r, sort_keys=True) + "\n" for r in self._records)

    @staticmethod
    def read_jsonl(text: str) -> list[dict[str, Any]]:
        """Parse `to_jsonl()` output."""
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def evidence_report(self) -> dict[str, Any]:
        """Decisions per policy version, plus the chain state. Counts only; no arguments."""
        per: dict[str, Counter[str]] = {}
        for r in self._records:
            per.setdefault(f"{r['policy_id']}@{r['policy_version']}", Counter())[r["decision"]] += 1
        v = self.verify()
        return {
            "records": len(self._records),
            "chain_ok": v.ok,
            "head": self.head,
            "policies": {k: dict(c) for k, c in sorted(per.items())},
        }
