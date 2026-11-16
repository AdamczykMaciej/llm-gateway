"""Case and sample schema (JSONL on disk, one object per line)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class Turn:
    """One conversation turn. ``role`` is ``user`` or ``assistant``."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Case:
    """An evaluation scenario, single- or multi-turn.

    ``turns`` is the scripted conversation. Every user turn that is not followed by a scripted
    assistant turn is a point where the system under test is asked for a reply, so a
    single-turn case has one user turn and a multi-turn case has several.
    ``must`` / ``must_not`` are plain-language rubric items for the judge; ``checks`` are
    deterministic checks (see ``checks.py``) that decide pass/fail without any model.
    """

    id: str
    turns: tuple[Turn, ...]
    must: tuple[str, ...] = ()
    must_not: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    language: str = "en"
    context: str = ""
    rubric_focus: str = ""
    checks: tuple[dict[str, Any], ...] = ()
    meta: dict[str, Any] = field(default_factory=lambda: {})

    @property
    def is_multi_turn(self) -> bool:
        return sum(1 for t in self.turns if t.role == "user") > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turns": [{"role": t.role, "content": t.content} for t in self.turns],
            "must": list(self.must),
            "must_not": list(self.must_not),
            "tags": list(self.tags),
            "language": self.language,
            "context": self.context,
            "rubric_focus": self.rubric_focus,
            "checks": list(self.checks),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Case:
        if "id" not in d:
            raise ValueError("case is missing 'id'")
        raw_turns = d.get("turns")
        if raw_turns is None and "message" in d:  # single-turn shorthand
            raw_turns = [{"role": "user", "content": d["message"]}]
        if not raw_turns:
            raise ValueError(f"case {d['id']!r} has no turns")
        turns: list[Turn] = []
        for t in raw_turns:
            if t.get("role") not in ("user", "assistant"):
                raise ValueError(f"case {d['id']!r}: bad turn role {t.get('role')!r}")
            turns.append(Turn(t["role"], str(t["content"])))
        if turns[-1].role != "user" and not any(t.role == "user" for t in turns):
            raise ValueError(f"case {d['id']!r} has no user turn")
        return cls(
            id=str(d["id"]),
            turns=tuple(turns),
            must=tuple(d.get("must", ())),
            must_not=tuple(d.get("must_not", ())),
            tags=tuple(d.get("tags", ())),
            language=str(d.get("language", d.get("lang", "en"))),
            context=str(d.get("context", "")),
            rubric_focus=str(d.get("rubric_focus", "")),
            checks=tuple(d.get("checks", ())),
            meta=dict(d.get("meta", {})),
        )


@dataclass(frozen=True)
class Sample:
    """Outputs of one system variant for one case: one reply per answered user turn."""

    case_id: str
    variant: str
    replies: tuple[str, ...]
    sample: int = 0
    meta: dict[str, Any] = field(default_factory=lambda: {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "variant": self.variant,
            "replies": list(self.replies),
            "sample": self.sample,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Sample:
        replies = d["replies"] if "replies" in d else [d["reply"]]
        return cls(
            case_id=str(d["case_id"]),
            variant=str(d["variant"]),
            replies=tuple(str(r) for r in replies),
            sample=int(d.get("sample", 0)),
            meta=dict(d.get("meta", {})),
        )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def load_cases(path: str | Path) -> list[Case]:
    """Load cases from JSONL; ids must be unique."""
    cases = [Case.from_dict(r) for r in _read_jsonl(path)]
    ids = [c.id for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate case ids: {dupes}")
    return cases


def dump_cases(path: str | Path, cases: Iterable[Case]) -> None:
    _write_jsonl(path, (c.to_dict() for c in cases))


def load_samples(path: str | Path) -> list[Sample]:
    return [Sample.from_dict(r) for r in _read_jsonl(path)]


def dump_samples(path: str | Path, samples: Iterable[Sample]) -> None:
    _write_jsonl(path, (s.to_dict() for s in samples))
