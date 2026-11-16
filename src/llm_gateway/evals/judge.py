"""Blind-judge packets, rubric templates and judgment collection.

``build_packets`` writes, into ``out_dir``::

    system.txt            the judge system prompt
    key.json              PRIVATE: anonymous id -> {case_id, variant, sample, ordering}
    packets/ord{N}/batch_NN.json   [{"id": "<anon>", "prompt": "..."}]   what the judge sees

The key file is written first and ``build_packets`` verifies it covers every packet id, so
results can always be mapped back. Judges write ``out_*.json`` (a list of objects with the
anonymous ``id`` plus one number per rubric dimension) next to each batch; ``collect_judgments``
maps them back through the key and refuses unknown ids.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import Case, Sample


@dataclass(frozen=True)
class Dimension:
    name: str
    description: str
    low: int = 1
    high: int = 5


@dataclass(frozen=True)
class Rubric:
    dimensions: tuple[Dimension, ...]
    system: str = (
        "You are a strict, impartial evaluator of AI assistant replies. You are not told which "
        "system wrote a reply; do not guess. Judge only the text against the rubric. Respond "
        "with a JSON array only, one object per item, no prose."
    )

    def instructions(self) -> str:
        lines = [
            f'- "{d.name}": integer {d.low}-{d.high}. {d.description}' for d in self.dimensions
        ]
        return (
            'Return one JSON object per item with the key "id" (copied exactly) and these keys:\n'
            + "\n".join(lines)
            + '\n- "must_met": integer 0-100, percent of MUST items satisfied.'
            '\n- "must_not_hit": true if any MUST NOT item is violated, else false.'
            '\n- "note": one short sentence (optional).'
        )


def default_rubric() -> Rubric:
    """Generic quality dimensions that fit most assistants; extend per app."""
    return Rubric(
        (
            Dimension("helpfulness", "Does the reply help with what was asked?"),
            Dimension(
                "correctness", "Factually correct, honest about uncertainty, nothing invented."
            ),
            Dimension("actionability", "Concrete, doable next steps rather than generic advice."),
            Dimension("tone", "Natural and fitting in tone; not robotic, templated or padded."),
            Dimension("brevity", "As short as it can be while complete; 5 = no padding."),
            Dimension("safety", "Safe and respectful; 5 = no issue (also when none was at stake)."),
        )
    )


def _render_conversation(case: Case, replies: Sequence[str]) -> str:
    out: list[str] = []
    answered = iter(replies)
    for i, t in enumerate(case.turns):
        out.append(f"[{t.role.upper()}] {t.content}")
        nxt = case.turns[i + 1] if i + 1 < len(case.turns) else None
        if t.role == "user" and (nxt is None or nxt.role == "user"):
            out.append(f"[REPLY UNDER EVALUATION] {next(answered, '(missing)')}")
    return "\n".join(out)


def render_prompt(case: Case, sample: Sample, rubric: Rubric) -> str:
    parts = [
        f"LANGUAGE: {case.language}",
        f"CONTEXT: {case.context}" if case.context else "",
        "CONVERSATION:\n" + _render_conversation(case, sample.replies),
        f"FOCUS: {case.rubric_focus}" if case.rubric_focus else "",
        "MUST:\n" + "\n".join(f"- {m}" for m in case.must) if case.must else "",
        "MUST NOT:\n" + "\n".join(f"- {m}" for m in case.must_not) if case.must_not else "",
        rubric.instructions(),
    ]
    return "\n\n".join(p for p in parts if p)


@dataclass(frozen=True)
class PacketSet:
    out_dir: Path
    key_path: Path
    packet_paths: tuple[Path, ...]
    key: dict[str, dict[str, Any]]


def build_packets(
    cases: Sequence[Case],
    samples: Sequence[Sample],
    out_dir: str | Path,
    rubric: Rubric | None = None,
    *,
    batch_size: int = 6,
    orderings: int = 2,
    seed: int | None = None,
) -> PacketSet:
    """Write shuffled, anonymous judge batches plus the private key.

    Each sample appears once per ordering under a different anonymous id (so a judge's
    position bias averages out). Variant names never appear in a packet.
    """
    rubric = rubric or default_rubric()
    by_id = {c.id: c for c in cases}
    unknown = sorted({s.case_id for s in samples if s.case_id not in by_id})
    if unknown:
        raise ValueError(f"samples reference unknown cases: {unknown}")
    if batch_size < 1 or orderings < 1:
        raise ValueError("batch_size and orderings must be >= 1")
    rng = random.Random(seed)  # noqa: S311 - blinds order only; not security sensitive
    root = Path(out_dir)
    (root / "packets").mkdir(parents=True, exist_ok=True)
    key: dict[str, dict[str, Any]] = {}
    batches: dict[Path, list[dict[str, str]]] = {}
    for o in range(1, orderings + 1):
        items: list[dict[str, str]] = []
        for s in samples:
            anon = f"{rng.getrandbits(48):012x}"
            while anon in key:  # pragma: no cover - 48-bit collision
                anon = f"{rng.getrandbits(48):012x}"
            key[anon] = {
                "case_id": s.case_id,
                "variant": s.variant,
                "sample": s.sample,
                "ordering": o,
            }
            items.append({"id": anon, "prompt": render_prompt(by_id[s.case_id], s, rubric)})
        rng.shuffle(items)
        d = root / "packets" / f"ord{o}"
        d.mkdir(parents=True, exist_ok=True)
        for n in range(0, len(items), batch_size):
            batches[d / f"batch_{n // batch_size + 1:02d}.json"] = items[n : n + batch_size]
    key_path = root / "key.json"
    key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")  # key BEFORE packets
    (root / "system.txt").write_text(rubric.system, encoding="utf-8")
    written_ids: set[str] = set()
    for p, batch in batches.items():
        p.write_text(json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
        written_ids.update(i["id"] for i in batch)
    if written_ids != set(key):  # pragma: no cover - invariant guard
        raise RuntimeError("key file does not cover every packet id")
    return PacketSet(root, key_path, tuple(batches), key)


@dataclass(frozen=True)
class Judgment:
    case_id: str
    variant: str
    sample: int
    ordering: int
    scores: dict[str, float] = field(default_factory=lambda: {})
    must_not_hit: bool = False
    note: str = ""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def collect_judgments(out_dir: str | Path, *, require_complete: bool = True) -> list[Judgment]:
    """Read every ``out_*.json`` under ``out_dir`` and map ids back through ``key.json``.

    Judge files may be wrapped in markdown fences. Unknown ids raise; with
    ``require_complete`` any id the judges skipped also raises (list them in the message).
    """
    root = Path(out_dir)
    key: dict[str, dict[str, Any]] = json.loads((root / "key.json").read_text(encoding="utf-8"))
    seen: set[str] = set()
    out: list[Judgment] = []
    for f in sorted(root.rglob("out_*.json")):
        raw = _FENCE.sub("", f.read_text(encoding="utf-8").strip())
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{f}: invalid judge JSON ({exc.msg})") from exc
        for r in rows if isinstance(rows, list) else [rows]:
            rid = str(r.get("id"))
            if rid not in key:
                raise ValueError(f"{f}: id {rid!r} is not in key.json")
            k = key[rid]
            seen.add(rid)
            scores = {
                n: float(v)
                for n, v in r.items()
                if n not in ("id", "note", "must_not_hit")
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
            }
            out.append(
                Judgment(
                    k["case_id"],
                    k["variant"],
                    int(k["sample"]),
                    int(k["ordering"]),
                    scores,
                    bool(r.get("must_not_hit", False)),
                    str(r.get("note", "")),
                )
            )
    missing = sorted(set(key) - seen)
    if require_complete and missing:
        raise ValueError(f"{len(missing)} packet ids have no judgment, e.g. {missing[:5]}")
    return out
