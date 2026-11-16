"""Human calibration: label files and judge-vs-human agreement.

Label file (JSON)::

    {"version": 1, "items": [{"id": "c01", "case_id": "...", "variant": "v1", "replies": ["..."],
      "human": {"refuses": true, "kind": null}}]}

``human`` maps a criterion to true/false (null = not labelled yet).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .judge import Judgment

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class HumanLabel:
    id: str
    case_id: str
    variant: str
    replies: tuple[str, ...]
    human: dict[str, bool | None] = field(default_factory=lambda: {})


def save_labels(path: str | Path, labels: Sequence[HumanLabel]) -> None:
    items: list[dict[str, Any]] = [
        {
            "id": x.id,
            "case_id": x.case_id,
            "variant": x.variant,
            "replies": list(x.replies),
            "human": x.human,
        }
        for x in labels
    ]
    Path(path).write_text(
        json.dumps({"version": 1, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_labels(path: str | Path) -> list[HumanLabel]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        HumanLabel(
            str(i["id"]),
            str(i["case_id"]),
            str(i["variant"]),
            tuple(i["replies"]),
            dict(i.get("human", {})),
        )
        for i in data["items"]
    ]


def agreement(a: Sequence[T], b: Sequence[T]) -> float | None:
    """Fraction of equal pairs; None for empty input."""
    if len(a) != len(b):
        raise ValueError("sequences differ in length")
    return None if not a else sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: Sequence[T], b: Sequence[T]) -> float | None:
    """Cohen's kappa; None when undefined (empty, or chance agreement is 1)."""
    if len(a) != len(b):
        raise ValueError("sequences differ in length")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def verdicts_from_judgments(
    judgments: Sequence[Judgment], thresholds: Mapping[str, float]
) -> dict[tuple[str, str], dict[str, bool]]:
    """Per (case_id, variant): criterion -> mean judge score >= threshold."""
    sums: dict[tuple[str, str], dict[str, list[float]]] = {}
    for j in judgments:
        cell = sums.setdefault((j.case_id, j.variant), {})
        for c in thresholds:
            if c in j.scores:
                cell.setdefault(c, []).append(j.scores[c])
    return {
        k: {c: sum(v) / len(v) >= thresholds[c] for c, v in cell.items()}
        for k, cell in sums.items()
    }


def judge_vs_human(
    labels: Sequence[HumanLabel],
    verdicts: Mapping[tuple[str, str], Mapping[str, bool]],
) -> dict[str, dict[str, float | int | None]]:
    """Per criterion: n compared, agreement and kappa (unlabelled / unjudged items skipped)."""
    pairs: dict[str, tuple[list[bool], list[bool]]] = {}
    for lab in labels:
        v = verdicts.get((lab.case_id, lab.variant), {})
        for crit, h in lab.human.items():
            if h is not None and crit in v:
                hs, js = pairs.setdefault(crit, ([], []))
                hs.append(h)
                js.append(v[crit])
    return {
        c: {"n": len(h), "agreement": agreement(h, j), "kappa": cohen_kappa(h, j)}
        for c, (h, j) in sorted(pairs.items())
    }
