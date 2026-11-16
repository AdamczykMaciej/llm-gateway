"""Aggregation and A/B comparison over deterministic results and judge scores."""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .judge import Judgment
from .schema import Case

NOISE = 0.3  # per-prompt score differences below this are within judge noise


@dataclass(frozen=True)
class CaseStats:
    case_id: str
    variant: str
    n: int
    means: dict[str, float]
    mech_pass: bool | None  # None when no deterministic result was supplied
    must_not_hit: bool


def _mean(xs: Sequence[float]) -> float:
    return st.fmean(xs)


def aggregate(
    judgments: Sequence[Judgment],
    mech: Mapping[tuple[str, str], bool] | None = None,
) -> list[CaseStats]:
    """Mean score per (case, variant), averaging samples and orderings.

    ``mech`` maps ``(case_id, variant)`` to the deterministic pass flag; a failed mechanical
    check is carried through and never overridden by judge scores.
    """
    cells: dict[tuple[str, str], list[Judgment]] = defaultdict(list)
    for j in judgments:
        cells[(j.case_id, j.variant)].append(j)
    out: list[CaseStats] = []
    for (cid, var), js in sorted(cells.items()):
        dims: dict[str, list[float]] = defaultdict(list)
        for j in js:
            for d, v in j.scores.items():
                dims[d].append(v)
        out.append(
            CaseStats(
                cid,
                var,
                len(js),
                {d: _mean(v) for d, v in sorted(dims.items())},
                None if mech is None else mech.get((cid, var)),
                any(j.must_not_hit for j in js),
            )
        )
    return out


def pooled(stats: Sequence[CaseStats], variant: str) -> dict[str, float]:
    dims: dict[str, list[float]] = defaultdict(list)
    for s in stats:
        if s.variant == variant:
            for d, v in s.means.items():
                dims[d].append(v)
    return {d: _mean(v) for d, v in sorted(dims.items())}


def by_tag(
    stats: Sequence[CaseStats], cases: Sequence[Case], variant: str
) -> dict[str, dict[str, float]]:
    tags = {c.id: c.tags for c in cases}
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in stats:
        if s.variant == variant:
            for t in tags.get(s.case_id, ()):
                for d, v in s.means.items():
                    groups[t][d].append(v)
    return {t: {d: _mean(v) for d, v in sorted(g.items())} for t, g in sorted(groups.items())}


@dataclass(frozen=True)
class Comparison:
    baseline: str
    candidate: str
    pooled_delta: dict[str, float]
    per_case_delta: dict[str, dict[str, float]]
    worse: list[str] = field(default_factory=lambda: [])
    better: list[str] = field(default_factory=lambda: [])
    mech_regressions: list[str] = field(default_factory=lambda: [])
    tag_delta: dict[str, dict[str, float]] = field(default_factory=lambda: {})


def compare(
    stats: Sequence[CaseStats],
    baseline: str,
    candidate: str,
    cases: Sequence[Case] = (),
    noise: float = NOISE,
) -> Comparison:
    """A/B: candidate minus baseline. A case is ``worse``/``better`` when its mean change over
    shared dimensions exceeds ``noise``. ``mech_regressions`` lists cases that passed the
    deterministic checks for the baseline and fail them for the candidate."""
    a = {s.case_id: s for s in stats if s.variant == baseline}
    b = {s.case_id: s for s in stats if s.variant == candidate}
    shared = sorted(set(a) & set(b))
    per_case: dict[str, dict[str, float]] = {}
    worse: list[str] = []
    better: list[str] = []
    regress: list[str] = []
    for cid in shared:
        dims = sorted(set(a[cid].means) & set(b[cid].means))
        per_case[cid] = {d: b[cid].means[d] - a[cid].means[d] for d in dims}
        if dims:
            delta = _mean(list(per_case[cid].values()))
            if delta <= -noise:
                worse.append(cid)
            elif delta >= noise:
                better.append(cid)
        if a[cid].mech_pass is True and b[cid].mech_pass is False:
            regress.append(cid)
    pa, pb = pooled(stats, baseline), pooled(stats, candidate)
    tags_a, tags_b = by_tag(stats, cases, baseline), by_tag(stats, cases, candidate)
    tag_delta = {
        t: {d: tags_b[t][d] - tags_a[t][d] for d in tags_a[t] if d in tags_b[t]}
        for t in tags_a
        if t in tags_b
    }
    return Comparison(
        baseline,
        candidate,
        {d: pb[d] - pa[d] for d in pa if d in pb},
        per_case,
        worse,
        better,
        regress,
        tag_delta,
    )
