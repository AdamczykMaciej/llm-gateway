"""Deterministic check registry. Checks decide pass/fail; a judge never overrides a failure.

A check is ``fn(case, replies, params) -> CheckResult | list[CheckResult]``. Register app-specific
ones with :func:`register_check`::

    @register_check("no_emoji")
    def no_emoji(case, replies, params):
        bad = [r for r in replies if any(ord(ch) > 0x1F000 for ch in r)]
        return CheckResult("no_emoji", not bad, "emoji found" if bad else "")

and reference them from a case: ``{"name": "no_emoji"}``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import Case


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


CheckSpec = dict[str, Any]
CheckFn = Callable[[Case, Sequence[str], dict[str, Any]], "CheckResult | list[CheckResult]"]

_REGISTRY: dict[str, CheckFn] = {}
_language_detector: Callable[[str], str | None] | None = None


def register_check(name: str) -> Callable[[CheckFn], CheckFn]:
    """Decorator registering a check under ``name`` (replaces an existing one)."""

    def deco(fn: CheckFn) -> CheckFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def registered_checks() -> list[str]:
    return sorted(_REGISTRY)


def get_check(name: str) -> CheckFn:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown check {name!r}; registered: {registered_checks()}") from None


def set_language_detector(fn: Callable[[str], str | None] | None) -> None:
    """Install ``fn(text) -> language code or None`` used by the ``language`` check
    (for example a wrapper around langdetect or lingua). ``None`` clears it."""
    global _language_detector
    _language_detector = fn


def _results(out: CheckResult | list[CheckResult]) -> list[CheckResult]:
    return out if isinstance(out, list) else [out]


def check_case(
    case: Case, replies: Sequence[str], extra: Sequence[CheckSpec] = ()
) -> list[CheckResult]:
    """Run the case's own checks (plus ``extra``) against the replies.

    An unknown check name or a crashing check is reported as a failed result, not raised, so
    one broken check cannot hide the others.
    """
    out: list[CheckResult] = []
    for spec in (*case.checks, *extra):
        name = str(spec.get("name", ""))
        params = {k: v for k, v in spec.items() if k != "name"}
        try:
            out.extend(_results(get_check(name)(case, replies, params)))
        except Exception as exc:  # noqa: BLE001 - report, never hide the other checks
            out.append(CheckResult(name or "?", False, f"check error: {exc!r}"))
    return out


def _flags(params: dict[str, Any]) -> int:
    return 0 if params.get("case_sensitive") else re.IGNORECASE


@register_check("max_chars")
def _max_chars(case: Case, replies: Sequence[str], params: dict[str, Any]) -> CheckResult:
    limit = int(params["limit"])
    longest = max((len(r) for r in replies), default=0)
    return CheckResult(
        "max_chars", longest <= limit, f"longest {longest} > {limit}" if longest > limit else ""
    )


@register_check("max_words")
def _max_words(case: Case, replies: Sequence[str], params: dict[str, Any]) -> CheckResult:
    limit = int(params["limit"])
    longest = max((len(r.split()) for r in replies), default=0)
    return CheckResult(
        "max_words",
        longest <= limit,
        f"longest {longest} words > {limit}" if longest > limit else "",
    )


@register_check("forbidden_patterns")
def _forbidden(case: Case, replies: Sequence[str], params: dict[str, Any]) -> CheckResult:
    hits = [p for p in params["patterns"] for r in replies if re.search(p, r, _flags(params))]
    return CheckResult(
        "forbidden_patterns", not hits, f"matched {sorted(set(hits))}" if hits else ""
    )


@register_check("required_patterns")
def _required(case: Case, replies: Sequence[str], params: dict[str, Any]) -> CheckResult:
    scope = params.get("scope", "any")  # any: in some reply; all: in every reply; last
    pool = list(replies[-1:]) if scope == "last" else list(replies)
    missing: list[str] = []
    for p in params["patterns"]:
        found = [bool(re.search(p, r, _flags(params))) for r in pool]
        ok = all(found) if scope == "all" and found else any(found)
        if not ok:
            missing.append(p)
    return CheckResult("required_patterns", not missing, f"missing {missing}" if missing else "")


@register_check("repetition")
def _repetition(case: Case, replies: Sequence[str], params: dict[str, Any]) -> CheckResult:
    """Fail when a later reply repeats an earlier one (shared sentences or word overlap)."""
    threshold = float(params.get("threshold", 0.6))
    min_words = int(params.get("min_words", 4))

    def sentences(t: str) -> set[str]:
        parts = (re.sub(r"\W+", " ", s).strip().lower() for s in re.split(r"[.!?\n]+", t))
        return {p for p in parts if len(p.split()) >= min_words}

    for i in range(1, len(replies)):
        cur = sentences(replies[i])
        for j in range(i):
            prev = sentences(replies[j])
            if cur and prev:
                share = len(cur & prev) / min(len(cur), len(prev))
                if share >= threshold:
                    return CheckResult(
                        "repetition", False, f"reply {i + 1} repeats reply {j + 1} ({share:.0%})"
                    )
    return CheckResult("repetition", True)


@register_check("language")
def _language(case: Case, replies: Sequence[str], params: dict[str, Any]) -> CheckResult:
    """Replies must be in ``params['expected']`` (default: the case language), per the
    detector installed with :func:`set_language_detector`. Without one the check is skipped
    (passes with a note) rather than guessing."""
    if _language_detector is None:
        return CheckResult("language", True, "skipped: no language detector installed")
    expected = str(params.get("expected", case.language))
    bad = [d for r in replies if (d := _language_detector(r)) not in (None, expected)]
    return CheckResult("language", not bad, f"detected {bad}, expected {expected}" if bad else "")
