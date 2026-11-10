"""Ergonomics pass 10: closed-set types, changelog continuity, version consistency."""

import importlib.metadata
import re
import typing
from pathlib import Path

import llm_gateway
from llm_gateway.errors import FAILURE_CLASSES, FailureClass, ProviderFailure, ResponseTooLargeError

ROOT = Path(__file__).resolve().parents[1]


def _vt(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def test_failure_class_literal_matches_runtime_set():
    assert FAILURE_CLASSES == frozenset(typing.get_args(FailureClass))
    assert llm_gateway.FailureClass is FailureClass
    hints = typing.get_type_hints(ProviderFailure)
    assert hints["kind"] is FailureClass


def test_response_too_large_kind_is_closed():
    kind = typing.get_type_hints(ResponseTooLargeError.__init__)["kind"]
    assert set(typing.get_args(kind)) == {"body", "sse_event", "stream"}


def test_changelog_has_every_version_from_0_29():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = set(re.findall(r"^## (\d+\.\d+\.\d+)$", text, re.M))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    current = re.search(r'^version = "([\d.]+)"', pyproject, re.M).group(1)  # type: ignore[union-attr]
    assert current in headings
    # Patch releases that were never cut (no tag, no code) may be skipped only if listed here.
    skipped: set[str] = set()
    missing = []
    majors = {(_vt(h)[0], _vt(h)[1]) for h in headings if _vt(h) >= (0, 29, 0)}
    for minor in sorted(majors):
        patches = sorted(_vt(h)[2] for h in headings if _vt(h)[:2] == minor)
        for patch in range(0, patches[-1] + 1):
            v = f"{minor[0]}.{minor[1]}.{patch}"
            # A bump that skipped a patch with no entry is a gap; 0.x.0 must always exist.
            if v not in headings and v not in skipped:
                missing.append(v)
    assert not missing, f"CHANGELOG has no entry for {missing}"


def test_installed_version_matches_pyproject():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    current = re.search(r'^version = "([\d.]+)"', pyproject, re.M).group(1)  # type: ignore[union-attr]
    try:
        installed = importlib.metadata.version("llm-gateway")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return
    assert installed == current or installed.startswith(current)
