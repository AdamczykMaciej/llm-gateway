"""The shared safety layer, assembled: one call gives a new app PII masking.

    from llm_gateway import PrivacyLayer, safe_privacy

    layer = PrivacyLayer(strict=True)
    privacy = safe_privacy(protected={"PERSON": ["Anna Nowak"]})
    completion = await layer.complete_with_usage(
        system="...", prompt=untrusted_text, privacy=privacy
    )

`safe_privacy` is a plain `Privacy`: placeholders replace personal data before
the provider sees it and (by default) are restored in the reply. Untrusted
content is the other half: fence it with `wrap_untrusted` and gate tools with
`gate_tool_calls` (see `untrusted.py`). Everything here is opt-in and offline.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping

from .pii_detect import PatternRedactor
from .privacy import DEFAULT_FORMAT, OutputPolicy, Privacy, Redactor, Span, TokenFormat

_MIN_VALUE_LEN = 2


class ProtectedValues:
    """A `Redactor` for values the caller already knows are sensitive (the
    user's name, an employer, an account id). `values` maps a kind such as
    "PERSON" to the strings to hide; matching is case-insensitive and on whole
    words. Empty and one-character values are ignored. The values are never
    shown by `repr`.

    Raises:
        ValueError: a kind is not 1-31 letters, digits, space, `_` or `-`."""

    def __init__(self, values: Mapping[str, Iterable[str]]) -> None:
        entries: list[tuple[str, str]] = []
        for kind, items in values.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,30}", kind):
                raise ValueError("a protected kind must be 1-31 letters, digits, space, '_' or '-'")
            entries += [(kind, v.strip()) for v in items if len(v.strip()) >= _MIN_VALUE_LEN]
        entries.sort(key=lambda e: -len(e[1]))  # longest first: "Anna Nowak" before "Anna"
        self._kinds = {v.casefold(): k for k, v in reversed(entries)}
        names = sorted(self._kinds, key=len, reverse=True)
        self._pattern = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)",
                re.IGNORECASE,
            )
            if names
            else None
        )

    def detect(self, text: str) -> Iterable[Span]:
        """Spans of every protected value in `text`."""
        if not text or self._pattern is None:
            return []
        return [
            Span(m.start(), m.end(), self._kinds[m.group(0).casefold()])
            for m in self._pattern.finditer(text)
        ]

    def __repr__(self) -> str:  # count only: never the values
        return f"<ProtectedValues entries={len(self._kinds)}>"


class CompositeRedactor:
    """Runs several redactors and merges their spans (the privacy layer keeps
    the earliest, longest span where they overlap)."""

    def __init__(self, *redactors: Redactor) -> None:
        self._redactors = redactors

    def detect(self, text: str) -> Iterable[Span]:
        """Every span any member reports."""
        out: list[Span] = []
        for r in self._redactors:
            out.extend(r.detect(text))
        return out


def safe_privacy(
    *,
    protected: Mapping[str, Iterable[str]] | None = None,
    kinds: Collection[str] | None = None,
    output: OutputPolicy = OutputPolicy.RESTORE,
    fmt: TokenFormat = DEFAULT_FORMAT,
    extra: Redactor | None = None,
    label: str = "",
) -> Privacy:
    """A ready `Privacy` policy: checksum-validated identifiers (`kinds`, all
    by default; see `PatternRedactor`), the caller's `protected` values, and
    an optional `extra` redactor, all replaced by placeholders before a model
    sees them. `output` decides what happens to placeholders in the reply
    (RESTORE for text only the data owner sees, STRIP for anything stored or
    shared, PASSTHROUGH to leave them).

    Raises:
        ValueError: an unknown PII kind or an invalid protected kind."""
    members: list[Redactor] = []
    if protected:
        members.append(ProtectedValues(protected))
    members.append(PatternRedactor(kinds))
    if extra is not None:
        members.append(extra)
    return Privacy(redactor=CompositeRedactor(*members), output=output, fmt=fmt, label=label)
