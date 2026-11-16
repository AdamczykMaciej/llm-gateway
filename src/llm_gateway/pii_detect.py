"""Checksum-validated personal-data detection, as a `privacy.Redactor`.

`mask_pii` (see `pii.py`) is a quick best-effort scrub for traces. This module
is the stronger, opt-in detector for text that goes to a model: every number-
shaped identifier is only reported when its checksum holds (Luhn for cards,
mod-97 for IBANs, the official weights for Polish PESEL and NIP), so an order
number or a salary is not mistaken for an ID, while a real one is caught in
the usual written forms (grouped, spaced, with `+48` or a country prefix).

    from llm_gateway.privacy import Privacy, OutputPolicy
    privacy = Privacy(redactor=PatternRedactor(), output=OutputPolicy.RESTORE)

Pure, offline, no dependencies. Regex-based and deliberately conservative: not
a certified detector, and names / addresses need `ProtectedValues` (the caller
knows them) or a custom `Redactor`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable

from .privacy import Span

PII_KINDS = frozenset({"EMAIL", "PHONE", "IBAN", "CARD", "PESEL", "NIP"})

# IBAN length per country (EU/EEA plus the common non-EU SEPA members).
_IBAN_LENGTHS = {
    "AT": 20, "BE": 16, "BG": 22, "HR": 21, "CY": 28, "CZ": 24, "DK": 18, "EE": 20,
    "FI": 18, "FR": 27, "DE": 22, "GR": 27, "HU": 28, "IE": 22, "IT": 27, "LV": 21,
    "LT": 20, "LU": 20, "MT": 31, "NL": 18, "PL": 28, "PT": 25, "RO": 24, "SK": 24,
    "SI": 19, "ES": 24, "SE": 24, "IS": 26, "LI": 21, "NO": 15, "CH": 21, "GB": 22,
}  # fmt: skip

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,}\b")
_IBAN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?"
    r"|(?:PL ?)?\d{2}(?: ?\d{4}){6})(?![A-Za-z0-9])"
)
_CARD_RE = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_PESEL_RE = re.compile(r"\b\d{11}\b")
_NIP_RE = re.compile(r"(?<![\w@.])(?:PL ?)?\d(?:[- ]?\d){9}(?![\w@-])")
# Polish numbers: optional +48/0048, then 9 digits grouped 3-3-3 or 2-3-2-2 (or
# bare, mobile range only); international: "+" then 8-15 digits in groups.
_PHONE_PL_RE = re.compile(
    r"(?<![\w@.+])(?:(?:\+|00)48[ -]?)?(?:[4-8]\d{2}[ -]?\d{3}[ -]?\d{3}"
    r"|\d{2}[ -]\d{3}[ -]\d{2}[ -]\d{2})(?![\w@-])"
)
_PHONE_INTL_RE = re.compile(r"(?<![\w@.+])\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}(?![\w@-])")


def _digits(value: str) -> list[int]:
    return [int(c) for c in value if c.isdigit()]


def luhn_valid(digits: list[int]) -> bool:
    """Luhn checksum (payment cards)."""
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - (9 if d > 4 else 0)
        total += d
    return bool(digits) and total % 10 == 0


def iban_valid(value: str) -> bool:
    """A known country, its exact length and a passing mod-97 check."""
    iban = value.replace(" ", "").upper()
    if len(iban) < 15 or _IBAN_LENGTHS.get(iban[:2]) != len(iban) or not iban.isalnum():
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def pesel_valid(value: str) -> bool:
    """Polish PESEL: 11 digits, weighted checksum."""
    d = _digits(value)
    if len(d) != 11:
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    return (10 - sum(a * b for a, b in zip(d, weights, strict=False)) % 10) % 10 == d[10]


def nip_valid(value: str) -> bool:
    """Polish NIP (tax number): 10 digits, weighted checksum."""
    d = _digits(value)
    if len(d) != 10:
        return False
    check = sum(a * b for a, b in zip(d, (6, 5, 7, 2, 3, 4, 5, 6, 7), strict=False)) % 11
    return check != 10 and check == d[9]


def _iban_prefix(text: str) -> int | None:
    """Length of the longest valid IBAN at the start of `text`: a candidate may
    have swallowed following uppercase groups, which are trimmed group by group."""
    if iban_valid(text):
        return len(text)
    for m in re.finditer(r" ", text[::-1]):
        cut = len(text) - m.end()
        if cut > 0 and iban_valid(text[:cut]):
            return cut
    return None


def _iban_spans(text: str) -> Iterable[Span]:
    for m in _IBAN_RE.finditer(text):
        raw = m.group(0)
        if raw[0].isdigit():  # a Polish IBAN written without its country code
            if iban_valid("PL" + raw):
                yield Span(m.start(), m.end(), "IBAN")
            continue
        length = _iban_prefix(raw)
        if length:
            yield Span(m.start(), m.start() + length, "IBAN")


def _simple(
    pattern: re.Pattern[str], kind: str, ok: Callable[[str], bool], text: str
) -> Iterable[Span]:
    for m in pattern.finditer(text):
        if ok(m.group(0)):
            yield Span(m.start(), m.end(), kind)


def _is_date_like(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}", value))


def _phone_ok(value: str) -> bool:
    n = len(_digits(value))
    return 8 <= n <= 15 and not _is_date_like(value)


class PatternRedactor:
    """A `Redactor` for emails, phone numbers, IBANs (any EU country), payment
    cards, PESEL and NIP. `kinds` picks a subset of `PII_KINDS` (default all).

    Raises:
        ValueError: `kinds` holds a name outside `PII_KINDS`."""

    def __init__(self, kinds: Collection[str] | None = None) -> None:
        chosen = frozenset(PII_KINDS if kinds is None else kinds)
        unknown = chosen - PII_KINDS
        if unknown:
            raise ValueError(f"unknown PII kinds: {sorted(unknown)}")
        self.kinds = chosen

    def detect(self, text: str) -> Iterable[Span]:
        """Spans of personal data in `text`, checksum-validated; may overlap."""
        if not text:
            return []
        spans: list[Span] = []
        k = self.kinds
        if "EMAIL" in k:
            spans += [Span(m.start(), m.end(), "EMAIL") for m in _EMAIL_RE.finditer(text)]
        if "IBAN" in k:
            spans += _iban_spans(text)
        if "CARD" in k:
            spans += _simple(
                _CARD_RE,
                "CARD",
                lambda v: 13 <= len(_digits(v)) <= 19 and luhn_valid(_digits(v)),
                text,
            )
        if "PESEL" in k:
            spans += _simple(_PESEL_RE, "PESEL", pesel_valid, text)
        if "NIP" in k:
            spans += _simple(_NIP_RE, "NIP", nip_valid, text)
        if "PHONE" in k:
            spans += _simple(_PHONE_PL_RE, "PHONE", _phone_ok, text)
            spans += _simple(_PHONE_INTL_RE, "PHONE", _phone_ok, text)
        return spans
