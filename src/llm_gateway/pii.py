"""Best-effort PII redaction for text before it is traced or logged.

Not a certified PII detector — regex-based, deliberately conservative about
false positives (ordinary prose, dates, prices) at the cost of occasionally
missing edge-case formats. Order matters: more specific patterns (IBAN, card,
national ID) run before the generic phone-number pattern so a long digit run
isn't partially masked twice.
"""

import re

_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PESEL_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<![\w@.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){2,4}(?![\w@])"
)


def mask_pii(text: str | None) -> str | None:
    """Return text with emails, phone numbers, and ID-like sequences redacted."""
    if not text:
        return text

    masked = _IBAN_RE.sub("[IBAN]", text)
    masked = _CARD_RE.sub("[CARD]", masked)
    masked = _PESEL_RE.sub("[ID]", masked)
    masked = _EMAIL_RE.sub("[EMAIL]", masked)
    masked = _PHONE_RE.sub("[PHONE]", masked)
    return masked
