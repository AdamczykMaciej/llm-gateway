"""One format for configuration errors: which field, what was given, how to fix.

    <field> <problem>. Got <value>. Fix: <how>.

`<value>` is redacted when the field looks like a secret or the value is a URL
with embedded credentials, and truncated when long, so an error message that
ends up in a log never leaks a credential.
"""

import re

_SECRET_FIELD = re.compile(r"(^|_)(api_?)?(keys?|secrets?|passwords?|token)(_|$)", re.IGNORECASE)
_URL_USERINFO = re.compile(r"^([a-z][a-z0-9+.-]*://)[^/?#]*@", re.IGNORECASE)
_MAX_SHOWN = 200


def describe_given(field: str, value: object) -> str:
    """How to show `value` of `field` in an error message: `repr`, truncated;
    `<redacted, N chars>` for secret-like fields; URL credentials hidden."""
    if _SECRET_FIELD.search(field):
        size = len(value) if isinstance(value, str) else None
        return f"<redacted, {size} chars>" if size is not None else "<redacted>"
    if isinstance(value, str):
        value = _URL_USERINFO.sub(r"\1<redacted>@", value)
    shown = repr(value)
    if len(shown) > _MAX_SHOWN:
        shown = shown[: _MAX_SHOWN - 3] + "..."
    return shown


def config_error(
    field: str,
    problem: str,
    *,
    fix: str,
    given: object = None,
    shown: str | None = None,
) -> ValueError:
    """A `ValueError` reading `<field> <problem>. Got <value>. Fix: <fix>.`
    `shown` overrides the rendering of `given` (for a pre-formatted list)."""
    text = shown if shown is not None else describe_given(field, given)
    return ValueError(f"{field} {problem}. Got {text}. Fix: {fix}")
