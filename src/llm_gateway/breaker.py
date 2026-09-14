"""In-process, per-provider circuit breaker.

Trips a provider after consecutive failures so callers stop paying the
latency cost of retrying a provider that's clearly down on every request.
Module-level state (not per-instance) is intentional: it's meant to be
shared process-wide across every call to `router.complete()`, the same way
a single process shares one HTTP connection pool.

Only failures that say something about the *provider* count — see
`record_error()` and `errors.POLICIES`. A caller's own malformed request
(400/422) must not take a healthy provider out of rotation for everyone.
"""

import time

from .errors import policy_for

_failures: dict[str, int] = {}
_opened_until: dict[str, float] = {}


def is_open(provider: str) -> bool:
    return time.monotonic() < _opened_until.get(provider, 0.0)


def record_success(provider: str) -> None:
    _failures[provider] = 0
    _opened_until.pop(provider, None)


def record_failure(provider: str, *, threshold: int, cooldown_seconds: float) -> None:
    failures = _failures.get(provider, 0) + 1
    _failures[provider] = failures
    if failures >= threshold:
        _opened_until[provider] = time.monotonic() + cooldown_seconds


def record_error(
    provider: str, error: BaseException, *, threshold: int, cooldown_seconds: float
) -> bool:
    """Record `error` as a failure only if its classification trips the
    breaker. Returns whether it was counted."""
    if not policy_for(error).trips_breaker:
        return False
    record_failure(provider, threshold=threshold, cooldown_seconds=cooldown_seconds)
    return True


def reset() -> None:
    """Clear all breaker state. Used by tests to isolate runs."""
    _failures.clear()
    _opened_until.clear()
