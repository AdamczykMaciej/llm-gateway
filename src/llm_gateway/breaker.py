"""In-process, per-provider circuit breaker.

Trips a provider after consecutive failures so callers stop paying the
latency cost of retrying a provider that's clearly down on every request.
Module-level state (not per-instance) is intentional: it's meant to be
shared process-wide across every call to `router.complete()`, the same way
a single process shares one HTTP connection pool.
"""

import time

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


def reset() -> None:
    """Clear all breaker state. Used by tests to isolate runs."""
    _failures.clear()
    _opened_until.clear()
