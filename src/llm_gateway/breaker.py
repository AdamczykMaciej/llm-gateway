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

import threading
import time

from .errors import policy_for

# Closed -> open (cooldown) -> half-open (one probe at a time) -> closed.
# `_opened_until` is kept after the cooldown ends: its presence marks the
# half-open state until a success clears it. `_probe_until` is the lease on
# the single in-flight probe; it expires after one cooldown so a probe whose
# caller was cancelled can't wedge the provider open.
_lock = threading.Lock()
_failures: dict[str, int] = {}
_opened_until: dict[str, float] = {}
_probe_until: dict[str, float] = {}
_cooldowns: dict[str, float] = {}


def is_open(provider: str) -> bool:
    """Pure check (no side effects): in cooldown, or a probe is in flight."""
    now = time.monotonic()
    return now < _opened_until.get(provider, 0.0) or now < _probe_until.get(provider, 0.0)


def allow(provider: str) -> bool:
    """Admit a call. Closed: always. Open: never. Half-open (cooldown over,
    not yet proven healthy): exactly one caller, the probe, until it reports
    back with `record_success`/`record_error`; the rest stay out, so a
    recovering provider isn't hit by every queued request at once."""
    with _lock:
        now = time.monotonic()
        if now < _opened_until.get(provider, 0.0):
            return False
        if provider not in _opened_until:
            return True
        if now < _probe_until.get(provider, 0.0):
            return False
        _probe_until[provider] = now + max(_cooldowns.get(provider, 1.0), 1.0)
        return True


def record_success(provider: str) -> None:
    with _lock:
        _failures[provider] = 0
        _opened_until.pop(provider, None)
        _probe_until.pop(provider, None)


def record_failure(provider: str, *, threshold: int, cooldown_seconds: float) -> None:
    with _lock:
        failures = _failures.get(provider, 0) + 1
        _failures[provider] = failures
        _probe_until.pop(provider, None)
        if failures >= threshold:
            _opened_until[provider] = time.monotonic() + cooldown_seconds
            _cooldowns[provider] = cooldown_seconds


def record_error(
    provider: str, error: BaseException, *, threshold: int, cooldown_seconds: float
) -> bool:
    """Record `error` as a failure only if its classification trips the
    breaker. Returns whether it was counted."""
    policy = policy_for(error)
    if not policy.trips_breaker:
        # The provider answered (e.g. a 400): a probe's lease is released so
        # the next caller can probe; the state is otherwise unchanged.
        with _lock:
            _probe_until.pop(provider, None)
        return False
    record_failure(
        provider,
        threshold=1 if policy.opens_immediately else threshold,
        cooldown_seconds=cooldown_seconds,
    )
    return True


def reset() -> None:
    """Clear all breaker state. Used by tests to isolate runs."""
    with _lock:
        _failures.clear()
        _opened_until.clear()
        _probe_until.clear()
        _cooldowns.clear()
