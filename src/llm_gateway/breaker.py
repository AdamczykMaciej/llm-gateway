"""In-process circuit breaker, per provider and (for model-specific failures)
per (provider, model).

Trips a provider after consecutive failures so callers stop paying the
latency cost of retrying a provider that's clearly down on every request.
Module-level state (not per-instance) is intentional: it's meant to be
shared process-wide across every call to `router.complete()`, the same way
a single process shares one HTTP connection pool.

Auth, billing, rate-limit, timeout and connection failures say something about
the whole provider and open it for every model. A failure that is specific to
one model (a model-not-found 404, an overloaded model: see
`errors.is_model_scoped`) opens only that (provider, model) key when the caller
passes `model=`, so one retired or overloaded model does not take the other
models of a healthy provider out of rotation. Every `model` argument is
optional: without it the functions behave exactly as before.

Only failures that say something about the *provider* count — see
`record_error()` and `errors.POLICIES`. A caller's own malformed request
(400/422) must not take a healthy provider out of rotation for everyone.
"""

import threading
import time

from .errors import is_model_scoped, policy_for

# Closed -> open (cooldown) -> half-open (one probe at a time) -> closed.
# `_opened_until` is kept after the cooldown ends: its presence marks the
# half-open state until a success clears it. `_probe_until` is the lease on
# the single in-flight probe; it expires after one cooldown so a probe whose
# caller was cancelled can't wedge the provider open.
#
# Keys are the provider id, or `provider + _SEP + model` for a model-scoped
# state. Model ids can be caller-chosen (`provider/model` in the HTTP service),
# so model-scoped keys are capped at MAX_MODEL_KEYS (oldest evicted first);
# provider keys are bounded by the provider registry.
_SEP = "\x1f"
MAX_MODEL_KEYS = 256
_lock = threading.Lock()
_failures: dict[str, int] = {}
_opened_until: dict[str, float] = {}
_probe_until: dict[str, float] = {}
_cooldowns: dict[str, float] = {}
_model_keys: dict[str, None] = {}  # insertion-ordered set of model-scoped keys


def _keys(provider: str, model: str | None) -> list[str]:
    return [provider] if not model else [provider, f"{provider}{_SEP}{model}"]


def _state(key: str, now: float) -> str:
    """'open' (no admission), 'closed' (free) or 'probe' (half-open, no probe
    in flight: the next caller becomes the probe)."""
    if now < _opened_until.get(key, 0.0):
        return "open"
    if key not in _opened_until:
        return "closed"
    return "open" if now < _probe_until.get(key, 0.0) else "probe"


def _forget(key: str) -> None:
    _failures.pop(key, None)
    _opened_until.pop(key, None)
    _probe_until.pop(key, None)
    _cooldowns.pop(key, None)
    _model_keys.pop(key, None)


def _track(key: str, provider: str) -> None:
    """Bound the model-scoped state: called with the lock held, before a
    model key gains an entry."""
    if key == provider or key in _model_keys:
        return
    _model_keys[key] = None
    while len(_model_keys) > MAX_MODEL_KEYS:
        _forget(next(iter(_model_keys)))


def is_open(provider: str, model: str | None = None) -> bool:
    """Pure check (no side effects): in cooldown, or a probe is in flight."""
    now = time.monotonic()
    return any(_state(k, now) == "open" for k in _keys(provider, model))


def allow(provider: str, model: str | None = None) -> bool:
    """Admit a call. Closed: always. Open: never. Half-open (cooldown over,
    not yet proven healthy): exactly one caller, the probe, until it reports
    back with `record_success`/`record_error`; the rest stay out, so a
    recovering provider isn't hit by every queued request at once. With
    `model`, both the provider's and the (provider, model) state must admit;
    a probe lease is only taken when the call is admitted."""
    with _lock:
        now = time.monotonic()
        keys = _keys(provider, model)
        states = [_state(k, now) for k in keys]
        if "open" in states:
            return False
        for key, state in zip(keys, states, strict=True):
            if state == "probe":
                _probe_until[key] = now + max(_cooldowns.get(key, 1.0), 1.0)
        return True


def record_success(provider: str, model: str | None = None) -> None:
    with _lock:
        for key in _keys(provider, model):
            _forget(key)


def record_failure(
    provider: str, *, threshold: int, cooldown_seconds: float, model: str | None = None
) -> None:
    """Count one failure. With `model`, against that (provider, model) only."""
    with _lock:
        key = _keys(provider, model)[-1]
        _track(key, provider)
        failures = _failures.get(key, 0) + 1
        _failures[key] = failures
        _probe_until.pop(key, None)
        if failures >= threshold:
            _opened_until[key] = time.monotonic() + cooldown_seconds
            _cooldowns[key] = cooldown_seconds


def record_error(
    provider: str,
    error: BaseException,
    *,
    threshold: int,
    cooldown_seconds: float,
    model: str | None = None,
) -> bool:
    """Record `error` as a failure only if its classification trips the
    breaker. Returns whether it was counted. With `model`, a model-specific
    error (`errors.is_model_scoped`) is counted against that (provider, model)
    alone; a model-not-found opens it at once, since it cannot heal by itself."""
    scope = is_model_scoped(error) if model else None
    if scope:
        record_failure(
            provider,
            threshold=1 if scope == "missing" else threshold,
            cooldown_seconds=cooldown_seconds,
            model=model,
        )
        return True
    policy = policy_for(error)
    if not policy.trips_breaker:
        # The provider answered (e.g. a 400): a probe's lease is released so
        # the next caller can probe; the state is otherwise unchanged.
        with _lock:
            for key in _keys(provider, model):
                _probe_until.pop(key, None)
        return False
    record_failure(
        provider,
        threshold=1 if policy.opens_immediately else threshold,
        cooldown_seconds=cooldown_seconds,
    )
    if model:
        with _lock:  # a probe on the model key got a provider-level answer
            _probe_until.pop(_keys(provider, model)[-1], None)
    return True


def reset() -> None:
    """Clear all breaker state. Used by tests to isolate runs."""
    with _lock:
        _failures.clear()
        _opened_until.clear()
        _probe_until.clear()
        _cooldowns.clear()
        _model_keys.clear()
