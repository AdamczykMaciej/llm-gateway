"""In-process sliding-window rate limiter, keyed by API key.

Per-instance only — Cloud Run can scale out to multiple replicas, so the
effective ceiling under horizontal scale-out is up to
`max_instances * rate_limit_per_minute`, not an exact global limit. That's
an acceptable tradeoff for basic abuse protection on a single-tenant
gateway; a shared store (Redis) would be needed for an exact cross-replica
limit, which is more than this needs today.
"""

import time
from collections import deque

from fastapi import Depends, HTTPException

from ..config import GatewayConfig
from .auth import require_api_key

_windows: dict[str, deque[float]] = {}


def _check(api_key: str, limit_per_minute: int) -> None:
    if limit_per_minute <= 0:
        return  # disabled
    now = time.monotonic()
    window = _windows.setdefault(api_key, deque())
    cutoff = now - 60.0
    while window and window[0] < cutoff:
        window.popleft()
    if len(window) >= limit_per_minute:
        retry_after = int(window[0] + 60.0 - now) + 1
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
    window.append(now)


def enforce_rate_limit(config: GatewayConfig):
    """Return a FastAPI dependency that authenticates, then rate-limits by key."""
    auth_dep = require_api_key(config)

    async def _dep(api_key: str = Depends(auth_dep)) -> str:
        _check(api_key, config.rate_limit_per_minute)
        return api_key

    return _dep


def reset() -> None:
    """Clear all rate-limit state. Used by tests to isolate runs."""
    _windows.clear()
