"""Per-key cumulative usage tracking — in-process, ephemeral (resets on
restart, not shared across Cloud Run replicas — same tradeoff as
rate_limit.py's sliding window). Enough for basic per-caller cost
visibility on a single-tenant gateway; not a billing system, and a valid
key still has unlimited spend within its rate-limit window (see README).
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class _Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    since: float = field(default_factory=time.time)


_usage: dict[str, _Usage] = {}


def record(api_key: str, input_tokens: int, output_tokens: int) -> None:
    u = _usage.setdefault(api_key, _Usage())
    u.requests += 1
    u.input_tokens += input_tokens
    u.output_tokens += output_tokens


def get(api_key: str) -> dict:
    u = _usage.get(api_key)
    if u is None:
        return {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "since": None,
        }
    return {
        "requests": u.requests,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.input_tokens + u.output_tokens,
        "since": datetime.fromtimestamp(u.since, tz=UTC).isoformat(),
    }


def reset() -> None:
    """Clear all usage state. Used by tests to isolate runs."""
    _usage.clear()
