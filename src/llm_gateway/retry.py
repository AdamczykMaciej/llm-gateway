"""Short retry-with-backoff for a single provider, before the caller falls
over to the next one in the chain — plus the per-attempt timeout and the
overall call deadline that bound how long any of that may take.

Only errors `errors.policy_for()` marks retryable (connection errors, 5xx,
overloaded) are retried: an auth failure or a malformed request fails the
same way on the next try, and a timeout or a 429 is better spent on the next
provider than on waiting for this one again. No jitter — just a small fixed
number of attempts with a doubling delay between them. Not a
general-purpose retry library.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .errors import is_retryable

T = TypeVar("T")


def _bounded(seconds: float | None) -> float | None:
    """`None` for "no bound" — 0/negative settings disable a bound."""
    return seconds if seconds is not None and seconds > 0 else None


class Deadline:
    """Wall-clock budget for one gateway call, shared by every attempt and
    failover within it. `seconds` of 0/negative/None means unbounded."""

    def __init__(self, seconds: float | None) -> None:
        bound = _bounded(seconds)
        self._expires_at = time.monotonic() + bound if bound is not None else None

    def remaining(self) -> float | None:
        if self._expires_at is None:
            return None
        return max(self._expires_at - time.monotonic(), 0.0)

    @property
    def expired(self) -> bool:
        return self.remaining() == 0.0

    def cap(self, seconds: float | None) -> float | None:
        """The tighter of `seconds` and the time left, `None` if neither bounds."""
        limits = [s for s in (_bounded(seconds), self.remaining()) if s is not None]
        return min(limits) if limits else None


async def call_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay_seconds: float,
    attempt_timeout_seconds: float | None = None,
    deadline: Deadline | None = None,
    should_retry: Callable[[Exception], bool] = is_retryable,
) -> T:
    """Call `call()` up to `attempts` times (minimum 1), sleeping
    `base_delay_seconds * 2**i` between attempts, but only while
    `should_retry(error)` holds and the backoff still fits in `deadline`.
    Each attempt is cut off after `attempt_timeout_seconds` (or when the
    deadline runs out, whichever is sooner) with a builtin `TimeoutError`.
    Raises the *last* exception if no attempt succeeds."""
    deadline = deadline or Deadline(None)
    attempts = max(attempts, 1)
    last_error: Exception | None = None
    for i in range(attempts):
        try:
            async with asyncio.timeout(deadline.cap(attempt_timeout_seconds)):
                return await call()
        except Exception as e:  # noqa: BLE001 — classified below; the caller decides what happens after
            last_error = e
            if i == attempts - 1 or not should_retry(e):
                break
            delay = base_delay_seconds * (2**i)
            remaining = deadline.remaining()
            if remaining is not None and remaining <= delay:
                break
            await asyncio.sleep(delay)
    assert last_error is not None  # attempts >= 1, so the loop ran and raised at least once
    raise last_error
