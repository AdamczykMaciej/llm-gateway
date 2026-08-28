"""Short retry-with-backoff for a single provider, before the caller falls
over to the next one in the chain.

Deliberately minimal: no error-type classification (a dropped connection and
a genuine "your API key is wrong" both retry the same way) and no jitter —
just a small fixed number of attempts with a linearly-doubling delay between
them. Enough to absorb a momentary blip without paying for a full failover
(and its cold-start latency on the next provider) when the same provider
would have succeeded on the very next try. Not a general-purpose retry
library; if that distinction ever matters, this is the place to add it.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def call_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay_seconds: float,
) -> T:
    """Call `call()` up to `attempts` times (minimum 1), sleeping
    `base_delay_seconds * 2**i` between attempts. Raises the *last*
    exception if every attempt fails."""
    attempts = max(attempts, 1)
    last_error: Exception | None = None
    for i in range(attempts):
        try:
            return await call()
        except Exception as e:  # noqa: BLE001 — any failure is retried; the caller decides what happens after
            last_error = e
            if i < attempts - 1:
                await asyncio.sleep(base_delay_seconds * (2**i))
    assert last_error is not None  # attempts >= 1, so the loop ran and raised at least once
    raise last_error
