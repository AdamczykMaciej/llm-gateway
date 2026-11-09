"""Idempotency keys: in-flight de-duplication of identical concurrent calls.

Process-local and in-flight only: nothing is remembered once the call ends,
so a later call with the same key runs again (a durable store is the host's
job). Keys never leave the process except as OpenAI's `Idempotency-Key`.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from .errors import InvalidRequestError

__all__ = ["IDEMPOTENCY_HEADER_PROVIDERS", "single_flight", "stream_slot"]

T = TypeVar("T")

# Providers that take the key as a request header (their `chat` accepts `idempotency_key`).
IDEMPOTENCY_HEADER_PROVIDERS = frozenset({"openai"})

# (event loop id, operation, key) -> [fingerprint, task, waiters]
_inflight: dict[tuple[int, str, str], list[Any]] = {}


def _fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=repr)


async def single_flight(
    key: str, payload: dict[str, Any], run: Callable[[], Awaitable[T]], *, scope: str = "chat"
) -> T:
    """Run `run()` once for all concurrent callers with the same `key` and
    `payload`; they all get its result (or its exception). `scope` names the
    operation (chat, complete), so one key may be used by different operations. The same key with a
    different payload raises `InvalidRequestError`. The shared call is
    cancelled only when its last waiter is."""
    slot_key = (id(asyncio.get_running_loop()), scope, key)
    fp = _fingerprint(payload)
    slot = _inflight.get(slot_key)
    if slot is None:
        task = asyncio.ensure_future(run())
        slot = [fp, task, 0]
        _inflight[slot_key] = slot
        task.add_done_callback(
            lambda _t: _inflight.get(slot_key) is slot and _inflight.pop(slot_key, None)
        )
    elif slot[0] != fp:
        raise InvalidRequestError(
            "idempotency_key is already in use by an in-flight call with a different request."
        )
    slot[2] += 1
    try:
        return await asyncio.shield(slot[1])
    finally:
        slot[2] -= 1
        if slot[2] == 0 and not slot[1].done():
            slot[1].cancel()


# (event loop id, key) of streams currently open.
_open_streams: set[tuple[int, str]] = set()


@contextmanager
def stream_slot(key: str) -> Iterator[None]:
    """Claim `key` for one open stream. A stream cannot be shared between
    callers (each consumer drives its own upstream request and the chunks are
    consumed, not replayed), so a second concurrent stream with the same key is
    rejected with `InvalidRequestError` instead of being fanned out or run
    twice. Released when the stream ends, fails or is closed."""
    slot = (id(asyncio.get_running_loop()), key)
    if slot in _open_streams:
        raise InvalidRequestError(
            "idempotency_key is already in use by an open stream; wait for it to finish "
            "or use a different key."
        )
    _open_streams.add(slot)
    try:
        yield
    finally:
        _open_streams.discard(slot)
