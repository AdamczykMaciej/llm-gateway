"""Transport-level size caps for every SDK client.

The SDKs read response bodies and SSE streams without any bound, so a hostile
or broken endpoint could exhaust memory. `response_guard` is an httpx2
response event hook: it runs once the headers arrived and, before the SDK
reads anything, wraps the response's `aiter_bytes` (what `aread()`, `aiter_text`,
`aiter_lines` and both SDKs' SSE decoders all pull from). It therefore counts
*decoded* bytes, so a compression bomb is stopped at its decoded size, and it
closes the response the moment a cap is crossed.

Residual: one decoder step can expand a single wire chunk (<= ~64 KiB, at most
~1000x for gzip) before the cap is checked, so transient memory can exceed a
cap by that much. A declared Content-Length over the body cap is refused
before any byte is read.
"""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from typing import Any

import httpx2

from ..config import GatewayConfig
from ..errors import ResponseTooLargeError


def _positive(value: int) -> int | None:
    return value if value > 0 else None


class _SSEMeter:
    """Counts the size of the SSE event in progress and of the whole stream.
    Lines end at LF, CRLF or a lone CR (all legal in SSE); a blank line ends
    the event. A CR at the end of a chunk may be half of a CRLF, so the LF
    that opens the next chunk is not a second line break."""

    def __init__(self, event_cap: int | None, stream_cap: int | None) -> None:
        self.event_cap = event_cap
        self.stream_cap = stream_cap
        self.total = 0
        self.event = 0  # bytes of the completed lines of the current event
        self.line = 0  # bytes of the line in progress
        self.after_cr = False

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if self.stream_cap is not None and self.total > self.stream_cap:
            raise ResponseTooLargeError(
                f"SSE stream passed {self.stream_cap} bytes", limit=self.stream_cap, kind="stream"
            )
        if self.after_cr and chunk.startswith(b"\n"):
            chunk = chunk[1:]
        if not chunk:
            return
        self.after_cr = chunk.endswith(b"\r")
        parts = chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
        for index, part in enumerate(parts):
            if index:  # a line break came before this part: close the line
                self.event = 0 if self.line == 0 else self.event + self.line + 1
                self.line = 0
            self.line += len(part)
            if self.event_cap is not None and self.event + self.line > self.event_cap:
                raise ResponseTooLargeError(
                    f"SSE event passed {self.event_cap} bytes",
                    limit=self.event_cap,
                    kind="sse_event",
                )


def response_guard(config: GatewayConfig) -> Callable[[httpx2.Response], Awaitable[None]]:
    """The response event hook enforcing `max_response_body_bytes`,
    `max_sse_event_bytes` and `max_stream_bytes` (0 disables each)."""
    body_cap = _positive(config.max_response_body_bytes)
    event_cap = _positive(config.max_sse_event_bytes)
    stream_cap = _positive(config.max_stream_bytes)

    async def bound_response_guard(response: httpx2.Response) -> None:
        is_sse = response.headers.get("content-type", "").lower().startswith("text/event-stream")
        cap = stream_cap if is_sse else body_cap
        if is_sse and event_cap is None and stream_cap is None:
            return
        if not is_sse and body_cap is None:
            return
        declared = response.headers.get("content-length", "")
        if cap is not None and declared.isdigit() and int(declared) > cap:
            await response.aclose()
            raise ResponseTooLargeError(
                f"response declares {declared} bytes, over the {cap}-byte cap",
                limit=cap,
                kind="stream" if is_sse else "body",
            )
        original = response.aiter_bytes
        meter = _SSEMeter(event_cap, stream_cap) if is_sse else None

        async def capped(chunk_size: int | None = None) -> AsyncGenerator[bytes, None]:
            total = 0
            try:
                async with aclosing(original(chunk_size)) as chunks:
                    async for chunk in chunks:
                        if meter is not None:
                            meter.feed(chunk)
                        else:
                            total += len(chunk)
                            if body_cap is not None and total > body_cap:
                                raise ResponseTooLargeError(
                                    f"response body passed {body_cap} bytes",
                                    limit=body_cap,
                                    kind="body",
                                )
                        yield chunk
            except ResponseTooLargeError:
                await response.aclose()
                raise

        guarded: Any = response
        guarded.aiter_bytes = capped

    return bound_response_guard
