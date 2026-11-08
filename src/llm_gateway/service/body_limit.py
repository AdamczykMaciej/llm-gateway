"""Request-body ceiling at the ASGI level.

Rejects an oversized body before the framework buffers and parses it: by the
declared Content-Length when there is one, and by counting bytes as they
arrive when there is not (chunked uploads). The response never echoes any of
the request.
"""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_BODY = json.dumps(
    {
        "error": {
            "message": "Request body exceeds the configured ceiling",
            "type": "invalid_request_error",
            "code": 413,
        }
    }
).encode()


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > self.max_bytes:
                    await self._reject(send)
                    return

        received = 0
        started = False
        rejected = False

        async def counting_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # The framework turns an exception raised here into a 400,
                    # so answer 413 ourselves and tell it the client is gone.
                    rejected = True
                    if not started:
                        await self._reject(send)
                    return {"type": "http.disconnect"}
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if rejected:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self.app(scope, counting_receive, tracking_send)

    async def _reject(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_BODY)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _BODY})
