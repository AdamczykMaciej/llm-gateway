"""Reliability pass 3: slow SSE clients, stream cap, graceful shutdown,
keepalive heartbeats and bounded buffering in the HTTP service."""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest

from llm_gateway import GatewayConfig, reset_circuit_breakers
from llm_gateway.providers.base import ChatResult, StreamDelta
from llm_gateway.service import app as app_module
from llm_gateway.service import create_app
from llm_gateway.service.rate_limit import reset as reset_rate_limits
from llm_gateway.service.usage import reset as reset_usage

BODY = {"messages": [{"role": "user", "content": "hi"}], "stream": True}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    reset_rate_limits()
    reset_usage()
    yield


def _app(**kw):
    kw.setdefault("anthropic_api_key", "k")
    kw.setdefault("provider_order", "anthropic")
    kw.setdefault("rate_limit_per_minute", 0)
    return create_app(GatewayConfig(**kw))


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


class Tracker:
    """Fake stream engine recording whether it was closed."""

    def __init__(self, deltas=None, gate: asyncio.Event | None = None, forever=False):
        self.closed = asyncio.Event()
        self.started = asyncio.Event()
        self.deltas = deltas or []
        self.gate = gate
        self.forever = forever
        self.produced = 0

    def __call__(self, **_kw):
        async def gen():
            self.started.set()
            try:
                for d in self.deltas:
                    self.produced += 1
                    yield d
                if self.gate is not None:
                    await self.gate.wait()
                i = 0
                while self.forever:
                    i += 1
                    self.produced += 1
                    yield StreamDelta(content="x" * 10)
                    await asyncio.sleep(0)
            finally:
                self.closed.set()

        return gen()


# 1a. slow client: the write timeout closes the upstream stream.
async def test_slow_reader_write_timeout_closes_upstream():
    tracker = Tracker(forever=True)
    app = _app(stream_write_timeout_seconds=0.2)
    sent = []

    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message["type"])
        if message["type"] == "http.response.body" and len(sent) > 2:
            await asyncio.sleep(3600)  # client stopped reading

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("c", 1),
        "server": ("s", 80),
        "scheme": "http",
    }
    body = json.dumps(BODY).encode()
    first = True

    async def receive_body():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    with patch.object(app_module, "stream_engine", tracker):
        task = asyncio.create_task(app(scope, receive_body, send))
        await asyncio.wait_for(tracker.closed.wait(), 5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert tracker.closed.is_set()


# 1b. cap on concurrent streams.
async def test_concurrent_stream_cap_returns_503():
    gate = asyncio.Event()
    tracker = Tracker(deltas=[StreamDelta(content="a")], gate=gate)
    app = _app(max_concurrent_streams=1)
    with patch.object(app_module, "stream_engine", tracker):
        async with _client(app) as c:
            first = asyncio.create_task(c.post("/v1/chat/completions", json=BODY))
            await asyncio.wait_for(tracker.started.wait(), 5)
            second = await c.post("/v1/chat/completions", json=BODY)
            assert second.status_code == 503
            assert "concurrent" in second.json()["error"]["message"]
            gate.set()
            r = await first
            assert r.status_code == 200
            # slot released: a third request is accepted
            third = await c.post("/v1/chat/completions", json=BODY)
            assert third.status_code == 200


# 2. graceful shutdown.
async def test_shutdown_waits_for_nonstreaming_then_errors_streams_then_closes_providers():
    order = []
    gate = asyncio.Event()
    tracker = Tracker(deltas=[StreamDelta(content="a")], gate=asyncio.Event())

    async def slow_chat(**_kw):
        order.append("chat-start")
        await gate.wait()
        order.append("chat-end")
        return ChatResult(content="ok", model="m", input_tokens=1, output_tokens=1)

    async def fake_aclose():
        order.append("aclose")

    app = _app(shutdown_grace_seconds=5)
    with (
        patch.object(app_module, "stream_engine", tracker),
        patch.object(app_module, "chat_engine", slow_chat),
        patch.object(app_module, "aclose_all", fake_aclose),
    ):
        async with app.router.lifespan_context(app) as _:
            c = _client(app)
            stream_task = asyncio.create_task(c.post("/v1/chat/completions", json=BODY))
            await asyncio.wait_for(tracker.started.wait(), 5)
            chat_task = asyncio.create_task(
                c.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
            )
            while "chat-start" not in order:
                await asyncio.sleep(0.01)

            async def release():
                await asyncio.sleep(0.2)
                gate.set()

            rel = asyncio.create_task(release())
        # lifespan exit ran: shutdown finished
        await rel
        assert order.index("chat-end") < order.index("aclose")
        chat_resp = await chat_task
        assert chat_resp.status_code == 200
        stream_resp = await stream_task
        assert '"error"' in stream_resp.text
        assert "shutting down" in stream_resp.text
        assert stream_resp.text.rstrip().endswith("data: [DONE]")
        assert tracker.closed.is_set()
        # new requests are refused with 503 while draining
        refused = await c.post("/v1/chat/completions", json=BODY)
        assert refused.status_code == 503
        await c.aclose()


# 3. heartbeat comments on an idle stream.
async def test_heartbeat_comment_on_idle_stream():
    gate = asyncio.Event()
    tracker = Tracker(deltas=[StreamDelta(content="a")], gate=gate)
    app = _app(stream_heartbeat_seconds=0.05)
    with patch.object(app_module, "stream_engine", tracker):
        async with _client(app) as c:

            async def release():
                await asyncio.sleep(0.3)
                gate.set()

            rel = asyncio.create_task(release())
            r = await c.post("/v1/chat/completions", json=BODY)
            await rel
    assert r.text.count(": keepalive\n\n") >= 2
    assert r.text.rstrip().endswith("data: [DONE]")


async def test_heartbeat_disabled_with_zero():
    gate = asyncio.Event()
    tracker = Tracker(deltas=[StreamDelta(content="a")], gate=gate)
    app = _app(stream_heartbeat_seconds=0)
    with patch.object(app_module, "stream_engine", tracker):
        async with _client(app) as c:

            async def release():
                await asyncio.sleep(0.15)
                gate.set()

            rel = asyncio.create_task(release())
            r = await c.post("/v1/chat/completions", json=BODY)
            await rel
    assert ": keepalive" not in r.text


# 4. bounded buffering: a stalled consumer stops the producer.
async def test_producer_is_bounded_when_consumer_stalls():
    tracker = Tracker(forever=True)
    pump = app_module._pump  # noqa: SLF001
    state = app_module._StreamState()  # noqa: SLF001
    with patch.object(app_module, "stream_engine", tracker):
        gen = pump(tracker(), state, heartbeat=0)
        await gen.__anext__()  # consume one, then stall
        await asyncio.sleep(0.2)
        assert tracker.produced <= app_module._QUEUE_SIZE + 3  # noqa: SLF001
        await gen.aclose()
        await asyncio.wait_for(tracker.closed.wait(), 2)


async def test_producer_error_is_reraised_to_consumer():
    async def boom(**_kw):
        yield StreamDelta(content="a")
        raise RuntimeError("upstream broke")

    app = _app()
    with patch.object(app_module, "stream_engine", boom):
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json=BODY)
    assert "failed mid-response" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")
