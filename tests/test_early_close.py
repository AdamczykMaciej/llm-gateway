"""Consumer stops reading a stream early: break, exception in the loop body,
task cancel, client disconnect. Fakes only (no network): a fake provider
generator with a connection-pool counter."""

import asyncio
import gc
import warnings
from unittest.mock import patch

import pytest

from llm_gateway import GatewayConfig, breaker, reset_circuit_breakers, stream_chat
from llm_gateway.idempotency import _open_streams
from llm_gateway.providers import DEFAULT_MODEL
from llm_gateway.providers.base import StreamDelta

_MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw) -> GatewayConfig:
    return GatewayConfig(anthropic_api_key="k", provider_order="anthropic", retry_attempts=1, **kw)


class Pool:
    """Counts open fake upstream connections."""

    def __init__(self) -> None:
        self.open = 0
        self.opened = 0

    def fn(self, *items, hang_first=False):
        pool = self

        async def gen(*a, **k):
            pool.open += 1
            pool.opened += 1
            try:
                if hang_first:
                    await asyncio.sleep(3600)
                for item in items:
                    yield item
                await asyncio.sleep(3600)
            finally:
                pool.open -= 1

        return gen


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)
    gc.collect()
    for _ in range(5):
        await asyncio.sleep(0)


_DELTAS = (
    StreamDelta(content="a"),
    StreamDelta(content="b", usage=(3, 4)),
    StreamDelta(content="c"),
)


async def test_1000_breaks_leak_no_connections_and_keep_breaker_closed():
    pool = Pool()
    reports = []
    asyncio.get_running_loop().set_debug(True)
    with (
        warnings.catch_warnings(),
        patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": pool.fn(*_DELTAS)}),
    ):
        warnings.simplefilter("error")
        for i in range(1000):
            kind = i % 3
            if kind == 0:  # plain break, no aclosing
                async for _ in stream_chat(
                    messages=_MSGS, config=_cfg(), on_attempt=reports.append
                ):
                    break
            elif kind == 1:  # exception in the loop body
                with pytest.raises(ValueError):
                    async for _ in stream_chat(
                        messages=_MSGS, config=_cfg(), on_attempt=reports.append
                    ):
                        raise ValueError("boom")
            else:  # explicit aclose
                s = stream_chat(messages=_MSGS, config=_cfg(), on_attempt=reports.append)
                await s.__anext__()
                await s.aclose()
            if i % 50 == 0:
                await _settle()
        await _settle()
    assert pool.opened == 1000
    assert pool.open == 0
    assert len(reports) == 1000
    assert {r.outcome for r in reports} == {"cancelled"}
    assert all(r.partial for r in reports)
    assert not breaker.tripped("anthropic")
    assert breaker.state("anthropic") == "closed"


async def test_usage_known_at_break_is_reported_once():
    pool = Pool()
    reports = []
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": pool.fn(*_DELTAS)}):
        s = stream_chat(messages=_MSGS, config=_cfg(), on_attempt=reports.append)
        await s.__anext__()
        await s.__anext__()  # carries usage
        await s.aclose()
    (r,) = reports
    assert r.outcome == "cancelled" and r.usage is not None
    assert (r.usage.input_tokens, r.usage.output_tokens) == (3, 4)


async def test_cancelled_task_releases_everything():
    pool = Pool()
    reports = []
    started = asyncio.Event()

    async def consume():
        async for _ in stream_chat(
            messages=_MSGS, config=_cfg(), on_attempt=reports.append, idempotency_key="k-1"
        ):
            started.set()
            await asyncio.sleep(3600)

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": pool.fn(*_DELTAS)}):
        task = asyncio.create_task(consume())
        await started.wait()
        assert pool.open == 1 and _open_streams
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _settle()
    assert pool.open == 0
    assert not _open_streams
    assert [r.outcome for r in reports] == ["cancelled"]
    assert not breaker.tripped("anthropic")


async def test_cancel_before_first_chunk_releases_probe_and_is_not_a_failure():
    pool = Pool()
    reports = []
    cfg = _cfg()
    model = DEFAULT_MODEL["anthropic"](cfg)
    # Trip the breaker with a tiny cooldown so it is half-open right away.
    breaker.record_failure("anthropic", threshold=1, cooldown_seconds=0.01, model=model)
    await asyncio.sleep(0.05)
    assert breaker.state("anthropic", model) == "probe"
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": pool.fn(hang_first=True)}):

        async def consume():
            async for _ in stream_chat(messages=_MSGS, config=cfg, on_attempt=reports.append):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        assert pool.open == 1
        assert not breaker.allow("anthropic", model)  # probe lease is held
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert pool.open == 0
    assert [r.outcome for r in reports] == ["cancelled"]
    assert breaker.allow("anthropic", model)  # lease given back, not a new failure


async def test_break_after_probe_success_closes_breaker():
    pool = Pool()
    cfg = _cfg()
    model = DEFAULT_MODEL["anthropic"](cfg)
    breaker.record_failure("anthropic", threshold=1, cooldown_seconds=0.01, model=model)
    await asyncio.sleep(0.05)
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": pool.fn(*_DELTAS)}):
        async for _ in stream_chat(messages=_MSGS, config=cfg):
            break
        await _settle()
    assert pool.open == 0
    assert breaker.state("anthropic", model) == "closed"


async def test_service_client_disconnect_stops_upstream():
    from llm_gateway.service import create_app
    from llm_gateway.service.rate_limit import reset as reset_rate_limits

    reset_rate_limits()
    pool = Pool()
    app = create_app(_cfg(stream_heartbeat_seconds=0))
    body = b'{"messages":[{"role":"user","content":"hi"}],"stream":true}'
    sent: list[dict] = []
    first_chunk = asyncio.Event()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await first_chunk.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "server": ("test", 80),
        "client": ("c", 1),
        "scheme": "http",
        "root_path": "",
        "state": {},
    }
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": pool.fn(*_DELTAS)}):
        await asyncio.wait_for(app(scope, receive, send), 10)
        await _settle()
    assert first_chunk.is_set()
    assert pool.opened == 1 and pool.open == 0
    assert not breaker.tripped("anthropic")
    assert not _open_streams


async def test_spans_of_an_abandoned_stream_end_without_error_status():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    pool = Pool()
    with (
        patch("llm_gateway.tracing.get_tracer", lambda: provider.get_tracer("t")),
        patch("llm_gateway.streaming.get_tracer", lambda: provider.get_tracer("t")),
        patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": pool.fn(*_DELTAS)}),
    ):
        async for _ in stream_chat(messages=_MSGS, config=_cfg()):
            break
        await _settle()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"llm_gateway.stream_chat", "llm_gateway.attempt"}
    assert spans["llm_gateway.attempt"].attributes["llm_gateway.outcome"] == "cancelled"
    assert spans["llm_gateway.attempt"].status.status_code.name != "ERROR"
    assert spans["llm_gateway.stream_chat"].status.status_code.name != "ERROR"


# ── SDK-backed providers: the HTTP response is released on early close ──────


class _FakeSdkStream:
    """Mimics openai's AsyncStream: iterating does NOT close the response on an
    early exit; only `close()` / leaving `async with` does."""

    def __init__(self, pool: Pool) -> None:
        self.pool = pool
        pool.open += 1
        pool.opened += 1
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self):
        if not self._closed:
            self._closed = True
            self.pool.open -= 1

    def __aiter__(self):
        return self._chunks()

    async def _chunks(self):
        from openai.types.chat import ChatCompletionChunk

        for text in ("a", "b", "c"):
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "m",
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
            )
        await asyncio.sleep(3600)


def _sdk_for(pool: Pool):
    async def create(**kwargs):
        return _FakeSdkStream(pool)

    class _Completions:
        pass

    ns = _Completions()
    ns.create = create
    chat = _Completions()
    chat.completions = ns
    sdk = _Completions()
    sdk.chat = chat
    return sdk


@pytest.mark.parametrize("name", ["openai", "groq", "azure", "mistral"])
async def test_sdk_providers_close_the_http_stream_on_early_break(name):
    import importlib

    module = importlib.import_module(f"llm_gateway.providers.{name}")
    pool = Pool()
    sdk = _sdk_for(pool)
    cfg = _cfg(azure_endpoint="https://e.example", azure_model="m", azure_api_key="k")
    if name == "mistral":
        module._provider.client_factory = lambda config: sdk
        make = lambda: module._provider._stream_chat(cfg, _MSGS, None, 10)  # noqa: E731
    else:
        make = lambda: module._stream_chat(cfg, _MSGS, None, 10)  # noqa: E731
    with patch.object(module, "_client", lambda config: sdk, create=True):
        try:
            for _ in range(100):
                gen = make()
                await gen.__anext__()
                await gen.aclose()
        finally:
            if name == "mistral":
                module._provider.client_factory = lambda config: module._client(config)
    assert pool.opened == 100
    assert pool.open == 0
