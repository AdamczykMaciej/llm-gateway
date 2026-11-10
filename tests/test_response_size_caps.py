"""0.32.0: bounded response reading. A hostile or broken endpoint (above all a
configurable base URL) must not be able to exhaust memory with an endless body
or SSE line. Everything runs over in-memory httpx2 transports."""

import gzip
import json

import httpx2
import openai as openai_sdk
import pytest

from llm_gateway import (
    GatewayConfig,
    InvalidOutputError,
    LLMError,
    ResponseTooLargeError,
    StreamTooLargeError,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.errors import ErrorKind, classify
from llm_gateway.providers import (
    anthropic,
    azure,
    groq,
    huggingface,
    openai,
    openai_compat,
    vertex,
)
from llm_gateway.providers.base import sdk_http_client

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


class _Endless(httpx2.AsyncByteStream):
    """A body that never ends; counts what was pulled and records the close."""

    def __init__(self, chunk: bytes, limit_chunks: int = 10_000):
        self.chunk = chunk
        self.pulled = 0
        self.closed = False
        self.limit_chunks = limit_chunks

    async def __aiter__(self):
        for _ in range(self.limit_chunks):
            self.pulled += 1
            yield self.chunk

    async def aclose(self):
        self.closed = True


def _client(handler, **cfg) -> httpx2.AsyncClient:
    config = GatewayConfig(_env_file=None, **cfg)
    return sdk_http_client(httpx2.AsyncClient, config, transport=httpx2.MockTransport(handler))


async def _get(client, **kw):
    async with client.stream("GET", "https://x.test/") as response:
        return await response.aread()


# ─── Non-streamed bodies ───────────────────────────────────────────────────


async def test_endless_body_stops_at_the_cap_and_closes_the_connection():
    stream = _Endless(b"x" * 1000)
    client = _client(
        lambda r: httpx2.Response(200, stream=stream, headers={"content-type": "application/json"}),
        max_response_body_bytes=10_000,
    )
    with pytest.raises(ResponseTooLargeError) as ei:
        await _get(client)
    assert stream.pulled < 50  # stopped promptly, not after draining everything
    assert stream.closed
    assert ei.value.limit == 10_000
    assert isinstance(ei.value, StreamTooLargeError)
    assert isinstance(ei.value, InvalidOutputError)


async def test_declared_content_length_over_cap_is_refused_before_any_read():
    stream = _Endless(b"x", limit_chunks=1)
    client = _client(
        lambda r: httpx2.Response(
            200, stream=stream, headers={"content-length": "999999", "content-type": "text/plain"}
        ),
        max_response_body_bytes=1000,
    )
    with pytest.raises(ResponseTooLargeError):
        await _get(client)
    assert stream.pulled == 0
    assert stream.closed


async def test_body_under_cap_is_untouched():
    client = _client(
        lambda r: httpx2.Response(200, json={"ok": "y" * 500}), max_response_body_bytes=10_000
    )
    assert json.loads(await _get(client))["ok"] == "y" * 500


async def test_gzip_bomb_is_capped_on_decoded_size():
    bomb = gzip.compress(b"\0" * 5_000_000)
    assert len(bomb) < 10_000  # tiny on the wire, large once decoded
    client = _client(
        lambda r: httpx2.Response(
            200,
            stream=httpx2.ByteStream(bomb),
            headers={"content-encoding": "gzip", "content-type": "text/plain"},
        ),
        max_response_body_bytes=100_000,
    )
    with pytest.raises(ResponseTooLargeError):
        await _get(client)


async def test_zero_disables_the_body_cap():
    client = _client(
        lambda r: httpx2.Response(200, content=b"x" * 50_000), max_response_body_bytes=0
    )
    assert len(await _get(client)) == 50_000


# ─── SSE ───────────────────────────────────────────────────────────────────

_SSE = {"content-type": "text/event-stream"}


def _sse_client(chunks, **cfg):
    stream = _Endless(b"", 0)

    async def gen():
        for c in chunks:
            stream.pulled += 1
            yield c

    class _S(httpx2.AsyncByteStream):
        async def __aiter__(self):
            async for c in gen():
                yield c

        async def aclose(self):
            stream.closed = True

    return _client(lambda r: httpx2.Response(200, stream=_S(), headers=_SSE), **cfg), stream


async def test_one_endless_sse_line_is_capped():
    client, stream = _sse_client([b"data: " + b"a" * 1000] * 500, max_sse_event_bytes=5000)
    with pytest.raises(ResponseTooLargeError) as ei:
        await _get(client)
    assert stream.pulled < 20
    assert stream.closed
    assert ei.value.limit == 5000


async def test_lone_cr_line_breaks_do_not_evade_the_event_cap():
    # "\r" alone is a legal SSE line terminator, but never a blank line here.
    client, _ = _sse_client([b"data: a\r" * 500] * 10, max_sse_event_bytes=2000)
    with pytest.raises(ResponseTooLargeError):
        await _get(client)


async def test_many_small_lines_without_a_blank_line_are_one_event():
    client, _ = _sse_client([b"data: x\n"] * 1000, max_sse_event_bytes=2000)
    with pytest.raises(ResponseTooLargeError):
        await _get(client)


async def test_many_small_events_pass_the_event_cap_but_hit_the_stream_cap():
    chunks = [b"data: x\n\n"] * 1000  # each event is tiny; the whole stream is not
    client, _ = _sse_client(chunks, max_sse_event_bytes=100, max_stream_bytes=3000)
    with pytest.raises(ResponseTooLargeError) as ei:
        await _get(client)
    assert ei.value.limit == 3000


async def test_event_cap_resets_across_blank_lines_split_over_chunks():
    chunks = [b"data: " + b"a" * 90 + b"\r", b"\n\r", b"\n"] * 20  # CRLF split in two chunks
    client, _ = _sse_client(chunks, max_sse_event_bytes=150)
    assert await _get(client)


async def test_normal_sse_is_untouched_with_defaults():
    events = b"".join(
        f"data: {json.dumps({'i': i, 't': 'w' * 40})}\n\n".encode() for i in range(500)
    )
    client, _ = _sse_client(
        [events],
    )
    assert (await _get(client)) == events


# ─── Wiring through the real providers ─────────────────────────────────────


def _oai(handler, config):
    return openai_sdk.AsyncOpenAI(
        api_key="k",
        base_url="https://llm.example.com/v1",
        max_retries=0,
        http_client=sdk_http_client(
            httpx2.AsyncClient, config, transport=httpx2.MockTransport(handler)
        ),
    )


def _cfg(**kw) -> GatewayConfig:
    base = dict(
        openai_compat_api_key="k",
        openai_compat_base_url="https://llm.example.com/v1",
        openai_compat_model="m",
        provider_order="openai_compat",
        retry_attempts=2,
        retry_base_delay_seconds=0,
        _env_file=None,
    )
    base.update(kw)
    return GatewayConfig(**base)


def _chain(exc: BaseException):
    while exc is not None:
        yield exc
        exc = exc.__cause__ or exc.__context__


async def test_complete_with_an_oversized_body_fails_typed_and_is_not_retried(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(200, content=b"x" * 200_000)

    config = _cfg(max_response_body_bytes=50_000)
    monkeypatch.setattr(openai_compat, "_client", lambda c: _oai(handler, config))
    with pytest.raises(LLMError) as ei:
        await complete_with_usage(system="s", prompt="p", config=config)
    assert any(isinstance(e, ResponseTooLargeError) for e in _chain(ei.value))
    assert len(calls) == 1  # INVALID_OUTPUT is not retried on the same provider


async def test_sdk_wrapped_error_classifies_as_invalid_output_not_an_outage():
    # No Content-Length, so the cap fires while the SDK reads the body.
    client = _client(
        lambda r: httpx2.Response(200, stream=_Endless(b"x" * 100, 100)),
        max_response_body_bytes=500,
    )
    sdk = openai_sdk.AsyncOpenAI(
        api_key="k", base_url="https://h.test/v1", max_retries=0, http_client=client
    )
    with pytest.raises(Exception) as ei:  # noqa: PT011 — SDK may or may not wrap
        await sdk.chat.completions.create(model="m", messages=MESSAGES)
    assert any(isinstance(e, ResponseTooLargeError) for e in _chain(ei.value))
    assert classify(ei.value) is ErrorKind.INVALID_OUTPUT
    # an SDK-style wrapper around the typed error is looked through too
    wrapper = openai_sdk.APIConnectionError(request=httpx2.Request("GET", "https://h.test"))
    wrapper.__cause__ = ResponseTooLargeError("x", limit=1)
    assert classify(wrapper) is ErrorKind.INVALID_OUTPUT


async def test_stream_chat_with_an_endless_sse_line_ends_typed(monkeypatch):
    stream = _Endless(b"data: " + b"a" * 1000, limit_chunks=100_000)
    config = _cfg(max_sse_event_bytes=20_000)

    def handler(request):
        return httpx2.Response(200, stream=stream, headers=_SSE)

    monkeypatch.setattr(openai_compat, "_client", lambda c: _oai(handler, config))
    with pytest.raises(Exception) as ei:  # noqa: PT011 — the chain is checked below
        async for _ in stream_chat(messages=MESSAGES, config=config):
            pass
    assert any(isinstance(e, ResponseTooLargeError) for e in _chain(ei.value))
    assert stream.pulled < 100
    assert stream.closed


@pytest.mark.parametrize(
    "build",
    [
        lambda c: anthropic._client(c),
        lambda c: openai._client(c),
        lambda c: groq._client(c),
        lambda c: azure._client(c),
        lambda c: huggingface._client(c),
        lambda c: openai_compat._client(c),
        lambda c: vertex._client(c),
    ],
    ids=["anthropic", "openai", "groq", "azure", "huggingface", "openai_compat", "vertex"],
)
async def test_every_sdk_client_carries_the_response_guard(build, monkeypatch):
    config = GatewayConfig(
        anthropic_api_key="k",
        openai_api_key="k",
        groq_api_key="k",
        azure_openai_api_key="k",
        azure_endpoint="https://r.openai.azure.com",
        azure_deployment="d",
        huggingface_api_key="k",
        huggingface_model="m",
        openai_compat_api_key="k",
        openai_compat_base_url="https://llm.example.com/v1",
        vertex_project="p",
        _env_file=None,
    )
    for module in (anthropic, openai, groq, azure):
        module._clients.clear()
    try:
        client = build(config)
        if hasattr(client, "__await__"):
            client = await client
    except ImportError:
        pytest.skip("optional SDK extra missing")
    hooks = client._client.event_hooks["response"]
    assert any(getattr(h, "__name__", "") == "bound_response_guard" for h in hooks)


def test_defaults_are_generous_and_configurable(monkeypatch):
    config = GatewayConfig(_env_file=None)
    assert config.max_response_body_bytes == 32 * 1024 * 1024
    assert config.max_sse_event_bytes == 8 * 1024 * 1024
    assert config.max_stream_bytes == 128 * 1024 * 1024
    monkeypatch.setenv("MAX_RESPONSE_BODY_BYTES", "123")
    assert GatewayConfig(_env_file=None).max_response_body_bytes == 123
