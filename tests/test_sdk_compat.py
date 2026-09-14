"""Contract tests against the *real* provider SDKs (anthropic 1.x, openai 3.x).

Everything else in this suite mocks the SDK client object, which is exactly
what hid the anthropic 1.0 / openai 3.0 breaking changes: a MagicMock accepts
any keyword argument and any http_client. These tests instead build genuine
SDK clients over an in-memory httpx2 MockTransport (no network, no API keys),
so a removed parameter, a rejected http_client, a renamed exception, or a
changed response/stream model fails here.
"""

import asyncio
import contextlib
import json
import time
from unittest.mock import AsyncMock, patch

import anthropic as anthropic_sdk
import httpx2
import openai as openai_sdk
import pytest

from llm_gateway import (
    GatewayConfig,
    LLMDeadlineExceeded,
    LLMError,
    breaker,
    chat,
    complete,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.errors import ErrorKind, classify
from llm_gateway.providers import anthropic, groq, openai
from llm_gateway.providers.base import ProviderResult

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


class _Recorder:
    """httpx2 MockTransport handler that records request bodies and replies
    with a canned response (or raises a canned transport error)."""

    def __init__(self, respond):
        self.respond = respond
        self.bodies: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content or b"{}"))
        return self.respond(request)


def _anthropic_client(recorder: _Recorder) -> anthropic_sdk.AsyncAnthropic:
    return anthropic_sdk.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)),
    )


def _openai_client(recorder: _Recorder) -> openai_sdk.AsyncOpenAI:
    return openai_sdk.AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)),
    )


def _sse(events: list[dict], *, named: bool) -> bytes:
    out = []
    for e in events:
        prefix = f"event: {e['type']}\n" if named else ""
        out.append(f"{prefix}data: {json.dumps(e)}\n\n")
    if not named:
        out.append("data: [DONE]\n\n")
    return "".join(out).encode()


# ─── Client construction ───────────────────────────────────────────────────


@pytest.mark.parametrize("provider", [anthropic, openai, groq])
async def test_client_builds_with_ssl_verify_disabled(provider):
    """anthropic>=1.0 raises TypeError for a plain httpx.AsyncClient; the
    ssl_verify=False path must hand every SDK an httpx2 client instead."""
    config = GatewayConfig(
        anthropic_api_key="k", openai_api_key="k", groq_api_key="k", ssl_verify=False
    )
    provider._clients.clear()
    try:
        client = provider._client(config)
        assert isinstance(client._client, httpx2.AsyncClient)
    finally:
        provider._clients.clear()


# ─── Anthropic ─────────────────────────────────────────────────────────────

_ANTHROPIC_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "content": [{"type": "text", "text": " Hello there. "}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 4},
}


async def test_anthropic_call_and_chat_with_sampling_through_real_sdk():
    recorder = _Recorder(lambda req: httpx2.Response(200, request=req, json=_ANTHROPIC_MESSAGE))
    config = GatewayConfig(anthropic_api_key="test-key")
    with patch.object(anthropic, "_client", return_value=_anthropic_client(recorder)):
        plain = await anthropic.call(config, "sys", "hi", 50)
        result = await anthropic.chat(
            config,
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=50,
            sampling={"temperature": 0.2, "top_p": 0.9, "stop": "END", "seed": 7},
        )

    assert plain == ProviderResult("Hello there.", config.claude_model, 12, 4)
    assert result.content == "Hello there."
    assert (result.input_tokens, result.output_tokens) == (12, 4)
    body = recorder.bodies[-1]
    # The request still carries sampling params even though the 1.x SDK no
    # longer accepts temperature/top_p as named keyword arguments.
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["stop_sequences"] == ["END"]
    assert "seed" not in body
    assert body["system"] == "sys"


async def test_anthropic_stream_chat_through_real_sdk():
    events = [
        {
            "type": "message_start",
            "message": {
                **_ANTHROPIC_MESSAGE,
                "content": [],
                "usage": {"input_tokens": 9, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"q": "x"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 6},
        },
        {"type": "message_stop"},
    ]
    recorder = _Recorder(
        lambda req: httpx2.Response(
            200,
            request=req,
            headers={"content-type": "text/event-stream"},
            content=_sse(events, named=True),
        )
    )
    config = GatewayConfig(anthropic_api_key="test-key")
    with patch.object(anthropic, "_client", return_value=_anthropic_client(recorder)):
        deltas = [
            d
            async for d in anthropic.stream_chat(
                config,
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "lookup"}}],
                max_tokens=50,
                sampling={"temperature": 0.0},
            )
        ]

    # The 1.x MessageStream also yields synthesized helper events ("text",
    # "input_json", ...); those must not produce duplicate deltas.
    assert [d.content for d in deltas if d.content] == ["Hi"]
    tool_deltas = [d.tool_call_deltas[0] for d in deltas if d.tool_call_deltas]
    assert tool_deltas[0]["id"] == "toolu_1"
    assert tool_deltas[0]["function"]["name"] == "lookup"
    assert [t["function"]["arguments"] for t in tool_deltas[1:]] == ['{"q": "x"}']
    assert deltas[-1].finish_reason == "tool_calls"
    assert deltas[-1].usage == (9, 6)
    assert recorder.bodies[-1]["temperature"] == 0.0
    assert recorder.bodies[-1]["stream"] is True


# ─── OpenAI (and Groq, same SDK) ───────────────────────────────────────────


async def test_openai_chat_with_tool_calls_through_real_sdk():
    completion = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-x",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }
    recorder = _Recorder(lambda req: httpx2.Response(200, request=req, json=completion))
    config = GatewayConfig(openai_api_key="test-key")
    with patch.object(openai, "_client", return_value=_openai_client(recorder)):
        result = await openai.chat(
            config,
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            max_tokens=50,
            tool_choice="auto",
            sampling={"temperature": 0.1, "seed": 3},
        )

    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].arguments == {"q": "x"}
    assert (result.input_tokens, result.output_tokens) == (7, 3)
    body = recorder.bodies[-1]
    assert body["max_tokens"] == 50
    assert body["temperature"] == 0.1
    assert body["seed"] == 3


async def test_openai_stream_chat_with_usage_chunk_through_real_sdk():
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-x"}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
        {**base, "choices": [{"index": 0, "delta": {"content": "lo"}}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **base,
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    ]
    for c in chunks:
        c["type"] = "chunk"  # only used by _sse's (unused) event-name prefix
    recorder = _Recorder(
        lambda req: httpx2.Response(
            200,
            request=req,
            headers={"content-type": "text/event-stream"},
            content=_sse(chunks, named=False),
        )
    )
    config = GatewayConfig(openai_api_key="test-key")
    with patch.object(openai, "_client", return_value=_openai_client(recorder)):
        deltas = [
            d
            async for d in openai.stream_chat(
                config, [{"role": "user", "content": "hi"}], tools=None, max_tokens=50
            )
        ]

    assert "".join(d.content for d in deltas if d.content) == "Hello"
    assert any(d.finish_reason == "stop" for d in deltas)
    assert deltas[-1].usage == (5, 2)
    assert recorder.bodies[-1]["stream_options"] == {"include_usage": True}


# ─── Error classification ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "anthropic_error", "openai_error", "kind"),
    [
        (400, "BadRequestError", "BadRequestError", ErrorKind.INVALID_REQUEST),
        (401, "AuthenticationError", "AuthenticationError", ErrorKind.AUTH),
        (403, "PermissionDeniedError", "PermissionDeniedError", ErrorKind.AUTH),
        (404, "NotFoundError", "NotFoundError", ErrorKind.INVALID_REQUEST),
        (408, "APIStatusError", "APIStatusError", ErrorKind.TRANSIENT),
        (413, "RequestTooLargeError", "APIStatusError", ErrorKind.INVALID_REQUEST),
        (422, "UnprocessableEntityError", "UnprocessableEntityError", ErrorKind.INVALID_REQUEST),
        (429, "RateLimitError", "RateLimitError", ErrorKind.RATE_LIMITED),
        (500, "InternalServerError", "InternalServerError", ErrorKind.TRANSIENT),
        (503, "InternalServerError", "InternalServerError", ErrorKind.TRANSIENT),
        (529, "OverloadedError", "InternalServerError", ErrorKind.TRANSIENT),
    ],
)
async def test_status_errors_keep_their_sdk_class_names_and_classify(
    status, anthropic_error, openai_error, kind
):
    """The router records `type(e).__name__` as the span's error_code, so the
    SDK exception class names are part of the gateway's observable output.
    Both SDKs' errors for the same status must land in the same ErrorKind."""
    recorder = _Recorder(
        lambda req: httpx2.Response(status, request=req, json={"error": {"message": "nope"}})
    )
    config = GatewayConfig(anthropic_api_key="k", openai_api_key="k")

    with patch.object(anthropic, "_client", return_value=_anthropic_client(recorder)):
        with pytest.raises(anthropic_sdk.APIStatusError) as a_exc:
            await anthropic.call(config, "s", "p", 10)
    assert type(a_exc.value).__name__ == anthropic_error
    assert a_exc.value.status_code == status

    with patch.object(openai, "_client", return_value=_openai_client(recorder)):
        with pytest.raises(openai_sdk.APIStatusError) as o_exc:
            await openai.call(config, "s", "p", 10)
    assert type(o_exc.value).__name__ == openai_error
    assert o_exc.value.status_code == status
    assert classify(a_exc.value) is kind
    assert classify(o_exc.value) is kind


@pytest.mark.parametrize(
    ("transport_error", "expected", "kind"),
    [
        (httpx2.ConnectError("refused"), "APIConnectionError", ErrorKind.TRANSIENT),
        (httpx2.ReadTimeout("slow"), "APITimeoutError", ErrorKind.TIMEOUT),
    ],
)
async def test_transport_errors_are_wrapped_by_both_sdks(transport_error, expected, kind):
    def boom(request):
        raise transport_error

    config = GatewayConfig(anthropic_api_key="k", openai_api_key="k")
    with patch.object(anthropic, "_client", return_value=_anthropic_client(_Recorder(boom))):
        with pytest.raises(anthropic_sdk.APIConnectionError) as a_exc:
            await anthropic.call(config, "s", "p", 10)
    with patch.object(openai, "_client", return_value=_openai_client(_Recorder(boom))):
        with pytest.raises(openai_sdk.APIConnectionError) as o_exc:
            await openai.call(config, "s", "p", 10)
    assert type(a_exc.value).__name__ == expected
    assert type(o_exc.value).__name__ == expected
    assert classify(a_exc.value) is kind
    assert classify(o_exc.value) is kind


async def test_real_sdk_5xx_is_retried_then_fails_over_and_trips_breaker_once():
    recorder = _Recorder(
        lambda req: httpx2.Response(500, request=req, json={"error": {"message": "down"}})
    )
    groq_call = AsyncMock(return_value=ProviderResult("from groq", "llama", 1, 1))
    config = GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        provider_order="anthropic,groq",
        retry_attempts=2,
        breaker_failure_threshold=1,
    )
    sleep = AsyncMock()
    with (
        patch.object(anthropic, "_client", return_value=_anthropic_client(recorder)),
        patch("llm_gateway.router.CALLS", {"anthropic": anthropic.call, "groq": groq_call}),
        patch("llm_gateway.retry.asyncio.sleep", sleep),
    ):
        assert await complete(system="s", prompt="p", config=config) == "from groq"
        # One gateway retry (SDK retries disabled on this client) → 2 requests.
        assert len(recorder.bodies) == 2
        sleep.assert_awaited_once_with(config.retry_base_delay_seconds)

        # Threshold 1: that single logical failure opened the breaker, so the
        # next call goes straight to groq without touching anthropic.
        assert await complete(system="s", prompt="p", config=config) == "from groq"
        assert len(recorder.bodies) == 2


async def test_all_real_sdk_providers_failing_raises_llm_error():
    recorder = _Recorder(
        lambda req: httpx2.Response(529, request=req, json={"error": {"message": "overloaded"}})
    )
    config = GatewayConfig(anthropic_api_key="k", provider_order="anthropic", retry_attempts=1)
    with (
        patch.object(anthropic, "_client", return_value=_anthropic_client(recorder)),
        patch("llm_gateway.router.CALLS", {"anthropic": anthropic.call}),
    ):
        with pytest.raises(LLMError) as exc:
            await complete(system="s", prompt="p", config=config)
    assert isinstance(exc.value.__cause__, anthropic_sdk.OverloadedError)


# ─── Retry / failover / breaker policy, end to end through the real SDKs ───
#
# Anthropic is the primary and Groq (openai SDK) the fallback in every test
# below; both are genuine SDK clients over in-memory transports, and each
# policy is exercised through all three engines.

_ENGINES = ["complete", "chat", "stream_chat"]
_MESSAGES = [{"role": "user", "content": "p"}]


def _openai_ok(request: httpx2.Request) -> httpx2.Response:
    """A successful Groq/OpenAI reply saying "from groq", streamed or not."""
    if json.loads(request.content or b"{}").get("stream"):
        base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "llama"}
        chunks = [
            {**base, "choices": [{"index": 0, "delta": {"content": "from groq"}}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        return httpx2.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content=_sse(chunks, named=False),
        )
    completion = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "llama",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "from groq"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return httpx2.Response(200, request=request, json=completion)


def _status(status: int, **headers: str):
    return lambda req: httpx2.Response(
        status, request=req, headers=headers, json={"error": {"message": f"status {status}"}}
    )


def _connection_refused(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("refused")


async def _never_responds(request: httpx2.Request) -> httpx2.Response:
    await asyncio.sleep(3600)
    raise AssertionError("unreachable")


def _chain_config(**overrides) -> GatewayConfig:
    settings = dict(
        anthropic_api_key="k",
        groq_api_key="k",
        provider_order="anthropic,groq",
        retry_attempts=2,
        breaker_failure_threshold=1,
    )
    settings.update(overrides)
    return GatewayConfig(**settings)


@contextlib.contextmanager
def _real_sdk_providers(**recorders: _Recorder):
    """Route each named provider through a real SDK client over its recorder."""
    builders = {"anthropic": _anthropic_client, "groq": _openai_client, "openai": _openai_client}
    modules = {"anthropic": anthropic, "groq": groq, "openai": openai}
    with contextlib.ExitStack() as stack:
        for name, recorder in recorders.items():
            stack.enter_context(
                patch.object(modules[name], "_client", return_value=builders[name](recorder))
            )
        yield


async def _run(engine: str, config: GatewayConfig) -> str:
    if engine == "complete":
        return await complete(system="s", prompt="p", config=config)
    if engine == "chat":
        return (await chat(messages=_MESSAGES, config=config)).content
    return "".join([d.content or "" async for d in stream_chat(messages=_MESSAGES, config=config)])


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error_is_not_retried_fails_over_and_counts_for_breaker(engine, status):
    primary, fallback = _Recorder(_status(status)), _Recorder(_openai_ok)
    with (
        _real_sdk_providers(anthropic=primary, groq=fallback),
        patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep,
    ):
        assert await _run(engine, _chain_config()) == "from groq"
    assert len(primary.bodies) == 1
    sleep.assert_not_awaited()
    assert breaker.is_open("anthropic")


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("status", [400, 404, 413, 422])
async def test_request_error_is_not_retried_fails_over_and_never_trips_breaker(engine, status):
    """Documented decision: a 4xx other than 401/403/429 still fails over.
    The gateway translates each request per provider (sampling params,
    images, tools, structured output) and providers differ in context
    windows, model names and content policy, so "invalid here" does not
    mean "invalid everywhere" — and the extra call is cheap because it is
    never retried. It must not count toward the breaker: it's the caller's
    request, not the provider's health."""
    primary, fallback = _Recorder(_status(status)), _Recorder(_openai_ok)
    config = _chain_config()
    with (
        _real_sdk_providers(anthropic=primary, groq=fallback),
        patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep,
    ):
        for _ in range(3):
            assert await _run(engine, config) == "from groq"
    # Threshold is 1, yet all three calls still reached anthropic exactly once.
    assert len(primary.bodies) == 3
    sleep.assert_not_awaited()
    assert not breaker.is_open("anthropic")


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize(
    "respond",
    [_status(500), _status(503), _status(529), _connection_refused],
    ids=["500", "503", "529-overloaded", "connection-error"],
)
async def test_transient_error_is_retried_once_then_fails_over(engine, respond):
    primary, fallback = _Recorder(respond), _Recorder(_openai_ok)
    config = _chain_config()
    with (
        _real_sdk_providers(anthropic=primary, groq=fallback),
        patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep,
    ):
        assert await _run(engine, config) == "from groq"
    assert len(primary.bodies) == 2
    sleep.assert_awaited_once_with(config.retry_base_delay_seconds)
    assert breaker.is_open("anthropic")


@pytest.mark.parametrize("engine", _ENGINES)
async def test_rate_limit_fails_over_immediately_and_counts_for_breaker(engine):
    """The router has no provider-429 handling beyond the fallback chain
    (service/rate_limit.py only limits *inbound* callers). Retrying a
    throttled provider 0.2 s later almost always 429s again, so a 429 fails
    over at once and, like before, counts toward the breaker."""
    primary, fallback = _Recorder(_status(429, **{"retry-after": "1"})), _Recorder(_openai_ok)
    with (
        _real_sdk_providers(anthropic=primary, groq=fallback),
        patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep,
    ):
        assert await _run(engine, _chain_config()) == "from groq"
    assert len(primary.bodies) == 1
    sleep.assert_not_awaited()
    assert breaker.is_open("anthropic")


async def test_breaker_ignores_bad_requests_between_real_failures():
    statuses = iter([400, 500, 400, 422, 500])
    primary = _Recorder(lambda req: _status(next(statuses))(req))
    config = _chain_config(retry_attempts=1, breaker_failure_threshold=2)
    with _real_sdk_providers(anthropic=primary, groq=_Recorder(_openai_ok)):
        for expected_open in (False, False, False, False, True):
            await complete(system="s", prompt="p", config=config)
            assert breaker.is_open("anthropic") is expected_open
    assert len(primary.bodies) == 5


@pytest.mark.parametrize(
    ("provider", "body", "kind"),
    [
        (
            anthropic,
            {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
            ErrorKind.TRANSIENT,
        ),
        (
            anthropic,
            {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
            ErrorKind.INVALID_REQUEST,
        ),
        (openai, {"error": {"type": "server_error", "message": "oops"}}, ErrorKind.TRANSIENT),
        (
            openai,
            {"error": {"type": "invalid_request_error", "message": "bad"}},
            ErrorKind.INVALID_REQUEST,
        ),
    ],
)
async def test_errors_arriving_inside_a_200_stream_are_classified_by_body(provider, body, kind):
    named = provider is anthropic
    prefix = "event: error\n" if named else ""
    payload = f"{prefix}data: {json.dumps(body)}\n\n".encode()
    recorder = _Recorder(
        lambda req: httpx2.Response(
            200, request=req, headers={"content-type": "text/event-stream"}, content=payload
        )
    )
    build = _anthropic_client if named else _openai_client
    config = GatewayConfig(anthropic_api_key="k", openai_api_key="k")
    with patch.object(provider, "_client", return_value=build(recorder)):
        with pytest.raises(Exception) as exc:  # noqa: B017 — the SDK-specific class is the point
            async for _ in provider.stream_chat(config, _MESSAGES, None, 10):
                pass
    assert classify(exc.value) is kind


# ─── Timeouts and the overall call deadline ────────────────────────────────


@pytest.mark.parametrize("provider", [anthropic, openai, groq])
async def test_provider_clients_get_gateway_timeout_and_sdk_retries(provider):
    keys = dict(anthropic_api_key="k", openai_api_key="k", groq_api_key="k")
    provider._clients.clear()
    try:
        default = provider._client(GatewayConfig(**keys))
        assert default.max_retries == 0
        assert (default.timeout.read, default.timeout.connect) == (45.0, 5.0)

        tuned = provider._client(
            GatewayConfig(**keys, request_timeout_seconds=12, sdk_max_retries=1)
        )
        assert tuned is not default  # settings are part of the client cache key
        assert (tuned.max_retries, tuned.timeout.read) == (1, 12.0)

        # 0 disables the gateway bound: the SDK's own default timeout applies.
        unbounded = provider._client(GatewayConfig(**keys, request_timeout_seconds=0))
        assert unbounded.timeout.read == 600
    finally:
        provider._clients.clear()


@pytest.mark.parametrize("engine", ["complete", "chat"])
async def test_never_responding_provider_hits_per_attempt_timeout_and_fails_over(engine):
    primary, fallback = _Recorder(_never_responds), _Recorder(_openai_ok)
    config = _chain_config(request_timeout_seconds=0.2)
    started = time.monotonic()
    with _real_sdk_providers(anthropic=primary, groq=fallback):
        assert await _run(engine, config) == "from groq"
    elapsed = time.monotonic() - started
    assert 0.2 <= elapsed < 2.0
    # A timeout is not retried on the same provider: a second attempt would
    # spend another full timeout that the next provider could use instead.
    assert len(primary.bodies) == 1
    assert breaker.is_open("anthropic")


@contextlib.asynccontextmanager
async def _silent_server():
    """A real localhost TCP server that accepts requests and never answers —
    unlike MockTransport, this exercises the SDK's own httpx timeouts."""

    async def swallow(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while await reader.read(65536):
            pass
        writer.close()

    server = await asyncio.start_server(swallow, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.close()


@pytest.mark.parametrize("provider", [anthropic, openai])
@pytest.mark.parametrize("streaming", [False, True], ids=["request", "stream"])
async def test_silent_provider_trips_the_sdk_timeout(provider, streaming):
    config = GatewayConfig(
        anthropic_api_key="k",
        openai_api_key="k",
        # Only the timeout for the mode under test is short, which proves
        # streaming uses its own per-request timeout, not the client's.
        request_timeout_seconds=30.0 if streaming else 0.3,
        stream_idle_timeout_seconds=0.3 if streaming else 30.0,
    )
    provider._clients.clear()
    try:
        async with _silent_server() as base_url:
            client = provider._client(config).with_options(base_url=base_url)
            started = time.monotonic()
            with patch.object(provider, "_client", return_value=client):
                with pytest.raises(Exception) as exc:  # noqa: B017 — SDK-specific class
                    if streaming:
                        async for _ in provider.stream_chat(config, _MESSAGES, None, 10):
                            pass
                    else:
                        await provider.call(config, "s", "p", 10)
            elapsed = time.monotonic() - started
    finally:
        provider._clients.clear()
    assert type(exc.value).__name__ == "APITimeoutError"
    assert classify(exc.value) is ErrorKind.TIMEOUT
    assert elapsed < 5.0


@pytest.mark.parametrize("engine", ["complete", "chat"])
async def test_call_deadline_stops_the_chain(engine):
    primary, secondary, last = (
        _Recorder(_never_responds),
        _Recorder(_never_responds),
        _Recorder(_openai_ok),
    )
    config = _chain_config(
        openai_api_key="k",
        provider_order="anthropic,groq,openai",
        request_timeout_seconds=0.3,
        call_deadline_seconds=0.5,
    )
    started = time.monotonic()
    with _real_sdk_providers(anthropic=primary, groq=secondary, openai=last):
        with pytest.raises(LLMDeadlineExceeded) as exc:
            await _run(engine, config)
    assert time.monotonic() - started < 2.0
    assert isinstance(exc.value, LLMError)  # callers keep a single error path
    assert (len(primary.bodies), len(secondary.bodies), len(last.bodies)) == (1, 1, 0)
    # Anthropic hit its own per-attempt timeout; groq was only cut short by
    # the deadline, which says nothing about groq's health.
    assert breaker.is_open("anthropic")
    assert not breaker.is_open("groq")


async def test_call_deadline_bounds_stream_preflight():
    primary = _Recorder(_never_responds)
    config = _chain_config(provider_order="anthropic", call_deadline_seconds=0.3)
    started = time.monotonic()
    with _real_sdk_providers(anthropic=primary):
        with pytest.raises(LLMDeadlineExceeded):
            await _run("stream_chat", config)
    assert time.monotonic() - started < 2.0
    assert not breaker.is_open("anthropic")


async def test_call_deadline_ends_a_stream_that_stalls_after_starting():
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt"}
    first = {**base, "choices": [{"index": 0, "delta": {"content": "Hel"}}]}

    async def stalls(request: httpx2.Request) -> httpx2.Response:
        async def body():
            yield f"data: {json.dumps(first)}\n\n".encode()
            await asyncio.sleep(3600)

        return httpx2.Response(
            200, request=request, headers={"content-type": "text/event-stream"}, content=body()
        )

    config = GatewayConfig(openai_api_key="k", provider_order="openai", call_deadline_seconds=0.3)
    received: list[str] = []
    started = time.monotonic()
    with _real_sdk_providers(openai=_Recorder(stalls)):
        with pytest.raises(LLMDeadlineExceeded):
            async for delta in stream_chat(messages=_MESSAGES, config=config):
                received.append(delta.content)
    assert time.monotonic() - started < 2.0
    assert received == ["Hel"]


# ─── Multi-block Anthropic responses ───────────────────────────────────────
#
# A model that thinks (e.g. extended thinking, or Sonnet 5 where thinking is
# on by default) returns `thinking` / `redacted_thinking` blocks *before* its
# text. These tests use only the 0.3 public API so they also run against it.

_THINKING_FIRST_CONTENT = [
    {"type": "thinking", "thinking": "The user wants a greeting.", "signature": "sig-1"},
    {"type": "redacted_thinking", "data": "opaque-encrypted-reasoning"},
    {"type": "text", "text": "Hello, "},
    {"type": "text", "text": "world."},
]


def _anthropic_content_reply(content: list[dict], stop_reason: str = "end_turn"):
    body = {**_ANTHROPIC_MESSAGE, "content": content, "stop_reason": stop_reason}
    return lambda req: httpx2.Response(200, request=req, json=body)


async def test_complete_returns_text_blocks_after_thinking_blocks():
    recorder = _Recorder(_anthropic_content_reply(_THINKING_FIRST_CONTENT))
    config = GatewayConfig(anthropic_api_key="k", provider_order="anthropic", retry_attempts=1)
    with _real_sdk_providers(anthropic=recorder):
        text = await complete(system="s", prompt="p", config=config)
    assert text == "Hello, world."


async def test_anthropic_call_skips_thinking_and_tool_use_blocks():
    content = [
        {"type": "thinking", "thinking": "Plan.", "signature": "sig-1"},
        {"type": "text", "text": "Answer."},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
    ]
    recorder = _Recorder(_anthropic_content_reply(content, stop_reason="tool_use"))
    config = GatewayConfig(anthropic_api_key="k")
    with patch.object(anthropic, "_client", return_value=_anthropic_client(recorder)):
        result = await anthropic.call(config, "s", "p", 50)
    assert result.text == "Answer."


async def test_chat_ignores_thinking_blocks_and_keeps_tool_calls():
    content = [*_THINKING_FIRST_CONTENT[:2], {"type": "text", "text": "Looking it up."}]
    content.append({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}})
    recorder = _Recorder(_anthropic_content_reply(content, stop_reason="tool_use"))
    config = GatewayConfig(anthropic_api_key="k")
    with patch.object(anthropic, "_client", return_value=_anthropic_client(recorder)):
        result = await anthropic.chat(
            config,
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            max_tokens=50,
        )
    assert result.content == "Looking it up."
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("lookup", {"q": "x"})]


async def test_thinking_only_response_fails_over_to_the_next_provider():
    primary = _Recorder(_anthropic_content_reply(_THINKING_FIRST_CONTENT[:2], "max_tokens"))
    fallback = _Recorder(_openai_ok)
    with _real_sdk_providers(anthropic=primary, groq=fallback):
        text = await complete(system="s", prompt="p", config=_chain_config())
    assert text == "from groq"
    assert len(primary.bodies) == 1  # an unusable answer is not re-rolled
