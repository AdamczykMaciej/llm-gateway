"""Contract tests against the *real* provider SDKs (anthropic 1.x, openai 3.x).

Everything else in this suite mocks the SDK client object, which is exactly
what hid the anthropic 1.0 / openai 3.0 breaking changes: a MagicMock accepts
any keyword argument and any http_client. These tests instead build genuine
SDK clients over an in-memory httpx2 MockTransport (no network, no API keys),
so a removed parameter, a rejected http_client, a renamed exception, or a
changed response/stream model fails here.
"""

import json
from unittest.mock import AsyncMock, patch

import anthropic as anthropic_sdk
import httpx2
import openai as openai_sdk
import pytest

from llm_gateway import GatewayConfig, LLMError, complete, reset_circuit_breakers
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
    ("status", "anthropic_error", "openai_error"),
    [
        (400, "BadRequestError", "BadRequestError"),
        (401, "AuthenticationError", "AuthenticationError"),
        (429, "RateLimitError", "RateLimitError"),
        (500, "InternalServerError", "InternalServerError"),
        (529, "OverloadedError", "InternalServerError"),
    ],
)
async def test_status_errors_keep_their_sdk_class_names(status, anthropic_error, openai_error):
    """The router records `type(e).__name__` as the span's error_code, so the
    SDK exception class names are part of the gateway's observable output."""
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


@pytest.mark.parametrize(
    ("transport_error", "expected"),
    [
        (httpx2.ConnectError("refused"), "APIConnectionError"),
        (httpx2.ReadTimeout("slow"), "APITimeoutError"),
    ],
)
async def test_transport_errors_are_wrapped_by_both_sdks(transport_error, expected):
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
