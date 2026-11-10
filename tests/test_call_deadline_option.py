"""Per-call `deadline_seconds=`: one total budget across retries, Retry-After
waits, failover and repair calls (fake providers and a fake clock only)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx2
import openai
import pytest

from llm_gateway import (
    GatewayConfig,
    InvalidRequestError,
    LLMDeadlineExceeded,
    chat,
    complete,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers.base import ProviderResult

pytestmark = pytest.mark.asyncio
_REQ = httpx2.Request("POST", "https://api.example.test")
_MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    c = Clock()
    with patch("llm_gateway.retry.time", SimpleNamespace(monotonic=c)):
        yield c


def _cfg(**kw):
    kw.setdefault("provider_order", "anthropic,groq")
    return GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        retry_attempts=3,
        retry_base_delay_seconds=0.01,
        call_deadline_seconds=90.0,
        **kw,
    )


def _slow_503(clock, took, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else None
    err = openai.AsyncOpenAI(api_key="k")._make_status_error(
        "overloaded",
        body=None,
        response=httpx2.Response(503, request=_REQ, headers=headers),
    )

    async def call(*a, **k):
        clock.now += took
        raise err

    return AsyncMock(side_effect=call)


async def test_budget_spans_retry_after_and_failover(clock):
    first = _slow_503(clock, 3, retry_after=10)  # 10s wait does not fit the 2s left
    second = _slow_503(clock, 3)
    sleep = AsyncMock()
    with (
        patch("llm_gateway.router.CALLS", {"anthropic": first, "groq": second}),
        patch("llm_gateway.retry.asyncio.sleep", sleep),
    ):
        with pytest.raises(LLMDeadlineExceeded, match=r"deadline of 5s"):
            await complete(system="s", prompt="p", config=_cfg(), deadline_seconds=5)
    sleep.assert_not_awaited()  # never waited out the Retry-After
    assert first.await_count == 1
    assert second.await_count == 1  # started with 2s left, then the budget was gone


async def test_no_fallback_started_without_time_left(clock):
    first = _slow_503(clock, 6)
    second = AsyncMock(
        return_value=ProviderResult(text="x", model="m", input_tokens=1, output_tokens=1)
    )
    with patch("llm_gateway.router.CALLS", {"anthropic": first, "groq": second}):
        with pytest.raises(LLMDeadlineExceeded):
            await complete(system="s", prompt="p", config=_cfg(), deadline_seconds=5)
    second.assert_not_awaited()


async def test_default_keeps_config_deadline(clock):
    first = _slow_503(clock, 6)
    second = AsyncMock(
        return_value=ProviderResult(text="served", model="m", input_tokens=1, output_tokens=1)
    )
    with (
        patch("llm_gateway.router.CALLS", {"anthropic": first, "groq": second}),
        patch("llm_gateway.retry.asyncio.sleep", AsyncMock()),
    ):
        assert await complete(system="s", prompt="p", config=_cfg()) == "served"


async def test_repair_calls_share_the_budget(clock):
    async def call(*a, **k):
        clock.now += 3
        return ProviderResult(text="not json", model="m", input_tokens=1, output_tokens=1)

    provider = AsyncMock(side_effect=call)
    with patch("llm_gateway.router.CALLS", {"anthropic": provider}):
        with pytest.raises(LLMDeadlineExceeded):
            await complete_with_usage(
                system="s",
                prompt="p",
                config=_cfg(provider_order="anthropic"),
                validate={"type": "object"},
                repair_attempts=5,
                deadline_seconds=5,
            )
    assert provider.await_count == 2  # first call + one repair; then 6s > 5s


async def test_chat_honours_deadline_seconds(clock):
    first = _slow_503(clock, 6)
    second = AsyncMock(return_value=None)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": first, "groq": second}):
        with pytest.raises(LLMDeadlineExceeded, match=r"deadline of 5s"):
            await chat(messages=_MSGS, config=_cfg(), deadline_seconds=5)
    second.assert_not_awaited()


async def test_stream_chat_honours_deadline_seconds(clock):
    async def failing(*a, **k):
        clock.now += 6
        raise ConnectionResetError("boom")
        yield  # pragma: no cover

    def second(*a, **k):  # pragma: no cover
        raise AssertionError("no time left for a fallback")

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": failing, "groq": second}):
        with pytest.raises(LLMDeadlineExceeded, match=r"deadline of 5s"):
            async for _ in stream_chat(messages=_MSGS, config=_cfg(), deadline_seconds=5):
                pass


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "5"])
async def test_invalid_deadline_seconds_rejected_before_any_call(bad):
    provider = AsyncMock()
    with patch("llm_gateway.router.CALLS", {"anthropic": provider}):
        with pytest.raises(InvalidRequestError, match="deadline_seconds"):
            await complete(
                system="s",
                prompt="p",
                config=_cfg(provider_order="anthropic"),
                deadline_seconds=bad,  # type: ignore[arg-type]
            )
    provider.assert_not_awaited()
