"""Idempotency keys on chat(): correlation, OpenAI header, in-flight dedupe. Fakes only."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_gateway import GatewayConfig, InvalidRequestError, chat, reset_circuit_breakers
from llm_gateway.providers import openai as openai_provider
from llm_gateway.providers.base import ChatResult

pytestmark = pytest.mark.asyncio

_MSGS = [{"role": "user", "content": "hi"}]
_OK = ChatResult(content="hello", model="m", input_tokens=1, output_tokens=1)


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(order="openai") -> GatewayConfig:
    return GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        openai_api_key="k",
        retry_attempts=1,
        provider_order=order,
    )


async def test_bad_key_rejected():
    with pytest.raises(InvalidRequestError):
        await chat(messages=_MSGS, config=_cfg(), idempotency_key="bad key!")


async def test_key_on_attempt_reports_across_fallback():
    reports = []
    calls = {
        "openai": AsyncMock(side_effect=RuntimeError("boom")),
        "groq": AsyncMock(return_value=_OK),
    }
    with patch("llm_gateway.chat.CHAT_CALLS", calls):
        await chat(
            messages=_MSGS,
            config=_cfg("openai,groq"),
            idempotency_key="k-1",
            on_attempt=reports.append,
        )
    assert [r.idempotency_key for r in reports] == ["k-1", "k-1"]
    assert len(reports) == 2


async def test_no_key_reports_none():
    reports = []
    with patch("llm_gateway.chat.CHAT_CALLS", {"openai": AsyncMock(return_value=_OK)}):
        await chat(messages=_MSGS, config=_cfg(), on_attempt=reports.append)
    assert reports[0].idempotency_key is None


async def test_key_only_forwarded_to_openai():
    openai_call = AsyncMock(side_effect=RuntimeError("boom"))
    groq_call = AsyncMock(return_value=_OK)
    with patch("llm_gateway.chat.CHAT_CALLS", {"openai": openai_call, "groq": groq_call}):
        await chat(messages=_MSGS, config=_cfg("openai,groq"), idempotency_key="k-2")
    assert openai_call.call_args.kwargs["idempotency_key"] == "k-2"
    assert "idempotency_key" not in groq_call.call_args.kwargs


async def test_openai_sends_idempotency_header():
    create = AsyncMock(return_value=MagicMock())
    client = MagicMock()
    client.chat.completions.create = create
    with (
        patch.object(openai_provider, "_client", return_value=client),
        patch.object(openai_provider, "parse_openai_style_response", return_value=_OK),
    ):
        await openai_provider.chat(_cfg(), _MSGS, None, 10, idempotency_key="k-3")
        assert create.call_args.kwargs["extra_headers"] == {"Idempotency-Key": "k-3"}
        await openai_provider.chat(_cfg(), _MSGS, None, 10)
        assert "extra_headers" not in create.call_args.kwargs


async def test_concurrent_same_key_shares_one_call():
    gate = asyncio.Event()

    async def slow(*a, **k):
        await gate.wait()
        return _OK

    call = AsyncMock(side_effect=slow)
    with patch("llm_gateway.chat.CHAT_CALLS", {"openai": call}):
        t1 = asyncio.create_task(chat(messages=_MSGS, config=_cfg(), idempotency_key="same"))
        t2 = asyncio.create_task(chat(messages=_MSGS, config=_cfg(), idempotency_key="same"))
        await asyncio.sleep(0.01)
        gate.set()
        r1, r2 = await asyncio.gather(t1, t2)
    assert call.await_count == 1
    assert r1 is r2


async def test_same_key_different_request_conflicts_and_nothing_cached():
    gate = asyncio.Event()

    async def slow(*a, **k):
        await gate.wait()
        return _OK

    call = AsyncMock(side_effect=slow)
    with patch("llm_gateway.chat.CHAT_CALLS", {"openai": call}):
        t1 = asyncio.create_task(chat(messages=_MSGS, config=_cfg(), idempotency_key="x"))
        await asyncio.sleep(0.01)
        with pytest.raises(InvalidRequestError):
            await chat(
                messages=[{"role": "user", "content": "other"}],
                config=_cfg(),
                idempotency_key="x",
            )
        gate.set()
        await t1
        await chat(messages=_MSGS, config=_cfg(), idempotency_key="x")
    assert call.await_count == 2  # finished calls are not cached


async def test_one_waiter_cancelling_does_not_cancel_the_other():
    gate = asyncio.Event()

    async def slow(*a, **k):
        await gate.wait()
        return _OK

    with patch("llm_gateway.chat.CHAT_CALLS", {"openai": AsyncMock(side_effect=slow)}):
        t1 = asyncio.create_task(chat(messages=_MSGS, config=_cfg(), idempotency_key="c"))
        t2 = asyncio.create_task(chat(messages=_MSGS, config=_cfg(), idempotency_key="c"))
        await asyncio.sleep(0.01)
        t1.cancel()
        await asyncio.sleep(0.01)
        gate.set()
        assert (await t2).content == "hello"


async def test_failure_shared_then_retryable():
    call = AsyncMock(side_effect=[RuntimeError("boom"), _OK])
    with patch("llm_gateway.chat.CHAT_CALLS", {"openai": call}):
        with pytest.raises(Exception):  # noqa: B017
            await chat(messages=_MSGS, config=_cfg(), idempotency_key="r")
        out = await chat(messages=_MSGS, config=_cfg(), idempotency_key="r")
    assert out.content == "hello"
