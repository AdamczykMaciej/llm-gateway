import asyncio
import time
from unittest.mock import AsyncMock, patch

import anthropic
import httpx2
import pytest

from llm_gateway.retry import Deadline, call_with_retry

pytestmark = pytest.mark.asyncio

_REQUEST = httpx2.Request("POST", "https://api.example.test/v1/messages")


def _transient(message: str = "blip") -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(message=message, request=_REQUEST)


def _status_error(status: int) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=_REQUEST)
    return anthropic.AsyncAnthropic(api_key="k")._make_status_error(
        "nope", body=None, response=response
    )


async def test_returns_result_on_first_success():
    call = AsyncMock(return_value="ok")
    result = await call_with_retry(call, attempts=3, base_delay_seconds=0.01)
    assert result == "ok"
    call.assert_awaited_once()


async def test_retries_after_a_transient_failure_then_succeeds():
    call = AsyncMock(side_effect=[_transient(), "ok"])
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep:
        result = await call_with_retry(call, attempts=3, base_delay_seconds=0.01)
    assert result == "ok"
    assert call.await_count == 2
    sleep.assert_awaited_once_with(0.01)


async def test_raises_the_last_error_when_every_attempt_fails():
    errors = [_transient("first"), _transient("second"), _transient("last")]
    call = AsyncMock(side_effect=errors)
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()):
        with pytest.raises(anthropic.APIConnectionError, match="last"):
            await call_with_retry(call, attempts=3, base_delay_seconds=0.01)
    assert call.await_count == 3


async def test_delay_doubles_between_attempts():
    call = AsyncMock(side_effect=[_transient("a"), _transient("b"), "ok"])
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep:
        await call_with_retry(call, attempts=3, base_delay_seconds=0.1)
    sleep.assert_any_call(0.1)
    sleep.assert_any_call(0.2)
    assert sleep.await_count == 2


async def test_attempts_below_one_still_calls_once():
    call = AsyncMock(return_value="ok")
    result = await call_with_retry(call, attempts=0, base_delay_seconds=0.01)
    assert result == "ok"
    call.assert_awaited_once()


async def test_no_sleep_when_attempts_is_one():
    call = AsyncMock(side_effect=_transient())
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep:
        with pytest.raises(anthropic.APIConnectionError):
            await call_with_retry(call, attempts=1, base_delay_seconds=0.01)
    sleep.assert_not_awaited()
    call.assert_awaited_once()


@pytest.mark.parametrize(
    "error",
    [
        _status_error(400),
        _status_error(401),
        _status_error(403),
        _status_error(422),
        _status_error(429),
        TimeoutError(),
        RuntimeError("a bug, not a blip"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_non_retryable_errors_are_raised_without_a_second_attempt(error):
    call = AsyncMock(side_effect=[error, "never reached"])
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep:
        with pytest.raises(type(error)):
            await call_with_retry(call, attempts=3, base_delay_seconds=0.01)
    call.assert_awaited_once()
    sleep.assert_not_awaited()


async def test_should_retry_can_be_overridden():
    call = AsyncMock(side_effect=[RuntimeError("x"), "ok"])
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()):
        result = await call_with_retry(
            call, attempts=2, base_delay_seconds=0.01, should_retry=lambda e: True
        )
    assert result == "ok"


async def test_attempt_timeout_cuts_off_a_hung_call_and_is_not_retried():
    calls = 0

    async def hang():
        nonlocal calls
        calls += 1
        await asyncio.sleep(3600)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await call_with_retry(
            hang, attempts=3, base_delay_seconds=0.01, attempt_timeout_seconds=0.05
        )
    assert time.monotonic() - started < 1.0
    assert calls == 1


async def test_deadline_caps_a_longer_attempt_timeout():
    async def hang():
        await asyncio.sleep(3600)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await call_with_retry(
            hang,
            attempts=1,
            base_delay_seconds=0.01,
            attempt_timeout_seconds=30.0,
            deadline=Deadline(0.05),
        )
    assert time.monotonic() - started < 1.0


async def test_no_retry_when_the_backoff_would_overrun_the_deadline():
    call = AsyncMock(side_effect=[_transient(), "ok"])
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep:
        with pytest.raises(anthropic.APIConnectionError):
            await call_with_retry(call, attempts=2, base_delay_seconds=5.0, deadline=Deadline(1.0))
    call.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.parametrize("seconds", [None, 0, -1])
async def test_non_positive_deadline_is_unbounded(seconds):
    deadline = Deadline(seconds)
    assert deadline.remaining() is None
    assert deadline.expired is False
    assert deadline.cap(5.0) == 5.0
    assert deadline.cap(0) is None


async def test_deadline_cap_picks_the_tighter_bound():
    deadline = Deadline(10.0)
    assert deadline.cap(2.0) == 2.0
    assert 9.0 < deadline.cap(30.0) <= 10.0
    assert 9.0 < deadline.cap(None) <= 10.0
