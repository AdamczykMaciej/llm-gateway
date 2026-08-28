from unittest.mock import AsyncMock, patch

import pytest

from llm_gateway.retry import call_with_retry

pytestmark = pytest.mark.asyncio


async def test_returns_result_on_first_success():
    call = AsyncMock(return_value="ok")
    result = await call_with_retry(call, attempts=3, base_delay_seconds=0.01)
    assert result == "ok"
    call.assert_awaited_once()


async def test_retries_after_a_failure_then_succeeds():
    call = AsyncMock(side_effect=[RuntimeError("boom"), "ok"])
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep:
        result = await call_with_retry(call, attempts=3, base_delay_seconds=0.01)
    assert result == "ok"
    assert call.await_count == 2
    sleep.assert_awaited_once_with(0.01)


async def test_raises_the_last_error_when_every_attempt_fails():
    errors = [RuntimeError("first"), RuntimeError("second"), RuntimeError("last")]
    call = AsyncMock(side_effect=errors)
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()):
        with pytest.raises(RuntimeError, match="last"):
            await call_with_retry(call, attempts=3, base_delay_seconds=0.01)
    assert call.await_count == 3


async def test_delay_doubles_between_attempts():
    call = AsyncMock(side_effect=[RuntimeError("a"), RuntimeError("b"), "ok"])
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
    call = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("llm_gateway.retry.asyncio.sleep", AsyncMock()) as sleep:
        with pytest.raises(RuntimeError):
            await call_with_retry(call, attempts=1, base_delay_seconds=0.01)
    sleep.assert_not_awaited()
    call.assert_awaited_once()
