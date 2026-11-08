"""Circuit breaker state machine: closed -> open -> half-open (one probe) -> closed."""

import threading
from unittest.mock import patch

import pytest

from llm_gateway import breaker


@pytest.fixture(autouse=True)
def _reset():
    breaker.reset()
    yield
    breaker.reset()


def _trip(provider="p", cooldown=10.0):
    breaker.record_failure(provider, threshold=1, cooldown_seconds=cooldown)


def test_opens_then_half_open_admits_exactly_one_probe():
    with patch("llm_gateway.breaker.time.monotonic", return_value=100.0):
        _trip()
        assert breaker.is_open("p")
        assert breaker.allow("p") is False
    with patch("llm_gateway.breaker.time.monotonic", return_value=111.0):
        assert breaker.allow("p") is True  # the probe
        assert breaker.allow("p") is False  # everyone else still waits
        assert breaker.is_open("p")  # and the pure check agrees


def test_probe_success_closes_and_probe_failure_reopens():
    with patch("llm_gateway.breaker.time.monotonic", return_value=100.0):
        _trip()
    with patch("llm_gateway.breaker.time.monotonic", return_value=111.0):
        assert breaker.allow("p")
        breaker.record_success("p")
        assert not breaker.is_open("p")
        assert breaker.allow("p") and breaker.allow("p")

    breaker.reset()
    with patch("llm_gateway.breaker.time.monotonic", return_value=100.0):
        _trip()
    with patch("llm_gateway.breaker.time.monotonic", return_value=111.0):
        assert breaker.allow("p")
        _trip()  # probe failed
        assert breaker.is_open("p")
    with patch("llm_gateway.breaker.time.monotonic", return_value=122.0):
        assert breaker.allow("p")  # next window, next probe


def test_lost_probe_lease_expires_so_provider_is_not_stuck_open():
    with patch("llm_gateway.breaker.time.monotonic", return_value=100.0):
        _trip()
    with patch("llm_gateway.breaker.time.monotonic", return_value=111.0):
        assert breaker.allow("p")  # probe claimed, caller vanishes (cancelled)
    with patch("llm_gateway.breaker.time.monotonic", return_value=122.0):
        assert breaker.allow("p")


def test_closed_breaker_always_allows():
    assert all(breaker.allow("fresh") for _ in range(5))


def test_concurrent_callers_get_a_single_probe():
    with patch("llm_gateway.breaker.time.monotonic", return_value=100.0):
        _trip()
    results = []
    with patch("llm_gateway.breaker.time.monotonic", return_value=111.0):
        threads = [
            threading.Thread(target=lambda: results.append(breaker.allow("p"))) for _ in range(32)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert results.count(True) == 1


def test_non_tripping_error_releases_the_probe():
    import anthropic
    import httpx2

    req = httpx2.Request("POST", "https://x.test")
    bad = anthropic.AsyncAnthropic(api_key="k")._make_status_error(
        "bad", body=None, response=httpx2.Response(400, request=req)
    )
    with patch("llm_gateway.breaker.time.monotonic", return_value=100.0):
        _trip()
    with patch("llm_gateway.breaker.time.monotonic", return_value=111.0):
        assert breaker.allow("p")
        assert breaker.record_error("p", bad, threshold=1, cooldown_seconds=10.0) is False
        assert breaker.allow("p")  # provider answered; another probe may go
