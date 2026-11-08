"""Reliability pass 2: model-scoped breaker keys, half-open concurrency at the
engine level, timeouts on every adapter, client lifecycle and bounded state."""

import asyncio
from unittest.mock import AsyncMock, patch

import anthropic
import httpx2
import pytest

from llm_gateway import GatewayConfig, breaker, chat, complete, reset_circuit_breakers
from llm_gateway.providers import (
    aclose_all,
    azure,
    base,
    gemini,
    groq,
    huggingface,
    mistral,
    openai_compat,
    openrouter,
    vertex,
)
from llm_gateway.providers import (
    anthropic as anthropic_provider,
)
from llm_gateway.providers import (
    openai as openai_provider,
)
from llm_gateway.providers.base import ChatResult, ProviderResult
from tests.test_vertex_provider import _fake_google_credentials


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


_REQ = httpx2.Request("POST", "https://x.test")


def _status(cls, status: int, body: dict | None = None):
    return cls("boom", response=httpx2.Response(status, request=_REQ), body=body)


def _overloaded():
    return _status(anthropic.APIStatusError, 529, {"error": {"type": "overloaded_error"}})


def _model_missing():
    return _status(
        anthropic.NotFoundError,
        404,
        {"error": {"type": "not_found_error", "message": "model: claude-gone"}},
    )


def _auth():
    return _status(anthropic.AuthenticationError, 401)


def _trip(error, model="m1", provider="p", threshold=3):
    return breaker.record_error(
        provider, error, threshold=threshold, cooldown_seconds=30.0, model=model
    )


# ─── 1. Model-scoped breaker keys ──────────────────────────────────────────


def test_overloaded_model_opens_only_that_model_after_threshold():
    for _ in range(2):
        assert _trip(_overloaded()) is True
    assert not breaker.is_open("p", "m1")  # below the threshold
    _trip(_overloaded())
    assert breaker.is_open("p", "m1")
    assert not breaker.is_open("p", "m2")
    assert not breaker.is_open("p")
    assert breaker.allow("p", "m2") and breaker.allow("p")
    assert not breaker.allow("p", "m1")


def test_model_not_found_opens_that_model_at_once_but_a_plain_404_does_not():
    _trip(_model_missing())
    assert breaker.is_open("p", "m1") and not breaker.is_open("p", "m2")
    breaker.reset()
    plain = _status(anthropic.NotFoundError, 404, {"error": {"message": "no such route"}})
    assert _trip(plain) is False
    assert not breaker.is_open("p", "m1")


@pytest.mark.parametrize("error", [_auth(), _status(anthropic.APIStatusError, 500)])
def test_auth_and_ordinary_failures_still_open_the_whole_provider(error):
    _trip(error, threshold=1)
    assert breaker.is_open("p") and breaker.is_open("p", "m2")


def test_billing_opens_the_whole_provider_for_every_model():
    billing = _status(
        anthropic.BadRequestError, 400, {"error": {"message": "Your credit balance is too low"}}
    )
    _trip(billing)
    assert breaker.is_open("p", "other-model")


def test_model_argument_is_optional_and_provider_only_calls_are_unchanged():
    breaker.record_error("p", _overloaded(), threshold=1, cooldown_seconds=30.0)
    # No model given: legacy behaviour, a provider-wide failure.
    assert breaker.is_open("p") and breaker.is_open("p", "m1")


def test_half_open_model_probe_does_not_leak_a_provider_probe_lease():
    with patch("llm_gateway.breaker.time.monotonic", return_value=100.0):
        _trip(_model_missing())
    with patch("llm_gateway.breaker.time.monotonic", return_value=140.0):
        assert breaker.allow("p", "m1") is True  # the model's probe
        assert breaker.allow("p", "m1") is False
        assert breaker.allow("p", "m2") is True  # other models unaffected
        breaker.record_success("p", "m1")
        assert breaker.allow("p", "m1")


def test_breaker_state_is_bounded_per_model_id():
    for i in range(breaker.MAX_MODEL_KEYS * 3):
        _trip(_model_missing(), model=f"user-chosen-{i}")
    assert len(breaker._model_keys) == breaker.MAX_MODEL_KEYS
    for state in (breaker._failures, breaker._opened_until, breaker._cooldowns):
        assert len(state) <= breaker.MAX_MODEL_KEYS
    assert breaker.is_open("p", f"user-chosen-{breaker.MAX_MODEL_KEYS * 3 - 1}")
    assert not breaker.is_open("p", "user-chosen-0")  # evicted, closed again


def test_success_leaves_no_state_behind():
    for i in range(50):
        breaker.record_success("p", f"m{i}")
    assert not breaker._failures and not breaker._cooldowns


async def test_engine_default_model_not_found_fails_over_without_opening_the_provider():
    gone = AsyncMock(side_effect=_model_missing())
    ok = AsyncMock(return_value=ProviderResult("from groq", "llama", 1, 1))
    config = GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        provider_order="anthropic,groq",
        retry_attempts=1,
        claude_model="claude-gone",
    )
    with patch("llm_gateway.router.CALLS", {"anthropic": gone, "groq": ok}):
        assert await complete(system="s", prompt="p", config=config) == "from groq"
        assert breaker.is_open("anthropic", "claude-gone")
        assert not breaker.is_open("anthropic")
        await complete(system="s", prompt="p", config=config)
        assert gone.await_count == 1  # parked: the second call skipped it
        # The operator fixes the model: same provider, a different model key.
        gone.reset_mock()
        gone.side_effect = None
        gone.return_value = ProviderResult("from claude", "claude-new", 1, 1)
        fixed = config.model_copy(update={"claude_model": "claude-new"})
        assert await complete(system="s", prompt="p", config=fixed) == "from claude"


# ─── 2. Engine-level half-open concurrency ─────────────────────────────────


async def _half_open(provider: str, model: str) -> None:
    breaker.record_failure(provider, threshold=1, cooldown_seconds=0.05)
    await asyncio.sleep(0.08)


async def test_half_open_complete_admits_exactly_one_probe_under_concurrency():
    n = 25
    release = asyncio.Event()
    probe_started = asyncio.Event()
    calls = []

    async def anthropic_call(config, system, prompt, max_tokens, model, **kw):
        calls.append(prompt)
        probe_started.set()
        await release.wait()
        return ProviderResult("probe ok", "claude-x", 1, 1)

    groq_call = AsyncMock(return_value=ProviderResult("from groq", "llama", 1, 1))
    config = GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        provider_order="anthropic,groq",
        retry_attempts=1,
        breaker_cooldown_seconds=0.05,
    )
    await _half_open("anthropic", config.claude_model)
    with patch("llm_gateway.router.CALLS", {"anthropic": anthropic_call, "groq": groq_call}):
        tasks = [
            asyncio.create_task(complete(system="s", prompt=f"p{i}", config=config))
            for i in range(n)
        ]
        await probe_started.wait()
        # Everyone but the probe has failed over while the probe is in flight.
        for _ in range(100):
            if groq_call.await_count == n - 1:
                break
            await asyncio.sleep(0.01)
        assert groq_call.await_count == n - 1
        release.set()
        texts = sorted(await asyncio.gather(*tasks))
    assert len(calls) == 1
    assert texts.count("probe ok") == 1 and texts.count("from groq") == n - 1
    assert not breaker.is_open("anthropic")  # the probe's success closed it


async def test_half_open_chat_admits_exactly_one_probe_under_concurrency():
    n = 15
    release = asyncio.Event()
    probe_started = asyncio.Event()
    calls = []

    async def anthropic_chat(config, messages, tools, max_tokens, **kw):
        calls.append(1)
        probe_started.set()
        await release.wait()
        return ChatResult(content="probe ok", model="claude-x", input_tokens=1, output_tokens=1)

    groq_chat = AsyncMock(
        return_value=ChatResult(content="from groq", model="llama", input_tokens=1, output_tokens=1)
    )
    config = GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        provider_order="anthropic,groq",
        retry_attempts=1,
        breaker_cooldown_seconds=0.05,
    )
    await _half_open("anthropic", config.claude_model)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": anthropic_chat, "groq": groq_chat}):
        tasks = [
            asyncio.create_task(chat(messages=[{"role": "user", "content": "hi"}], config=config))
            for _ in range(n)
        ]
        await probe_started.wait()
        for _ in range(100):
            if groq_chat.await_count == n - 1:
                break
            await asyncio.sleep(0.01)
        assert groq_chat.await_count == n - 1
        release.set()
        results = await asyncio.gather(*tasks)
    assert len(calls) == 1
    assert sorted(r.content for r in results).count("probe ok") == 1


# ─── 3. Timeouts on every adapter ──────────────────────────────────────────

_KEYS = dict(
    anthropic_api_key="k",
    openai_api_key="k",
    groq_api_key="k",
    mistral_api_key="k",
    openrouter_api_key="k",
    huggingface_api_key="k",
    openai_compat_api_key="k",
    openai_compat_base_url="https://compat.test/v1",
    azure_endpoint="https://r.openai.azure.com",
    azure_model="d",
    azure_api_key="k",
    gemini_project_id="my-proj",
    gemini_model="gemini-x",
    vertex_project_id="my-proj",
    _env_file=None,
)


@pytest.mark.parametrize(
    "module",
    [
        anthropic_provider,
        openai_provider,
        groq,
        mistral,
        openrouter,
        huggingface,
        openai_compat,
        azure,
    ],
)
async def test_every_sync_adapter_client_carries_connect_and_read_deadlines(module):
    module._clients.clear()
    config = GatewayConfig(**_KEYS, request_timeout_seconds=12)
    client = module._client(config)
    assert (client.timeout.read, client.timeout.connect) == (12, 5.0)
    assert client.timeout.pool == 12 and client.timeout.write == 12
    assert client.max_retries == 0
    await aclose_all()


async def test_vertex_client_carries_deadlines():
    vertex._clients.clear()
    credentials = _fake_google_credentials()
    config = GatewayConfig(**_KEYS, request_timeout_seconds=12)
    with patch("google.auth.default", return_value=(credentials, "my-proj")):
        client = await vertex._client(config)
    assert (client.timeout.read, client.timeout.connect) == (12, 5.0)
    await aclose_all()


async def test_gemini_client_carries_deadlines():
    gemini._clients.clear()
    credentials = _fake_google_credentials()
    config = GatewayConfig(**_KEYS, request_timeout_seconds=12)
    with patch("google.auth.default", return_value=(credentials, "my-proj")):
        client = gemini._client(config)
    assert (client.timeout.read, client.timeout.connect) == (12, 5.0)
    await aclose_all()


async def test_stream_open_is_bounded_even_if_the_adapter_never_yields():
    """Token acquisition or a connect that hangs before any byte is not
    covered by httpx's read timeout; the engine bounds opening the stream."""
    from llm_gateway import stream_chat

    async def hangs(*args, **kwargs):
        await asyncio.sleep(30)
        yield  # pragma: no cover

    ok = AsyncMock()

    async def good(*args, **kwargs):
        from llm_gateway.providers.base import StreamDelta

        yield StreamDelta(content="hi", finish_reason="stop")

    config = GatewayConfig(
        anthropic_api_key="k",
        groq_api_key="k",
        provider_order="anthropic,groq",
        retry_attempts=1,
        stream_idle_timeout_seconds=0.1,
    )
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": hangs, "groq": good}):
        deltas = [
            d async for d in stream_chat(messages=[{"role": "user", "content": "x"}], config=config)
        ]
    assert [d.content for d in deltas] == ["hi"]
    del ok


# ─── 4. Client lifecycle ───────────────────────────────────────────────────


async def test_clients_are_reused_across_calls_not_rebuilt_per_call():
    config = GatewayConfig(**_KEYS)
    anthropic_provider._clients.clear()
    first = anthropic_provider._client(config)
    assert all(anthropic_provider._client(config) is first for _ in range(20))
    assert len(anthropic_provider._clients) == 1
    await aclose_all()


async def test_aclose_all_closes_every_cached_client_and_empties_every_cache():
    config = GatewayConfig(**_KEYS)
    modules = [
        anthropic_provider,
        openai_provider,
        groq,
        mistral,
        openrouter,
        huggingface,
        openai_compat,
        azure,
    ]
    clients = [m._client(config) for m in modules]
    with (
        patch.object(type(clients[0]), "close", new=AsyncMock()) as close_anthropic,
        patch.object(type(clients[1]), "close", new=AsyncMock()) as close_openai,
    ):
        await aclose_all()
    assert close_anthropic.await_count + close_openai.await_count >= len(modules)
    assert all(not m._clients for m in modules)


async def test_aclose_all_survives_one_failing_close():
    config = GatewayConfig(**_KEYS)
    bad = groq._client(config)
    good = mistral._client(config)
    with (
        patch.object(bad, "close", AsyncMock(side_effect=RuntimeError("boom"))),
        patch.object(good, "close", AsyncMock()) as good_close,
    ):
        await aclose_all()
    good_close.assert_awaited_once()
    assert not groq._clients and not mistral._clients


async def test_service_shutdown_closes_every_provider_client():
    from fastapi.testclient import TestClient

    from llm_gateway.service.app import create_app

    config = GatewayConfig(**_KEYS)
    anthropic_provider._clients.clear()
    client = anthropic_provider._client(config)
    with patch.object(client, "close", AsyncMock()) as close:
        with TestClient(create_app(config)):
            pass
    close.assert_awaited_once()


async def test_aclose_is_cancellation_safe_and_idempotent():
    config = GatewayConfig(**_KEYS)
    client = openai_provider._client(config)
    started = asyncio.Event()

    async def slow_close():
        started.set()
        await asyncio.sleep(30)

    with patch.object(client, "close", slow_close):
        task = asyncio.create_task(aclose_all())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not openai_provider._clients  # emptied before closing: nothing half-registered
    await aclose_all()
    await aclose_all()


async def test_concurrent_vertex_client_builds_share_one_client_and_close_the_loser():
    vertex._clients.clear()
    config = GatewayConfig(**_KEYS)
    credentials = _fake_google_credentials()
    with patch("google.auth.default", return_value=(credentials, "my-proj")):
        clients = await asyncio.gather(*(vertex._client(config) for _ in range(5)))
    assert len({id(c) for c in clients}) == 1
    assert len(vertex._clients) == 1
    await aclose_all()


# ─── 5. Bounded state ──────────────────────────────────────────────────────


async def test_client_cache_is_capped_when_credentials_rotate_and_evicted_clients_close():
    groq._clients.clear()
    with patch.object(base, "EVICTED_CLIENT_GRACE_SECONDS", 0.0):
        first = groq._client(GatewayConfig(groq_api_key="key-0", _env_file=None))
        with patch.object(first, "close", AsyncMock()) as closed:
            for i in range(1, base.MAX_CACHED_CLIENTS + 5):
                groq._client(GatewayConfig(groq_api_key=f"key-{i}", _env_file=None))
            await asyncio.sleep(0.05)
            assert len(groq._clients) == base.MAX_CACHED_CLIENTS
            closed.assert_awaited_once()
    await aclose_all()
