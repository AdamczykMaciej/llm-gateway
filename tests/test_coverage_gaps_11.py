"""Coverage pass 11 (0.42.7): privacy default entry points and stream tail,
Azure Entra failure paths, thinking/effort, id remapping, rate-limit window,
SSE chunk shapes and small guards. Fakes only; no provider is ever called."""

import asyncio
import re
import sys
from datetime import date
from types import SimpleNamespace as NS

import pytest

from llm_gateway import GatewayConfig
from llm_gateway.errors import ProviderAuthError
from llm_gateway.privacy import OutputPolicy, Privacy, PrivacyLayer, Span, Vault
from llm_gateway.providers import _anthropic_caching, _anthropic_translate, anthropic, azure
from llm_gateway.providers.base import ChatResult, StreamDelta
from llm_gateway.routing import _fallback_for

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


class _Rx:
    def detect(self, text):
        for m in EMAIL.finditer(text):
            yield Span(m.start(), m.end(), "email")


# -- privacy ---------------------------------------------------------------


def test_vault_tokens_lists_every_issued_token():
    vault = Vault()
    assert vault.tokens() == frozenset()
    token = vault.token_for("email", "a@b.co")
    assert vault.tokens() == frozenset({token})
    assert vault.original(token) == "a@b.co"


async def test_default_chat_and_stream_entry_points_are_used_without_injection(monkeypatch):
    seen = {}

    async def fake_chat(**kw):
        seen["chat"] = kw["messages"][0]["content"]
        return ChatResult(content="ok", model="m", input_tokens=1, output_tokens=1)

    async def fake_stream(**kw):
        seen["stream"] = kw["messages"][0]["content"]
        yield StreamDelta(content="hi")
        yield StreamDelta(finish_reason="stop")

    monkeypatch.setattr(sys.modules["llm_gateway.chat"], "chat", fake_chat)
    monkeypatch.setattr(sys.modules["llm_gateway.streaming"], "stream_chat", fake_stream)
    layer = PrivacyLayer()
    policy = Privacy(redactor=_Rx(), output=OutputPolicy.PASSTHROUGH)
    msgs = [{"role": "user", "content": "mail a@b.co"}]
    result = await layer.chat(messages=msgs, privacy=policy)
    assert result.content == "ok"  # PASSTHROUGH leaves the reply untouched
    assert "a@b.co" not in seen["chat"]
    out = [d async for d in layer.stream_chat(messages=msgs, privacy=policy)]
    assert "".join(d.content or "" for d in out) == "hi"
    assert "a@b.co" not in seen["stream"]


async def test_stream_ending_inside_a_token_flushes_the_held_tail():
    async def fake(**kw):
        yield StreamDelta(content="see [emai")  # no finish_reason ever arrives

    layer = PrivacyLayer(stream_chat=lambda **kw: fake())
    deltas = [
        d
        async for d in layer.stream_chat(
            messages=[{"role": "user", "content": "x"}],
            privacy=Privacy(redactor=_Rx(), output=OutputPolicy.RESTORE),
        )
    ]
    assert "".join(d.content or "" for d in deltas) == "see [emai"
    assert deltas[-1].finish_reason is None


async def test_stream_without_aclose_and_argumentless_tool_fragment():
    class Plain:  # an async iterator with no aclose()
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return StreamDelta(
                tool_call_deltas=[{"index": 0, "id": "c1", "function": {"name": "f"}}],
                finish_reason="tool_calls",
            )

    layer = PrivacyLayer(stream_chat=lambda **kw: Plain())
    deltas = [
        d
        async for d in layer.stream_chat(
            messages=[{"role": "user", "content": "x"}],
            privacy=Privacy(redactor=_Rx(), output=OutputPolicy.RESTORE),
        )
    ]
    assert deltas[0].tool_call_deltas[0]["function"] == {"name": "f"}


# -- azure -----------------------------------------------------------------


@pytest.fixture
def _clean_azure_credentials():
    azure._credentials.clear()
    yield
    azure._credentials.clear()


def _fake_identity(*, credential_error=None, token_error=None):
    def _credential(**kw):
        if credential_error:
            raise credential_error
        return object()

    def provider(cred, scope):
        async def get():
            raise token_error

        return get

    return NS(
        DefaultAzureCredential=_credential,
        ManagedIdentityCredential=_credential,
        get_bearer_token_provider=provider,
    )


def test_entra_credential_needing_aiohttp_is_a_clear_auth_error(
    monkeypatch, _clean_azure_credentials
):
    ident = _fake_identity(credential_error=ImportError("aiohttp"))
    monkeypatch.setattr(azure, "_load_identity", lambda: (ident, RuntimeError))
    with pytest.raises(ProviderAuthError, match="azure"):
        azure._token_provider(GatewayConfig())


async def test_entra_token_failure_and_late_import_error(monkeypatch, _clean_azure_credentials):
    class AuthFail(Exception):
        pass

    ident = _fake_identity(token_error=AuthFail("denied"))
    monkeypatch.setattr(azure, "_load_identity", lambda: (ident, AuthFail))
    with pytest.raises(ProviderAuthError, match="Entra token") as info:
        await azure._token_provider(GatewayConfig())()
    assert "denied" not in str(info.value)  # only the exception type is surfaced

    azure._credentials.clear()
    ident = _fake_identity(token_error=ImportError("aiohttp"))
    monkeypatch.setattr(azure, "_load_identity", lambda: (ident, AuthFail))
    with pytest.raises(ProviderAuthError, match="azure-identity"):
        await azure._token_provider(GatewayConfig())()


def test_azure_request_kwargs_carry_response_format():
    fmt = {"type": "json_object"}
    kwargs = azure._request_kwargs(GatewayConfig(), None, None, None, fmt)
    assert kwargs["response_format"] == fmt
    assert "tools" not in kwargs


# -- anthropic -------------------------------------------------------------


def test_adaptive_thinking_is_sent_as_extra_body():
    kwargs = {"extra_body": {"keep": 1}}
    anthropic._apply_thinking_effort(kwargs, "claude-sonnet-5-5", "adaptive", "")
    assert kwargs["extra_body"] == {"keep": 1, "thinking": {"type": "adaptive"}}


def test_only_model_rejections_trigger_fallback():
    assert anthropic.is_model_rejection(RuntimeError("model not found")) is False


# -- caching / translate / routing ----------------------------------------


def test_system_block_below_the_cacheable_minimum_stays_a_plain_string():
    msgs = [{"role": "system", "content": "short"}, {"role": "system", "content": "more"}]
    out = _anthropic_caching.system_param_from_messages(msgs, "claude-sonnet-5-5", True)
    assert out == "short\nmore"


def test_remapped_tool_id_never_collides_with_an_existing_valid_id():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "a_b"}, {"id": "a.b"}, {"id": "a:b"}]},
    ]
    mapping = _anthropic_translate._anthropic_tool_ids(messages)
    assert mapping == {"a.b": "a_b_2", "a:b": "a_b_3"}


def test_retired_fallback_model_is_not_used():
    cfg = GatewayConfig(
        anthropic_api_key="k",
        claude_model="claude-sonnet-5-5",
        claude_model_fallback="claude-3-opus-20240229",
        block_retired_models=True,
    )
    assert _fallback_for("anthropic", cfg, None) is None
    assert date.today() > date(2026, 1, 5)
    open_cfg = cfg.model_copy(update={"block_retired_models": False})
    assert _fallback_for("anthropic", open_cfg, None) == "claude-3-opus-20240229"


# -- small guards ------------------------------------------------------------


def test_cost_estimate_excluded_defaults_to_empty():
    from llm_gateway.estimate import CostEstimate

    assert CostEstimate(models=(), max_cost_usd=None).excluded == {}


def test_rate_limit_window_drops_entries_older_than_a_minute(monkeypatch):
    from fastapi import HTTPException

    from llm_gateway.service import rate_limit

    clock = [1000.0]
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: clock[0])
    rate_limit._windows.clear()
    rate_limit._check("k", 1)
    with pytest.raises(HTTPException) as info:
        rate_limit._check("k", 1)
    assert info.value.status_code == 429
    clock[0] += 61.0
    rate_limit._check("k", 1)  # the old hit aged out
    rate_limit._windows.clear()


def test_usage_only_delta_emits_no_choice_chunk():
    from llm_gateway.service.schemas import stream_chunk_sse

    lines = stream_chunk_sse(StreamDelta(usage=(1, 2), model="m"), chunk_id="c", created=1)
    assert len(lines) == 1 and '"choices": []' in lines[0]
    lines = stream_chunk_sse(
        StreamDelta(tool_call_deltas=[{"index": 0}], model="m"), chunk_id="c", created=1
    )
    assert '"tool_calls"' in lines[0]


async def test_idempotency_shared_call_is_cancelled_only_with_its_last_waiter():
    from llm_gateway.idempotency import single_flight

    started, cancelled = asyncio.Event(), asyncio.Event()

    async def run():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    w1 = asyncio.ensure_future(single_flight("k11", {"a": 1}, run))
    w2 = asyncio.ensure_future(single_flight("k11", {"a": 1}, run))
    await started.wait()
    w1.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not cancelled.is_set()  # w2 is still waiting
    w2.cancel()
    await asyncio.wait_for(cancelled.wait(), 2)


# -- config / capabilities / provider helpers -------------------------------


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("0x7f.1", True),  # numeric shorthand for 127.0.0.1
        ("2130706433", True),  # the same address as one integer
        ("08.8.8.8", False),  # numeric-looking but not a valid address: not local
        ("example.com", False),
    ],
)
def test_is_local_host_numeric_forms(host, local):
    from llm_gateway.config import _is_local_host

    assert _is_local_host(host) is local


def test_unknown_openai_model_gets_conservative_capabilities():
    from llm_gateway.capabilities import capabilities_for

    caps = capabilities_for("openai", "gpt-4o-realtime-preview")
    assert caps.streaming is True
    assert caps.structured_output is None


async def test_release_sdk_stream_closes_sync_async_and_missing_closers():
    from llm_gateway.providers.base import release_sdk_stream

    calls = []

    class Sync:
        def close(self):
            calls.append("sync")

    class Async:
        async def aclose(self):
            calls.append("async")

    for stream in (Sync(), Async(), object()):
        async with release_sdk_stream(stream):
            pass
    assert calls == ["sync", "async"]


def test_evicting_a_client_without_a_running_loop_does_not_raise():
    from llm_gateway.providers import base

    cache: dict = {}
    for i in range(base.MAX_CACHED_CLIENTS + 2):
        base.remember_client(cache, (i,), NS(close=lambda: None))
    assert len(cache) == base.MAX_CACHED_CLIENTS


async def test_aclose_all_continues_past_a_failing_provider(monkeypatch, caplog):
    from llm_gateway import providers

    closed = []

    async def boom():
        raise RuntimeError("close failed")

    async def ok():
        closed.append("ok")

    monkeypatch.setattr(providers.anthropic, "aclose", boom)
    monkeypatch.setattr(providers.vertex, "aclose", ok)
    with caplog.at_level("WARNING", logger="llm_gateway"):
        await providers.aclose_all()
    assert "ok" in closed  # later providers were still closed
    assert "closing" in caplog.text


def test_gemini_credentials_without_a_token_are_an_auth_error():
    from llm_gateway.providers import gemini

    class Creds:
        valid = False
        token = None

        def refresh(self, request):
            pass

    with pytest.raises(ProviderAuthError, match="no access token"):
        gemini._fresh_token(Creds())


def test_think_stripper_drops_a_block_split_across_pieces():
    from llm_gateway.providers.base import _ThinkStripper

    s = _ThinkStripper()
    assert [s.feed(x) for x in ("<th", "ink>plan</th", "ink>  hi", " there")] == [
        "",
        "",
        "hi",
        " there",
    ]


# -- anthropic streaming fallback ------------------------------------------


def _not_found():
    import httpx
    from anthropic import NotFoundError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return NotFoundError(
        "model: claude-haiku-5-5",
        response=httpx.Response(404, request=request),
        body={"error": {"message": "model"}},
    )


def _fake_messages_stream(models, behaviours):
    def fake(*args, model, **kw):
        models.append(model)
        behaviour = behaviours[len(models) - 1]

        async def gen():
            if behaviour == "fail":
                raise _not_found()
            if behaviour == "fail_late":
                yield StreamDelta(content="partial")
                raise _not_found()
            yield StreamDelta(content="from fallback")

        return gen()

    return fake


async def test_stream_retries_once_on_the_fallback_model_before_any_chunk(monkeypatch):
    models: list[str] = []
    monkeypatch.setattr(anthropic, "messages_stream", _fake_messages_stream(models, ["fail", "ok"]))
    monkeypatch.setattr(anthropic, "_client_factory", lambda config: None)
    cfg = GatewayConfig(
        anthropic_api_key="k",
        claude_model="claude-haiku-5-5",
        claude_model_fallback="claude-haiku-4-5-20251001",
    )
    out = [
        d async for d in anthropic._stream_chat(cfg, [{"role": "user", "content": "x"}], None, 10)
    ]
    assert [d.content for d in out] == ["from fallback"]
    assert models == ["claude-haiku-5-5", "claude-haiku-4-5-20251001"]


async def test_stream_does_not_switch_models_after_a_chunk_was_delivered(monkeypatch):
    from anthropic import NotFoundError

    models: list[str] = []
    monkeypatch.setattr(
        anthropic, "messages_stream", _fake_messages_stream(models, ["fail_late", "ok"])
    )
    monkeypatch.setattr(anthropic, "_client_factory", lambda config: None)
    cfg = GatewayConfig(
        anthropic_api_key="k",
        claude_model="claude-haiku-5-5",
        claude_model_fallback="claude-haiku-4-5-20251001",
    )
    seen = []
    with pytest.raises(NotFoundError):
        async for d in anthropic._stream_chat(cfg, [{"role": "user", "content": "x"}], None, 10):
            seen.append(d.content)
    assert seen == ["partial"] and len(models) == 1


# -- service: shutdown grace and guarded streaming response ------------------


async def test_shutdown_grace_expiry_is_logged_and_does_not_hang(caplog):
    from unittest.mock import patch

    import httpx

    from llm_gateway.service import app as app_module
    from llm_gateway.service import create_app

    gate = asyncio.Event()
    started = asyncio.Event()

    async def stuck_chat(**_kw):
        started.set()
        await gate.wait()

    closed = []

    async def fake_aclose():
        closed.append(1)

    app = create_app(GatewayConfig(anthropic_api_key="k", shutdown_grace_seconds=0.05))
    with (
        patch.object(app_module, "chat_engine", stuck_chat),
        patch.object(app_module, "aclose_all", fake_aclose),
        caplog.at_level("WARNING", logger="llm_gateway.service.app"),
    ):
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(
                client.post(
                    "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
                )
            )
            await asyncio.wait_for(started.wait(), 5)
        assert "grace expired" in caplog.text
        assert closed == [1]  # providers are still closed
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.aclose()


@pytest.mark.parametrize("write_timeout", [0.0, 5.0])
async def test_guarded_response_closes_body_without_aclose_and_releases_slot_once(write_timeout):
    from llm_gateway.service.app import _GuardedStreamingResponse

    class Body:  # an async iterator with no aclose()
        def __init__(self):
            self.items = [b"a", b"b"]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    released = []
    response = _GuardedStreamingResponse(
        Body(), write_timeout=write_timeout, on_close=lambda: released.append(1)
    )
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        await asyncio.sleep(3600)

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}}
    await response(scope, receive, send)
    bodies = [m["body"] for m in sent if m["type"] == "http.response.body"]
    assert b"".join(bodies) == b"ab"
    assert released == [1]


# -- remaining guards --------------------------------------------------------


def test_metadata_host_numeric_but_malformed_is_not_flagged():
    from llm_gateway.config import _is_metadata_host

    assert _is_metadata_host("0xa9.0xfe.0xa9.0xfe") is True  # hex form of 169.254.169.254
    assert _is_metadata_host("09.9.9.9") is False


def test_system_messages_stay_plain_when_caching_is_off_or_absent():
    msgs = [{"role": "system", "content": "a"}, {"role": "user", "content": "u"}]
    assert _anthropic_caching.system_param_from_messages(msgs, "claude-sonnet-5-5", False) == "a"
    assert _anthropic_caching.system_param_from_messages(msgs[1:], "claude-sonnet-5-5", True) == ""


async def test_response_guard_leaves_bodies_alone_when_the_body_cap_is_off():
    import httpx

    from llm_gateway.providers._limits import response_guard

    guard = response_guard(GatewayConfig(max_response_body_bytes=0))
    response = httpx.Response(200, content=b"x" * 100, headers={"content-type": "application/json"})
    original = response.aiter_bytes
    await guard(response)
    assert response.aiter_bytes == original  # not wrapped


def test_openai_compat_response_format_without_tools():
    from llm_gateway.providers import mistral

    fmt = {"type": "json_object"}
    kwargs = mistral._provider._chat_kwargs(GatewayConfig(), None, None, None, fmt)
    assert kwargs["response_format"] == fmt
    assert "tools" not in kwargs and "tool_choice" not in kwargs


def test_vault_reserve_ignores_empty_text():
    vault = Vault()
    vault.reserve("")
    assert vault.token_for("email", "a@b.co") == "[email_1]"


def test_vertex_without_google_auth_is_a_clear_auth_error(monkeypatch):
    from llm_gateway.providers import vertex

    monkeypatch.setitem(sys.modules, "google.auth", None)
    with pytest.raises(ProviderAuthError, match="vertex"):
        vertex.load_credentials(GatewayConfig())


def test_attempt_span_failures_never_reach_the_caller():
    from llm_gateway import tracing

    class Broken:
        def start_span(self, *a, **k):
            raise RuntimeError("exporter down")

    tracing.emit_attempt_span(
        Broken(),
        provider="p",
        model="m",
        request_id="r",
        attempt=1,
        started_ns=0,
        error=None,
        input_tokens=0,
        output_tokens=0,
    )


def test_provider_failure_without_a_failure_list_is_accepted():
    from llm_gateway.retry import Deadline
    from llm_gateway.router import record_provider_failure

    record_provider_failure(
        "groq", RuntimeError("x"), config=GatewayConfig(), deadline=Deadline(None), failures=None
    )


def test_scrubbing_a_request_without_cached_content_still_redacts_its_url():
    import httpx

    from llm_gateway.errors import _scrub_http_object

    class Req:
        method = "POST"
        stream = object()  # no _content attribute: nothing cached to clear

        def __init__(self):
            self.url = httpx.URL("https://h.example/v1?key=sk-secret-canary-123456")
            self.headers = {}

    req = Req()
    _scrub_http_object(req, GatewayConfig(anthropic_api_key="sk-secret-canary-123456"), set())
    assert "sk-secret-canary-123456" not in str(req.url)


async def test_response_guard_leaves_sse_alone_when_both_stream_caps_are_off():
    import httpx

    from llm_gateway.providers._limits import response_guard

    guard = response_guard(GatewayConfig(max_sse_event_bytes=0, max_stream_bytes=0))
    response = httpx.Response(
        200, content=b"data: x\n\n", headers={"content-type": "text/event-stream"}
    )
    original = response.aiter_bytes
    await guard(response)
    assert response.aiter_bytes == original
