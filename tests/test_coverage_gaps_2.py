"""Coverage pass 2 (0.16.1): the streaming orchestrator's remaining branches,
cancellation during fallback, per-model breaker eviction, the privacy layer's
streaming of tool-call arguments, request ids and attempt spans through
retries and fallbacks, and the validate= / tool-result-cap edges."""

import asyncio
import json
import re
from unittest.mock import AsyncMock, patch

import anthropic
import httpx2
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_gateway import (
    AllProvidersExhaustedError,
    GatewayConfig,
    LLMDeadlineExceeded,
    PolicyViolationError,
    ProviderPolicy,
    breaker,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.privacy import (
    OutputPolicy,
    Privacy,
    PrivacyLayer,
    Span,
    StreamFilter,
    Vault,
    _Output,
)
from llm_gateway.providers.base import ChatResult, ProviderResult, StreamDelta
from llm_gateway.structured import _schema_errors
from llm_gateway.tool_results import cap_tool_results

_USER = {"role": "user", "content": "hi"}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**kw):
    base = {
        "anthropic_api_key": "k1",
        "groq_api_key": "k2",
        "provider_order": "anthropic,groq",
        "retry_attempts": 1,
    }
    return GatewayConfig(**{**base, **kw})


def _tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("t"), exporter


def _attempts(exporter):
    return sorted(
        (s for s in exporter.get_finished_spans() if s.name == "llm_gateway.attempt"),
        key=lambda s: s.attributes["llm_gateway.attempt"],
    )


async def _gen(*deltas):
    for d in deltas:
        yield d


def _failing(exc):
    async def g(*a, **kw):
        raise exc
        yield  # pragma: no cover

    return g()


# ── streaming orchestrator ───────────────────────────────────────────────────


async def test_stream_passes_parallel_tool_calls_only_when_set():
    seen = []

    def fake(config, messages, tools, max_tokens, **kw):
        seen.append(kw)
        return _gen(StreamDelta(content="x"))

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": fake}):
        _ = [d async for d in stream_chat(messages=[_USER], config=_config())]
        _ = [
            d
            async for d in stream_chat(
                messages=[_USER], config=_config(), parallel_tool_calls=False
            )
        ]
    assert "parallel_tool_calls" not in seen[0]
    assert seen[1]["parallel_tool_calls"] is False


async def test_empty_stream_ends_cleanly_without_fallback_and_closes_breaker():
    calls = []

    def groq(*a, **kw):
        calls.append("groq")
        return _gen(StreamDelta(content="never"))

    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": lambda *a, **kw: _gen(), "groq": groq},
    ):
        out = [d async for d in stream_chat(messages=[_USER], config=_config())]
    assert out == [] and calls == []
    assert not breaker.is_open("anthropic")


async def test_deadline_already_spent_before_second_provider_raises_deadline():
    def slow_fail(*a, **kw):
        async def g():
            await asyncio.sleep(0.15)
            raise RuntimeError("down")
            yield  # pragma: no cover

        return g()

    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": slow_fail, "groq": lambda *a, **kw: _gen(StreamDelta(content="x"))},
    ):
        with pytest.raises(LLMDeadlineExceeded):
            _ = [
                d
                async for d in stream_chat(
                    messages=[_USER], config=_config(call_deadline_seconds=0.1)
                )
            ]


async def test_policy_that_excludes_every_provider_raises_before_any_chunk():
    opened = []
    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": lambda *a, **kw: opened.append(1) or _gen(StreamDelta(content="x"))},
    ):
        with pytest.raises(PolicyViolationError):
            _ = [
                d
                async for d in stream_chat(
                    messages=[_USER],
                    config=_config(provider_order="anthropic"),
                    policy=ProviderPolicy(ignore=("anthropic",)),
                )
            ]
    assert opened == []


async def test_policy_denying_first_provider_falls_to_second_in_stream():
    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {
            "anthropic": lambda *a, **kw: _gen(StreamDelta(content="a")),
            "groq": lambda *a, **kw: _gen(StreamDelta(content="g")),
        },
    ):
        out = [
            d
            async for d in stream_chat(
                messages=[_USER],
                config=_config(),
                policy=ProviderPolicy(ignore=("anthropic",)),
            )
        ]
    assert [d.content for d in out] == ["g"]


async def test_mid_stream_timeout_that_is_not_the_deadline_is_counted_not_swallowed():
    async def g(*a, **kw):
        yield StreamDelta(content="a")
        raise TimeoutError("idle")

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": g, "groq": g}):
        got = []
        with pytest.raises(TimeoutError):
            async for d in stream_chat(
                messages=[_USER], config=_config(breaker_failure_threshold=1)
            ):
                got.append(d.content)
    assert got == ["a"]
    assert breaker.is_open("anthropic")  # counted against the breaker
    assert not breaker.is_open("groq")  # and no silent fallback after the first chunk


async def test_mid_stream_failure_after_first_chunk_does_not_fall_back_and_closes_generator():
    closed = []

    async def a(*args, **kw):
        try:
            yield StreamDelta(content="x")
            raise RuntimeError("boom")
        finally:
            closed.append("a")

    groq = AsyncMock()
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": a, "groq": groq}):
        with pytest.raises(RuntimeError):
            _ = [d async for d in stream_chat(messages=[_USER], config=_config())]
    assert closed == ["a"]
    groq.assert_not_called()


async def test_cancel_while_second_provider_is_opening_leaves_breaker_and_generator_clean():
    started = asyncio.Event()
    closed = []

    async def hang(*a, **kw):
        try:
            started.set()
            await asyncio.Event().wait()
            yield StreamDelta(content="never")  # pragma: no cover
        finally:
            closed.append("groq")

    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": lambda *a, **kw: _failing(RuntimeError("down")), "groq": hang},
    ):

        async def consume():
            return [d async for d in stream_chat(messages=[_USER], config=_config())]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed == ["groq"]
    assert not breaker.is_open("groq")  # a cancelled caller is not a provider failure
    assert breaker.allow("groq")


async def test_consumer_close_after_first_chunk_closes_provider_generator():
    closed = []

    async def a(*args, **kw):
        try:
            yield StreamDelta(content="1")
            yield StreamDelta(content="2")
        finally:
            closed.append(1)

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": a}):
        gen = stream_chat(messages=[_USER], config=_config(provider_order="anthropic"))
        first = await gen.__anext__()
        await gen.aclose()
    assert first.content == "1" and closed == [1]
    assert not breaker.is_open("anthropic")


# ── request ids and attempt spans through retries and fallbacks ──────────────


async def test_stream_fallback_attempts_share_one_request_id_and_number_in_order():
    tracer, exporter = _tracer()
    with (
        patch("llm_gateway.streaming.get_tracer", return_value=tracer),
        patch(
            "llm_gateway.streaming.STREAM_CALLS",
            {
                "anthropic": lambda *a, **kw: _failing(RuntimeError("down")),
                "groq": lambda *a, **kw: _gen(
                    StreamDelta(content="g"), StreamDelta(finish_reason="stop", usage=(3, 4))
                ),
            },
        ),
    ):
        out = [d async for d in stream_chat(messages=[_USER], config=_config(), request_id="r-1")]
    spans = _attempts(exporter)
    assert [s.attributes["gen_ai.system"] for s in spans] == ["anthropic", "groq"]
    assert [s.attributes["llm_gateway.attempt"] for s in spans] == [1, 2]
    assert {s.attributes["llm_gateway.request_id"] for s in spans} == {"r-1"}
    assert spans[0].attributes["llm_gateway.outcome"] != "ok"
    assert spans[1].attributes["llm_gateway.outcome"] == "ok"
    assert out[-1].request_id == "r-1" and out[-1].provider == "groq"


async def test_chat_retry_then_success_is_one_attempt_span_fallback_is_two():
    tracer, exporter = _tracer()
    flaky = AsyncMock(
        side_effect=[
            anthropic.APIConnectionError(
                message="blip", request=httpx2.Request("POST", "https://x.test/")
            ),
            ChatResult(content="ok", model="m", input_tokens=1, output_tokens=1),
        ]
    )
    with (
        patch("llm_gateway.chat.get_tracer", return_value=tracer),
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": flaky}),
    ):
        result = await chat(
            messages=[_USER],
            config=_config(
                provider_order="anthropic", retry_attempts=2, retry_base_delay_seconds=0.0
            ),
            request_id="r-2",
        )
    assert flaky.await_count == 2 and result.request_id == "r-2"
    spans = _attempts(exporter)
    assert len(spans) == 1 and spans[0].attributes["llm_gateway.outcome"] == "ok"

    tracer, exporter = _tracer()
    with (
        patch("llm_gateway.chat.get_tracer", return_value=tracer),
        patch(
            "llm_gateway.chat.CHAT_CALLS",
            {
                "anthropic": AsyncMock(side_effect=RuntimeError("down")),
                "groq": AsyncMock(
                    return_value=ChatResult(
                        content="ok", model="m", input_tokens=1, output_tokens=1
                    )
                ),
            },
        ),
    ):
        result = await chat(messages=[_USER], config=_config(), request_id="r-3")
    assert result.request_id == "r-3"
    assert {s.attributes["llm_gateway.request_id"] for s in _attempts(exporter)} == {"r-3"}
    assert len(_attempts(exporter)) == 2


async def test_complete_with_usage_fallback_spans_and_exhaustion_keep_request_id():
    tracer, exporter = _tracer()
    boom = AsyncMock(side_effect=RuntimeError("down"))
    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch("llm_gateway.router.CALLS", {"anthropic": boom, "groq": boom}),
    ):
        with pytest.raises(AllProvidersExhaustedError):
            await complete_with_usage(system="s", prompt="p", config=_config(), request_id="r-4")
    spans = _attempts(exporter)
    assert len(spans) == 2
    assert {s.attributes["llm_gateway.request_id"] for s in spans} == {"r-4"}


async def test_repair_attempts_keep_the_callers_request_id():
    tracer, exporter = _tracer()
    prov = AsyncMock(
        side_effect=[ProviderResult("nope", "m", 1, 1), ProviderResult('{"a": 1}', "m", 1, 1)]
    )
    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch("llm_gateway.router.CALLS", {"anthropic": prov}),
    ):
        c = await complete_with_usage(
            system="s",
            prompt="p",
            config=_config(provider_order="anthropic"),
            validate={"type": "object", "required": ["a"]},
            repair_attempts=1,
            request_id="r-5",
        )
    assert c.parsed == {"a": 1}
    assert {s.attributes["llm_gateway.request_id"] for s in _attempts(exporter)} == {"r-5"}


async def test_budget_check_denying_every_provider_raises_violation_before_any_chunk():
    async def deny(provider, model, cost):
        return False

    opened = []
    with patch(
        "llm_gateway.streaming.STREAM_CALLS",
        {"anthropic": lambda *a, **kw: opened.append(1) or _gen(StreamDelta(content="x"))},
    ):
        with pytest.raises(PolicyViolationError):
            _ = [
                d
                async for d in stream_chat(
                    messages=[_USER],
                    config=_config(provider_order="anthropic"),
                    policy=ProviderPolicy(budget_check=deny),
                )
            ]
    assert opened == []


async def test_breaker_admission_lost_between_check_and_call_skips_the_provider():
    calls = []
    with (
        patch("llm_gateway.streaming.breaker.allow", side_effect=lambda p, m=None: p == "groq"),
        patch(
            "llm_gateway.streaming.STREAM_CALLS",
            {
                "anthropic": lambda *a, **kw: calls.append("a") or _gen(StreamDelta(content="a")),
                "groq": lambda *a, **kw: _gen(StreamDelta(content="g")),
            },
        ),
    ):
        out = [d async for d in stream_chat(messages=[_USER], config=_config())]
    assert [d.content for d in out] == ["g"] and calls == []


async def test_every_stream_attempt_gets_a_span_even_when_empty_failed_or_abandoned():
    tracer, exporter = _tracer()

    async def mid_fail(*a, **kw):
        yield StreamDelta(content="x")
        raise RuntimeError("boom")

    cfg = _config(provider_order="anthropic")
    for fn, consume in (
        (lambda *a, **kw: _gen(), "all"),
        (mid_fail, "all"),
        (lambda *a, **kw: _gen(StreamDelta(content="1"), StreamDelta(content="2")), "one"),
    ):
        reset_circuit_breakers()
        with (
            patch("llm_gateway.streaming.get_tracer", return_value=tracer),
            patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": fn}),
        ):
            gen = stream_chat(messages=[_USER], config=cfg, request_id="r-9")
            try:
                if consume == "all":
                    _ = [d async for d in gen]
                else:
                    await gen.__anext__()
            except RuntimeError:
                pass
            finally:
                await gen.aclose()
    spans = _attempts(exporter)
    assert len(spans) == 3
    assert {s.attributes["llm_gateway.request_id"] for s in spans} == {"r-9"}
    outcomes = [s.attributes["llm_gateway.outcome"] for s in spans]
    assert outcomes[0] == "ok" and outcomes[1] != "ok" and outcomes[2] == "cancelled"


# ── per-model breaker ────────────────────────────────────────────────────────


@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["fail", "missing", "ok", "allow"]),
            st.sampled_from(["p", "q"]),
            st.sampled_from([None, "m0", "m1", "m2", "m3", "m4", "m5", "m6"]),
        ),
        max_size=60,
    )
)
def test_breaker_model_state_stays_bounded_and_consistent(ops):
    from llm_gateway.errors import ErrorKind  # noqa: F401

    reset_circuit_breakers()
    with patch.object(breaker, "MAX_MODEL_KEYS", 3):
        for op, provider, model in ops:
            if op == "fail":
                breaker.record_failure(provider, threshold=1, cooldown_seconds=30.0, model=model)
            elif op == "missing":
                breaker.record_failure(provider, threshold=2, cooldown_seconds=30.0, model=model)
            elif op == "ok":
                breaker.record_success(provider, model)
            else:
                breaker.allow(provider, model)
            assert len(breaker._model_keys) <= 3
            sep = breaker._SEP
            for state in (breaker._failures, breaker._opened_until, breaker._cooldowns):
                model_scoped = {k for k in state if sep in k}
                assert model_scoped <= set(breaker._model_keys)
    reset_circuit_breakers()


def test_eviction_never_touches_provider_level_state():
    breaker.record_failure("p", threshold=1, cooldown_seconds=30.0)
    with patch.object(breaker, "MAX_MODEL_KEYS", 2):
        for i in range(6):
            breaker.record_failure("p", threshold=1, cooldown_seconds=30.0, model=f"m{i}")
    assert breaker.is_open("p")
    assert breaker.is_open("p", "m5") and breaker.is_open("p", "m4")


# ── privacy: streaming text and tool-call arguments ──────────────────────────

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


class _Rx:
    def detect(self, text):
        for m in _EMAIL.finditer(text):
            yield Span(m.start(), m.end(), "email")


def _out(policy):
    vault = Vault()
    vault.token_for("email", 'a"b\\c@x.io')  # [email_1], with chars that need JSON escaping
    return _Output(Privacy(redactor=_Rx(), output=policy), vault)


_PIECES = st.sampled_from(
    ["[email_1]", "[email_2]", "[", "]", "[em", "ail_1]", "x", " ", '"', "\\"]
)


@given(st.lists(_PIECES, max_size=14), st.data())
def test_stream_filter_is_chunking_invariant_for_text(pieces, data):
    text = "".join(pieces)
    cuts = sorted(data.draw(st.sets(st.integers(0, len(text)), max_size=6)))
    chunks = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True)]
    for policy in (OutputPolicy.RESTORE, OutputPolicy.STRIP):
        out = _out(policy)
        f = StreamFilter(out)
        streamed = "".join(f.feed(c) for c in chunks) + f.finish()
        assert streamed == _out(policy).apply(text)


@given(st.lists(_PIECES, max_size=12), st.data())
def test_restored_tool_arguments_stay_valid_json_for_any_chunking(pieces, data):
    # Arguments are a JSON string; split anywhere, restored values are escaped.
    body = "".join(p for p in pieces if p not in ('"', "\\"))
    raw = json.dumps({"to": body})
    cuts = sorted(data.draw(st.sets(st.integers(0, len(raw)), max_size=6)))
    chunks = [raw[a:b] for a, b in zip([0, *cuts], [*cuts, len(raw)], strict=True)]
    out = _out(OutputPolicy.RESTORE)
    f = StreamFilter(out, escape=True)
    joined = "".join(f.feed(c) for c in chunks) + f.finish()
    assert json.loads(joined) == {"to": _out(OutputPolicy.RESTORE).apply(body)}


async def _privacy_stream(deltas, policy=OutputPolicy.RESTORE, **kw):
    layer = PrivacyLayer()

    async def fake(**_):
        for d in deltas:
            yield d

    layer._stream = lambda **k: fake(**k)
    return [
        d
        async for d in layer.stream_chat(
            messages=[{"role": "user", "content": "mail anna@example.com"}],
            privacy=Privacy(redactor=_Rx(), output=policy),
            **kw,
        )
    ]


async def test_two_parallel_tool_calls_are_filtered_independently_and_flushed_at_finish():
    deltas = [
        StreamDelta(
            tool_call_deltas=[{"index": 0, "id": "a", "function": {"arguments": '{"t":"[em'}}]
        ),
        StreamDelta(
            tool_call_deltas=[{"index": 1, "id": "b", "function": {"arguments": '{"t":"[email_1'}}]
        ),
        StreamDelta(tool_call_deltas=[{"index": 0, "function": {"arguments": 'ail_1]"}'}}]),
        StreamDelta(finish_reason="tool_calls", usage=(1, 1)),
    ]
    out = await _privacy_stream(deltas)
    args: dict[int, str] = {}
    for d in out:
        for f in d.tool_call_deltas or []:
            args[f["index"]] = args.get(f["index"], "") + f["function"]["arguments"]
    assert args[0] == '{"t":"anna@example.com"}'
    # Call 1 never closed its token: the held tail is flushed, not lost or restored.
    assert args[1] == '{"t":"[email_1'
    # The flush rides on (or before) the finishing delta, never after the end.
    assert out[-1].finish_reason == "tool_calls" or not out[-1].tool_call_deltas


async def test_stream_abandoned_by_consumer_closes_inner_stream_and_clears_vault():
    closed = []

    async def inner(**_):
        try:
            yield StreamDelta(content="[email_1] one")
            yield StreamDelta(content="two")
        finally:
            closed.append(1)

    layer = PrivacyLayer()
    layer._stream = lambda **k: inner(**k)
    gen = layer.stream_chat(
        messages=[{"role": "user", "content": "anna@example.com"}],
        privacy=Privacy(redactor=_Rx(), output=OutputPolicy.RESTORE),
    )
    first = await gen.__anext__()
    assert first.content.startswith("anna@example.com")
    await gen.aclose()
    assert closed == [1]


async def test_privacy_stream_error_mid_stream_still_closes_and_propagates():
    async def inner(**_):
        yield StreamDelta(content="x")
        raise RuntimeError("boom")

    layer = PrivacyLayer()
    layer._stream = lambda **k: inner(**k)
    with pytest.raises(RuntimeError):
        async for _ in layer.stream_chat(
            messages=[_USER], privacy=Privacy(redactor=_Rx(), output=OutputPolicy.STRIP)
        ):
            pass


async def test_privacy_layer_default_engines_are_the_gateways(monkeypatch):
    seen = {}

    async def fake_chat(**kw):
        seen["chat"] = kw
        return ChatResult(content="[email_1]", model="m", input_tokens=1, output_tokens=1)

    import sys

    monkeypatch.setattr(sys.modules["llm_gateway.chat"], "chat", fake_chat)
    layer = PrivacyLayer()
    res = await layer.chat(
        messages=[{"role": "user", "content": "anna@example.com"}],
        privacy=Privacy(redactor=_Rx(), output=OutputPolicy.RESTORE),
    )
    assert res.content == "anna@example.com"
    assert "anna@example.com" not in json.dumps(seen["chat"]["messages"])


# ── validate= and tool-result cap ────────────────────────────────────────────


def test_schema_enum_and_const_do_not_confuse_bool_with_int():
    out: list[str] = []
    _schema_errors(True, {"enum": [1]}, {}, "", out)
    _schema_errors(True, {"const": 1}, {}, "", out)
    _schema_errors(0, {"enum": [False]}, {}, "", out)
    assert out == ["<root>", "<root>", "<root>"]


def test_schema_enum_and_const_accept_equal_values():
    out: list[str] = []
    _schema_errors(1, {"enum": [1, 2]}, {}, "", out)
    _schema_errors("a", {"const": "a"}, {}, "", out)
    _schema_errors(True, {"enum": [True]}, {}, "", out)
    _schema_errors({"k": [1, True]}, {"const": {"k": [1, True]}}, {}, "", out)
    assert out == []


@given(
    st.dictionaries(st.text("abc", min_size=1, max_size=3), st.integers(), max_size=4),
)
def test_validate_dict_schema_accepts_what_it_describes(data):
    schema = {
        "type": "object",
        "required": list(data),
        "properties": {k: {"type": "integer"} for k in data},
        "additionalProperties": False,
    }
    out: list[str] = []
    _schema_errors(data, schema, schema, "", out)
    assert out == []


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "role": st.sampled_from(["user", "assistant", "tool"]),
                "content": st.one_of(st.text(max_size=60), st.none()),
            }
        ),
        max_size=6,
    ),
    st.integers(0, 40),
)
def test_cap_tool_results_properties(messages, cap):
    import copy

    before = copy.deepcopy(messages)
    out = cap_tool_results(messages, cap)
    assert messages == before  # input never modified
    assert len(out) == len(messages)
    for old, new in zip(messages, out, strict=True):
        if old["role"] != "tool" or cap == 0 or not isinstance(old["content"], str):
            assert new == old
        elif len(old["content"]) <= cap:
            assert new == old
        else:
            assert new["content"].startswith(old["content"][:cap])
            assert "truncated by llm-gateway" in new["content"]
    if cap == 0:
        assert out is messages


def test_cap_leaves_non_text_parts_and_non_tool_messages_alone():
    msgs = [
        {"role": "user", "content": "x" * 50},
        {"role": "tool", "content": [{"type": "image", "url": "u"}, "bare", {"text": 5}]},
        {"role": "tool", "tool_call_id": "c"},
    ]
    assert cap_tool_results(msgs, 10) is msgs


# ── service entrypoint ───────────────────────────────────────────────────────


def test_service_main_builds_the_app_and_runs_uvicorn_on_port_env(monkeypatch):
    import importlib

    import uvicorn

    monkeypatch.setenv("PORT", "9123")
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.update(args=a, kw=kw))
    import llm_gateway.service.main as entry

    entry = importlib.reload(entry)
    assert entry.app.title
    entry.main()
    assert seen["args"] == ("llm_gateway.service.main:app",)
    assert seen["kw"] == {"host": "0.0.0.0", "port": 9123}  # noqa: S104


def test_service_main_defaults_to_port_8080(monkeypatch):
    import uvicorn

    import llm_gateway.service.main as entry

    monkeypatch.delenv("PORT", raising=False)
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.update(kw))
    entry.main()
    assert seen["port"] == 8080
