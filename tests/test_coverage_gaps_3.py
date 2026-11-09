"""Coverage pass 3: behaviours a caller hits on error, fallback, stream and
translation paths that no earlier test exercised. Fakes only; no provider is
ever called."""

import os
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from llm_gateway import (
    EmptyCompletionError,
    GatewayConfig,
    InvalidOutputError,
    InvalidRequestError,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.policy import ProviderPolicy
from llm_gateway.providers._anthropic_translate import (
    from_anthropic_response,
    to_anthropic_tool_choice,
    to_anthropic_tools,
    usage_from_anthropic_response,
)
from llm_gateway.providers.base import (
    StreamDelta,
    Usage,
    _strip_think_stream,
    parse_openai_style_response,
)
from llm_gateway.routing import ChainPlan, record_cost, record_plan

PROFILE = settings(
    max_examples=int(os.environ.get("PROPERTY_EXAMPLES", "60")),
    deadline=None,
    derandomize=bool(os.environ.get("CI")),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,
)

# ── Anthropic translation: malformed tool definitions ───────────────────


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "function"}],  # no "function" key
        [{"type": "function", "function": None}],
        [{"type": "function", "function": {"description": "no name"}}],
        [{"type": "function", "function": {"name": 5}}],
        ["not-a-dict"],
    ],
)
def test_malformed_tool_definition_is_an_invalid_request(tools):
    with pytest.raises(InvalidRequestError):
        to_anthropic_tools(tools)


@pytest.mark.parametrize(
    "choice",
    [
        {"type": "function"},
        {"type": "function", "function": {}},
        {"type": "function", "function": None},
        {"type": "function", "function": {"name": 3}},
    ],
)
def test_malformed_tool_choice_is_an_invalid_request(choice):
    with pytest.raises(InvalidRequestError):
        to_anthropic_tool_choice(choice)


def test_well_formed_tool_choice_and_tools_still_translate():
    assert to_anthropic_tool_choice({"type": "function", "function": {"name": "f"}}) == {
        "type": "tool",
        "name": "f",
    }
    tools = to_anthropic_tools([{"type": "function", "function": {"name": "f"}}])
    assert tools == [
        {"name": "f", "description": "", "input_schema": {"type": "object", "properties": {}}}
    ]
    assert to_anthropic_tool_choice({"type": "other"}) is None


_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=6),
    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=6), c, max_size=3),
    max_leaves=8,
)
_tool_like = st.dictionaries(
    st.sampled_from(["type", "function", "name", "description", "parameters"]), _json, max_size=4
)


@PROFILE
@given(st.lists(_tool_like | _json, max_size=4))
def test_tool_translation_only_ever_raises_invalid_request(tools):
    try:
        out = to_anthropic_tools(tools)
    except InvalidRequestError:
        return
    assert out is None or all(isinstance(t["name"], str) for t in out)


@PROFILE
@given(_tool_like | _json)
def test_tool_choice_translation_only_ever_raises_invalid_request(choice):
    try:
        out = to_anthropic_tool_choice(choice)
    except InvalidRequestError:
        return
    assert out is None or out["type"] in {"auto", "any", "tool"}


# ── Anthropic response mapping over odd shapes ──────────────────────────

_block = st.one_of(
    st.builds(lambda t: NS(type="text", text=t), st.text(max_size=10)),
    st.builds(lambda t: NS(type="thinking", thinking=t, signature="s"), st.text(max_size=10)),
    st.just(NS(type="redacted_thinking", data="x")),
    st.builds(
        lambda i, n: NS(type="tool_use", id=i, name=n, input={"a": 1}),
        st.text(min_size=1, max_size=6),
        st.text(min_size=1, max_size=6),
    ),
)
_usage = st.one_of(
    st.none(),
    st.builds(
        lambda i, o, r, w: NS(
            input_tokens=i,
            output_tokens=o,
            cache_read_input_tokens=r,
            cache_creation_input_tokens=w,
        ),
        st.none() | st.integers(0, 10**6),
        st.none() | st.integers(0, 10**6),
        st.none() | st.integers(0, 10**6),
        st.none() | st.integers(0, 10**6),
    ),
)
_stop = st.sampled_from(
    [None, "end_turn", "max_tokens", "tool_use", "stop_sequence", "refusal", "weird"]
)


@PROFILE
@given(st.lists(_block, max_size=5), _usage, _stop)
def test_anthropic_response_mapping_is_total_and_consistent(content, usage, stop):
    resp = NS(content=content, usage=usage, stop_reason=stop)
    calls = [b for b in content if b.type == "tool_use"]
    try:
        result = from_anthropic_response(resp, "m")
    except InvalidOutputError:
        assert calls and stop == "max_tokens"  # only a cut-off tool call is an error here
        return
    assert (result.finish_reason == "tool_calls") == bool(calls)
    assert len(result.tool_calls) == len(calls)
    assert result.input_tokens >= 0 and result.output_tokens >= 0
    assert result.content is None or result.content == result.content.strip()


def test_anthropic_usage_none_is_zero():
    assert usage_from_anthropic_response(None) == Usage()


# ── OpenAI-style response parsing ───────────────────────────────────────


def _openai_resp(message, finish="stop", usage=None):
    return NS(
        choices=[NS(message=message, finish_reason=finish)],
        usage=usage or NS(prompt_tokens=7, completion_tokens=3),
    )


def test_openai_refusal_without_tools_is_invalid_output_and_billed():
    msg = NS(content=None, refusal="I can't help with that", tool_calls=None)
    with pytest.raises(InvalidOutputError) as info:
        parse_openai_style_response(_openai_resp(msg), "m")
    assert info.value.usage is not None and info.value.usage.input_tokens == 7


def test_openai_refusal_alongside_tool_calls_is_not_an_error():
    call = NS(id="c1", function=NS(name="f", arguments="{}"))
    msg = NS(content=None, refusal="partial", tool_calls=[call])
    assert (
        parse_openai_style_response(_openai_resp(msg, "tool_calls"), "m").tool_calls[0].name == "f"
    )


def test_openai_tool_calls_finish_without_calls_is_stop():
    msg = NS(content="hello", refusal=None, tool_calls=None)
    assert parse_openai_style_response(_openai_resp(msg, "tool_calls"), "m").finish_reason == "stop"


def test_openai_tool_calls_present_forces_tool_calls_finish():
    call = NS(id="c1", function=NS(name="f", arguments='{"x": 1}'))
    msg = NS(content=None, refusal=None, tool_calls=[call])
    result = parse_openai_style_response(_openai_resp(msg, "stop"), "m")
    assert result.finish_reason == "tool_calls" and result.tool_calls[0].arguments == {"x": 1}


_call = st.builds(
    lambda i, n, a: NS(id=i, function=NS(name=n, arguments=a)),
    st.text(max_size=5),
    st.text(max_size=5),
    st.none()
    | st.text(max_size=12)
    | st.sampled_from(['{"a":1}', "{}", "[]", '{"a":1,"a":2}', "NaN"]),
)
_msg = st.builds(
    lambda c, r, t: NS(content=c, refusal=r, tool_calls=t),
    st.none() | st.text(max_size=20) | st.just("<think>x</think>"),
    st.none() | st.text(max_size=5),
    st.none() | st.lists(_call, max_size=3),
)


@PROFILE
@given(_msg, st.sampled_from([None, "stop", "length", "tool_calls", "content_filter", "x"]))
def test_openai_response_parsing_is_total(msg, finish):
    try:
        result = parse_openai_style_response(_openai_resp(msg, finish), "m")
    except (InvalidOutputError, EmptyCompletionError):
        return
    assert (result.finish_reason == "tool_calls") == bool(result.tool_calls)
    assert result.content or result.tool_calls


def test_usage_add_with_foreign_type_is_not_implemented():
    assert Usage().__add__(5) is NotImplemented
    with pytest.raises(TypeError):
        Usage() + 5  # type: ignore[operator]


# ── Think-tag stripping across stream boundaries ────────────────────────


async def _strip(pieces, final=None):
    async def gen():
        for p in pieces:
            yield StreamDelta(content=p)
        if final:
            yield final

    return [d async for d in _strip_think_stream(gen())]


async def test_stream_that_ends_inside_a_possible_think_tag_flushes_it():
    out = await _strip(["<thi"])
    assert "".join(d.content or "" for d in out) == "<thi"


async def test_partial_tag_is_flushed_before_the_finish_chunk():
    out = await _strip(["<th"], StreamDelta(finish_reason="stop"))
    assert [d.content for d in out] == ["<th", None]
    assert out[-1].finish_reason == "stop"


async def test_think_block_split_over_chunks_is_removed():
    out = await _strip(["<think>pl", "an</thi", "nk>  Hel", "lo"])
    assert "".join(d.content or "" for d in out) == "Hello"


async def test_unclosed_think_block_leaves_nothing():
    out = await _strip(["<think>never", " closed"])
    assert "".join(d.content or "" for d in out) == ""


# ── Policy ──────────────────────────────────────────────────────────────


def test_narrowed_budget_checks_either_side_alone_and_both():
    def deny(*a):
        return False

    def allow(*a):
        return True

    only_global = ProviderPolicy(budget_check=deny).narrowed_by(ProviderPolicy())
    assert only_global.budget_check is deny
    only_call = ProviderPolicy().narrowed_by(ProviderPolicy(budget_check=allow))
    assert only_call.budget_check is allow
    both = ProviderPolicy(budget_check=allow).narrowed_by(ProviderPolicy(budget_check=deny))
    assert both.budget_check is not allow and both.budget_check is not deny


async def test_narrowed_budget_checks_must_both_allow():
    from llm_gateway.policy import call_budget_check

    async def allow(*a):
        return True

    def deny(*a):
        return False

    merged = ProviderPolicy(budget_check=allow).narrowed_by(ProviderPolicy(budget_check=deny))
    assert merged.budget_check is not None
    assert await call_budget_check(merged.budget_check, "openai", "m", 0.1) is False
    ok = ProviderPolicy(budget_check=allow).narrowed_by(ProviderPolicy(budget_check=allow))
    assert ok.budget_check is not None
    assert await call_budget_check(ok.budget_check, "openai", "m", 0.1) is True


def test_narrowed_only_lists_intersect_or_inherit():
    assert ProviderPolicy(only=("openai", "groq")).narrowed_by(ProviderPolicy()).only == (
        "openai",
        "groq",
    )
    assert ProviderPolicy().narrowed_by(ProviderPolicy(only=("openai",))).only == ("openai",)
    assert ProviderPolicy(only=("openai", "groq")).narrowed_by(
        ProviderPolicy(only=("groq", "azure"))
    ).only == ("groq",)
    assert ProviderPolicy(only=("openai",)).narrowed_by(ProviderPolicy(only=("groq",))).only == ()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"sort": "random"}, ValueError),
        ({"budget_check": "nope"}, TypeError),
        ({"max_cost_usd": float("nan")}, ValueError),
        ({"max_cost_usd": -1}, ValueError),
        ({"max_cost_usd": True}, ValueError),
        ({"max_cost_usd": "1"}, ValueError),
    ],
)
def test_policy_rejects_bad_settings(kwargs, error):
    with pytest.raises(error):
        ProviderPolicy(**kwargs)


# ── Tracing helpers must never break a request ──────────────────────────


class _BrokenSpan:
    def set_attribute(self, *a):
        raise RuntimeError("exporter down")


def test_record_cost_swallows_span_errors_and_skips_none():
    record_cost(_BrokenSpan(), 0.5)  # type: ignore[arg-type]
    record_cost(_BrokenSpan(), None)  # type: ignore[arg-type]


def test_record_plan_swallows_span_errors_and_records_exclusions():
    seen: dict[str, Any] = {}

    class Span:
        def set_attribute(self, key, value):
            seen[key] = value

    plan = ChainPlan(policy=ProviderPolicy(ignore=("openai",)))
    plan.providers.append("anthropic")
    plan.exclusions["openai"] = [("policy", "ignored by policy")]
    record_plan(Span(), plan)  # type: ignore[arg-type]
    assert seen["llm_gateway.policy.excluded_providers"] == ["openai"]
    assert "ignored by policy" in seen["llm_gateway.policy.exclusion_reasons"][0]
    record_plan(_BrokenSpan(), plan)  # type: ignore[arg-type]
    record_plan(None, plan)


# ── Streaming: gateway errors are never restarted ───────────────────────

_MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def test_restartable_excludes_gateway_and_invalid_output_errors():
    from llm_gateway import LLMDeadlineExceeded
    from llm_gateway.streaming import _restartable

    assert not _restartable(LLMDeadlineExceeded("late"))
    assert not _restartable(InvalidOutputError("filtered"))
    assert _restartable(ConnectionResetError("reset"))


async def test_first_token_timeout_that_is_not_ours_propagates():
    """A TimeoutError raised by the provider itself (not the first-token
    scope) must not be rewritten into FirstTokenTimeoutError."""

    async def times_out(*a, **k):
        raise TimeoutError("provider-side")
        yield  # pragma: no cover

    cfg = GatewayConfig(anthropic_api_key="k", provider_order="anthropic", retry_attempts=1)
    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": times_out}):
        with pytest.raises(Exception) as info:
            async for _ in stream_chat(messages=_MSGS, config=cfg, first_token_timeout_seconds=30):
                pass
    assert "first_token_timeout_seconds" not in str(info.value)


# ── Structured output: lenient parsing and validation ───────────────────


def test_strict_schema_drops_keywords_the_provider_would_reject():
    from llm_gateway.structured import to_strict_schema

    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 3},
            "one": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "when": {"type": "string", "format": "made-up"},
            "mail": {"type": "string", "format": "email"},
            "n": {"type": "integer", "minimum": 0},
        },
    }
    props = to_strict_schema(schema, require_all_properties=True)["properties"]
    assert "minItems" not in props["tags"] and props["one"]["minItems"] == 1
    assert "format" not in props["when"] and props["mail"]["format"] == "email"
    assert "minimum" not in props["n"]


def test_strict_schema_leaves_non_object_nodes_alone():
    from llm_gateway.structured import to_strict_schema

    assert to_strict_schema({"type": "string"}, require_all_properties=True) == {"type": "string"}


def test_lenient_parse_skips_an_invalid_fence_and_finds_the_embedded_object():
    from llm_gateway.structured import _loads_lenient

    text = 'Here: ```json\nnot json\n``` and the answer {"ok": true} thanks'
    assert _loads_lenient(text) == {"ok": True}


def test_lenient_parse_raises_the_strict_error_when_nothing_is_usable():
    import json

    from llm_gateway.structured import _loads_lenient

    with pytest.raises(json.JSONDecodeError):
        _loads_lenient("no json here ```json\nstill not\n```")


def test_validator_reports_bad_json_and_deep_nesting_with_a_root_path():
    from llm_gateway.structured import OutputValidator

    validator = OutputValidator.from_spec({"type": "object"})
    with pytest.raises(InvalidOutputError) as bad:
        validator.check("definitely not json")
    assert bad.value.paths == ("<root>",)
    with pytest.raises(InvalidOutputError, match="nested too deeply"):
        validator.check("[" * 200000 + "]" * 200000)


def test_validator_rejects_non_object_missing_key_and_non_array():
    from llm_gateway.structured import OutputValidator

    obj = OutputValidator.from_spec({"type": "object", "required": ["a", "b"]})
    with pytest.raises(InvalidOutputError, match="failed validation"):
        obj.check("[1]")
    with pytest.raises(InvalidOutputError, match="failed validation"):
        obj.check('{"a": 1}')
    arr = OutputValidator.from_spec({"type": "array"})
    with pytest.raises(InvalidOutputError):
        arr.check('{"a": 1}')
    assert arr.check("[1, 2]") == [1, 2]


def test_validator_rejects_unsupported_spec():
    from llm_gateway.structured import OutputValidator

    with pytest.raises(TypeError):
        OutputValidator.from_spec("nope")  # type: ignore[arg-type]


@PROFILE
@given(st.text(max_size=60))
def test_validator_never_raises_anything_but_invalid_output(text):
    from llm_gateway.structured import OutputValidator

    try:
        OutputValidator.from_spec({"type": "object", "required": ["a"]}).check(text)
    except InvalidOutputError:
        pass


# ── Retry: backoff without Retry-After and deadline-bounded give-up ─────


async def test_retry_backs_off_between_attempts_when_no_retry_after():
    from llm_gateway.retry import call_with_retry

    calls = []
    sleeps: list[float] = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionResetError("reset")
        return "ok"

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    with patch("llm_gateway.retry.asyncio.sleep", fake_sleep):
        result = await call_with_retry(
            flaky, attempts=3, base_delay_seconds=1.0, should_retry=lambda e: True
        )
    assert result == "ok" and len(calls) == 3 and len(sleeps) == 2
    assert sleeps[1] > 0


async def test_retry_gives_up_early_when_the_backoff_would_outlast_the_deadline():
    from llm_gateway.retry import Deadline, call_with_retry

    calls = []

    async def always_fails():
        calls.append(1)
        raise ConnectionResetError("reset")

    with pytest.raises(ConnectionResetError):
        await call_with_retry(
            always_fails,
            attempts=5,
            base_delay_seconds=60.0,
            deadline=Deadline(0.5),
            should_retry=lambda e: True,
        )
    assert calls == [1]


# ── Routing: provider listed twice, unregistered provider ───────────────


def test_plan_chain_keeps_a_duplicated_provider_and_unknown_ones():
    from llm_gateway.routing import RequestProfile, plan_chain

    cfg = GatewayConfig(anthropic_api_key="k", groq_api_key="k")
    registry = {"anthropic": object(), "groq": object()}
    plan = plan_chain(
        ["anthropic", "groq", "anthropic", "mystery"],
        config=cfg,
        policy=None,
        profile=RequestProfile(),
        registry=registry,
        model_override=None,
    )
    assert plan.providers == ["anthropic", "groq", "anthropic", "mystery"]
