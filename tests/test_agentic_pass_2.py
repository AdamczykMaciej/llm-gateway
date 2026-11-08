"""Agentic pass 2 (0.16.0): request ids and attempt spans, the parallel
tool-call switch, opt-in output validation with a repair knob, and the tool
result size guard. No provider is ever called: fakes and patched SDK clients."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from llm_gateway import (
    GatewayConfig,
    InvalidOutputError,
    InvalidRequestError,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers import (
    PARALLEL_TOOL_CALLS_SUPPORTED,
    anthropic,
    azure,
    gemini,
    groq,
    huggingface,
    mistral,
    openai,
    openai_compat,
    openrouter,
    vertex,
)
from llm_gateway.providers.base import ChatResult, ProviderResult, StreamDelta

_USER = {"role": "user", "content": "hi"}
_TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**kwargs):
    base = {
        "anthropic_api_key": "sk-ant-KEYLEAK-1",
        "groq_api_key": "gsk-KEYLEAK-2",
        "provider_order": "anthropic,groq",
        "retry_attempts": 1,
    }
    return GatewayConfig(**{**base, **kwargs})


def _ok(text="x", model="m"):
    return ChatResult(content=text, model=model, input_tokens=7, output_tokens=3)


def _tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("t"), exporter


# ── 1. request ids and attempt spans ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_result_carries_generated_and_supplied_request_id():
    fake = AsyncMock(return_value=_ok())
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": fake}):
        generated = await chat(messages=[_USER], config=_config())
        supplied = await chat(messages=[_USER], config=_config(), request_id="req-abc.123")
    assert len(generated.request_id) == 32
    assert supplied.request_id == "req-abc.123"
    # Never forwarded to the provider call.
    assert "request_id" not in fake.call_args.kwargs


@pytest.mark.parametrize("bad", ["", "x" * 129, "has space", "new\nline", 5])
@pytest.mark.asyncio
async def test_invalid_request_id_is_rejected_before_any_call(bad):
    fake = AsyncMock(return_value=_ok())
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": fake}):
        with pytest.raises(InvalidRequestError, match="request_id"):
            await chat(messages=[_USER], config=_config(), request_id=bad)
    fake.assert_not_called()


@pytest.mark.asyncio
async def test_attempt_spans_carry_id_attempt_outcome_and_no_content():
    tracer, exporter = _tracer()
    failing = AsyncMock(side_effect=TimeoutError("boom"))
    serving = AsyncMock(return_value=_ok(model="served"))
    secret = {"role": "user", "content": "PROMPT-SECRET anna@example.com"}
    with (
        patch("llm_gateway.chat.get_tracer", return_value=tracer),
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": failing, "groq": serving}),
    ):
        result = await chat(messages=[secret], config=_config(), request_id="rid-1")
    spans = exporter.get_finished_spans()
    attempts = sorted(
        (s for s in spans if s.name == "llm_gateway.attempt"),
        key=lambda s: s.attributes["llm_gateway.attempt"],
    )
    assert [s.attributes["llm_gateway.attempt"] for s in attempts] == [1, 2]
    assert [s.attributes["gen_ai.system"] for s in attempts] == ["anthropic", "groq"]
    assert [s.attributes["llm_gateway.outcome"] for s in attempts] == ["timeout", "ok"]
    assert attempts[1].attributes["gen_ai.request.model"] == "served"
    assert attempts[1].attributes["gen_ai.usage.input_tokens"] == 7
    assert attempts[1].attributes["gen_ai.usage.output_tokens"] == 3
    assert all(s.attributes["llm_gateway.latency_ms"] >= 0 for s in attempts)
    root = next(s for s in spans if s.name == "llm_gateway.chat")
    assert root.attributes["llm_gateway.request_id"] == "rid-1"
    assert all(s.attributes["llm_gateway.request_id"] == "rid-1" for s in attempts)
    assert result.request_id == "rid-1"
    blob = repr([dict(s.attributes) for s in spans]) + repr([s.events for s in spans])
    for needle in ("PROMPT-SECRET", "anna@example.com", "KEYLEAK", "boom"):
        assert needle not in blob


@pytest.mark.asyncio
async def test_complete_with_usage_exposes_request_id_and_spans():
    tracer, exporter = _tracer()
    fake = AsyncMock(return_value=ProviderResult("hi", "m", 4, 2))
    with (
        patch("llm_gateway.router.get_tracer", return_value=tracer),
        patch("llm_gateway.router.CALLS", {"anthropic": fake}),
    ):
        completion = await complete_with_usage(
            system="SYS-SECRET", prompt="PROMPT-SECRET", config=_config(), request_id="rid-2"
        )
    assert completion.request_id == "rid-2"
    spans = exporter.get_finished_spans()
    attempt = next(s for s in spans if s.name == "llm_gateway.attempt")
    assert attempt.attributes["llm_gateway.request_id"] == "rid-2"
    assert attempt.attributes["llm_gateway.outcome"] == "ok"
    assert "SECRET" not in repr([dict(s.attributes) for s in spans])


@pytest.mark.asyncio
async def test_stream_final_delta_and_spans_carry_request_id():
    tracer, exporter = _tracer()

    async def fake_stream(config, messages, tools, max_tokens, **kwargs):
        yield StreamDelta(content="a")
        yield StreamDelta(finish_reason="stop", usage=(5, 2))

    with (
        patch("llm_gateway.streaming.get_tracer", return_value=tracer),
        patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": fake_stream}),
    ):
        deltas = [
            d async for d in stream_chat(messages=[_USER], config=_config(), request_id="rid-3")
        ]
    assert next(d for d in deltas if d.usage).request_id == "rid-3"
    attempt = next(s for s in exporter.get_finished_spans() if s.name == "llm_gateway.attempt")
    assert attempt.attributes["llm_gateway.request_id"] == "rid-3"
    assert attempt.attributes["gen_ai.usage.output_tokens"] == 2


# ── 2. parallel tool calls ───────────────────────────────────────────────────


def test_parallel_tool_call_support_table():
    assert PARALLEL_TOOL_CALLS_SUPPORTED == {
        "anthropic": True,
        "vertex": True,
        "openai": True,
        "azure": True,
        "groq": True,
        "mistral": True,
        "openrouter": True,
        "gemini": False,
        "huggingface": False,
        "openai_compat": False,
    }


@pytest.mark.asyncio
async def test_engine_passes_switch_only_when_set():
    fake = AsyncMock(return_value=_ok())
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": fake}):
        await chat(messages=[_USER], tools=_TOOLS, config=_config())
        assert "parallel_tool_calls" not in fake.call_args.kwargs
        await chat(messages=[_USER], tools=_TOOLS, config=_config(), parallel_tool_calls=False)
        assert fake.call_args.kwargs["parallel_tool_calls"] is False


class _Sentinel(Exception):
    pass


def _openai_style_client():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_Sentinel)
    return client


def _cfg_all():
    return GatewayConfig(
        openai_api_key="k",
        groq_api_key="k",
        mistral_api_key="k",
        openrouter_api_key="k",
        azure_endpoint="https://r.openai.azure.com",
        azure_api_key="k",
        azure_model="dep",
        openai_compat_base_url="https://host.example/v1",
        openai_compat_api_key="k",
        openai_compat_model="m",
        huggingface_api_key="k",
        gemini_project_id="my-project",
        vertex_project_id="my-project",
    )


_OPENAI_STYLE = [
    ("openai", openai, True),
    ("azure", azure, True),
    ("groq", groq, True),
    ("mistral", mistral, True),
    ("openrouter", openrouter, True),
    ("gemini", gemini, False),
    ("huggingface", huggingface, False),
    ("openai_compat", openai_compat, False),
]


@pytest.mark.parametrize(("name", "module", "supported"), _OPENAI_STYLE)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_openai_style_adapters_send_parallel_tool_calls(name, module, supported, streaming):
    client = _openai_style_client()
    fn = module.stream_chat if streaming else module.chat
    for flag in (False, True, None):
        client.chat.completions.create.reset_mock()
        with patch.object(module, "_client", return_value=client):
            with pytest.raises(_Sentinel):
                if streaming:
                    await anext(fn(_cfg_all(), [_USER], _TOOLS, 10, parallel_tool_calls=flag))
                else:
                    await fn(_cfg_all(), [_USER], _TOOLS, 10, parallel_tool_calls=flag)
        sent = client.chat.completions.create.call_args.kwargs
        if supported and flag is not None:
            assert sent["parallel_tool_calls"] is flag
        else:
            assert "parallel_tool_calls" not in sent


@pytest.mark.parametrize("name", ["openai", "groq", "mistral"])
@pytest.mark.asyncio
async def test_parallel_flag_not_sent_without_tools(name):
    module = {"openai": openai, "groq": groq, "mistral": mistral}[name]
    client = _openai_style_client()
    with patch.object(module, "_client", return_value=client):
        with pytest.raises(_Sentinel):
            await module.chat(_cfg_all(), [_USER], None, 10, parallel_tool_calls=False)
    assert "parallel_tool_calls" not in client.chat.completions.create.call_args.kwargs


def _anthropic_client():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=_Sentinel)
    return client


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (None, {"type": "auto", "disable_parallel_tool_use": True}),
        ("auto", {"type": "auto", "disable_parallel_tool_use": True}),
        ("required", {"type": "any", "disable_parallel_tool_use": True}),
        (
            {"type": "function", "function": {"name": "f"}},
            {"type": "tool", "name": "f", "disable_parallel_tool_use": True},
        ),
    ],
)
@pytest.mark.asyncio
async def test_anthropic_and_vertex_disable_parallel_tool_use(tool_choice, expected):
    for module in (anthropic, vertex):
        client = _anthropic_client()
        with patch.object(module, "_client", return_value=client):
            with pytest.raises(_Sentinel):
                await module.chat(
                    GatewayConfig(anthropic_api_key="k", vertex_project_id="my-project"),
                    [_USER],
                    _TOOLS,
                    10,
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,
                )
        assert client.messages.create.call_args.kwargs["tool_choice"] == expected


@pytest.mark.asyncio
async def test_anthropic_leaves_tool_choice_alone_by_default_and_for_none():
    client = _anthropic_client()
    with patch.object(anthropic, "_client", return_value=client):
        for kwargs in ({}, {"parallel_tool_calls": True}):
            with pytest.raises(_Sentinel):
                await anthropic.chat(
                    GatewayConfig(anthropic_api_key="k"), [_USER], _TOOLS, 10, **kwargs
                )
            assert "tool_choice" not in client.messages.create.call_args.kwargs
        with pytest.raises(_Sentinel):
            await anthropic.chat(
                GatewayConfig(anthropic_api_key="k"),
                [_USER],
                _TOOLS,
                10,
                tool_choice="none",
                parallel_tool_calls=False,
            )
        assert "disable_parallel_tool_use" not in repr(client.messages.create.call_args.kwargs)


@pytest.mark.asyncio
async def test_anthropic_stream_disables_parallel_tool_use():
    client = MagicMock()
    client.messages.stream = MagicMock(side_effect=_Sentinel)
    with patch.object(anthropic, "_client", return_value=client):
        with pytest.raises(_Sentinel):
            await anext(
                anthropic.stream_chat(
                    GatewayConfig(anthropic_api_key="k"),
                    [_USER],
                    _TOOLS,
                    10,
                    parallel_tool_calls=False,
                )
            )
    assert client.messages.stream.call_args.kwargs["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


# ── 3. validate= and repair_attempts ─────────────────────────────────────────


class Person(BaseModel):
    name: str
    age: int


def _provider(*texts):
    return AsyncMock(side_effect=[ProviderResult(t, "m", 10, 5, stop_reason="stop") for t in texts])


async def _complete(provider, **kwargs):
    with patch("llm_gateway.router.CALLS", {"anthropic": provider}):
        return await complete_with_usage(
            system="s", prompt="p", config=_config(provider_order="anthropic"), **kwargs
        )


@pytest.mark.asyncio
async def test_validate_pydantic_returns_typed_result():
    result = await _complete(_provider('{"name": "Ann", "age": 3}'), validate=Person)
    assert result.parsed == Person(name="Ann", age=3)


@pytest.mark.asyncio
async def test_validate_raises_with_path_and_never_echoes_output():
    provider = _provider('{"name": "SECRET-NAME", "age": "old"}')
    with pytest.raises(InvalidOutputError) as info:
        await _complete(provider, validate=Person)
    assert info.value.paths == ("age",)
    assert "age" in str(info.value)
    assert "SECRET-NAME" not in str(info.value) and "old" not in str(info.value)
    assert provider.call_count == 1  # no paid retry by default


@pytest.mark.asyncio
async def test_validate_json_schema_reports_nested_path():
    schema = {
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["qty"],
                    "properties": {"qty": {"type": "integer"}},
                },
            }
        },
    }
    ok = await _complete(_provider('{"items": [{"qty": 2}]}'), validate=schema)
    assert ok.parsed == {"items": [{"qty": 2}]}
    with pytest.raises(InvalidOutputError) as info:
        await _complete(_provider('{"items": [{"qty": 2}, {"qty": "x"}]}'), validate=schema)
    assert info.value.paths == ("items.1.qty",)
    with pytest.raises(InvalidOutputError) as info:
        await _complete(_provider('{"items": [{}]}'), validate=schema)
    assert info.value.paths == ("items.0.qty",)


@pytest.mark.asyncio
async def test_validate_rejects_non_json():
    with pytest.raises(InvalidOutputError):
        await _complete(_provider("not json at all"), validate=Person)


@pytest.mark.asyncio
async def test_repair_attempts_zero_is_default_and_negative_is_rejected():
    with pytest.raises(InvalidRequestError, match="repair_attempts"):
        await _complete(_provider("{}"), validate=Person, repair_attempts=-1)


@pytest.mark.asyncio
async def test_repair_attempts_retries_with_error_path_and_sums_usage():
    provider = _provider('{"name": "Ann", "age": "x"}', '{"name": "Ann", "age": 4}')
    result = await _complete(provider, validate=Person, repair_attempts=1)
    assert provider.call_count == 2
    assert result.parsed == Person(name="Ann", age=4)
    repair_prompt = provider.call_args.args[2]
    assert "age" in repair_prompt and repair_prompt.startswith("p")
    assert result.usage.input_tokens == 20 and result.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_repair_exhausted_raises_last_error():
    provider = _provider('{"name": 1}', '{"name": 2}', '{"name": 3}')
    with pytest.raises(InvalidOutputError):
        await _complete(provider, validate=Person, repair_attempts=1)
    assert provider.call_count == 2


# ── 4. tool result size guard ────────────────────────────────────────────────


def _tool_history(content):
    return [
        _USER,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": content},
    ]


async def _sent_messages(messages, **kwargs):
    fake = AsyncMock(return_value=_ok())
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": fake}):
        await chat(
            messages=messages, tools=_TOOLS, config=_config(**kwargs.pop("cfg", {})), **kwargs
        )
    return fake.call_args.args[1]


@pytest.mark.asyncio
async def test_oversized_tool_result_is_truncated_with_marker_and_input_untouched():
    big = "A" * 5000
    history = _tool_history(big)
    sent = await _sent_messages(history, cfg={"tool_result_max_chars": 100})
    content = sent[2]["content"]
    assert content.startswith("A" * 100) and "A" * 101 not in content
    assert "truncated" in content and "4900" in content
    assert history[2]["content"] == big
    assert sent[0] == _USER


@pytest.mark.asyncio
async def test_small_and_disabled_results_pass_through_unchanged():
    history = _tool_history("short")
    assert (await _sent_messages(history, cfg={"tool_result_max_chars": 100}))[2][
        "content"
    ] == "short"
    big = _tool_history("B" * 5000)
    assert (await _sent_messages(big))[2]["content"] == "B" * 5000  # default: off


@pytest.mark.asyncio
async def test_per_call_cap_overrides_config_and_covers_text_parts():
    parts = [{"type": "text", "text": "C" * 300}, {"type": "text", "text": "ok"}]
    sent = await _sent_messages(
        _tool_history(parts), cfg={"tool_result_max_chars": 1000}, max_tool_result_chars=50
    )
    out = sent[2]["content"]
    assert out[0]["text"].startswith("C" * 50) and "C" * 51 not in out[0]["text"]
    assert out[1]["text"] == "ok"


@pytest.mark.asyncio
async def test_stream_chat_also_truncates():
    seen = {}

    async def fake_stream(config, messages, tools, max_tokens, **kwargs):
        seen["messages"] = messages
        yield StreamDelta(finish_reason="stop", usage=(1, 1))

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": fake_stream}):
        _ = [
            d
            async for d in stream_chat(
                messages=_tool_history("D" * 900),
                tools=_TOOLS,
                config=_config(tool_result_max_chars=40),
            )
        ]
    assert "D" * 41 not in seen["messages"][2]["content"]
    assert "truncated" in seen["messages"][2]["content"]


@pytest.mark.asyncio
async def test_invalid_cap_rejected():
    with pytest.raises(InvalidRequestError, match="max_tool_result_chars"):
        await _sent_messages(_tool_history("x"), max_tool_result_chars=-5)


def test_json_roundtrip_of_marker_is_plain_text():
    json.dumps("…[truncated]")


def _paths(schema, text):
    from llm_gateway.structured import OutputValidator

    try:
        OutputValidator.from_spec(schema).check(text)
    except InvalidOutputError as error:
        return error.paths
    return ()


def test_json_schema_subset_enum_const_ref_anyof_additional_properties():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "v": {"const": 1},
            "ref": {"$ref": "#/$defs/N"},
            "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "maybe": {"type": ["string", "null"]},
        },
        "$defs": {"N": {"type": "number"}},
    }
    good = '{"kind": "a", "v": 1, "ref": 2.5, "either": 3, "maybe": null}'
    assert _paths(schema, good) == ()
    bad = '{"kind": "z", "v": 2, "ref": "x", "either": [], "maybe": 1, "extra": 1}'
    assert _paths(schema, bad) == ("kind", "v", "ref", "either", "maybe", "extra")
    assert _paths({"type": "array", "items": {"type": "string"}}, '["a", 1]') == ("1",)
    assert _paths({"type": "object"}, "[1]") == ("<root>",)
    assert _paths({"type": "object"}, "no json") == ("<root>",)
    assert _paths({"$ref": "http://elsewhere"}, "1") == ()


def test_validate_spec_must_be_model_or_dict():
    from llm_gateway.structured import OutputValidator

    with pytest.raises(TypeError):
        OutputValidator.from_spec("nope")  # type: ignore[arg-type]
