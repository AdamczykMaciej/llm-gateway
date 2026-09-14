"""0.4.0 features through the *real* provider SDKs over in-memory transports:
usage reporting (including after failover and when streaming), structured
output per provider, and Anthropic prompt caching.

Helpers come from test_sdk_compat.py, which explains the approach.
"""

import json
from unittest.mock import patch

import httpx2
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, Field

from llm_gateway import (
    Completion,
    GatewayConfig,
    InvalidOutputError,
    LLMError,
    Usage,
    breaker,
    complete,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.errors import POLICIES, ErrorKind, classify
from llm_gateway.providers import anthropic
from llm_gateway.providers._anthropic_caching import min_cacheable_tokens
from llm_gateway.structured import OutputSchema, to_strict_schema
from tests.test_sdk_compat import (
    _ANTHROPIC_MESSAGE,
    _anthropic_client,
    _chain_config,
    _openai_ok,
    _real_sdk_providers,
    _Recorder,
    _sse,
    _status,
)


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _anthropic_reply(content=None, *, usage=None, stop_reason="end_turn", text=None):
    body = {
        **_ANTHROPIC_MESSAGE,
        "content": content if content is not None else [{"type": "text", "text": text or "ok"}],
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 12, "output_tokens": 4},
    }
    return lambda req: httpx2.Response(200, request=req, json=body)


def _openai_reply(content="ok", *, usage=None, refusal=None, finish_reason="stop"):
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
        "usage": usage or {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    return lambda req: httpx2.Response(200, request=req, json=body)


def _single(provider: str, **overrides) -> GatewayConfig:
    settings = {
        f"{provider}_api_key": "k",
        "provider_order": provider,
        "retry_attempts": 1,
    }
    settings.update(overrides)
    return GatewayConfig(**settings)


# ─── Usage reporting ───────────────────────────────────────────────────────


async def test_complete_with_usage_reports_anthropic_usage_including_cache():
    usage = {
        "input_tokens": 50,
        "output_tokens": 7,
        "cache_read_input_tokens": 4000,
        "cache_creation_input_tokens": 100,
    }
    recorder = _Recorder(_anthropic_reply(usage=usage, text=" Hi. "))
    config = _single("anthropic")
    with _real_sdk_providers(anthropic=recorder):
        result = await complete_with_usage(system="s", prompt="p", config=config)

    assert result == Completion(
        text="Hi.",
        provider="anthropic",
        model=config.claude_model,
        usage=Usage(
            input_tokens=4150,
            output_tokens=7,
            cache_read_input_tokens=4000,
            cache_creation_input_tokens=100,
        ),
        stop_reason="end_turn",
    )
    assert result.usage.uncached_input_tokens == 50


async def test_complete_with_usage_reports_openai_cached_tokens():
    usage = {
        "prompt_tokens": 1200,
        "completion_tokens": 30,
        "total_tokens": 1230,
        "prompt_tokens_details": {"cached_tokens": 1024},
    }
    recorder = _Recorder(_openai_reply("Hello", usage=usage))
    config = _single("openai")
    with _real_sdk_providers(openai=recorder):
        result = await complete_with_usage(system="s", prompt="p", config=config)

    assert (result.provider, result.model, result.text) == ("openai", config.openai_model, "Hello")
    assert result.usage == Usage(1200, 30, cache_read_input_tokens=1024)
    assert result.stop_reason == "stop"


async def test_usage_after_failover_is_the_serving_providers():
    primary = _Recorder(_status(500))
    fallback = _Recorder(_openai_ok)
    config = _chain_config(retry_attempts=1)
    with _real_sdk_providers(anthropic=primary, groq=fallback):
        result = await complete_with_usage(system="s", prompt="p", config=config)

    assert result.provider == "groq"
    assert result.model == config.groq_model
    assert result.usage == Usage(input_tokens=1, output_tokens=1)


async def test_complete_is_unchanged_for_existing_callers():
    recorder = _Recorder(_anthropic_reply(text="plain"))
    with _real_sdk_providers(anthropic=recorder):
        text = await complete(system="sys", prompt="p", config=_single("anthropic"))

    assert text == "plain"
    body = recorder.bodies[-1]
    assert set(body) == {"model", "max_tokens", "system", "messages"}
    assert body["system"] == "sys"


async def test_span_records_token_counts_but_not_prompt_text():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    usage = {
        "input_tokens": 5,
        "output_tokens": 3,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 0,
    }
    recorder = _Recorder(_anthropic_reply(usage=usage))
    with (
        _real_sdk_providers(anthropic=recorder),
        patch("llm_gateway.router.get_tracer", return_value=provider.get_tracer("t")),
    ):
        await complete_with_usage(
            system="secret system", prompt="secret prompt", config=_single("anthropic")
        )

    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.usage.input_tokens"] == 2005
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 2000
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 0
    assert not any("secret" in str(value) for value in attrs.values())


async def test_chat_result_carries_the_cache_breakdown():
    usage = {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 1500}
    recorder = _Recorder(_anthropic_reply(usage=usage))
    with patch.object(anthropic, "_client", return_value=_anthropic_client(recorder)):
        result = await anthropic.chat(
            GatewayConfig(anthropic_api_key="k"),
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=50,
        )
    assert result.usage == Usage(1510, 2, cache_read_input_tokens=1500)


# ─── Streaming usage ───────────────────────────────────────────────────────


def _anthropic_stream(events: list[dict]):
    return lambda req: httpx2.Response(
        200,
        request=req,
        headers={"content-type": "text/event-stream"},
        content=_sse(events, named=True),
    )


async def test_anthropic_stream_reports_cache_usage_and_skips_thinking():
    events = [
        {
            "type": "message_start",
            "message": {
                **_ANTHROPIC_MESSAGE,
                "content": [],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 3000,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Considering."},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]
    recorder = _Recorder(_anthropic_stream(events))
    with _real_sdk_providers(anthropic=recorder):
        deltas = [
            d
            async for d in stream_chat(
                messages=[{"role": "user", "content": "hi"}], config=_single("anthropic")
            )
        ]

    assert "".join(d.content or "" for d in deltas) == "Hi"
    final = deltas[-1]
    assert final.usage == (3020, 9)
    assert final.usage_details == Usage(3020, 9, cache_read_input_tokens=3000)
    assert final.provider == "anthropic"


def _openai_stream_with_usage(usage: dict):
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt"}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"content": "streamed"}}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**base, "choices": [], "usage": usage},
    ]
    return lambda req: httpx2.Response(
        200,
        request=req,
        headers={"content-type": "text/event-stream"},
        content=_sse(chunks, named=False),
    )


async def test_stream_usage_after_failover_is_attributed_to_the_serving_provider():
    usage = {
        "prompt_tokens": 1100,
        "completion_tokens": 12,
        "total_tokens": 1112,
        "prompt_tokens_details": {"cached_tokens": 1024},
    }
    primary = _Recorder(_status(401))
    fallback = _Recorder(_openai_stream_with_usage(usage))
    with _real_sdk_providers(anthropic=primary, groq=fallback):
        deltas = [
            d
            async for d in stream_chat(
                messages=[{"role": "user", "content": "hi"}], config=_chain_config()
            )
        ]

    assert fallback.bodies[-1]["stream_options"] == {"include_usage": True}
    final = deltas[-1]
    assert final.provider == "groq"
    assert final.usage == (1100, 12)
    assert final.usage_details == Usage(1100, 12, cache_read_input_tokens=1024)


# ─── Structured output ─────────────────────────────────────────────────────


class Verdict(BaseModel):
    score: int = Field(ge=0, le=100)
    summary: str = Field(min_length=1)
    tags: list[str] = []


_VERDICT_JSON = json.dumps({"score": 80, "summary": "Solid answer.", "tags": ["star"]})


async def test_anthropic_structured_output_uses_native_output_config():
    recorder = _Recorder(_anthropic_reply(text=_VERDICT_JSON))
    with _real_sdk_providers(anthropic=recorder):
        result = await complete_with_usage(
            system="s", prompt="p", config=_single("anthropic"), output_schema=Verdict
        )

    assert result.parsed == Verdict(score=80, summary="Solid answer.", tags=["star"])
    assert result.text == _VERDICT_JSON
    body = recorder.bodies[-1]
    assert "tools" not in body and "tool_choice" not in body
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert fmt["schema"]["required"] == ["score", "summary"]
    sent = json.dumps(fmt["schema"])
    for keyword in ("minimum", "maximum", "minLength", "default"):
        assert f'"{keyword}"' not in sent


async def test_openai_structured_output_uses_strict_json_schema():
    recorder = _Recorder(_openai_reply(_VERDICT_JSON))
    with _real_sdk_providers(openai=recorder):
        result = await complete_with_usage(
            system="s", prompt="p", config=_single("openai"), output_schema=Verdict
        )

    assert result.parsed.score == 80
    response_format = recorder.bodies[-1]["response_format"]
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert (json_schema["name"], json_schema["strict"]) == ("Verdict", True)
    assert json_schema["schema"]["required"] == ["score", "summary", "tags"]
    assert json_schema["schema"]["additionalProperties"] is False


async def test_groq_non_strict_model_uses_json_mode_with_schema_in_system_prompt():
    # Since 0.4.2 the default groq_model is openai/gpt-oss-120b (strict
    # json_schema); models without strict support still get JSON mode.
    fenced = f"```json\n{_VERDICT_JSON}\n```"
    recorder = _Recorder(_openai_reply(fenced))
    config = _single("groq", groq_model="llama-3.1-8b-instant")
    with _real_sdk_providers(groq=recorder):
        result = await complete_with_usage(
            system="Be strict.", prompt="p", config=config, output_schema=Verdict
        )

    assert result.parsed.summary == "Solid answer."
    body = recorder.bodies[-1]
    assert body["response_format"] == {"type": "json_object"}
    system = body["messages"][0]["content"]
    assert system.startswith("Be strict.\n\n") and "JSON Schema" in system


async def test_groq_gpt_oss_uses_strict_json_schema():
    recorder = _Recorder(_openai_reply(_VERDICT_JSON))
    config = _single("groq", groq_model="openai/gpt-oss-120b")
    with _real_sdk_providers(groq=recorder):
        await complete_with_usage(system="s", prompt="p", config=config, output_schema=Verdict)

    body = recorder.bodies[-1]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][0]["content"] == "s"


async def test_dict_schema_returns_parsed_json():
    schema = {
        "title": "Pick",
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": ["a", "b"]}},
        "required": ["choice"],
    }
    recorder = _Recorder(_anthropic_reply(text='{"choice": "b"}'))
    with _real_sdk_providers(anthropic=recorder):
        result = await complete_with_usage(
            system="s", prompt="p", config=_single("anthropic"), output_schema=schema
        )
    assert result.parsed == {"choice": "b"}


async def test_invalid_json_fails_over_without_retry_or_breaker_trip():
    primary = _Recorder(_anthropic_reply(text='{"score": 80, "summary": '))
    fallback = _Recorder(_openai_reply(_VERDICT_JSON))
    config = _chain_config(retry_attempts=2, breaker_failure_threshold=1)
    with _real_sdk_providers(anthropic=primary, groq=fallback):
        result = await complete_with_usage(
            system="s", prompt="p", config=config, output_schema=Verdict
        )

    assert result.provider == "groq"
    assert result.parsed.score == 80
    assert len(primary.bodies) == 1
    assert not breaker.is_open("anthropic")


async def test_schema_validation_failure_everywhere_raises_without_leaking_output():
    leaky = json.dumps({"score": 150, "summary": "Call Jane on 555-0100"})
    primary = _Recorder(_anthropic_reply(text=leaky))
    fallback = _Recorder(_openai_reply(leaky))
    with (
        _real_sdk_providers(anthropic=primary, groq=fallback),
        pytest.raises(LLMError) as excinfo,
    ):
        await complete_with_usage(
            system="s", prompt="p", config=_chain_config(), output_schema=Verdict
        )

    assert isinstance(excinfo.value.__cause__, InvalidOutputError)
    assert "Jane" not in str(excinfo.value)
    assert "score" in str(excinfo.value)
    assert not breaker.is_open("anthropic") and not breaker.is_open("groq")


async def test_missing_required_key_in_dict_schema_fails_over():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    primary = _Recorder(_anthropic_reply(text='{"y": 1}'))
    fallback = _Recorder(_openai_reply('{"x": 2}'))
    with _real_sdk_providers(anthropic=primary, groq=fallback):
        result = await complete_with_usage(
            system="s", prompt="p", config=_chain_config(), output_schema=schema
        )
    assert (result.provider, result.parsed) == ("groq", {"x": 2})


async def test_groq_json_validate_failed_is_classified_as_invalid_output():
    body = {
        "error": {
            "message": "Failed to generate JSON.",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
        }
    }
    primary = _Recorder(_anthropic_reply(text="not json"))
    fallback = _Recorder(lambda req: httpx2.Response(400, request=req, json=body))
    config = _chain_config(breaker_failure_threshold=1)
    with (
        _real_sdk_providers(anthropic=primary, groq=fallback),
        pytest.raises(LLMError) as excinfo,
    ):
        await complete_with_usage(system="s", prompt="p", config=config, output_schema=Verdict)

    assert classify(excinfo.value.__cause__) is ErrorKind.INVALID_OUTPUT
    assert POLICIES[ErrorKind.INVALID_OUTPUT].trips_breaker is False
    assert not breaker.is_open("groq")


async def test_openai_refusal_fails_over():
    primary = _Recorder(_openai_reply(None, refusal="I can't help with that."))
    fallback = _Recorder(_openai_reply(_VERDICT_JSON))
    config = GatewayConfig(
        openai_api_key="k", groq_api_key="k", provider_order="openai,groq", retry_attempts=1
    )
    with _real_sdk_providers(openai=primary, groq=fallback):
        result = await complete_with_usage(
            system="s", prompt="p", config=config, output_schema=Verdict
        )
    assert result.provider == "groq"


async def test_anthropic_structured_refusal_fails_over():
    primary = _Recorder(_anthropic_reply(text="I can't help with that.", stop_reason="refusal"))
    fallback = _Recorder(_openai_reply(_VERDICT_JSON))
    with _real_sdk_providers(anthropic=primary, groq=fallback):
        result = await complete_with_usage(
            system="s", prompt="p", config=_chain_config(), output_schema=Verdict
        )
    assert result.provider == "groq"


def test_strict_schema_handles_nested_models_and_unions():
    class Item(BaseModel):
        name: str = Field(max_length=10)

    class Envelope(BaseModel):
        items: list[Item] = Field(min_length=2)
        mode: str | int

    schema = OutputSchema.from_spec(Envelope).json_schema
    schema["properties"]["mode"] = {"oneOf": schema["properties"]["mode"].pop("anyOf")}
    strict = to_strict_schema(schema, require_all_properties=True)

    assert strict["additionalProperties"] is False
    assert strict["$defs"]["Item"]["additionalProperties"] is False
    assert strict["$defs"]["Item"]["required"] == ["name"]
    assert "maxLength" not in strict["$defs"]["Item"]["properties"]["name"]
    assert "minItems" not in strict["properties"]["items"]
    assert "anyOf" in strict["properties"]["mode"] and "oneOf" not in strict["properties"]["mode"]
    # The caller's schema is not mutated.
    assert "additionalProperties" not in schema


def test_output_schema_rejects_other_types():
    with pytest.raises(TypeError):
        OutputSchema.from_spec("not a schema")  # type: ignore[arg-type]


# ─── Prompt caching (Anthropic) ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "system_chars", "cache_system", "expect_marker"),
    [
        # 3 chars per estimated token, so 4096 * 3 chars estimates to 4096.
        ("claude-haiku-4-5-20251001", 4096 * 3, True, True),
        ("claude-haiku-4-5-20251001", 4096 * 3 - 3, True, False),
        ("claude-sonnet-5", 1024 * 3, True, True),
        ("claude-sonnet-5", 1024 * 3 - 3, True, False),
        ("claude-haiku-4-5-20251001", 20_000, False, False),
    ],
)
async def test_cache_control_is_sent_only_at_or_above_the_model_minimum(
    model, system_chars, cache_system, expect_marker
):
    system = "x" * system_chars
    recorder = _Recorder(_anthropic_reply())
    with _real_sdk_providers(anthropic=recorder):
        await complete_with_usage(
            system=system,
            prompt="p",
            config=_single("anthropic"),
            force_provider="anthropic",
            model=model,
            cache_system=cache_system,
        )

    sent = recorder.bodies[-1]["system"]
    if expect_marker:
        assert sent == [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        assert sent == system


async def test_cache_system_is_a_no_op_for_openai():
    recorder = _Recorder(_openai_reply())
    with _real_sdk_providers(openai=recorder):
        await complete(system="x" * 20_000, prompt="p", config=_single("openai"), cache_system=True)
    assert set(recorder.bodies[-1]) == {"model", "max_tokens", "messages"}


@pytest.mark.parametrize(
    ("model", "minimum"),
    [
        ("claude-haiku-4-5-20251001", 4096),
        ("claude-haiku-4-5", 4096),
        ("claude-sonnet-4-5-20250929", 1024),
        ("claude-sonnet-4-20250514", 1024),
        ("claude-opus-4-1-20250805", 1024),
        ("claude-opus-4-5-20251101", 4096),
        ("claude-opus-4-7", 2048),
        ("claude-opus-5", 512),
        ("claude-fable-5-1", 512),
        ("some-future-model", 4096),
    ],
)
def test_min_cacheable_tokens_by_model(model, minimum):
    assert min_cacheable_tokens(model) == minimum
