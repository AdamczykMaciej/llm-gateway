"""Edge-path tests for the least-covered branches: translation of odd
message shapes, error scrubbing, think-stripping, the privacy stream wrapper
and the HTTP service. No network; fakes only."""

import asyncio
import re

import pytest

from llm_gateway import GatewayConfig
from llm_gateway.errors import (
    InvalidRequestError,
    is_model_scoped,
    scrub_exception_chain,
)
from llm_gateway.privacy import OutputPolicy, PrivacyLayer, _map_strings
from llm_gateway.providers import base
from llm_gateway.providers._anthropic_translate import (
    _translate_content,
    to_anthropic_messages,
)
from llm_gateway.providers.base import StreamDelta, _ThinkStripper, strip_think
from llm_gateway.service import create_app

# ── anthropic translation: caller-data shapes ────────────────────────────────


def test_content_list_with_only_blank_text_is_rejected():
    with pytest.raises(InvalidRequestError, match="no non-empty"):
        _translate_content([{"type": "text", "text": "   "}])


def test_unsupported_content_part_type_is_rejected():
    with pytest.raises(InvalidRequestError, match="not supported"):
        _translate_content([{"type": "audio", "data": "x"}])


def test_image_url_part_without_url_is_rejected():
    with pytest.raises(InvalidRequestError, match="no url"):
        _translate_content([{"type": "image_url", "image_url": {"url": ""}}])


@pytest.mark.parametrize(
    "messages, fragment",
    [
        ([{"role": "wizard", "content": "x"}], "no valid role"),
        (["just a string"], "no valid role"),
        ([{"role": "tool", "content": "x"}], "tool_call_id"),
        ([{"role": "system", "content": 5}], "system content"),
        (
            [{"role": "assistant", "content": None, "tool_calls": "nope"}],
            "malformed tool call",
        ),
        (
            [{"role": "assistant", "content": None, "tool_calls": [{"id": 1}]}],
            "malformed tool call",
        ),
    ],
)
def test_malformed_history_raises_typed_error_naming_the_index(messages, fragment):
    with pytest.raises(InvalidRequestError, match=re.escape(fragment)) as exc:
        to_anthropic_messages(messages)
    assert "messages[0]" in str(exc.value)


def test_assistant_with_text_list_and_tool_calls_keeps_both():
    _, msgs = to_anthropic_messages(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "calling"}],
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
        ]
    )
    kinds = [b["type"] for b in msgs[1]["content"]]
    assert kinds == ["text", "tool_use"]


def test_tool_message_after_user_text_starts_its_own_turn():
    _, msgs = to_anthropic_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ]
    )
    assert [m["role"] for m in msgs] == ["user", "user"]
    assert msgs[1]["content"][0]["type"] == "tool_result"


# ── errors: model-scoped classification and scrubbing ────────────────────────


def _sdk_error(**attrs):
    from llm_gateway import errors

    cls = errors._SDK_ERRORS[0]
    exc = cls.__new__(cls)
    Exception.__init__(exc, attrs.pop("message", "boom"))
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


def test_non_sdk_error_is_never_model_scoped():
    assert is_model_scoped(ValueError("model not found")) is None


def test_overloaded_by_status_and_by_body_type():
    assert is_model_scoped(_sdk_error(status_code=529, body=None)) == "overloaded"
    body = {"error": {"type": "overloaded_error", "message": "busy"}}
    assert is_model_scoped(_sdk_error(status_code=200, body=body)) == "overloaded"
    flat = {"type": "overloaded_error"}
    assert is_model_scoped(_sdk_error(status_code=500, body=flat)) == "overloaded"


def test_invalid_model_type_is_missing():
    err = _sdk_error(status_code=400, body={"error": {"type": "invalid_model"}})
    assert is_model_scoped(err) == "missing"


def test_ordinary_5xx_with_non_dict_body_stays_provider_wide():
    assert is_model_scoped(_sdk_error(status_code=500, body="<html>bad gateway</html>")) is None


def test_scrub_handles_nested_containers_depth_and_non_str_keys():
    secret = "sk-ant-api03-" + "A" * 40
    deep: object = secret
    for _ in range(12):
        deep = [deep]
    body = {"error": {"message": f"bad key {secret}"}, "api_key": "hunter2", 3: ("x", secret)}
    exc = _sdk_error(body=body, deep=deep, code=None)
    scrub_exception_chain(exc, GatewayConfig(anthropic_api_key=secret))
    assert secret not in str(exc.body)
    assert exc.body["api_key"] == "[REDACTED]"


def test_scrub_http_object_redacts_headers_url_and_buffered_body():
    secret = "sk-ant-api03-" + "B" * 40

    class Headers(dict):
        pass

    class Req:
        method = "POST"
        stream = object()
        _content = b"prompt"

        def __init__(self):
            self.headers = Headers({"authorization": "Bearer abc", "x-note": secret})
            self.url = f"https://h/x?key={secret}"

    class Resp:
        status_code = 401
        _content = f"bad {secret}".encode()

        def __init__(self):
            self.headers = Headers({"retry-after": "3"})
            self.request = Req()

    resp = Resp()
    scrub_exception_chain(_sdk_error(response=resp), GatewayConfig(anthropic_api_key=secret))
    assert secret.encode() not in resp._content
    assert resp.headers["retry-after"] == "3"
    assert resp.request.headers["authorization"] == "[REDACTED]"
    assert secret not in resp.request.headers["x-note"]
    assert resp.request._content == b""


# ── base: think stripping and client eviction ────────────────────────────────


def test_unclosed_think_block_leaves_nothing():
    assert strip_think("<think>still going") == ""
    assert strip_think("  <think>a</think>\n answer ") == "answer "
    assert strip_think(None) is None


def test_think_stripper_streams_across_chunk_boundaries():
    s = _ThinkStripper()
    pieces = ["<th", "ink>reason", "ing</th", "ink>", "  ", "\n", "Hello", " world"]
    assert "".join(s.feed(p) for p in pieces) == "Hello world"


def test_think_stripper_passes_through_non_think_text_held_back_as_prefix():
    s = _ThinkStripper()
    assert s.feed("<thi") == ""
    assert s.feed("s is not a tag") == "<this is not a tag"
    assert s.feed("!") == "!"


async def test_close_clients_survives_a_failing_close_and_cancels_pending():
    closed = []

    class Good:
        async def close(self):
            closed.append("good")

    class Bad:
        async def close(self):
            raise RuntimeError("nope")

    cache = {("a",): Bad(), ("b",): Good()}
    await base.close_clients(cache)
    assert cache == {} and closed == ["good"]

    # An evicted client waiting out its grace is closed immediately.
    waiter = asyncio.get_running_loop().create_task(base._close_later(Good(), 60))
    base._closing.add(waiter)
    await asyncio.sleep(0)  # let it reach its sleep
    await base.close_clients({})
    assert closed == ["good", "good"]


async def test_close_later_swallows_a_failing_close():
    class Bad:
        async def close(self):
            raise RuntimeError("nope")

    await base._close_later(Bad(), 0)  # must not raise


# ── privacy layer ────────────────────────────────────────────────────────────


def test_map_strings_refuses_pathologically_deep_values():
    value: object = "x"
    for _ in range(300):
        value = [value]
    with pytest.raises(ValueError, match="nested too deeply"):
        _map_strings(value, str.upper)


async def test_stream_with_only_usage_and_empty_content_deltas_is_forwarded():
    layer = PrivacyLayer()

    async def fake(**kw):
        yield StreamDelta(content="")
        yield StreamDelta(usage=(3, 4))
        yield StreamDelta(finish_reason="length")

    layer._stream = lambda **kw: fake(**kw)
    from tests.test_privacy import pol

    deltas = [
        d
        async for d in layer.stream_chat(
            messages=[{"role": "user", "content": "hello"}], privacy=pol(OutputPolicy.RESTORE)
        )
    ]
    assert deltas[-1].finish_reason == "length"
    assert any(d.usage == (3, 4) for d in deltas)


async def test_stream_closes_the_upstream_even_if_the_consumer_stops_early():
    layer = PrivacyLayer()
    closed = []

    async def fake(**kw):
        try:
            for i in range(5):
                yield StreamDelta(content=f"w{i} ")
        finally:
            closed.append(True)

    layer._stream = lambda **kw: fake(**kw)
    from tests.test_privacy import pol

    gen = layer.stream_chat(
        messages=[{"role": "user", "content": "x"}], privacy=pol(OutputPolicy.RESTORE)
    )
    await gen.__anext__()
    await gen.aclose()
    assert closed == [True]


async def test_stream_flushes_held_tail_before_the_finishing_delta():
    layer = PrivacyLayer()

    async def fake(**kw):
        yield StreamDelta(content="ends with [per")
        yield StreamDelta(finish_reason="stop")

    layer._stream = lambda **kw: fake(**kw)
    from tests.test_privacy import pol

    text = ""
    async for d in layer.stream_chat(
        messages=[{"role": "user", "content": "x"}], privacy=pol(OutputPolicy.RESTORE)
    ):
        text += d.content or ""
        if d.finish_reason:
            break
    assert text == "ends with [per"


# ── service ──────────────────────────────────────────────────────────────────


def test_openapi_version_matches_the_package():
    from importlib.metadata import version

    app = create_app(GatewayConfig(anthropic_api_key="k", provider_order="anthropic"))
    assert app.version == version("llm-gateway")


def test_unknown_provider_prefix_is_a_400():
    from fastapi.testclient import TestClient

    client = TestClient(
        create_app(GatewayConfig(anthropic_api_key="k", provider_order="anthropic"))
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "nosuch/model-x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400
    assert "Unknown provider" in resp.json()["error"]["message"]
