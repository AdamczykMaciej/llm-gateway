"""Property-based tests (Hypothesis) for the Anthropic Messages stream, the
OpenAI -> Anthropic translation, and the Vertex Claude path.

Bounded and fast like tests/test_properties.py: no network, no SDK calls. The
events are validated through the SDK's own event union, then replayed through a
fake `client.messages.stream(...)`.
"""

import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import anthropic
import httpx2
import pytest
from anthropic.types import RawMessageStreamEvent
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import TypeAdapter

from llm_gateway.config import GatewayConfig
from llm_gateway.errors import InvalidRequestError
from llm_gateway.providers import vertex
from llm_gateway.providers._anthropic_translate import to_anthropic_messages
from llm_gateway.providers.anthropic import messages_stream

PROFILE = settings(
    max_examples=int(os.environ.get("PROPERTY_EXAMPLES", "60")),
    deadline=None,
    derandomize=bool(os.environ.get("CI")),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,
)

_EVENT = TypeAdapter(RawMessageStreamEvent)
_text = st.text(max_size=30)
_json_leaf = st.none() | st.booleans() | st.integers(-(2**31), 2**31) | _text
_args = st.dictionaries(
    st.text(min_size=1, max_size=8),
    st.recursive(_json_leaf, lambda c: st.lists(c, max_size=3), max_leaves=6),
    max_size=4,
)
_tokens = st.integers(0, 10**6)


@st.composite
def _splits(draw: st.DrawFn, text: str) -> list[str]:
    cuts = sorted(draw(st.lists(st.integers(0, len(text)), max_size=6)))
    bounds = [0, *cuts, len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]


# --- a valid stream --------------------------------------------------------


@st.composite
def _blocks(draw: st.DrawFn) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(draw(st.integers(0, 5))):
        kind = draw(st.sampled_from(["text", "thinking", "tool_use"]))
        if kind == "tool_use":
            raw = json.dumps(draw(_args))
            out.append(
                {"kind": kind, "id": f"toolu_{i}", "name": f"fn_{i}", "parts": draw(_splits(raw))}
            )
        else:
            out.append({"kind": kind, "parts": draw(_splits(draw(_text)))})
    return out


def _events(
    blocks: list[dict[str, Any]],
    usage: dict[str, int],
    stop_reason: str,
    pings: list[int],
) -> list[Any]:
    raw: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage["input"],
                    "output_tokens": 1,
                    "cache_read_input_tokens": usage["read"],
                    "cache_creation_input_tokens": usage["write"],
                },
            },
        }
    ]
    for index, block in enumerate(blocks):
        if block["kind"] == "text":
            start = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": p} for p in block["parts"]]
        elif block["kind"] == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [{"type": "thinking_delta", "thinking": p} for p in block["parts"]]
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": p} for p in block["parts"]]
        raw.append({"type": "content_block_start", "index": index, "content_block": start})
        raw.extend({"type": "content_block_delta", "index": index, "delta": d} for d in deltas)
        raw.append({"type": "content_block_stop", "index": index})
    raw.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage["output"]},
        }
    )
    raw.append({"type": "message_stop"})
    events = [_EVENT.validate_python(e) for e in raw]
    for position in pings:  # keep-alives and unknown event types anywhere
        at = position % (len(events) + 1)
        events.insert(at, SimpleNamespace(type="ping" if position % 2 else "future_event"))
    return events


class _Stream:
    def __init__(self, events: list[Any], error: BaseException | None = None):
        self._events = events
        self._error = error

    async def __aenter__(self) -> "_Stream":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event
        if self._error is not None:
            raise self._error


def _client(events: list[Any], error: BaseException | None = None) -> Any:
    client = SimpleNamespace()
    client.messages = SimpleNamespace(stream=lambda **_: _Stream(events, error))
    return client


async def _run(
    events: list[Any], error: BaseException | None = None, messages: Any = None, **kw: Any
) -> list[Any]:
    async def get_client() -> Any:
        return _client(events, error)

    gen = messages_stream(
        get_client,
        model="claude-opus-4-5",
        messages=messages or [{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=100,
        tool_choice=None,
        sampling=None,
        response_format=None,
        request_options={},
        **kw,
    )

    async def drain() -> list[Any]:
        return [d async for d in gen]

    return await asyncio.wait_for(drain(), timeout=5)  # a hang fails, not blocks


def _summarize(deltas: list[Any]) -> tuple[str, list[dict[str, Any]], Any, Any]:
    content = ""
    calls: dict[int, dict[str, Any]] = {}
    finish = usage = None
    for d in deltas:
        content += d.content or ""
        for frag in d.tool_call_deltas or []:
            call = calls.setdefault(frag["index"], {"id": None, "name": None, "args": ""})
            call["id"] = frag.get("id", call["id"])
            call["name"] = frag["function"].get("name", call["name"])
            call["args"] += frag["function"]["arguments"]
        finish = d.finish_reason or finish
        usage = d.usage_details or usage
    return content, [c for _, c in sorted(calls.items())], finish, usage


_usage_st = st.fixed_dictionaries(
    {"input": _tokens, "output": _tokens, "read": _tokens, "write": _tokens}
)
_pings = st.lists(st.integers(0, 10_000), max_size=4)


@PROFILE
@given(
    blocks=_blocks(),
    usage=_usage_st,
    pings=_pings,
    reason=st.sampled_from(["end_turn", "max_tokens", "stop_sequence", "pause_turn", "refusal"]),
)
def test_valid_streams_reassemble(blocks, usage, pings, reason):
    deltas = asyncio.run(_run(_events(blocks, usage, reason, pings)))
    content, calls, finish, final_usage = _summarize(deltas)

    assert content == "".join("".join(b["parts"]) for b in blocks if b["kind"] == "text")
    wanted = [b for b in blocks if b["kind"] == "tool_use"]
    assert [(c["id"], c["name"], json.loads(c["args"])) for c in calls] == [
        (b["id"], b["name"], json.loads("".join(b["parts"]))) for b in wanted
    ]
    assert final_usage.input_tokens == usage["input"] + usage["read"] + usage["write"]
    assert final_usage.output_tokens == usage["output"]
    assert deltas[-1].finish_reason == finish and finish is not None
    # exactly one terminal delta carries the finish reason
    assert sum(1 for d in deltas if d.finish_reason) == 1


@PROFILE
@given(blocks=_blocks(), usage=_usage_st, data=st.data())
def test_stream_split_does_not_change_the_result(blocks, usage, data):
    """Re-cutting every block's fragments differently yields the same totals."""
    resplit = [{**b, "parts": data.draw(_splits("".join(b["parts"])))} for b in blocks]
    a = _summarize(asyncio.run(_run(_events(blocks, usage, "end_turn", []))))
    b = _summarize(asyncio.run(_run(_events(resplit, usage, "end_turn", []))))
    assert a[0] == b[0]
    assert [(c["id"], c["name"], c["args"]) for c in a[1]] == [
        (c["id"], c["name"], c["args"]) for c in b[1]
    ]
    assert a[2:] == b[2:]


@PROFILE
@given(blocks=_blocks(), usage=_usage_st, cut=st.integers(0, 60))
def test_error_mid_stream_surfaces_typed(blocks, usage, cut):
    """An error event mid-stream (the SDK raises it from the iterator) comes
    out as that typed error, after the deltas already produced; no hang."""
    events = _events(blocks, usage, "end_turn", [])[: cut % 40]
    request = httpx2.Request("POST", "https://api.example.test")
    error = anthropic.APIConnectionError(request=request)
    with pytest.raises(anthropic.APIError):
        asyncio.run(_run(events, error))


# --- garbage ---------------------------------------------------------------

_junk_text = st.one_of(st.none(), st.integers(), _text, st.booleans(), st.lists(_text, max_size=2))


@st.composite
def _junk_event(draw: st.DrawFn) -> Any:
    kind = draw(
        st.sampled_from(
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "message_delta",
                "message_stop",
                "content_block_stop",
                "ping",
                "error",
                "zzz",
            ]
        )
    )
    attrs: dict[str, Any] = {"index": draw(st.integers(-3, 9))}
    block_type = draw(st.sampled_from(["text", "tool_use", "thinking", "weird"]))
    attrs["content_block"] = SimpleNamespace(
        type=block_type, id=draw(_junk_text), name=draw(_junk_text)
    )
    delta_type = draw(st.sampled_from(["text_delta", "input_json_delta", "thinking_delta", "x"]))
    attrs["delta"] = SimpleNamespace(
        type=delta_type,
        text=draw(_text),
        partial_json=draw(_text),
        stop_reason=draw(st.none() | st.sampled_from(["end_turn", "tool_use", "max_tokens", "?"])),
    )
    usage = SimpleNamespace(
        input_tokens=draw(_junk_text),
        output_tokens=draw(_junk_text),
        cache_read_input_tokens=draw(_junk_text),
        cache_creation_input_tokens=draw(_junk_text),
    )
    attrs["usage"] = usage
    attrs["message"] = SimpleNamespace(usage=usage)
    return SimpleNamespace(type=kind, **attrs)


@PROFILE
@given(events=st.lists(_junk_event(), max_size=12))
def test_garbage_events_never_raise_untyped_lookups(events):
    """Well-formed-but-odd and unknown events must not hit KeyError/IndexError
    or hang; the stream either completes or raises something else deliberate."""
    try:
        asyncio.run(_run(events))
    except (KeyError, IndexError):
        raise
    except Exception:  # noqa: BLE001 - only lookup errors are the bug
        pass


@PROFILE
@given(
    fragments=st.lists(_text, max_size=6),
    index=st.integers(0, 4),
)
def test_orphan_json_delta_without_a_start_does_not_crash(fragments, index):
    events = [
        SimpleNamespace(
            type="content_block_delta",
            index=index,
            delta=SimpleNamespace(type="input_json_delta", partial_json=f),
        )
        for f in fragments
    ]
    events.append(
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=1),
        )
    )
    deltas = asyncio.run(_run(events))
    assert deltas[-1].finish_reason == "stop"


# --- the Vertex Claude path --------------------------------------------------


@PROFILE
@given(blocks=_blocks(), usage=_usage_st, pings=_pings)
def test_vertex_stream_matches_the_direct_path(blocks, usage, pings):
    events = _events(blocks, usage, "end_turn", pings)
    config = GatewayConfig(vertex_project_id="my-project", provider_order="vertex", _env_file=None)

    async def run_vertex() -> list[Any]:
        async def fake_client(_config: Any) -> Any:
            return _client(events)

        with patch.object(vertex, "_client", fake_client):
            return [
                d
                async for d in vertex._stream_chat(  # pyright: ignore[reportPrivateUsage]
                    config, [{"role": "user", "content": "hi"}], None, 100
                )
            ]

    assert _summarize(asyncio.run(run_vertex())) == _summarize(asyncio.run(_run(events)))


@PROFILE
@given(url=st.text(min_size=1, max_size=40).filter(lambda s: not s.startswith("data:")))
def test_vertex_rejects_any_non_base64_image_url_before_the_network(url):
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
    with pytest.raises(InvalidRequestError):
        vertex.check_images_are_base64(messages)


# --- OpenAI -> Anthropic translation ----------------------------------------

_data_uri = st.builds(
    lambda mt, d: f"data:{mt};base64,{d}",
    st.sampled_from(["image/png", "image/jpeg", "image/webp"]),
    st.text("ABCDEFabcdef0123456789+/=", min_size=1, max_size=20),
)
_nonblank = st.text(min_size=1, max_size=20).filter(lambda s: s.strip() != "")
_parts = st.lists(
    st.one_of(
        st.builds(lambda t: {"type": "text", "text": t}, _nonblank),
        st.builds(lambda u: {"type": "image_url", "image_url": {"url": u}}, _data_uri),
    ),
    min_size=1,
    max_size=3,
)
_user_content = _nonblank | _parts


@st.composite
def _history(draw: st.DrawFn) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": draw(_nonblank)} for _ in range(draw(st.integers(0, 2)))
    ]
    counter = 0
    for _ in range(draw(st.integers(1, 4))):
        messages.append({"role": "user", "content": draw(_user_content)})
        if draw(st.booleans()):
            ids = [f"call_{counter + i}" for i in range(draw(st.integers(1, 3)))]
            counter += len(ids)
            messages.append(
                {
                    "role": "assistant",
                    "content": draw(st.none() | _nonblank),
                    "tool_calls": [
                        {
                            "id": i,
                            "type": "function",
                            "function": {"name": "f", "arguments": json.dumps(draw(_args))},
                        }
                        for i in ids
                    ],
                }
            )
            messages.extend(
                {"role": "tool", "tool_call_id": i, "content": draw(_nonblank | _parts)}
                for i in ids
            )
        messages.append({"role": "assistant", "content": draw(_nonblank)})
    messages.append({"role": "user", "content": draw(_user_content)})
    return messages


@PROFILE
@given(history=_history())
def test_translation_invariants(history):
    system, out = to_anthropic_messages(history)

    expected_system = "\n".join(m["content"] for m in history if m["role"] == "system")
    assert system == expected_system
    assert all(m["role"] in ("user", "assistant") for m in out)  # system merged out
    assert out[0]["role"] == "user"
    assert [m["role"] for m in out[1:]] == [
        "assistant" if prev["role"] == "user" else "user" for prev in out[:-1]
    ], "roles must alternate"

    for position, message in enumerate(out):
        content = message["content"]
        blocks = content if isinstance(content, list) else []
        for block in blocks:
            if block["type"] == "image":
                assert block["source"]["type"] == "base64"
                assert block["source"]["data"]
            if block["type"] == "text":
                assert block["text"].strip()
        uses = [b["id"] for b in blocks if b["type"] == "tool_use"]
        if uses:  # the very next message answers every tool_use id, in order
            nxt = out[position + 1]["content"]
            assert [b["tool_use_id"] for b in nxt if b["type"] == "tool_result"] == uses
        results = [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"]
        if results:
            prev = out[position - 1]["content"]
            assert results == [b["id"] for b in prev if b["type"] == "tool_use"]


_junk = st.recursive(
    st.none() | st.booleans() | st.integers() | _text,
    lambda c: (
        st.lists(c, max_size=3)
        | st.dictionaries(
            st.sampled_from(
                [
                    "role",
                    "content",
                    "tool_calls",
                    "tool_call_id",
                    "type",
                    "text",
                    "image_url",
                    "url",
                    "function",
                    "name",
                    "arguments",
                    "id",
                ]
            ),
            c,
            max_size=5,
        )
    ),
    max_leaves=10,
)


@PROFILE
@given(history=st.lists(_junk, max_size=4))
def test_garbage_histories_raise_only_typed_errors(history):
    try:
        to_anthropic_messages(history)
    except InvalidRequestError:
        pass
