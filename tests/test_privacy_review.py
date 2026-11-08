"""Regression tests from the security review of the privacy layer."""

import asyncio
import json
import re

import pytest

from llm_gateway.privacy import (
    OutputPolicy,
    Privacy,
    PrivacyLayer,
    Span,
    TokenFormat,
    Vault,
    redact_messages,
)
from llm_gateway.providers.base import ChatResult, StreamDelta, Usage
from llm_gateway.router import Completion

ANNA = re.compile(r"Anna Nowak")


class Rx:
    def __init__(self, kind="person", pattern=ANNA):
        self.kind, self.pattern = kind, pattern

    def detect(self, text):
        return [Span(m.start(), m.end(), self.kind) for m in self.pattern.finditer(text)]


def comp(text):
    return Completion(text=text, provider="p", model="m", usage=Usage(1, 1))


async def stream_all(layer, deltas, privacy):
    async def fake(**kw):
        for d in deltas:
            yield d

    layer._stream = lambda **kw: fake(**kw)
    return [
        d
        async for d in layer.stream_chat(
            messages=[{"role": "user", "content": "Anna Nowak"}], privacy=privacy
        )
    ]


# 1. a literal placeholder already in the input must not restore to someone else
async def test_literal_placeholder_in_input_is_not_restored_to_another_value():
    async def fake(**kw):
        return comp(kw["prompt"])  # the model echoes its input

    layer = PrivacyLayer(complete_with_usage=fake)
    out = await layer.complete_with_usage(
        system="s",
        prompt="Keep the literal [person_1] and mention Anna Nowak",
        privacy=Privacy(Rx(), OutputPolicy.RESTORE),
    )
    assert out.text == "Keep the literal [person_1] and mention Anna Nowak"


async def test_literal_placeholder_in_a_chat_message_is_reserved_too():
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return ChatResult(content="[person_1]", model="m", input_tokens=1, output_tokens=1)

    layer = PrivacyLayer(chat=fake)
    res = await layer.chat(
        messages=[
            {"role": "system", "content": "template with [person_1]"},
            {"role": "user", "content": "Anna Nowak"},
        ],
        privacy=Privacy(Rx(), OutputPolicy.RESTORE),
    )
    assert res.content == "[person_1]"  # the literal, not Anna
    assert "Anna" not in json.dumps(seen["messages"])


# 2. a kind the token pattern cannot match would leak the token unrestored
async def test_kind_with_underscore_or_digit_round_trips():
    async def fake(**kw):
        return comp(kw["prompt"])

    layer = PrivacyLayer(complete_with_usage=fake)
    out = await layer.complete_with_usage(
        system="s", prompt="Hi Anna Nowak", privacy=Privacy(Rx("job_title2"), OutputPolicy.RESTORE)
    )
    assert out.text == "Hi Anna Nowak"


def test_unusable_kind_is_refused_not_silently_leaked():
    with pytest.raises(ValueError):
        Vault().token_for("bad]kind", "x")


# 3. restored values inside streamed tool-call arguments must stay valid JSON
async def test_streamed_tool_arguments_stay_valid_json_after_restore():
    tricky = 'Anna "Ann" Nowak\nline2'

    frags = ['{"who": "[perso', 'n_1]"}']
    deltas = [
        StreamDelta(tool_call_deltas=[{"index": 0, "function": {"arguments": f}}]) for f in frags
    ]
    deltas.append(StreamDelta(finish_reason="tool_calls"))
    layer = PrivacyLayer()

    # drive through redact: input text is the tricky value
    async def fake(**kw):
        for d in deltas:
            yield d

    layer._stream = lambda **kw: fake(**kw)
    out = [
        d
        async for d in layer.stream_chat(
            messages=[{"role": "user", "content": tricky}],
            privacy=Privacy(
                type("X", (), {"detect": lambda s, t: [Span(0, len(t), "person")]})(),
                OutputPolicy.RESTORE,
            ),
        )
    ]
    args = "".join(
        (f.get("function") or {}).get("arguments", "")
        for d in out
        for f in (d.tool_call_deltas or [])
    )
    assert json.loads(args) == {"who": tricky}


# 4. held-back text must be flushed before the finishing delta
async def test_held_tail_is_flushed_before_finish_reason():
    deltas = [StreamDelta(content="see ["), StreamDelta(finish_reason="stop", usage=(1, 1))]
    out = await stream_all(PrivacyLayer(), deltas, Privacy(Rx(), OutputPolicy.RESTORE))
    text_before_finish = ""
    for d in out:
        text_before_finish += d.content or ""
        if d.finish_reason:
            break
    assert text_before_finish == "see ["


# 5. multi-character delimiters split mid-delimiter
@pytest.mark.parametrize("cut", range(1, 14))
async def test_multichar_delimiters_split_anywhere(cut):
    fmt = TokenFormat(open="<<", close=">>")
    full = "Hi <<person_1>>!"
    deltas = [StreamDelta(content=full[:cut]), StreamDelta(content=full[cut:])]
    out = await stream_all(PrivacyLayer(), deltas, Privacy(Rx(), OutputPolicy.RESTORE, fmt=fmt))
    assert "".join(d.content or "" for d in out) == "Hi Anna Nowak!"


# 6. isolation between concurrent requests (no shared state)
async def test_concurrent_calls_never_see_each_others_values():
    names = [f"Anna Nowak{i}" for i in range(20)]
    seen = {}

    async def fake(**kw):
        await asyncio.sleep(0.001 * (hash(kw["prompt"]) % 5))
        seen[kw["prompt"]] = kw["prompt"]
        return comp("hello [person_1]")

    layer = PrivacyLayer(complete_with_usage=fake)
    r = Rx(pattern=re.compile(r"Anna Nowak\d+"))
    outs = await asyncio.gather(
        *(
            layer.complete_with_usage(
                system="s", prompt=n, privacy=Privacy(r, OutputPolicy.RESTORE)
            )
            for n in names
        )
    )
    assert [o.text for o in outs] == [f"hello {n}" for n in names]
    assert all("Anna" not in p for p in seen)


# 7. tool-call pairing survives redaction (0.14.0 validation still passes)
def test_redaction_keeps_tool_ids_and_names():
    msgs = [
        {"role": "user", "content": "Anna Nowak"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "Anna Nowak"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "lookup", "content": "Anna Nowak found"},
    ]
    out = redact_messages(msgs, Rx(), Vault())
    assert out[1]["tool_calls"][0]["id"] == "c1"
    assert out[1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert out[2]["tool_call_id"] == "c1" and out[2]["name"] == "lookup"
    assert "Anna" not in json.dumps(out)
    assert msgs[1]["tool_calls"][0]["function"]["arguments"].count("Anna") == 1
