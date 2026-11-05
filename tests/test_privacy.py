import copy
import pickle
import re

import pytest
from pydantic import BaseModel

from llm_gateway.errors import InvalidOutputError
from llm_gateway.privacy import (
    AuditEvent,
    OutputPolicy,
    Privacy,
    PrivacyLayer,
    PrivacyPolicyMissingError,
    Span,
    StreamFilter,
    TokenFormat,
    Vault,
    _Output,
    redact_messages,
    redact_text,
)
from llm_gateway.providers.base import ChatResult, StreamDelta, ToolCall, Usage
from llm_gateway.router import Completion

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
NAME = re.compile(r"Łukasz Żółć|Anna Nowak")


class Rx:
    """A tiny regex redactor for the tests."""

    def detect(self, text):
        for m in EMAIL.finditer(text):
            yield Span(m.start(), m.end(), "email")
        for m in NAME.finditer(text):
            yield Span(m.start(), m.end(), "person")


def pol(output, **kw):
    return Privacy(redactor=Rx(), output=output, **kw)


def comp(text, parsed=None):
    return Completion(text=text, provider="p", model="m", usage=Usage(1, 1), parsed=parsed)


# ── vault and redaction ──────────────────────────────────────────────────────


def test_same_value_same_token_distinct_values_distinct_tokens():
    v = Vault()
    t = redact_text("Anna Nowak, Łukasz Żółć, Anna Nowak", Rx(), v)
    assert t == "[person_1], [person_2], [person_1]"
    assert v.original("[person_2]") == "Łukasz Żółć"


def test_overlapping_and_unsorted_spans_are_merged():
    class R:
        def detect(self, text):
            return [Span(3, 6, "a"), Span(0, 5, "b"), Span(100, 200, "c")]

    assert redact_text("abcdefgh", R(), Vault()) == "[b_1]fgh"


def test_vault_never_shows_values_and_is_not_serialisable():
    v = Vault()
    redact_text("anna@example.com", Rx(), v)
    assert "anna" not in repr(v)
    for op in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            op(v)
    v.clear()
    assert len(v) == 0


def test_unnumbered_format_refuses_a_second_value_a_token():
    v = Vault(TokenFormat(numbered=False))
    assert redact_text("Anna Nowak and Łukasz Żółć", Rx(), v) == "[person] and "
    assert v.original("[person]") == "Anna Nowak"


def test_messages_system_user_assistant_tool_and_arguments_are_redacted():
    msgs = [
        {"role": "system", "content": "owner anna@example.com"},
        {"role": "user", "content": [{"type": "text", "text": "I am Anna Nowak"}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "mail", "arguments": '{"to": "anna@example.com", "n": 2}'},
                }
            ],
        },
        {"role": "tool", "content": "sent to anna@example.com"},
    ]
    before = copy.deepcopy(msgs)
    v = Vault()
    out = redact_messages(msgs, Rx(), v)
    assert msgs == before  # input untouched
    blob = repr(out)
    assert "anna@example.com" not in blob and "Anna Nowak" not in blob
    assert '"to": "[email_1]"' in out[2]["tool_calls"][0]["function"]["arguments"]
    assert len(v) == 2


# ── complete_with_usage ──────────────────────────────────────────────────────


def make_layer(reply, events=None, **kw):
    seen = {}

    async def fake(**kwargs):
        seen.update(kwargs)
        return reply(kwargs) if callable(reply) else reply

    layer = PrivacyLayer(
        complete_with_usage=fake,
        on_audit=(events.append if events is not None else None),
        **kw,
    )
    return layer, seen


async def test_strict_mode_refuses_a_call_without_a_policy():
    layer, seen = make_layer(comp("x"))
    with pytest.raises(PrivacyPolicyMissingError):
        await layer.complete_with_usage(system="s", prompt="p")
    assert not seen  # nothing reached the provider


async def test_non_strict_without_policy_is_a_passthrough():
    layer, seen = make_layer(comp("x"), strict=False)
    await layer.complete_with_usage(system="s", prompt="anna@example.com")
    assert seen["prompt"] == "anna@example.com"


def test_passthrough_needs_a_reason():
    with pytest.raises(ValueError):
        Privacy.passthrough("  ")


async def test_provider_never_sees_the_original_and_restore_puts_it_back():
    events: list[AuditEvent] = []
    layer, seen = make_layer(
        lambda kw: comp(f"Hello [person_1], write to [email_1]. Prompt was: {kw['prompt']}"), events
    )
    out = await layer.complete_with_usage(
        system="You help Anna Nowak",
        prompt="Anna Nowak <anna@example.com>",
        privacy=pol(OutputPolicy.RESTORE),
    )
    assert "Anna" not in seen["system"] + seen["prompt"] and "@" not in seen["prompt"]
    assert out.text.startswith("Hello Anna Nowak, write to anna@example.com.")
    (ev,) = events
    assert ev.redacted == {"person": 1, "email": 1} and ev.restored >= 2
    assert "Anna" not in repr(ev) and "example.com" not in repr(ev)


async def test_strip_removes_issued_tokens_and_keeps_foreign_text():
    layer, _ = make_layer(lambda kw: comp("Hi [person_1] [email_1] [other_1] [x]"))
    out = await layer.complete_with_usage(
        system="s", prompt="Anna Nowak anna@example.com", privacy=pol(OutputPolicy.STRIP)
    )
    assert out.text == "Hi   [other_1] [x]"


async def test_strip_pattern_removes_the_apps_own_vocabulary_too():
    p = pol(OutputPolicy.STRIP, strip_pattern=re.compile(r"\[(?:osoba|person)\]"))
    layer, _ = make_layer(comp("Hi [osoba] and [person]"))
    out = await layer.complete_with_usage(system="s", prompt="p", privacy=p)
    assert out.text == "Hi  and "


async def test_passthrough_is_untouched_and_audited_with_reason():
    events: list[AuditEvent] = []
    layer, seen = make_layer(comp("[person_1]"), events)
    out = await layer.complete_with_usage(
        system="s", prompt="Anna Nowak", privacy=Privacy.passthrough("public data", label="job-ad")
    )
    assert seen["prompt"] == "Anna Nowak" and out.text == "[person_1]"
    assert events[0].reason == "public data" and events[0].call == "job-ad"


class Card(BaseModel):
    name: str
    tips: list[str]


async def test_parsed_model_is_restored_and_revalidated():
    parsed = Card(name="[person_1]", tips=["mail [email_1]"])
    layer, _ = make_layer(comp('{"name":"[person_1]"}', parsed))
    out = await layer.complete_with_usage(
        system="s",
        prompt="Anna Nowak anna@example.com",
        privacy=pol(OutputPolicy.RESTORE),
        output_schema=Card,
    )
    assert out.parsed == Card(name="Anna Nowak", tips=["mail anna@example.com"])


async def test_parsed_dict_is_stripped():
    layer, _ = make_layer(comp("{}", {"a": ["x [person_1]"]}))
    out = await layer.complete_with_usage(
        system="s", prompt="Anna Nowak", privacy=pol(OutputPolicy.STRIP)
    )
    assert out.parsed == {"a": ["x "]}


async def test_values_the_policy_empties_fail_closed_not_leak():
    class Req(BaseModel):
        name: str

        def model_post_init(self, _):
            if not self.name.strip():
                raise ValueError("blank")

    layer, _ = make_layer(comp("{}", Req(name="[person_1]")))
    with pytest.raises(InvalidOutputError):
        await layer.complete_with_usage(
            system="s", prompt="Anna Nowak", privacy=pol(OutputPolicy.STRIP)
        )


async def test_the_vault_is_cleared_when_the_call_ends_even_on_error():
    holder = []

    class Spy(Rx):
        def detect(self, text):
            return super().detect(text)

    async def boom(**kw):
        raise RuntimeError("provider down")

    layer = PrivacyLayer(complete_with_usage=boom)
    original = Vault.clear
    Vault.clear = lambda self: (holder.append(len(self)), original(self))[1]
    try:
        with pytest.raises(RuntimeError):
            await layer.complete_with_usage(
                system="s", prompt="Anna Nowak", privacy=Privacy(Spy(), OutputPolicy.STRIP)
            )
    finally:
        Vault.clear = original
    assert holder == [1]


# ── streaming ────────────────────────────────────────────────────────────────


async def drain(layer, chunks, privacy, tool_chunks=None):
    async def fake(**kw):
        for c in chunks:
            yield StreamDelta(content=c)
        for frag in tool_chunks or []:
            yield StreamDelta(tool_call_deltas=[frag])
        yield StreamDelta(finish_reason="stop", usage=(1, 1))

    layer._stream = lambda **kw: fake(**kw)
    text, args = "", {}
    async for d in layer.stream_chat(
        messages=[{"role": "user", "content": "Anna Nowak anna@example.com"}], privacy=privacy
    ):
        text += d.content or ""
        for f in d.tool_call_deltas or []:
            args[f["index"]] = args.get(f["index"], "") + (f.get("function") or {}).get(
                "arguments", ""
            )
    return text, args


@pytest.mark.parametrize("cut", range(1, len("Hi [person_1], mail [email_1]!")))
async def test_a_token_split_at_any_position_is_restored(cut):
    full = "Hi [person_1], mail [email_1]!"
    layer = PrivacyLayer()
    text, _ = await drain(layer, [full[:cut], full[cut:]], pol(OutputPolicy.RESTORE))
    assert text == "Hi Anna Nowak, mail anna@example.com!"


async def test_a_token_split_one_char_at_a_time_is_stripped():
    full = "Hi [person_1], bye"
    text, _ = await drain(PrivacyLayer(), list(full), pol(OutputPolicy.STRIP))
    assert text == "Hi , bye"


async def test_a_lone_bracket_is_not_held_forever():
    text, _ = await drain(
        PrivacyLayer(),
        ["see [1", " and more text that is long enough to never be a token at all"],
        pol(OutputPolicy.RESTORE),
    )
    assert text.startswith("see [1 and more")


async def test_an_unclosed_bracket_at_the_end_is_flushed():
    text, _ = await drain(PrivacyLayer(), ["a [person"], pol(OutputPolicy.RESTORE))
    assert text == "a [person"


async def test_tool_call_arguments_split_across_chunks_are_restored():
    frags = [
        {"index": 0, "id": "c", "function": {"name": "mail", "arguments": '{"to":"[ema'}},
        {"index": 0, "function": {"arguments": 'il_1]"}'}},
    ]
    _, args = await drain(PrivacyLayer(), [], pol(OutputPolicy.RESTORE), frags)
    assert args[0] == '{"to":"anna@example.com"}'


async def test_stream_passthrough_and_strict():
    layer = PrivacyLayer()
    text, _ = await drain(layer, ["[person_1]"], Privacy.passthrough("x"))
    assert text == "[person_1]"
    with pytest.raises(PrivacyPolicyMissingError):
        async for _ in layer.stream_chat(messages=[]):
            pass


def test_stream_filter_unit():
    v = Vault()
    v.token_for("person", "Anna Nowak")
    f = StreamFilter(_Output(pol(OutputPolicy.RESTORE), v))
    got = f.feed("a [per") + f.feed("son_1] b") + f.finish()
    assert got == "a Anna Nowak b"


# ── chat ─────────────────────────────────────────────────────────────────────


async def test_chat_redacts_in_and_restores_tool_arguments_out():
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return ChatResult(
            content="Hi [person_1]",
            model="m",
            input_tokens=1,
            output_tokens=1,
            tool_calls=[ToolCall("1", "mail", {"to": "[email_1]"})],
            finish_reason="tool_calls",
        )

    layer = PrivacyLayer(chat=fake)
    res = await layer.chat(
        messages=[{"role": "user", "content": "Anna Nowak anna@example.com"}],
        privacy=pol(OutputPolicy.RESTORE),
    )
    assert "Anna" not in repr(seen["messages"])
    assert res.content == "Hi Anna Nowak" and res.tool_calls[0].arguments == {
        "to": "anna@example.com"
    }
