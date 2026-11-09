"""The TypedDict helpers in providers/_wire.py and the typed stream-delta parser."""

from types import SimpleNamespace

import pytest

from llm_gateway.errors import InvalidRequestError
from llm_gateway.providers._anthropic_translate import to_anthropic_messages
from llm_gateway.providers._wire import as_dict
from llm_gateway.providers.base import parse_openai_style_chunk


def test_as_dict_narrows_only_dicts():
    assert as_dict({"a": 1}) == {"a": 1}
    for value in (None, "x", 1, [1], ("a",)):
        assert as_dict(value) is None


def test_non_dict_content_part_is_rejected_not_trusted():
    with pytest.raises(InvalidRequestError):
        to_anthropic_messages([{"role": "user", "content": ["just a string"]}])


def test_chunk_tool_call_fragments_keep_openai_shape():
    first = SimpleNamespace(
        index=0, id="call_1", function=SimpleNamespace(name="f", arguments='{"a"')
    )
    later = SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments=":1}"))
    nameless = SimpleNamespace(index=1, id=None, function=None)

    def chunk(tc):
        delta = SimpleNamespace(content=None, tool_calls=[tc])
        choice = SimpleNamespace(delta=delta, finish_reason=None)
        return SimpleNamespace(usage=None, choices=[choice])

    assert parse_openai_style_chunk(chunk(first), "m").tool_call_deltas == [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "f", "arguments": '{"a"'},
        }
    ]
    assert parse_openai_style_chunk(chunk(later), "m").tool_call_deltas == [
        {"index": 0, "function": {"arguments": ":1}"}}
    ]
    assert parse_openai_style_chunk(chunk(nameless), "m").tool_call_deltas == [
        {"index": 1, "function": {"arguments": ""}}
    ]
