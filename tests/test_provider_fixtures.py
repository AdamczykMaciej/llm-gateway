"""Provider-shape fixtures: payloads from the providers' public documentation
examples (tests/fixtures/providers/*.json, each citing its URL) run through the
real parsing and translation code, so the hand-written fakes in the other test
files cannot drift unnoticed from what providers actually send. No network."""

import json
from pathlib import Path

import httpx
import openai
import pytest
from anthropic.types import Message, RawMessageStreamEvent
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import TypeAdapter

from llm_gateway.errors import classify
from llm_gateway.providers._anthropic_translate import from_anthropic_response
from llm_gateway.providers.anthropic import messages_stream
from llm_gateway.providers.base import (
    ToolIndexNormalizer,
    parse_openai_style_chunk,
    parse_openai_style_response,
)

FIXTURES = Path(__file__).parent / "fixtures" / "providers"
_EVENT = TypeAdapter(RawMessageStreamEvent)


def _load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _cases(name: str, key: str) -> list:
    return [pytest.param(c, id=f"{name}-{c['name']}") for c in _load(name).get(key, [])]


def test_every_fixture_cites_its_source():
    files = sorted(FIXTURES.glob("*.json"))
    assert files
    for path in files:
        assert json.loads(path.read_text())["source"].startswith("https://"), path.name


def _check_result(result, expect):
    assert result.content == expect["content"]
    assert result.finish_reason == expect["finish_reason"]
    assert result.input_tokens == expect["input_tokens"]
    assert result.output_tokens == expect["output_tokens"]
    assert [
        {"id": c.id, "name": c.name, "arguments": c.arguments} for c in result.tool_calls
    ] == expect["tool_calls"]


@pytest.mark.parametrize(
    "case",
    [c for n in ("openai", "mistral", "gemini") for c in _cases(n, "responses")],
)
def test_openai_style_response(case):
    resp = ChatCompletion.model_validate(case["payload"])
    _check_result(parse_openai_style_response(resp, resp.model), case["expect"])


@pytest.mark.parametrize("case", _cases("anthropic", "responses"))
def test_anthropic_response(case):
    resp = Message.model_validate(case["payload"])
    _check_result(from_anthropic_response(resp, resp.model), case["expect"])


def _aggregate(deltas, expect):
    content = ""
    calls: dict[int, dict] = {}
    finish = usage = None
    for d in deltas:
        content += d.content or ""
        for frag in d.tool_call_deltas or []:
            call = calls.setdefault(frag["index"], {"id": None, "name": None, "args": ""})
            call["id"] = frag.get("id", call["id"])
            call["name"] = frag["function"].get("name", call["name"])
            call["args"] += frag["function"]["arguments"]
        finish = d.finish_reason or finish
        usage = d.usage or usage
    assert content == expect["content"]
    assert finish == expect["finish_reason"]
    assert list(usage) == expect["usage"]
    assert [
        {"id": c["id"], "name": c["name"], "arguments": json.loads(c["args"])}
        for _, c in sorted(calls.items())
    ] == expect["tool_calls"]


@pytest.mark.parametrize(
    "case", [c for n in ("openai", "mistral", "gemini") for c in _cases(n, "streams")]
)
def test_openai_style_stream(case):
    normalizer = ToolIndexNormalizer()
    deltas = [
        normalizer.apply(parse_openai_style_chunk(ChatCompletionChunk.model_validate(c), "m"))
        for c in case["chunks"]
    ]
    _aggregate(deltas, case["expect"])


class _Stream:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for e in self._events:
            yield e


class _Messages:
    def __init__(self, events):
        self._events = events

    def stream(self, **kwargs):
        return _Stream(self._events)


class _Client:
    def __init__(self, events):
        self.messages = _Messages(events)


@pytest.mark.parametrize("case", _cases("anthropic", "streams"))
async def test_anthropic_stream(case):
    # The SDK consumes `ping` keep-alives itself, so they never reach the gateway.
    events = [_EVENT.validate_python(e) for e in case["events"] if e["type"] != "ping"]

    async def get_client():
        return _Client(events)

    deltas = [
        d
        async for d in messages_stream(
            get_client,
            model="claude-opus-4-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=100,
            tool_choice=None,
            sampling=None,
            response_format=None,
            request_options={},
        )
    ]
    _aggregate(deltas, case["expect"])


def _api_error(status: int, body: dict) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.example.invalid/v1/chat/completions")
    response = httpx.Response(status, request=request, json=body)
    return openai.APIStatusError("error", response=response, body=body.get("error", body))


@pytest.mark.parametrize("case", _cases("groq_errors", "errors"))
def test_error_bodies_classify(case):
    assert classify(_api_error(case["status"], case["body"])).value == case["kind"]
