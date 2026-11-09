"""Every `<!-- runnable: name -->` README block, executed as written against fake
providers. No network: the three provider seams are patched."""

import ast
import asyncio
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_gateway import reset_circuit_breakers
from llm_gateway.providers.base import ChatResult, ProviderResult, StreamDelta, ToolCall

pytestmark = pytest.mark.asyncio

_README = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
_BLOCKS = dict(re.findall(r"<!-- runnable: (\w+) -->\n```python\n(.*?)\n```", _README, re.S))

EXPECTED = {
    "quickstart",
    "failover",
    "stream_restart",
    "tool_loop",
    "structured",
    "on_attempt",
    "budgets",
    "timeouts",
    "pricing",
    "security",
}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


async def _anthropic_down(*a, **k):
    raise ConnectionResetError("boom")


async def _anthropic_text(*a, **k):
    return ProviderResult(text="Hi from anthropic", model="m-a", input_tokens=5, output_tokens=2)


async def _groq_text(*a, **k):
    return ProviderResult(
        text=json.dumps({"score": 90, "summary": "ok"}),
        model="m-g",
        input_tokens=5,
        output_tokens=2,
    )


def _fakes(name: str):
    """(CALLS, CHAT_CALLS, STREAM_CALLS) for one snippet."""
    calls: dict = {"anthropic": _anthropic_down, "groq": _groq_text}
    chat_calls: dict = {}
    stream_calls: dict = {}

    async def silent(*a, **k):
        await asyncio.sleep(5)
        yield StreamDelta(content="never")

    async def hello(*a, **k):
        yield StreamDelta(content="Hello ")
        yield StreamDelta(content="there")
        yield StreamDelta(finish_reason="stop", usage=(3, 2))

    async def dies(*a, **k):
        yield StreamDelta(content="Hel")
        raise ConnectionResetError("boom")

    if name == "structured":
        calls = {"anthropic": _groq_text}
    elif name == "stream_restart":
        stream_calls = {"anthropic": dies, "groq": hello}
    elif name == "timeouts":
        stream_calls = {"anthropic": silent, "groq": hello}
    elif name == "tool_loop":
        turns = iter(
            [
                ChatResult(
                    content=None,
                    model="m",
                    input_tokens=1,
                    output_tokens=1,
                    tool_calls=[ToolCall(id="t1", name="get_weather", arguments={"city": 5})],
                    finish_reason="tool_calls",
                ),
                ChatResult(
                    content=None,
                    model="m",
                    input_tokens=1,
                    output_tokens=1,
                    tool_calls=[
                        ToolCall(id="t2", name="get_weather", arguments={"city": "Warsaw"})
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(
                    content="It is 18C and clear in Warsaw.",
                    model="m",
                    input_tokens=1,
                    output_tokens=1,
                ),
            ]
        )

        async def scripted(*a, **k):
            return next(turns)

        chat_calls = {"anthropic": scripted}
    return calls, chat_calls, stream_calls


async def test_every_expected_snippet_is_in_the_readme():
    assert EXPECTED <= set(_BLOCKS), EXPECTED - set(_BLOCKS)


@pytest.mark.parametrize("name", sorted(EXPECTED))
async def test_readme_snippet_runs(name: str):
    calls, chat_calls, stream_calls = _fakes(name)
    code = compile(_BLOCKS[name], f"README:{name}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with (
        patch("llm_gateway.router.CALLS", calls),
        patch("llm_gateway.chat.CHAT_CALLS", chat_calls),
        patch("llm_gateway.streaming.STREAM_CALLS", stream_calls),
    ):
        result = eval(code, {"__name__": "readme_snippet"})  # noqa: S307 - our own README
        if asyncio.iscoroutine(result):
            await result
