"""The README's library examples, run against fake providers (no network)."""

import importlib.resources
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

import llm_gateway
from llm_gateway import (
    GatewayConfig,
    InvalidRequestError,
    chat,
    complete,
    complete_with_usage,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers.base import ChatResult, ProviderResult, StreamDelta, ToolCall

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config() -> GatewayConfig:
    # The README's first example, with fake keys.
    return GatewayConfig(
        anthropic_api_key="sk-ant-fake",
        groq_api_key="gsk-fake",
        openai_api_key="sk-fake",
        retry_attempts=1,
    )


async def test_public_api_is_exported_and_typed():
    for name in llm_gateway.__all__:
        assert hasattr(llm_gateway, name), name
    assert len(set(llm_gateway.__all__)) == len(llm_gateway.__all__)
    assert importlib.resources.files("llm_gateway").joinpath("py.typed").is_file()


async def test_readme_complete():
    result = ProviderResult(text="Hello!", model="m", input_tokens=1, output_tokens=1)
    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=result)}):
        text = await complete(system="You are a helpful assistant.", prompt="Hi!", config=_config())
    assert text == "Hello!"


async def test_readme_complete_with_usage_fields():
    result = ProviderResult(
        text="ok", model="m", input_tokens=3, output_tokens=2, stop_reason="end_turn"
    )
    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=result)}):
        out = await complete_with_usage(system="...", prompt="...", config=_config())
    assert (out.text, out.provider, out.model, out.stop_reason) == (
        "ok",
        "anthropic",
        "m",
        "end_turn",
    )
    assert out.usage.input_tokens == 3 and out.usage.uncached_input_tokens == 3


async def test_readme_structured_output():
    class Verdict(BaseModel):
        score: int = Field(ge=0, le=100)
        summary: str

    raw = json.dumps({"score": 90, "summary": "Good."})
    result = ProviderResult(text=raw, model="m", input_tokens=1, output_tokens=1)
    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=result)}):
        out = await complete_with_usage(
            system="Score the answer.", prompt="answer", config=_config(), output_schema=Verdict
        )
    assert out.parsed == Verdict(score=90, summary="Good.")
    assert out.text == raw


_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


async def test_readme_chat_with_tools():
    reply = ChatResult(
        content=None,
        model="m",
        input_tokens=1,
        output_tokens=1,
        tool_calls=[ToolCall(id="t1", name="get_weather", arguments={"city": "Warsaw"})],
        finish_reason="tool_calls",
    )
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(return_value=reply)}):
        result = await chat(
            messages=[{"role": "user", "content": "Weather in Warsaw?"}],
            tools=[_TOOL],
            config=_config(),
        )
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"city": "Warsaw"}


async def test_readme_stream_chat():
    async def fake(*args, **kwargs):
        yield StreamDelta(content="Hel", model="m")
        yield StreamDelta(content="lo", model="m")
        yield StreamDelta(finish_reason="stop", model="m", usage=(1, 2))

    with patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": fake}):
        deltas = [
            d
            async for d in stream_chat(
                messages=[{"role": "user", "content": "hi"}], config=_config()
            )
        ]
    assert "".join(d.content for d in deltas if d.content) == "Hello"


@pytest.mark.parametrize("bad", [5, ["auto"], {"function": "get_weather"}])
async def test_malformed_tool_choice_is_an_invalid_request(bad):
    # Malformed tool_choice values are rejected before any provider call.
    with pytest.raises(InvalidRequestError):
        await chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[_TOOL],
            tool_choice=bad,
            config=_config(),
        )


async def test_readme_error_table_covers_every_public_error():
    text = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    errors = [n for n in llm_gateway.__all__ if n.endswith(("Error", "Exceeded"))] + ["LLMError"]
    for name in errors:
        assert f"| `{name}`" in text, f"{name} missing from the README error table"
