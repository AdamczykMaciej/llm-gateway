"""The README's library examples, run against fake providers (no network)."""

import ast
import importlib.resources
import inspect
import json
import re
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


# ── README blocks tagged `<!-- example: name -->`, executed as written ───────

_README = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")


def _example(name: str) -> str:
    marker = f"<!-- example: {name} -->\n```python\n"
    assert marker in _README, f"README example {name!r} not found"
    return _README.split(marker, 1)[1].split("\n```", 1)[0]


async def _run(name: str, namespace: dict | None = None) -> dict:
    """Run a README block (top-level await allowed) and return its namespace."""
    ns: dict = dict(namespace or {})
    code = compile(_example(name), f"README:{name}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, ns)  # noqa: S307 - our own README
    if inspect.isawaitable(result):
        await result
    return ns


_FAKE_KEYS = {
    "ANTHROPIC_API_KEY": "sk-ant-fake",
    "GROQ_API_KEY": "gsk-fake",
    "OPENAI_API_KEY": "sk-fake",
    "RETRY_ATTEMPTS": "1",
}


@pytest.fixture
def fake_env(monkeypatch):
    for k, v in _FAKE_KEYS.items():
        monkeypatch.setenv(k, v)


def _ok(text: str = "ok") -> ProviderResult:
    return ProviderResult(text=text, model="m", input_tokens=1, output_tokens=1)


async def test_readme_validate_example(fake_env):
    good = json.dumps({"name": "Ada", "address": {"zip": "00-001"}})
    bad = json.dumps({"name": "Ada", "address": {}})
    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=_ok(good))}):
        ns = await _run("validate")
    assert ns["person"].address.zip == "00-001"
    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=_ok(bad))}):
        ns = await _run("validate")
    assert ns["paths"] == ("address.zip",)


async def test_readme_request_id_example(fake_env):
    with patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=_ok())}):
        await _run("request_id", {"order_id": 7})


async def test_readme_parallel_tool_calls_example(fake_env):
    reply = ChatResult(
        content=None,
        model="m",
        input_tokens=1,
        output_tokens=1,
        tool_calls=[ToolCall(id="t1", name="get_weather", arguments={"city": "Warsaw"})],
        finish_reason="tool_calls",
    )
    fake = AsyncMock(return_value=reply)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": fake}):
        ns = await _run("parallel_tool_calls", {"weather_tool": _TOOL})
    assert fake.call_args.kwargs["parallel_tool_calls"] is False
    assert ns["honoured"] is True


async def test_readme_tool_result_cap_example(fake_env):
    fake = AsyncMock(
        return_value=ChatResult(content="done", model="m", input_tokens=1, output_tokens=1)
    )
    tool = {
        "type": "function",
        "function": {"name": "read_file", "parameters": {"type": "object", "properties": {}}},
    }
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": fake}):
        await _run("tool_result_cap", {"huge_file_text": "x" * 50_000, "read_file_tool": tool})
    sent = fake.call_args.args[1]
    tool_msg = next(m for m in sent if m["role"] == "tool")
    assert len(tool_msg["content"]) < 21_000 and "truncated" in tool_msg["content"]


async def test_readme_privacy_example(fake_env):
    fake = AsyncMock(return_value=_ok("Dear [email_1], hello"))
    with patch("llm_gateway.router.CALLS", {"anthropic": fake}):
        ns = await _run("privacy")
    assert "ada@example.com" not in fake.call_args.args[2]
    assert ns["completion"].text == "Dear ada@example.com, hello"
    assert len(ns["events"]) == 1


async def test_readme_lifespan_example(fake_env):
    from starlette.testclient import TestClient

    ns = await _run("lifespan")
    closed = AsyncMock()
    with (
        patch("llm_gateway.router.CALLS", {"anthropic": AsyncMock(return_value=_ok("hi"))}),
        patch.dict(ns, {"aclose": closed}),
        TestClient(ns["app"]) as client,
    ):
        assert client.get("/hello").json() == {"text": "hi"}
        closed.assert_not_awaited()
    closed.assert_awaited_once()


async def test_readme_documented_names_are_exported():
    imported = set()
    for m in re.finditer(r"^from llm_gateway import ([^\n(]+)$", _README, re.M):
        imported |= {n.strip() for n in m.group(1).split(",")}
    assert imported
    assert imported <= set(llm_gateway.__all__), imported - set(llm_gateway.__all__)


async def test_nothing_private_is_exported_and_all_public_names_listed():
    assert not [n for n in llm_gateway.__all__ if n.startswith("_")]
    import types

    public = {
        n
        for n, v in vars(llm_gateway).items()
        if not n.startswith("_") and not isinstance(v, types.ModuleType)
    }
    assert public == set(llm_gateway.__all__), public ^ set(llm_gateway.__all__)
