"""Every fenced ```python block in the docs runs, against fake providers.

Blocks tagged `<!-- runnable: x -->` / `<!-- example: x -->` have dedicated tests
(`test_readme_snippets.py`, `test_readme_examples.py`). Every other block is a
fragment: it is executed here, as written, with a small prelude that supplies the
names a reader is expected to have (``config``, ``answer``, ``messages`` ...).
A block that is neither tagged nor runnable here fails the build, so a stale
example cannot ship.
"""

import ast
import asyncio
import json
import re
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_gateway import GatewayConfig, reset_circuit_breakers
from llm_gateway.providers.base import ChatResult, ProviderResult, StreamDelta, ToolCall

_ROOT = Path(__file__).parent.parent
_DOC_FILES = ["README.md", "MIGRATION.md", "CONTRIBUTING.md", "SECURITY.md"]
_FENCE = re.compile(r"(<!-- (?:runnable|example): [\w-]+ -->\n)?```python\n(.*?)\n```", re.S)


def _blocks() -> list[tuple[str, int, bool, str]]:
    out = []
    for name in _DOC_FILES:
        path = _ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for m in _FENCE.finditer(text):
            line = text.count("\n", 0, m.start(2)) + 1
            out.append((name, line, m.group(1) is not None, m.group(2)))
    docs = _ROOT / "docs"
    for path in sorted(docs.glob("**/*.md")) if docs.is_dir() else []:
        text = path.read_text(encoding="utf-8")
        for m in _FENCE.finditer(text):
            line = text.count("\n", 0, m.start(2)) + 1
            out.append((str(path.relative_to(_ROOT)), line, m.group(1) is not None, m.group(2)))
    return out


_ALL = _blocks()
_FRAGMENTS = [(f, ln, code) for f, ln, tagged, code in _ALL if not tagged]


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


async def _provider_text(*a, **k):
    return ProviderResult(
        text=json.dumps({"score": 90, "summary": "ok"}), model="m", input_tokens=5, output_tokens=2
    )


async def _chat_reply(*a, **k):
    return ChatResult(
        content=None,
        model="m",
        input_tokens=1,
        output_tokens=1,
        tool_calls=[ToolCall(id="t1", name="send_email", arguments={})],
        finish_reason="tool_calls",
    )


async def _stream(*a, **k):
    yield StreamDelta(content="Hello", model="m")
    yield StreamDelta(finish_reason="stop", model="m", usage=(1, 2))


class _Ledger:
    async def add(self, *args: object) -> None:
        return None


async def _deny(call: object) -> bool:
    return False


def _prelude() -> dict:
    user = [{"role": "user", "content": "hi"}]
    return {
        "__name__": "docs_fragment",
        "config": GatewayConfig(
            anthropic_api_key="sk-ant-fake", groq_api_key="gsk-fake", retry_attempts=1
        ),
        "answer": "my answer",
        "messages": user,
        "tools": [],
        "system": "Be brief.",
        "prompt": "Hi!",
        "LONG_STATIC_INSTRUCTIONS": "rules " * 1000,
        "STATIC_INSTRUCTIONS": "rules " * 1000,
        "per_request_context": "context",
        "user_input": "question",
        "html_text": "<p>posting</p>",
        "ledger": _Ledger(),
        "ask_the_user": _deny,
        "GATEWAY_API_KEY": "fake",
    }


class _FakeOpenAI:
    def __init__(self, **kwargs: object) -> None:
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kw: {"ok": True})
        )


@pytest.mark.asyncio
async def test_every_python_block_parses():
    assert _ALL, "no python blocks found; the extractor is broken"
    for name, line, _tagged, code in _ALL:
        try:
            compile(code, f"{name}:{line}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:  # pragma: no cover - failure path
            pytest.fail(f"{name}:{line} does not parse: {e}")


@pytest.mark.asyncio
async def test_tagged_blocks_have_a_dedicated_test():
    tests = _ROOT / "tests"
    haystack = (tests / "test_readme_snippets.py").read_text() + (
        tests / "test_readme_examples.py"
    ).read_text()
    tags = []
    for name in _DOC_FILES:
        path = _ROOT / name
        if path.exists():
            tags += re.findall(r"<!-- (?:runnable|example): ([\w-]+) -->", path.read_text())
    assert tags
    for tag in tags:
        assert f'"{tag}"' in haystack, f"doc block {tag!r} is tagged but no test runs it"


@pytest.mark.asyncio
@pytest.mark.parametrize(("file", "line", "code"), _FRAGMENTS, ids=lambda v: str(v)[:24])
async def test_doc_fragment_runs(file: str, line: int, code: str, monkeypatch):
    for key, value in {
        "ANTHROPIC_API_KEY": "sk-ant-fake",
        "AZURE_ENDPOINT": "https://x.services.ai.azure.com",
        "AZURE_MODEL": "dep",
    }.items():
        monkeypatch.setenv(key, value)
    compiled = compile(code, f"{file}:{line}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with (
        patch("llm_gateway.router.CALLS", {"anthropic": _provider_text, "groq": _provider_text}),
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": _chat_reply, "groq": _chat_reply}),
        patch("llm_gateway.streaming.STREAM_CALLS", {"anthropic": _stream, "groq": _stream}),
        patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=_FakeOpenAI)}),
    ):
        ns = _prelude()
        result = eval(compiled, ns)  # noqa: S307 - our own docs
        if asyncio.iscoroutine(result):
            await result
