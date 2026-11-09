"""Multi-step tool loops end to end (agentic robustness pass, 0.18.7).

A fake host runs model -> tool calls -> tool results -> model, repeatedly,
through `chat()` / `stream_chat()` with a failover chain (Anthropic, Vertex
Claude, OpenAI-style, Gemini via the OpenAI-compatible endpoint). The fake
providers validate every request the way their real wire format would (the
real Anthropic translation for the Claude pair), so a history that would be a
400 on the NEXT provider after a mid-loop failover fails the test. Fakes only:
no provider is ever called.
"""

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from llm_gateway import (
    AllProvidersExhaustedError,
    ChatResult,
    GatewayConfig,
    LLMDeadlineExceeded,
    ToolCall,
    ToolCallTruncatedError,
    Usage,
    chat,
    reset_circuit_breakers,
    stream_chat,
)
from llm_gateway.providers._anthropic_translate import to_anthropic_messages
from llm_gateway.providers.base import StreamDelta
from llm_gateway.tool_validation import check_tool_calls, tool_error_message

NAMES = ("anthropic", "vertex", "openai", "gemini")
ANTHROPIC_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
    {"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}},
]
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


# --- the scripted model -------------------------------------------------------


@dataclass
class Call:
    name: str  # "add" | "echo"
    bad: bool  # first emitted with invalid arguments, then re-emitted corrected
    result: str  # "text" | "image" | "error" | "empty"


@dataclass
class Plan:
    steps: list[list[Call]]
    usage: list[tuple[int, int, int, int, int]]
    id_style: dict[str, str]  # provider -> id style
    order: list[str]
    fail_first: dict[int, int]  # step -> how many leading providers fail with a 5xx
    truncate: set[int]  # steps whose first try (max_tokens=64) is cut off in a tool call
    cap: int

    def expanded(self) -> list[tuple[list[Call], bool]]:
        """(calls, emit_invalid): a step with bad calls is followed by a retry step."""
        out: list[tuple[list[Call], bool]] = []
        for step in self.steps:
            out.append((step, True))
            if any(c.bad for c in step):
                out.append(([c for c in step if c.bad], False))
        return out


@st.composite
def plans(draw) -> Plan:
    call = st.builds(
        Call,
        st.sampled_from(["add", "echo"]),
        st.booleans(),
        st.sampled_from(["text", "image", "error", "empty"]),
    )
    steps = draw(st.lists(st.lists(call, min_size=1, max_size=3), min_size=1, max_size=4))
    usage = st.tuples(*[st.integers(0, 5000)] * 3).map(lambda t: (t[0] + t[1], t[2], t[1], 0, 0))
    n = len(steps) * 2 + 2
    return Plan(
        steps=steps,
        usage=draw(st.lists(usage, min_size=n, max_size=n)),
        id_style={
            "anthropic": "toolu",
            "vertex": "toolu",
            "openai": draw(st.sampled_from(["call", "dotted", "long"])),
            "gemini": draw(st.sampled_from(["empty", "dup", "call", "dotted"])),
        },
        order=draw(st.permutations(NAMES)),
        fail_first={i: draw(st.integers(0, 3)) for i in range(n) if draw(st.booleans())},
        truncate=set(draw(st.lists(st.integers(0, n), max_size=3))),
        cap=draw(st.sampled_from([0, 0, 40])),
    )


def _make_id(style: str, name: str, step: int, k: int) -> str:
    return {
        "toolu": f"toolu_{step}_{k}",
        "call": f"call_{step}_{k}",
        "dotted": f"functions.{name}:{step}{k}",  # Kimi/OpenRouter-style ids
        "long": f"call_{'x' * 30}{step}_{k}",
        "empty": "",
        "dup": "same",
    }[style]


def _args(call: Call, emit_invalid: bool, step: int) -> dict:
    if call.name == "add":
        return {"a": "x", "b": 2} if (call.bad and emit_invalid) else {"a": step, "b": 2}
    return {"text": f"hi {step}"}


# --- provider-specific request validation ------------------------------------


def _walk_pairs(messages: list[dict]) -> list[str]:
    """OpenAI-style pairing problems (empty list = valid)."""
    problems: list[str] = []
    seen: set[str] = set()
    pending: set[str] = set()
    for i, m in enumerate(messages):
        if m["role"] == "tool":
            if m["tool_call_id"] not in pending:
                problems.append(f"{i}: orphan result {m['tool_call_id']!r}")
            pending.discard(m["tool_call_id"])
            continue
        if pending:
            problems.append(f"{i}: unanswered {sorted(pending)}")
            pending.clear()
        for tc in m.get("tool_calls") or []:
            if not tc["id"] or tc["id"] in seen:
                problems.append(f"{i}: empty/duplicate id {tc['id']!r}")
            seen.add(tc["id"])
            pending.add(tc["id"])
    if pending:
        problems.append(f"end: unanswered {sorted(pending)}")
    return problems


def _from_anthropic(system: str, msgs: list[dict]) -> list[dict]:
    """Anthropic wire back to OpenAI-shaped history (for the round-trip check)."""
    out: list[dict] = []
    for m in msgs:
        if isinstance(m["content"], str):
            out.append({"role": m["role"], "content": m["content"]})
        elif m["role"] == "assistant":
            out.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": b["id"],
                            "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                        }
                        for b in m["content"]
                        if b["type"] == "tool_use"
                    ],
                }
            )
        else:
            for b in m["content"]:
                if b["type"] == "tool_result":
                    out.append(
                        {"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]}
                    )
    return out


def _check_anthropic(messages: list[dict]) -> list[str]:
    problems: list[str] = []
    _, wire = to_anthropic_messages(messages)
    for i, m in enumerate(wire):
        if m["role"] != "assistant" or isinstance(m["content"], str):
            continue
        uses = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
        if not uses:
            continue
        for u in uses:
            if not ANTHROPIC_ID.fullmatch(u):
                problems.append(f"anthropic rejects tool_use id {u!r}")
        nxt = wire[i + 1] if i + 1 < len(wire) else None
        got = (
            [b["tool_use_id"] for b in nxt["content"] if b["type"] == "tool_result"]
            if nxt and nxt["role"] == "user" and not isinstance(nxt["content"], str)
            else []
        )
        if sorted(got) != sorted(uses):
            problems.append(f"{i}: tool_result ids {got} != tool_use ids {uses}")
    # Round trip: translating back keeps ids, names, arguments and result text.
    back = _from_anthropic("", wire)
    want_calls = [
        (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
        for m in messages
        for tc in m.get("tool_calls") or []
    ]
    got_calls = [
        (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
        for m in back
        for tc in m.get("tool_calls") or []
    ]
    if want_calls != got_calls:
        problems.append("tool calls changed in the Anthropic round trip")
    # Ids may be renamed to the Anthropic alphabet, but never merged or lost.
    ids_in = [tc["id"] for m in messages for tc in m.get("tool_calls") or []]
    ids_out = [tc["id"] for m in back for tc in m.get("tool_calls") or []]
    if len(set(ids_out)) != len(set(ids_in)):
        problems.append("two tool call ids became one")
    if sum(m["role"] == "tool" for m in messages) != sum(m["role"] == "tool" for m in back):
        problems.append("tool results changed in the Anthropic round trip")
    return problems


@dataclass
class World:
    plan: Plan
    problems: list[str] = field(default_factory=list)
    attempts: list[tuple[str, int]] = field(default_factory=list)  # (provider, step)
    in_step: int = 0
    max_tokens_seen: list[int] = field(default_factory=list)

    def provider(self, name: str):
        async def call(config, messages, tools, max_tokens, **kw):
            expanded = self.plan.expanded()
            step = sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
            self.attempts.append((name, step))
            self.max_tokens_seen.append(max_tokens)
            bad = (
                _check_anthropic(messages)
                if name in ("anthropic", "vertex")
                else _walk_pairs(messages)
            )
            if bad:
                self.problems.append(f"{name} step {step}: {bad}")
                raise ValueError("provider 400")
            if self.in_step < self.plan.fail_first.get(step, 0):
                self.in_step += 1
                raise ConnectionError("5xx")
            u = self.plan.usage[len(self.attempts) % len(self.plan.usage)]
            usage = dict(
                input_tokens=u[0],
                output_tokens=u[1],
                cache_read_input_tokens=u[2],
            )
            if step >= len(expanded):
                return ChatResult("done", "m", **usage)
            calls, invalid = expanded[step]
            if step in self.plan.truncate and max_tokens < 128:
                raise ToolCallTruncatedError("cut", tuple(c.name for c in calls))
            tool_calls = [
                ToolCall(
                    _make_id(self.plan.id_style[name], c.name, step, k),
                    c.name,
                    _args(c, invalid, step),
                )
                for k, c in enumerate(calls)
            ]
            return ChatResult(None, "m", tool_calls=tool_calls, finish_reason="tool_calls", **usage)

        return call


def _config(order, **kw) -> GatewayConfig:
    return GatewayConfig(
        provider_order=",".join(order),
        retry_attempts=1,
        retry_base_delay_seconds=0,
        breaker_failure_threshold=1000,
        **kw,
    )


def _result_content(kind: str, step: int):
    return {
        "text": f"ok {step}",
        "image": [{"type": "text", "text": "shot"}, IMAGE],
        "error": f"Error: boom {step}",
        "empty": "",
    }[kind]


async def run_loop(world: World, reports: list, usages: list[Usage]) -> list[dict]:
    plan = world.plan
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
    ]
    patches = {n: world.provider(n) for n in NAMES}
    with (
        patch.dict("llm_gateway.chat.CHAT_CALLS", patches),
        patch.dict("llm_gateway.chat.CONFIGURED", {n: (lambda c: True) for n in NAMES}),
    ):
        for _ in range(40):
            step = sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
            world.in_step = 0
            max_tokens = 64
            while True:
                try:
                    r = await chat(
                        messages=messages,
                        tools=TOOLS,
                        max_tokens=max_tokens,
                        config=_config(plan.order),
                        on_attempt=reports.append,
                        max_tool_result_chars=plan.cap,
                    )
                    break
                except AllProvidersExhaustedError as e:
                    if isinstance(e.__cause__, ToolCallTruncatedError) and max_tokens < 1024:
                        max_tokens *= 2
                        world.in_step = 0
                        continue
                    raise
            usages.append(r.usage)
            if not r.tool_calls:
                break
            issues = {i.call_id: i for i in check_tool_calls(TOOLS, r.tool_calls)}
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in r.tool_calls
                    ],
                }
            )
            expanded = plan.expanded()
            calls = expanded[step][0] if step < len(expanded) else []
            results = []
            for k, c in enumerate(r.tool_calls):
                if c.id in issues:
                    results.append(tool_error_message(issues[c.id]))
                else:
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": c.id,
                            "content": _result_content(calls[k].result, step),
                        }
                    )
            random.Random(step).shuffle(results)  # results may arrive in any order
            messages.extend(results)
    return messages


def _go(plan: Plan):
    world = World(plan)
    reports: list = []
    usages: list[Usage] = []
    reset_circuit_breakers()
    messages = asyncio.run(run_loop(world, reports, usages))
    return world, reports, usages, messages


@SETTINGS
@given(plans())
def test_tool_loop_survives_midloop_failover(plan: Plan):
    world, reports, usages, messages = _go(plan)
    # Every request, on whichever provider served it, was valid for that provider.
    assert world.problems == []
    # The loop ran to the final text answer through every scripted step.
    assert len(usages) == len(plan.expanded()) + 1
    assert _walk_pairs(messages) == []
    # Usage: the host's total is the sum of the served steps, and on_attempt
    # reported exactly the same numbers plus one report per attempt.
    assert Usage.total(usages) == Usage.total(
        r.usage for r in reports if r.outcome == "succeeded" and r.usage
    )
    assert len(reports) == len(world.attempts)
    assert all(r.outcome in ("succeeded", "failed") for r in reports)
    assert all(r.usage is None for r in reports if r.outcome == "failed")
    assert sum(r.outcome == "succeeded" for r in reports) == len(usages)
    by_call: dict[str, list] = {}
    for r in reports:
        by_call.setdefault(r.request_id, []).append(r)
    for rs in by_call.values():
        assert [r.attempt for r in rs] == list(range(1, len(rs) + 1))
        assert [r.is_fallback for r in rs] == [False] + [True] * (len(rs) - 1)
        assert all(r.outcome == "failed" for r in rs[:-1])
    # Truncated steps were retried with a larger max_tokens.
    if plan.truncate:
        assert max(world.max_tokens_seen) >= 64


def test_dotted_ids_survive_failover_to_claude():
    """OpenAI-compatible hosts mint ids like `functions.add:0`; Anthropic only
    accepts [a-zA-Z0-9_-]. The history built on one provider must be valid on
    the next."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "functions.add:0",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                },
                {
                    "id": "functions.add:1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a":3,"b":4}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "functions.add:0", "content": "3"},
        {"role": "tool", "tool_call_id": "functions.add:1", "content": "7"},
    ]
    assert _check_anthropic(messages) == []


# --- streaming: a restart mid-step never duplicates tool calls ----------------


def _fragments(calls: list[ToolCall]):
    for index, c in enumerate(calls):
        text = json.dumps(c.arguments)
        yield {
            "index": index,
            "id": c.id,
            "type": "function",
            "function": {"name": c.name, "arguments": text[:3]},
        }
        yield {"index": index, "function": {"arguments": text[3:]}}


async def _collect_calls(**kw) -> tuple[list[ToolCall], list, bool]:
    acc: dict[int, dict] = {}
    restarts = 0
    usage = None
    async for d in stream_chat(**kw):
        if d.restart:
            acc.clear()
            restarts += 1
            continue
        for f in d.tool_call_deltas or []:
            slot = acc.setdefault(f["index"], {"id": "", "name": "", "args": ""})
            slot["id"] = f.get("id") or slot["id"]
            fn = f.get("function") or {}
            slot["name"] = fn.get("name") or slot["name"]
            slot["args"] += fn.get("arguments") or ""
        if d.usage_details:
            usage = d.usage_details
    calls = [ToolCall(s["id"], s["name"], json.loads(s["args"])) for _, s in sorted(acc.items())]
    return calls, [usage], restarts > 0


@SETTINGS
@given(
    st.lists(st.integers(1, 3), min_size=1, max_size=4),
    st.lists(st.integers(0, 1), min_size=4, max_size=4),
)
def test_stream_loop_restart_never_duplicates_calls(per_step: list[int], deaths: list[int]):
    reset_circuit_breakers()
    order = ["anthropic", "openai", "gemini"]
    state = {"attempt": 0}
    reports: list = []

    def provider(name: str):
        async def gen(config, messages, tools, max_tokens, **kw):
            state["attempt"] += 1
            step = sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
            problems = _walk_pairs(messages)
            assert problems == [], problems
            if step >= len(per_step):
                yield StreamDelta(content="done")
                yield StreamDelta(finish_reason="stop", usage=(10, 2))
                return
            calls = [
                ToolCall(f"{name}_{step}_{k}", "add", {"a": k, "b": step})
                for k in range(per_step[step])
            ]
            dies = state["dies"] > 0
            for n, frag in enumerate(_fragments(calls)):
                yield StreamDelta(tool_call_deltas=[frag])
                if dies and n == 1:
                    state["dies"] -= 1
                    raise ConnectionResetError("cut")
            yield StreamDelta(
                finish_reason="tool_calls", usage=(10 + step, 3), usage_details=Usage(10 + step, 3)
            )

        return gen

    async def run():
        messages: list[dict] = [{"role": "user", "content": "go"}]
        total = Usage()
        served: list[ToolCall] = []
        with (
            patch.dict("llm_gateway.streaming.STREAM_CALLS", {n: provider(n) for n in order}),
            patch.dict("llm_gateway.streaming.CONFIGURED", {n: (lambda c: True) for n in order}),
        ):
            for step in range(len(per_step) + 1):
                state["dies"] = deaths[step % 4]
                calls, (usage,), _ = await _collect_calls(
                    messages=messages,
                    tools=TOOLS,
                    config=_config(order),
                    restart_on_failure=True,
                    on_attempt=reports.append,
                )
                if usage:
                    total = total + usage
                if not calls:
                    break
                served.extend(calls)
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                            }
                            for c in calls
                        ],
                    }
                )
                messages.extend(
                    {"role": "tool", "tool_call_id": c.id, "content": "ok"} for c in calls
                )
        return messages, served, total

    messages, served, total = asyncio.run(run())
    ids = [c.id for c in served]
    assert len(ids) == len(set(ids)) == sum(per_step)  # nothing doubled, nothing lost
    assert _walk_pairs(messages) == []
    assert sum(r.outcome == "succeeded" for r in reports) == len(per_step) + 1
    assert sum(r.outcome == "failed" for r in reports) == sum(
        deaths[step % 4] for step in range(len(per_step))
    )
    assert all(r.error_type == "ConnectionResetError" for r in reports if r.outcome == "failed")
    # Only the attempt that finished is billed in the host's total.
    assert total.input_tokens == sum(10 + step for step in range(len(per_step)))


# --- deadline bounds the whole loop (host passes the remaining budget) --------


def test_deadline_bounds_the_whole_loop_including_failover():
    reset_circuit_breakers()
    budget = 0.4
    started = time.monotonic()

    async def slow(config, messages, tools, max_tokens, **kw):
        await asyncio.sleep(0.12)
        return ChatResult(
            None, "m", 1, 1, [ToolCall(f"c{len(messages)}", "echo", {})], "tool_calls"
        )

    async def hang(config, messages, tools, max_tokens, **kw):
        await asyncio.sleep(30)

    async def run():
        messages: list[dict] = [{"role": "user", "content": "go"}]
        with (
            patch.dict("llm_gateway.chat.CHAT_CALLS", {"anthropic": hang, "openai": slow}),
            patch.dict(
                "llm_gateway.chat.CONFIGURED",
                {"anthropic": lambda c: True, "openai": lambda c: True},
            ),
        ):
            while True:
                left = budget - (time.monotonic() - started)
                cfg = _config(["anthropic", "openai"], call_deadline_seconds=max(left, 0.001))
                r = await chat(messages=messages, tools=TOOLS, config=cfg)
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": "{}"},
                            }
                            for c in r.tool_calls
                        ],
                    }
                )
                messages.extend(
                    {"role": "tool", "tool_call_id": c.id, "content": "ok"} for c in r.tool_calls
                )

    with pytest.raises(LLMDeadlineExceeded):
        asyncio.run(run())
    assert time.monotonic() - started < budget + 0.5


def test_repaired_ids_avoid_ids_already_in_the_history():
    from llm_gateway.tool_validation import unique_tool_call_ids

    fixed = unique_tool_call_ids(
        [ToolCall("", "echo", {}), ToolCall("a", "echo", {})], {"call_0", "a"}
    )
    assert [c.id for c in fixed] == ["call__2", "a__2"]
    same = [ToolCall("x", "echo", {})]
    assert unique_tool_call_ids(same, {"y"}) is same
