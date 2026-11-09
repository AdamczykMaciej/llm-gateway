"""A typed consumer: pyright must accept this file (checked by test_typing_consumer.py)."""

from typing import assert_type

from pydantic import BaseModel

from llm_gateway import (
    AttemptReport,
    Completion,
    GatewayConfig,
    Message,
    ToolChoice,
    ToolSpec,
    UserMessage,
    chat,
    complete_with_usage,
    stream_chat,
)


class Person(BaseModel):
    name: str


async def structured() -> None:
    result = await complete_with_usage(system="s", prompt="p", output_schema=Person)
    assert_type(result, Completion[Person])
    assert_type(result.parsed, Person | None)
    print(result.parsed.name if result.parsed else "")


async def validated() -> None:
    result = await complete_with_usage(system="s", prompt="p", validate=Person)
    assert_type(result, Completion[Person])


async def plain() -> None:
    result = await complete_with_usage(system="s", prompt="p")
    assert_type(result.text, str)
    # Without a model class, `parsed` is whatever JSON came back.
    reveal: Completion = result
    print(reveal.parsed)


async def schema_dict() -> None:
    result = await complete_with_usage(system="s", prompt="p", output_schema={"type": "object"})
    print(result.parsed)


def hook(report: AttemptReport) -> None:
    print(report.cost_usd)


config = GatewayConfig(on_attempt=hook)


TOOLS: list[ToolSpec] = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a person up",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    }
]
CHOICE: ToolChoice = {"type": "function", "function": {"name": "lookup"}}


async def typed_chat() -> None:
    user: UserMessage = {
        "role": "user",
        "content": [
            {"type": "text", "text": "who is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ],
    }
    history: list[Message] = [
        {"role": "system", "content": "be brief"},
        user,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Ada"},
    ]
    result = await chat(messages=history, tools=TOOLS, tool_choice=CHOICE, config=GatewayConfig())
    print(result.content)
    async for delta in stream_chat(messages=history, tool_choice="auto"):
        print(delta.content)


async def plain_dicts_still_work() -> None:
    plain = [{"role": "user", "content": "hi"}]
    await chat(messages=plain, tools=[{"type": "function", "function": {"name": "x"}}])
    await chat(messages=[{"role": "user", "content": "hi"}], tool_choice="none")
