"""A typed consumer: pyright must accept this file (checked by test_typing_consumer.py)."""

from typing import assert_type

from pydantic import BaseModel

from llm_gateway import (
    AttemptReport,
    Completion,
    GatewayConfig,
    complete_with_usage,
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
