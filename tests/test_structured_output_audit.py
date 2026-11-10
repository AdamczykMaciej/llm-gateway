"""Structured-output audit (0.41.0): truncation, BOM, depth, bounded repair
prompt. Fake payloads only; no provider is called."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from llm_gateway import (
    GatewayConfig,
    InvalidOutputError,
    OutputTruncatedError,
    complete_with_usage,
)
from llm_gateway.providers.base import ProviderResult
from llm_gateway.router import _repair_prompt
from llm_gateway.structured import OutputSchema, OutputValidator


class Person(BaseModel):
    name: str
    age: int


OBJ = {"type": "object"}


@pytest.mark.parametrize("stop", ["length", "max_tokens"])
def test_cut_off_reply_is_typed_even_with_a_complete_inner_object(stop):
    cut = '{"a": {"b": 1}, "c": [1, 2'  # an embedded {"b": 1} is complete
    for spec in (OBJ, Person):
        with pytest.raises(OutputTruncatedError) as info:
            OutputSchema.from_spec(spec).parse(cut, stop)
        assert info.value.paths == ("<root>",)
        assert '"b"' not in str(info.value)
    with pytest.raises(OutputTruncatedError):
        OutputValidator.from_spec(OBJ).check(cut, stop)


def test_without_a_length_stop_the_fragment_behaviour_is_unchanged():
    assert OutputSchema.from_spec(OBJ).parse('sure: {"b": 1} done', "stop") == {"b": 1}


def test_complete_json_with_length_stop_is_still_accepted():
    assert OutputSchema.from_spec(OBJ).parse('{"a": 1}', "length") == {"a": 1}


@pytest.mark.parametrize("text", ['﻿{"a": 1}', '﻿```json\n{"a": 1}\n```'])
def test_bom_is_ignored(text):
    assert OutputSchema.from_spec(OBJ).parse(text) == {"a": 1}
    assert OutputValidator.from_spec(OBJ).check(text) == {"a": 1}
    assert OutputSchema.from_spec({"type": "array"}).parse("﻿[1]") == [1]


def test_shallow_failures_carry_paths():
    schema = {"type": "object", "required": ["x"]}
    with pytest.raises(InvalidOutputError) as info:
        OutputSchema.from_spec(schema).parse("{}")
    assert info.value.paths == ("x",)
    with pytest.raises(InvalidOutputError) as info:
        OutputSchema.from_spec(schema).parse("[]")
    assert info.value.paths == ("<root>",)


def test_data_nested_past_the_validator_depth_is_not_silently_accepted():
    schema = {
        "$ref": "#/$defs/n",
        "$defs": {"n": {"type": "object", "properties": {"n": {"$ref": "#/$defs/n"}}}},
    }
    deep = "".join('{"n":' for _ in range(100)) + "{}" + "}" * 100
    with pytest.raises(InvalidOutputError) as info:
        OutputValidator.from_spec(schema).check(deep)
    assert info.value.paths


def test_very_deep_json_is_a_typed_error_for_model_and_dict():
    deep = "[" * 100000 + "]" * 100000
    for spec in (OBJ, Person):
        with pytest.raises(InvalidOutputError, match="deeply|valid JSON|not a"):
            OutputSchema.from_spec(spec).parse(deep)


def test_numbers_as_strings_dict_rejects_pydantic_coerces():
    schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
    with pytest.raises(InvalidOutputError) as info:
        OutputValidator.from_spec(schema).check('{"age": "3"}')
    assert info.value.paths == ("age",)
    assert OutputValidator.from_spec(Person).check('{"name": "a", "age": "3"}').age == 3


def test_repair_prompt_bounds_the_echoed_reply():
    error = InvalidOutputError("x", ("age",))
    prompt = _repair_prompt("p", "z" * 1_000_000, error)
    assert len(prompt) < 10_000 and "age" in prompt


def _provider(*results):
    return AsyncMock(side_effect=list(results))


@pytest.mark.asyncio
async def test_truncated_reply_is_not_repaired_with_a_paid_call():
    cut = ProviderResult('{"name": "A", "age": ', "m", 10, 5, stop_reason="max_tokens")
    provider = _provider(cut, cut)
    with patch("llm_gateway.router.CALLS", {"anthropic": provider}):
        with pytest.raises(OutputTruncatedError):
            await complete_with_usage(
                system="s",
                prompt="p",
                config=GatewayConfig(anthropic_api_key="k", provider_order="anthropic"),
                validate=Person,
                repair_attempts=2,
            )
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_output_schema_truncation_fails_over_as_invalid_output():
    cut = ProviderResult('{"name": "A", "age": ', "m", 10, 5, stop_reason="length")
    provider = _provider(cut)
    with patch("llm_gateway.router.CALLS", {"anthropic": provider}):
        with pytest.raises(Exception) as info:
            await complete_with_usage(
                system="s",
                prompt="p",
                config=GatewayConfig(anthropic_api_key="k", provider_order="anthropic"),
                output_schema=Person,
            )
    assert isinstance(info.value.__cause__, OutputTruncatedError)
