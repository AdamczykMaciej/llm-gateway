"""Public API audit: `__all__`, docstrings, py.typed, and backward-compatible call shapes."""

import importlib.resources
import inspect

import pytest

import llm_gateway
from llm_gateway import GatewayConfig, ModelPrice, Usage
from llm_gateway.errors import redact_secrets
from llm_gateway.pricing import cost_usd


def test_all_is_unique_resolvable_and_documented():
    assert len(set(llm_gateway.__all__)) == len(llm_gateway.__all__)
    for name in llm_gateway.__all__:
        obj = getattr(llm_gateway, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            assert obj.__doc__ and obj.__doc__.strip(), f"{name} has no docstring"


def test_every_error_class_is_exported_and_in_the_readme():
    from llm_gateway import errors

    readme = (importlib.resources.files("llm_gateway").parent.parent / "README.md").read_text(
        encoding="utf-8"
    )
    for name, obj in vars(errors).items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, Exception)
            and obj.__module__ == errors.__name__
        ):
            assert name in llm_gateway.__all__, name
            assert f"| `{name}`" in readme, name


def test_py_typed_ships():
    assert importlib.resources.files("llm_gateway").joinpath("py.typed").is_file()


def test_public_function_signatures_are_fully_annotated():
    for name in llm_gateway.__all__:
        obj = getattr(llm_gateway, name)
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            assert sig.return_annotation is not inspect.Signature.empty, name
            for p in sig.parameters.values():
                assert p.annotation is not inspect.Parameter.empty, f"{name}({p.name})"


def test_older_call_shapes_still_work():
    # Extended compatibly in 0.16.5-0.18.1 (no rename, so no deprecation alias is needed).
    assert redact_secrets("ok", GatewayConfig()) == "ok"
    price = ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0)
    assert cost_usd(price, Usage(input_tokens=1_000_000, output_tokens=0)) == pytest.approx(1.0)
