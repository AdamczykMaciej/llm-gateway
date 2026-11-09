"""Every public symbol in `__all__` (and every public method or property of an
exported class) carries a docstring of its own."""

import inspect

import pytest

import llm_gateway

# Type aliases and instances cannot carry a docstring; they are described in
# the module docstring or next to their definition.
_NON_DOCUMENTABLE = {"Message", "ContentPart", "ToolChoice", "AttemptHook", "BudgetCheck"}


def _own_doc(obj: object) -> str:
    return (inspect.getdoc(obj) or "").strip() if getattr(obj, "__doc__", None) else ""


def _symbols():
    for name in llm_gateway.__all__:
        obj = getattr(llm_gateway, name)
        if name in _NON_DOCUMENTABLE or not (inspect.isclass(obj) or inspect.isroutine(obj)):
            continue
        yield name, obj


def test_all_names_resolve_and_are_unique():
    assert len(set(llm_gateway.__all__)) == len(llm_gateway.__all__)
    for name in llm_gateway.__all__:
        assert hasattr(llm_gateway, name), name


@pytest.mark.parametrize(("name", "obj"), list(_symbols()), ids=lambda v: getattr(v, "__name__", v))
def test_public_symbol_has_its_own_docstring(name, obj):
    own = obj.__dict__.get("__doc__") if inspect.isclass(obj) else obj.__doc__
    assert own and own.strip(), f"{name} has no docstring"
    if inspect.isclass(obj):
        # dataclass / NamedTuple auto-generated "Name(field, ...)" does not count
        assert not own.strip().startswith(f"{obj.__name__}("), f"{name} has an auto docstring"


def test_public_members_of_exported_classes_are_documented():
    missing: list[str] = []
    for name, obj in _symbols():
        if not inspect.isclass(obj):
            continue
        for member_name, member in vars(obj).items():
            if member_name.startswith("_"):
                continue
            target = member.fget if isinstance(member, property) else member
            if isinstance(target, (staticmethod, classmethod)):
                target = target.__func__
            if not inspect.isroutine(target):
                continue
            if not (target.__doc__ or "").strip():
                missing.append(f"{name}.{member_name}")
    assert not missing, "undocumented public members: " + ", ".join(sorted(missing))


def test_type_aliases_are_documented_in_the_types_module():
    from llm_gateway import types

    assert types.__doc__ and "TypedDict" in types.__doc__
