"""The docs name every public symbol and every config field, with its default."""

import re
from pathlib import Path

import llm_gateway
from llm_gateway import GatewayConfig

_README = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")

# Not settable from the environment (a callable), documented in prose.
_NOT_ENV = {"on_attempt"}


def test_every_public_name_appears_in_the_readme():
    missing = [n for n in llm_gateway.__all__ if not re.search(rf"\b{n}\b", _README)]
    assert not missing, f"public names missing from README.md: {missing}"


def _env_rows() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in _README.splitlines():
        m = re.match(r"\|\s*(`[^|]+`)\s*\|\s*([^|]*)\|", line)
        if m:
            for env in re.findall(r"`([A-Z][A-Z0-9_]+)`", m.group(1)):
                rows.setdefault(env, []).append(m.group(2).strip().replace("`", ""))
    return rows


def _norm(default: object) -> list[str]:
    if default in (None, "", ()):
        return ["—", "unset", ""]
    if isinstance(default, bool):
        return [str(default).lower()]
    if isinstance(default, float) and default == int(default):
        return [str(int(default))]
    return [str(default)]


def test_every_config_field_has_an_env_row_with_its_default():
    rows = _env_rows()
    problems: list[str] = []
    for name, field in GatewayConfig.model_fields.items():
        if name in _NOT_ENV:
            continue
        cells = rows.get(name.upper())
        if cells is None:
            problems.append(f"{name}: no row for {name.upper()}")
            continue
        wanted = _norm(field.default)
        # a shared row such as "`true` / `90`" documents two defaults at once
        flat = [part.strip() for c in cells for part in c.split("/")]
        if not any(w in flat or w in cells for w in wanted):
            problems.append(f"{name}: README says {cells}, code default is {wanted}")
    assert not problems, "\n".join(problems)
