# Contributing

Setup: `python -m venv .venv && .venv/bin/pip install -e ".[dev,service,azure,vertex]"`

Checks (all must pass):

- Tests + coverage: `pytest -q --cov` (floor `fail_under` in `pyproject.toml`; raise it, never lower it)
- Lint/format: `ruff check . && ruff format --check .`
- Types: `pyright --pythonpath "$(which python)" src`
- Dependencies: `pip-audit`

Tests never call live or paid providers.

Release: bump `version` in `pyproject.toml`, add a `CHANGELOG.md` entry, merge, then tag `vX.Y.Z`.

PRs are merged by the maintainer after review.
