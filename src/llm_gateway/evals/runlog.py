"""Append-only run log: a human-readable markdown file plus a machine-readable JSONL twin."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunLog:
    """``RunLog("evals/log.md").append("v2 vs v1", "no regressions", data={...})``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.jsonl = self.path.with_suffix(".jsonl")

    def append(self, title: str, summary: str, data: dict[str, Any] | None = None) -> None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"## {stamp} - {title}\n\n{summary.strip()}\n\n")
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"at": stamp, "title": title, "summary": summary, "data": data or {}},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def entries(self) -> list[dict[str, Any]]:
        if not self.jsonl.exists():
            return []
        return [json.loads(x) for x in self.jsonl.read_text(encoding="utf-8").splitlines() if x]
