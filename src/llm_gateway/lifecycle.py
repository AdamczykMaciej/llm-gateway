"""Model lifecycle: which model ids a vendor has retired or announced to retire.

Every entry is an *announced* retirement from the vendor's own deprecation
page, with the page URL and the date it was read. Only exact model ids are
matched (an alias is listed only where the vendor says it resolves to the
retired snapshot), and a model that is not listed is simply unknown: nothing
here asserts that an unlisted model is safe to use. Dates a vendor calls
"not sooner than" are not retirements and are not recorded.

routing.py uses this to skip a provider whose model is retired (never routing
a fallback chain to it) and to log one WARNING per model that retires within
`GatewayConfig.retirement_warning_days`. `GatewayConfig.block_retired_models`
turns the skip off, e.g. for an enterprise Groq contract that keeps a model
alive past its public shutdown date.

Not covered, because the vendor's schedule for that platform could not be
confirmed: Azure deployments (operator-named), Claude on Vertex (Google sets
its own dates, which can differ from Anthropic's) and Mistral ids.
"""

import logging
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger("llm_gateway")


@dataclass(frozen=True)
class ModelLifecycle:
    provider: str
    model: str
    retires: date
    """The announced shutdown date: requests fail from this day on."""
    replacement: str
    """The vendor's recommended replacement ("" when none is named)."""
    source: str
    checked: date


_ANTHROPIC = "https://platform.claude.com/docs/en/about-claude/model-deprecations"
_GROQ = "https://console.groq.com/docs/deprecations"
_OPENAI = "https://developers.openai.com/api/docs/deprecations"
_GEMINI = "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions"
_CHECKED = date(2026, 11, 9)


def _rows(
    provider: str,
    source: str,
    entries: list[tuple[str, str, str]],
) -> list[ModelLifecycle]:
    return [
        ModelLifecycle(provider, model, date.fromisoformat(when), replacement, source, _CHECKED)
        for model, when, replacement in entries
    ]


_ENTRIES: list[ModelLifecycle] = [
    *_rows(
        "anthropic",
        _ANTHROPIC,
        [
            ("claude-sonnet-4-5-20250929", "2026-11-30", "claude-sonnet-5-5"),
            # Pre-4.6 aliases resolve to the dated snapshot (models overview).
            ("claude-sonnet-4-5", "2026-11-30", "claude-sonnet-5-5"),
            ("claude-opus-4-1-20250805", "2026-08-05", "claude-opus-4-8"),
            ("claude-opus-4-1", "2026-08-05", "claude-opus-4-8"),
            ("claude-opus-4-20250514", "2026-06-15", "claude-opus-4-8"),
            ("claude-sonnet-4-20250514", "2026-06-15", "claude-sonnet-5-5"),
            ("claude-3-7-sonnet-20250219", "2026-02-19", "claude-sonnet-5-5"),
            ("claude-3-5-haiku-20241022", "2026-02-19", "claude-haiku-4-5-20251001"),
            ("claude-3-haiku-20240307", "2026-04-20", "claude-haiku-4-5-20251001"),
            ("claude-3-opus-20240229", "2026-01-05", "claude-opus-4-8"),
            ("claude-3-5-sonnet-20241022", "2025-10-28", "claude-sonnet-5-5"),
            ("claude-3-5-sonnet-20240620", "2025-10-28", "claude-sonnet-5-5"),
            ("claude-3-sonnet-20240229", "2025-07-21", "claude-sonnet-5-5"),
            ("claude-2.1", "2025-07-21", "claude-opus-4-8"),
            ("claude-2.0", "2025-07-21", "claude-opus-4-8"),
        ],
    ),
    *_rows(
        "groq",
        _GROQ,
        [
            ("llama-3.3-70b-versatile", "2026-08-16", "openai/gpt-oss-120b"),
            ("llama-3.1-8b-instant", "2026-08-16", "openai/gpt-oss-20b"),
            ("meta-llama/llama-4-scout-17b-16e-instruct", "2026-07-17", "openai/gpt-oss-120b"),
            ("qwen/qwen3-32b", "2026-07-17", "openai/gpt-oss-120b"),
            ("qwen/qwen3.6-27b", "2026-09-14", "qwen/qwen3.8-27b"),
            ("meta-llama/llama-guard-4-12b", "2026-03-05", "openai/gpt-oss-safeguard-20b"),
            ("moonshotai/kimi-k2-instruct", "2025-10-10", "openai/gpt-oss-120b"),
            ("gemma2-9b-it", "2025-10-08", "llama-3.1-8b-instant"),
            ("deepseek-r1-distill-llama-70b", "2025-10-02", "openai/gpt-oss-120b"),
            ("llama3-70b-8192", "2025-08-30", "llama-3.3-70b-versatile"),
            ("llama3-8b-8192", "2025-08-30", "llama-3.1-8b-instant"),
            ("mistral-saba-24b", "2025-07-30", "qwen/qwen3-32b"),
            ("qwen-qwq-32b", "2025-07-14", "qwen/qwen3-32b"),
            ("groq/compound", "2026-09-21", ""),
            ("groq/compound-mini", "2026-09-21", ""),
            ("moonshotai/kimi-k2-instruct-0905", "2026-04-15", "openai/gpt-oss-120b"),
            (
                "meta-llama/llama-4-maverick-17b-128e-instruct",
                "2026-03-09",
                "openai/gpt-oss-120b",
            ),
        ],
    ),
    *_rows(
        "openai",
        _OPENAI,
        [
            ("gpt-4o-2024-05-13", "2026-10-23", "gpt-5.6-sol"),
            ("gpt-4.1-nano", "2026-10-23", "gpt-5.6-luna"),
            ("gpt-4.1-nano-2025-04-14", "2026-10-23", "gpt-5.6-luna"),
            ("gpt-4", "2026-10-23", "gpt-5.6-sol"),
            ("gpt-4-0613", "2026-10-23", "gpt-5.6-sol"),
            ("gpt-4-turbo", "2026-10-23", "gpt-5.6-sol"),
            ("gpt-4-turbo-2024-04-09", "2026-10-23", "gpt-5.6-sol"),
            ("o1", "2026-10-23", "gpt-5.6-sol"),
            ("o1-2024-12-17", "2026-10-23", "gpt-5.6-sol"),
            ("o3-mini", "2026-10-23", "gpt-5.6-sol"),
            ("o3-mini-2025-01-31", "2026-10-23", "gpt-5.6-sol"),
            ("gpt-3.5-turbo", "2026-10-23", "gpt-5.6-terra"),
            ("gpt-3.5-turbo-0125", "2026-10-23", "gpt-5.6-terra"),
            ("o4-mini", "2026-10-23", "gpt-5.6-terra"),
            ("o4-mini-2025-04-16", "2026-10-23", "gpt-5.6-terra"),
            ("gpt-5", "2026-12-11", "gpt-5.6-sol"),
            ("gpt-5-2025-08-07", "2026-12-11", "gpt-5.6-sol"),
            ("gpt-5-mini", "2026-12-11", "gpt-5.6-terra"),
            ("gpt-5-mini-2025-08-07", "2026-12-11", "gpt-5.6-terra"),
            ("gpt-5-nano", "2026-12-11", "gpt-5.6-luna"),
            ("gpt-5-nano-2025-08-07", "2026-12-11", "gpt-5.6-luna"),
            ("gpt-5-pro", "2026-12-11", "gpt-5.6-sol"),
            ("gpt-4o-2024-08-06", "2027-01-14", "gpt-6-luna"),
            ("gpt-4o-2024-11-20", "2027-01-14", "gpt-6-luna"),
            ("gpt-4o-mini-2024-07-18", "2027-01-14", "gpt-6-luna"),
            ("gpt-4.1-2025-04-14", "2027-01-14", "gpt-6-luna"),
            ("gpt-4.1-mini-2025-04-14", "2027-01-14", "gpt-6-luna"),
            ("chatgpt-4o-latest", "2026-02-17", "gpt-5.1-chat-latest"),
            ("gpt-5-chat-latest", "2026-07-23", "gpt-5.6-sol"),
            ("gpt-5.1-chat-latest", "2026-07-23", "gpt-5.6-sol"),
        ],
    ),
    # The "gemini" provider is Gemini on Vertex AI (ids may carry "google/").
    *_rows(
        "gemini",
        _GEMINI,
        [
            ("gemini-2.0-flash", "2026-06-01", "gemini-3.5-flash-lite"),
            ("gemini-2.0-flash-lite", "2026-06-01", "gemini-3.5-flash-lite"),
            ("gemini-1.5-pro-001", "2025-05-24", "gemini-3.5-flash"),
            ("gemini-1.5-pro-002", "2025-09-24", "gemini-3.5-flash"),
            ("gemini-1.5-flash-001", "2025-05-24", "gemini-3.5-flash-lite"),
            ("gemini-1.5-flash-002", "2025-09-24", "gemini-3.5-flash-lite"),
            ("gemini-2.5-pro", "2026-10-20", "gemini-3.5-flash"),
            ("gemini-2.5-flash", "2026-10-20", "gemini-3.5-flash-lite"),
            ("gemini-2.5-flash-lite", "2026-10-20", "gemini-3.5-flash-lite"),
            ("gemini-3.6-flash", "2026-11-19", "gemini-3.8-flash"),
            ("gemini-3.7-flash", "2027-01-28", "gemini-3.8-flash"),
        ],
    ),
]

LIFECYCLE: dict[tuple[str, str], ModelLifecycle] = {(e.provider, e.model): e for e in _ENTRIES}


def lifecycle_for(provider: str, model: str) -> ModelLifecycle | None:
    """The recorded lifecycle of `provider`'s `model`, or None when unknown."""
    name = model.strip().lower()
    if provider == "gemini":
        name = name.removeprefix("google/")
    return LIFECYCLE.get((provider, name))


def is_retired(provider: str, model: str, *, today: date | None = None) -> ModelLifecycle | None:
    """The lifecycle entry when the model's announced shutdown date has been
    reached, else None."""
    entry = lifecycle_for(provider, model)
    if entry is not None and entry.retires <= (today or date.today()):
        return entry
    return None


def retiring_within(
    provider: str, model: str, days: int, *, today: date | None = None
) -> ModelLifecycle | None:
    """The lifecycle entry when the model retires in the future, within `days`."""
    now = today or date.today()
    entry = lifecycle_for(provider, model)
    if entry is not None and 0 < (entry.retires - now).days <= days:
        return entry
    return None


def retired_reason(entry: ModelLifecycle) -> str:
    use = f"; use {entry.replacement}" if entry.replacement else ""
    return f"model {entry.model} was retired on {entry.retires.isoformat()}{use}"


_warned: set[tuple[str, str]] = set()


def warn_if_retiring(provider: str, model: str, days: int, *, today: date | None = None) -> None:
    """Log one WARNING per (provider, model) that retires within `days`."""
    entry = retiring_within(provider, model, days, today=today)
    if entry is None or (provider, entry.model) in _warned:
        return
    _warned.add((provider, entry.model))
    logger.warning(
        "llm_gateway model_retiring: provider=%s model=%s retires=%s replacement=%s",
        provider,
        entry.model,
        entry.retires.isoformat(),
        entry.replacement or "-",
    )


_warned_retired: set[tuple[str, str]] = set()


def warn_retired(provider: str, entry: ModelLifecycle) -> None:
    """Log one WARNING per (provider, model) skipped because it is retired."""
    if (provider, entry.model) in _warned_retired:
        return
    _warned_retired.add((provider, entry.model))
    logger.warning(
        "llm_gateway model_retired: provider=%s model=%s retired=%s replacement=%s; "
        "skipping it (block_retired_models)",
        provider,
        entry.model,
        entry.retires.isoformat(),
        entry.replacement or "-",
    )
