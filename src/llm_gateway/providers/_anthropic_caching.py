"""Prompt caching for Anthropic: when to mark the system prompt cacheable.

Minimum cacheable prompt length per model, from Anthropic's prompt-caching
docs (https://platform.claude.com/docs/en/build-with-claude/prompt-caching,
checked 2026-09-14). The docs say: "Shorter prompts cannot be cached, even if
marked with `cache_control`. Any requests to cache fewer than this number of
tokens will be processed without caching, and no error is returned."

So a marker on a prefix that's too short is harmless but does nothing. The
gateway sends one only when a cheap local estimate says the prefix reaches
the minimum. The estimate leans high (3 characters per token, where English
prose is roughly 3.5-4). Overestimating only sends a marker the API silently
ignores; underestimating would skip a cache that could have saved money.
Counting exactly with `messages.count_tokens` would add a separate API
request, and its latency, to every call. That costs more than the check can
save.
"""

import math

# Matched by the longest prefix of the model id, so dated snapshots
# ("claude-haiku-4-5-20251001") and point releases resolve correctly.
MIN_CACHEABLE_TOKENS: dict[str, int] = {
    "claude-fable-5": 512,
    "claude-mythos-5": 512,
    "claude-opus-5": 512,
    "claude-mythos-preview": 2048,
    "claude-opus-4-7": 2048,
    "claude-opus-4-6": 4096,
    "claude-opus-4-5": 4096,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-4-5": 1024,
    "claude-opus-4-1": 1024,
    "claude-opus-4": 1024,
    "claude-sonnet-4": 1024,
    "claude-haiku-4-5": 4096,
    "claude-3-5-haiku": 2048,
}
# Unknown models get the largest documented minimum.
DEFAULT_MIN_CACHEABLE_TOKENS = 4096
CHARS_PER_TOKEN_ESTIMATE = 3.0


def min_cacheable_tokens(model: str) -> int:
    matches = [prefix for prefix in MIN_CACHEABLE_TOKENS if model.startswith(prefix)]
    if not matches:
        return DEFAULT_MIN_CACHEABLE_TOKENS
    return MIN_CACHEABLE_TOKENS[max(matches, key=len)]


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def system_param(system: str, model: str, cache_system: bool) -> str | list[dict]:
    """The `system` argument for messages.create(): the plain string, or a
    single text block carrying `cache_control` when caching was requested and
    the prefix is long enough to be cached."""
    if cache_system and system and estimate_tokens(system) >= min_cacheable_tokens(model):
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    return system
