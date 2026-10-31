"""Anthropic's server-side web search tool, as one self-contained call.

`web_search()` asks Claude to search the web for a query with Anthropic's hosted
`web_search` tool and returns what the search found: result titles and URLs
(and the text snippets Claude cited), plus how many searches ran and the token
usage. It is a separate entry point on purpose: it does not change how
`complete()` / `chat()` / `stream_chat()` route, retry or fail over, and it
only ever calls the `anthropic` provider (a server tool exists nowhere else).

Everything returned is third-party web content. Treat it as untrusted data.

Cost: Anthropic bills each search ($10 per 1,000 at 2026-10) on top of tokens.
`searches` is read from `usage.server_tool_use.web_search_requests`; `cost_usd` is
the search fee only (tokens are in `usage`), at
`_PRICE_PER_SEARCH_USD`, so a caller can meter spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import GatewayConfig
from .errors import GatewayNotConfiguredError
from .providers import anthropic as _anthropic
from .providers._anthropic_translate import usage_from_anthropic_response
from .providers.base import Usage

TOOL_TYPE = "web_search_20250305"
TOOL_NAME = "web_search"
MAX_USES_CEILING = 5
_PRICE_PER_SEARCH_USD = 0.01


@dataclass(frozen=True)
class SearchHit:
    """One result: where it is, what it is called, and the cited text if any."""

    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class WebSearchResult:
    hits: list[SearchHit]
    text: str
    searches: int
    model: str
    usage: Usage
    cost_usd: float = field(default=0.0, compare=False)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def parse_response(resp: Any) -> tuple[list[SearchHit], str, int]:
    """Hits (deduplicated by URL, in order), the answer text and the number of
    searches that ran, from a Messages API response."""
    snippets: dict[str, str] = {}
    texts: list[str] = []
    results: list[tuple[str, str]] = []
    for block in _get(resp, "content", []) or []:
        kind = _get(block, "type")
        if kind == "text":
            texts.append(str(_get(block, "text", "") or ""))
            for cite in _get(block, "citations", None) or []:
                url = _get(cite, "url")
                quote = _get(cite, "cited_text")
                if url and quote and url not in snippets:
                    snippets[str(url)] = str(quote)
        elif kind == "web_search_tool_result":
            content = _get(block, "content", [])
            if isinstance(content, list):  # an error is an object, not a list
                for item in content:
                    if _get(item, "type") == "web_search_result" and _get(item, "url"):
                        results.append((str(_get(item, "title", "") or ""), str(_get(item, "url"))))
    seen: set[str] = set()
    hits: list[SearchHit] = []
    for title, url in results:
        if url not in seen:
            seen.add(url)
            hits.append(SearchHit(title=title, url=url, snippet=snippets.get(url, "")))
    server_use = _get(_get(resp, "usage"), "server_tool_use")
    searches = _get(server_use, "web_search_requests", 0)
    searches = searches if isinstance(searches, int) and not isinstance(searches, bool) else 0
    return hits, "".join(texts).strip(), searches


async def web_search(
    query: str,
    *,
    config: GatewayConfig | None = None,
    max_uses: int = 1,
    max_tokens: int = 800,
    model: str | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> WebSearchResult:
    """Search the web for `query` with Anthropic's hosted tool.

    `max_uses` caps how many searches Claude may run for this one call (1 to
    `MAX_USES_CEILING`). The caller decides what `query` is; the gateway sends it
    as given, so strip personal data before calling.

    Raises `GatewayNotConfiguredError` without an Anthropic key, and whatever the
    SDK raises for a rejected request (for example an organisation that has not
    enabled web search)."""
    config = config or GatewayConfig()
    if not _anthropic.configured(config):
        raise GatewayNotConfiguredError("web_search needs ANTHROPIC_API_KEY.")
    uses = max(1, min(int(max_uses), MAX_USES_CEILING))
    tool: dict[str, Any] = {"type": TOOL_TYPE, "name": TOOL_NAME, "max_uses": uses}
    if allowed_domains:
        tool["allowed_domains"] = list(allowed_domains)
    elif blocked_domains:
        tool["blocked_domains"] = list(blocked_domains)
    chosen = model or _anthropic.default_model(config)
    client = _anthropic._client(config)
    resp = await client.messages.create(
        model=chosen,
        max_tokens=max_tokens,
        system=(
            "Search the web for the user's query and report what the results say, "
            "briefly. Use only the search results; do not add facts from memory."
        ),
        messages=[{"role": "user", "content": query}],
        tools=[tool],
    )
    hits, text, searches = parse_response(resp)
    usage = usage_from_anthropic_response(resp.usage)
    return WebSearchResult(
        hits=hits,
        text=text,
        searches=searches,
        model=chosen,
        usage=usage,
        cost_usd=round(searches * _PRICE_PER_SEARCH_USD, 6),
    )
