"""Web search with a provider fallback chain.

`web_search()` tries its providers in order (default Tavily, then Anthropic's
hosted `web_search` server tool) and returns normalized hits plus usage and cost
metadata. It is a separate entry point on purpose: it does not change how
`complete()` / `chat()` / `stream_chat()` route, retry or fail over.

A provider is skipped, and the next one tried, when it is not configured (no
key), is not allowed by the caller (`allowed`), fails authentication or billing,
is rate limited, times out, answers 5xx or a connection error, or returns no
hits (`fall_through_on_empty`). A caller-side validation error (bad query,
bad arguments, a provider's own 400/422) is raised at once as
`WebSearchError(reason="invalid_request")`: the next provider would reject the
same request. Every skip is recorded in `SearchResults.attempts`.

Everything returned is third-party web content. Treat it as untrusted data, and
strip personal data from the query before calling.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx2

from .config import GatewayConfig
from .errors import ErrorKind, LLMError, classify
from .providers import anthropic as _anthropic
from .providers._anthropic_translate import usage_from_anthropic_response
from .providers.base import Usage

TOOL_TYPE = "web_search_20250305"
TOOL_NAME = "web_search"
MAX_USES_CEILING = 5
_PRICE_PER_SEARCH_USD = 0.01  # Anthropic: $10 per 1,000 searches (2026-10)
_PRICE_PER_CREDIT_USD = 0.008  # Tavily pay-as-you-go list price per credit
KNOWN_PROVIDERS = ("tavily", "anthropic", "vertex")
DEFAULT_PROVIDERS = ("tavily", "anthropic")
DEFAULT_TIMEOUT = 10.0
DEFAULT_SNIPPET_CHARS = 300

# Why a provider was skipped, or why the whole call failed.
Reason = Literal[
    "not_configured",
    "not_allowed",
    "not_implemented",
    "auth",
    "billing",
    "rate_limited",
    "timeout",
    "server_error",
    "empty",
    "unknown",
    "invalid_request",
    "all_failed",
]


class WebSearchError(LLMError):
    """A web search could not be served. `reason` says why (`invalid_request`
    for a caller-side validation error, `all_failed` when every provider in the
    chain was skipped or failed); `attempts` records each provider's outcome."""

    def __init__(
        self,
        message: str,
        reason: Reason,
        *,
        provider: str | None = None,
        attempts: Sequence[SearchAttempt] = (),
    ) -> None:
        super().__init__(message)
        self.reason: Reason = reason
        self.provider = provider
        self.attempts: tuple[SearchAttempt, ...] = tuple(attempts)


@dataclass(frozen=True)
class SearchHit:
    """One result, normalized across providers."""

    title: str
    url: str
    snippet: str = ""
    provider: str = ""


@dataclass(frozen=True)
class SearchUsage:
    """What the serving provider reports it used. `units` counts searches
    (Anthropic) or credits (Tavily); `cost_usd` is that fee only (Anthropic
    tokens are in `tokens`)."""

    provider: str
    units: int = 0
    unit: Literal["searches", "credits"] = "searches"
    model: str | None = None
    cost_usd: float = field(default=0.0, compare=False)
    tokens: Usage | None = None


@dataclass(frozen=True)
class SearchAttempt:
    """One provider's outcome: it `served` the call or was `skipped`/`failed`."""

    provider: str
    status: Literal["served", "skipped"]
    reason: Reason | None = None
    detail: str = ""


@dataclass(frozen=True)
class SearchResults:
    hits: list[SearchHit]
    provider: str
    usage: SearchUsage
    attempts: list[SearchAttempt]
    text: str = ""

    @property
    def skipped(self) -> list[SearchAttempt]:
        return [a for a in self.attempts if a.status == "skipped"]


# Back-compat alias for the 0.12.0 draft name.
WebSearchResult = SearchResults


class _Skip(Exception):
    def __init__(self, reason: Reason, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason: Reason = reason
        self.detail = detail


class _Invalid(Exception):
    pass


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
            hits.append(
                SearchHit(title=title, url=url, snippet=snippets.get(url, ""), provider="anthropic")
            )
    server_use = _get(_get(resp, "usage"), "server_tool_use")
    searches = _get(server_use, "web_search_requests", 0)
    searches = searches if isinstance(searches, int) and not isinstance(searches, bool) else 0
    return hits, "".join(texts).strip(), searches


_KIND_TO_REASON: dict[ErrorKind, Reason] = {
    ErrorKind.TRANSIENT: "server_error",
    ErrorKind.TIMEOUT: "timeout",
    ErrorKind.RATE_LIMITED: "rate_limited",
    ErrorKind.AUTH: "auth",
    ErrorKind.BILLING: "billing",
}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(limit - 1, 0)].rstrip() + "\u2026"


async def _search_tavily(
    query: str,
    *,
    config: GatewayConfig,
    max_results: int,
    timeout: float,
    retries: int,
    snippet_chars: int,
    client: httpx2.AsyncClient | None,
) -> tuple[list[SearchHit], str, SearchUsage]:
    if not config.tavily_api_key:
        raise _Skip("not_configured", "TAVILY_API_KEY is not set")
    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "include_usage": True,
    }
    headers = {"Authorization": f"Bearer {config.tavily_api_key}"}
    own = client is None
    http = client or httpx2.AsyncClient(timeout=timeout)
    try:
        resp: httpx2.Response | None = None
        for attempt in range(retries + 1):
            try:
                resp = await http.post(
                    config.tavily_url, json=payload, headers=headers, timeout=timeout
                )
            except httpx2.TimeoutException as exc:
                if attempt < retries:
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                raise _Skip("timeout", type(exc).__name__) from exc
            except httpx2.TransportError as exc:
                if attempt < retries:
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                raise _Skip("server_error", type(exc).__name__) from exc
            if resp.status_code >= 500 and attempt < retries:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            break
    finally:
        if own:
            await http.aclose()
    assert resp is not None
    status = resp.status_code
    if status >= 500:
        raise _Skip("server_error", f"HTTP {status}")
    if status in (401, 403):
        raise _Skip("auth", f"HTTP {status}")
    if status == 429:
        raise _Skip("rate_limited", "HTTP 429")
    if status in (432, 433):  # plan limit / pay-as-you-go limit exceeded
        raise _Skip("billing", f"HTTP {status}")
    if status >= 400:
        raise _Invalid(f"Tavily rejected the request (HTTP {status})")
    try:
        data = resp.json()
    except ValueError as exc:
        raise _Skip("server_error", "response was not JSON") from exc
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for item in (data.get("results") if isinstance(data, dict) else None) or []:
        url = str(_get(item, "url", "") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        hits.append(
            SearchHit(
                title=str(_get(item, "title", "") or ""),
                url=url,
                snippet=_clip(str(_get(item, "content", "") or ""), snippet_chars),
                provider="tavily",
            )
        )
    credits = _get(_get(data, "usage"), "credits", 1)
    credits = credits if isinstance(credits, int) and not isinstance(credits, bool) else 1
    usage = SearchUsage(
        provider="tavily",
        units=credits,
        unit="credits",
        cost_usd=round(credits * _PRICE_PER_CREDIT_USD, 6),
    )
    return hits[:max_results], "", usage


async def _search_anthropic(
    query: str,
    *,
    config: GatewayConfig,
    max_results: int,
    timeout: float,
    max_uses: int,
    model: str | None,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
    snippet_chars: int,
) -> tuple[list[SearchHit], str, SearchUsage]:
    if not _anthropic.configured(config):
        raise _Skip("not_configured", "ANTHROPIC_API_KEY is not set")
    uses = max(1, min(int(max_uses), MAX_USES_CEILING))
    tool: dict[str, Any] = {"type": TOOL_TYPE, "name": TOOL_NAME, "max_uses": uses}
    if allowed_domains:
        tool["allowed_domains"] = list(allowed_domains)
    elif blocked_domains:
        tool["blocked_domains"] = list(blocked_domains)
    chosen = model or _anthropic.default_model(config)
    try:
        client = _anthropic._client(config)
        resp = await client.messages.create(
            model=chosen,
            max_tokens=800,
            system=(
                "Search the web for the user's query and report what the results say, "
                "briefly. Use only the search results; do not add facts from memory."
            ),
            messages=[{"role": "user", "content": query}],
            tools=[tool],
            timeout=timeout,
        )
    except Exception as exc:
        kind = classify(exc)
        if kind is ErrorKind.INVALID_REQUEST:
            raise _Invalid(f"Anthropic rejected the request: {type(exc).__name__}") from exc
        raise _Skip(_KIND_TO_REASON.get(kind, "unknown"), type(exc).__name__) from exc
    hits, text, searches = parse_response(resp)
    hits = [
        SearchHit(h.title, h.url, _clip(h.snippet, snippet_chars), "anthropic")
        for h in hits[:max_results]
    ]
    usage = SearchUsage(
        provider="anthropic",
        units=searches,
        unit="searches",
        model=chosen,
        cost_usd=round(searches * _PRICE_PER_SEARCH_USD, 6),
        tokens=usage_from_anthropic_response(resp.usage),
    )
    return hits, text, usage


async def web_search(
    query: str,
    *,
    providers: Sequence[str] = DEFAULT_PROVIDERS,
    max_results: int = 3,
    timeout: float = DEFAULT_TIMEOUT,
    config: GatewayConfig | None = None,
    allowed: Mapping[str, bool] | None = None,
    fall_through_on_empty: bool = True,
    retries: int = 1,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    max_uses: int = 1,
    model: str | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    http_client: httpx2.AsyncClient | None = None,
) -> SearchResults:
    """Search the web for `query`, trying `providers` in order.

    `allowed` maps a provider to False to block it (for example Tavily until a
    DPA is signed); a provider missing from the mapping is allowed. `retries`
    applies to Tavily's transient failures. `max_uses` and `model` apply to
    Anthropic. `timeout` is per provider call, in seconds.

    Raises `WebSearchError`: `invalid_request` for bad arguments or a request a
    provider rejects as malformed (no fall-through), `all_failed` when no
    provider served the call."""
    if not isinstance(query, str) or not query.strip():
        raise WebSearchError("query must be a non-empty string.", "invalid_request")
    if not 1 <= max_results <= 20:
        raise WebSearchError("max_results must be between 1 and 20.", "invalid_request")
    if timeout <= 0:
        raise WebSearchError("timeout must be positive.", "invalid_request")
    order = [p.strip().lower() for p in providers if p and p.strip()]
    unknown = [p for p in order if p not in KNOWN_PROVIDERS]
    if not order or unknown:
        raise WebSearchError(
            f"providers must be a non-empty list of {', '.join(KNOWN_PROVIDERS)}"
            + (f"; got unknown {unknown}." if unknown else "."),
            "invalid_request",
        )
    config = config or GatewayConfig()
    allow = allowed or {}
    attempts: list[SearchAttempt] = []
    empty: tuple[str, str, SearchUsage] | None = None

    def skip(provider: str, reason: Reason, detail: str = "") -> None:
        attempts.append(SearchAttempt(provider, "skipped", reason, detail))

    for provider in dict.fromkeys(order):
        if not allow.get(provider, True):
            skip(provider, "not_allowed", "blocked by the caller")
            continue
        try:
            if provider == "tavily":
                hits, text, usage = await _search_tavily(
                    query,
                    config=config,
                    max_results=max_results,
                    timeout=timeout,
                    retries=max(0, retries),
                    snippet_chars=snippet_chars,
                    client=http_client,
                )
            elif provider == "anthropic":
                hits, text, usage = await _search_anthropic(
                    query,
                    config=config,
                    max_results=max_results,
                    timeout=timeout,
                    max_uses=max_uses,
                    model=model,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                    snippet_chars=snippet_chars,
                )
            else:
                raise _Skip("not_implemented", "Vertex/Google grounding is not implemented")
        except _Skip as s:
            skip(provider, s.reason, s.detail)
            continue
        except _Invalid as exc:
            skip(provider, "invalid_request", str(exc))
            raise WebSearchError(
                str(exc), "invalid_request", provider=provider, attempts=attempts
            ) from exc
        if not hits and fall_through_on_empty:
            # Try the next provider; if none does better, this empty answer is
            # what the caller gets (an honest "no results", not an error).
            skip(provider, "empty", "no results")
            if empty is None:
                empty = (provider, text, usage)
            continue
        attempts.append(SearchAttempt(provider, "served"))
        return SearchResults(
            hits=hits, provider=provider, usage=usage, attempts=attempts, text=text
        )

    if empty is not None:
        provider, text, usage = empty
        final = [
            SearchAttempt(a.provider, "served", "empty", a.detail)
            if a.provider == provider and a.reason == "empty"
            else a
            for a in attempts
        ]
        return SearchResults(hits=[], provider=provider, usage=usage, attempts=final, text=text)

    summary = "; ".join(
        f"{a.provider}: {a.reason}" + (f" ({a.detail})" if a.detail else "") for a in attempts
    )
    raise WebSearchError(
        f"No search provider served the call. {summary}", "all_failed", attempts=attempts
    )
