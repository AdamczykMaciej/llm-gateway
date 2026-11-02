"""web_search(): the request shape sent to Anthropic and the response parsing,
with the SDK client mocked (no network)."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from llm_gateway import GatewayConfig, WebSearchError, web_search
from llm_gateway.web_search import TOOL_NAME, TOOL_TYPE, parse_response

pytestmark = pytest.mark.asyncio


def _response(searches: int = 1):
    return NS(
        content=[
            NS(type="server_tool_use", name="web_search"),
            NS(
                type="web_search_tool_result",
                content=[
                    NS(type="web_search_result", title="NAJ", url="https://naj.com.pl"),
                    NS(type="web_search_result", title="NAJ again", url="https://naj.com.pl"),
                    NS(type="web_search_result", title="Other", url="https://example.org/a"),
                ],
            ),
            NS(
                type="text",
                text="NAJ is an executive search firm.",
                citations=[NS(url="https://naj.com.pl", cited_text="executive search")],
            ),
        ],
        usage=NS(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            server_tool_use=NS(web_search_requests=searches),
        ),
        stop_reason="end_turn",
    )


def _client(resp):
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=resp)
    return client


A = ["anthropic"]


async def test_request_uses_the_hosted_tool_with_max_uses():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search(
            "NAJ International company",
            providers=A,
            config=GatewayConfig(anthropic_api_key="k"),
            max_uses=2,
        )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tools"] == [{"type": TOOL_TYPE, "name": TOOL_NAME, "max_uses": 2}]
    assert kwargs["messages"] == [{"role": "user", "content": "NAJ International company"}]
    assert out.usage.units == 1 and out.usage.cost_usd == 0.01
    assert out.usage.model and out.usage.tokens.input_tokens == 10


async def test_hits_are_deduplicated_and_carry_cited_snippets():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search("q", providers=A, config=GatewayConfig(anthropic_api_key="k"))
    assert [h.url for h in out.hits] == ["https://naj.com.pl", "https://example.org/a"]
    assert out.hits[0].snippet == "executive search"
    assert out.hits[1].snippet == ""
    assert "executive search firm" in out.text


async def test_max_uses_is_clamped():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        await web_search("q", providers=A, config=GatewayConfig(anthropic_api_key="k"), max_uses=99)
        assert client.messages.create.call_args.kwargs["tools"][0]["max_uses"] == 5
        await web_search("q", providers=A, config=GatewayConfig(anthropic_api_key="k"), max_uses=0)
        assert client.messages.create.call_args.kwargs["tools"][0]["max_uses"] == 1


async def test_domain_filters_pass_through():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        await web_search(
            "q", providers=A, config=GatewayConfig(anthropic_api_key="k"), blocked_domains=["x.com"]
        )
    assert client.messages.create.call_args.kwargs["tools"][0]["blocked_domains"] == ["x.com"]


async def test_no_key_raises():
    with pytest.raises(WebSearchError) as ei:
        await web_search("q", providers=A, config=GatewayConfig(anthropic_api_key=""))
    assert ei.value.reason == "all_failed"
    assert ei.value.attempts[0].reason == "not_configured"


async def test_a_search_error_block_yields_no_hits():
    resp = NS(
        content=[
            NS(
                type="web_search_tool_result",
                content=NS(type="web_search_tool_result_error", error_code="max_uses_exceeded"),
            )
        ],
        usage=NS(server_tool_use=NS(web_search_requests=0)),
    )
    hits, text, searches = parse_response(resp)
    assert hits == [] and searches == 0


# ---- Tavily, chain, fall-through ----------------------------------------

import anthropic  # noqa: E402

CFG = GatewayConfig(anthropic_api_key="k", tavily_api_key="t")


def _tavily(handler):
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def _ok(request):
    return httpx2.Response(
        200,
        json={
            "results": [
                {"title": "Acme", "url": "https://acme.io", "content": "x" * 500},
                {"title": "dup", "url": "https://acme.io", "content": "d"},
                {"title": "B", "url": "https://b.io", "content": "b"},
            ],
            "usage": {"credits": 1},
        },
    )


async def test_tavily_normalization_request_and_usage():
    seen = {}

    def handler(request):
        seen["req"] = request
        return _ok(request)

    out = await web_search("Acme", config=CFG, http_client=_tavily(handler))
    import json

    body = json.loads(seen["req"].content)
    assert body["query"] == "Acme" and body["max_results"] == 3
    assert body["include_raw_content"] is False
    assert seen["req"].headers["authorization"] == "Bearer t"
    assert out.provider == "tavily"
    assert [h.url for h in out.hits] == ["https://acme.io", "https://b.io"]
    assert all(h.provider == "tavily" for h in out.hits)
    assert len(out.hits[0].snippet) <= 300
    assert out.usage.unit == "credits" and out.usage.units == 1 and out.usage.cost_usd > 0
    assert [a.status for a in out.attempts] == ["served"]


async def test_anthropic_hits_are_tagged_and_capped():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search("q", providers=A, max_results=1, config=CFG)
    assert len(out.hits) == 1 and out.hits[0].provider == "anthropic"
    assert client.messages.create.call_args.kwargs["timeout"] == 10.0


def _status(code):
    return lambda request: httpx2.Response(code, json={})


def _timeout(request):
    raise httpx2.ReadTimeout("slow")


@pytest.mark.parametrize(
    "handler,reason",
    [
        (_status(401), "auth"),
        (_status(429), "rate_limited"),
        (_status(432), "billing"),
        (_status(503), "server_error"),
        (_timeout, "timeout"),
        (lambda r: httpx2.Response(200, json={"results": []}), "empty"),
    ],
)
async def test_each_fall_through_reason_serves_from_anthropic(handler, reason):
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search("q", config=CFG, retries=0, http_client=_tavily(handler))
    assert out.provider == "anthropic"
    assert out.skipped[0].provider == "tavily" and out.skipped[0].reason == reason


async def test_tavily_retries_a_5xx_once():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(500) if len(calls) == 1 else _ok(request)

    out = await web_search("q", config=CFG, http_client=_tavily(handler))
    assert out.provider == "tavily" and len(calls) == 2


async def test_not_configured_falls_through():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search("q", config=GatewayConfig(anthropic_api_key="k"))
    assert out.provider == "anthropic" and out.skipped[0].reason == "not_configured"


async def test_anthropic_errors_fall_through_to_the_next_provider():
    def boom(status):
        resp = httpx2.Response(status, request=httpx2.Request("POST", "https://x"))
        cls = {401: anthropic.AuthenticationError, 429: anthropic.RateLimitError}[status]
        return cls("e", response=resp, body=None)

    for status, reason in [(401, "auth"), (429, "rate_limited")]:
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=boom(status))
        with patch("llm_gateway.providers.anthropic._client", return_value=client):
            out = await web_search(
                "q", providers=["anthropic", "tavily"], config=CFG, http_client=_tavily(_ok)
            )
        assert out.provider == "tavily" and out.skipped[0].reason == reason


async def test_disallowed_provider_is_skipped_without_a_call():
    def handler(request):
        raise AssertionError("tavily must not be called")

    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search(
            "q", config=CFG, allowed={"tavily": False}, http_client=_tavily(handler)
        )
    assert out.provider == "anthropic" and out.skipped[0].reason == "not_allowed"


async def test_validation_errors_do_not_fall_through():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        for kwargs in ({"max_results": 0}, {"providers": ["bing"]}, {"timeout": 0}):
            with pytest.raises(WebSearchError) as ei:
                await web_search("q", config=CFG, **kwargs)
            assert ei.value.reason == "invalid_request"
        with pytest.raises(WebSearchError) as ei:
            await web_search("  ", config=CFG)
        assert ei.value.reason == "invalid_request"
        # A provider's own 400 is the caller's problem: no fall-through.
        with pytest.raises(WebSearchError) as ei:
            await web_search("q", config=CFG, http_client=_tavily(_status(400)))
        assert ei.value.reason == "invalid_request" and ei.value.provider == "tavily"
    client.messages.create.assert_not_called()


async def test_all_providers_failing():
    client = MagicMock()
    client.messages.create = AsyncMock(
        side_effect=anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x"))
    )
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        with pytest.raises(WebSearchError) as ei:
            await web_search(
                "q",
                providers=["tavily", "vertex", "anthropic"],
                config=CFG,
                retries=0,
                http_client=_tavily(_status(503)),
            )
    err = ei.value
    assert err.reason == "all_failed"
    assert [(a.provider, a.reason) for a in err.attempts] == [
        ("tavily", "server_error"),
        ("vertex", "not_implemented"),
        ("anthropic", "timeout"),
    ]


async def test_empty_everywhere_returns_empty_results_unless_disabled():
    empty = lambda r: httpx2.Response(200, json={"results": []})  # noqa: E731
    resp = _response()
    resp.content = []
    client = _client(resp)
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search("q", config=CFG, http_client=_tavily(empty))
    assert out.hits == [] and out.provider == "tavily"
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search(
            "q", config=CFG, fall_through_on_empty=False, http_client=_tavily(empty)
        )
    assert out.hits == [] and out.attempts[0].status == "served"
