"""web_search(): the request shape sent to Anthropic and the response parsing,
with the SDK client mocked (no network)."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_gateway import GatewayConfig, web_search
from llm_gateway.errors import GatewayNotConfiguredError
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


async def test_request_uses_the_hosted_tool_with_max_uses():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search(
            "NAJ International company", config=GatewayConfig(anthropic_api_key="k"), max_uses=2
        )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tools"] == [{"type": TOOL_TYPE, "name": TOOL_NAME, "max_uses": 2}]
    assert kwargs["messages"] == [{"role": "user", "content": "NAJ International company"}]
    assert out.searches == 1 and out.cost_usd == 0.01


async def test_hits_are_deduplicated_and_carry_cited_snippets():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        out = await web_search("q", config=GatewayConfig(anthropic_api_key="k"))
    assert [h.url for h in out.hits] == ["https://naj.com.pl", "https://example.org/a"]
    assert out.hits[0].snippet == "executive search"
    assert out.hits[1].snippet == ""
    assert "executive search firm" in out.text


async def test_max_uses_is_clamped():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        await web_search("q", config=GatewayConfig(anthropic_api_key="k"), max_uses=99)
        assert client.messages.create.call_args.kwargs["tools"][0]["max_uses"] == 5
        await web_search("q", config=GatewayConfig(anthropic_api_key="k"), max_uses=0)
        assert client.messages.create.call_args.kwargs["tools"][0]["max_uses"] == 1


async def test_domain_filters_pass_through():
    client = _client(_response())
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        await web_search(
            "q", config=GatewayConfig(anthropic_api_key="k"), blocked_domains=["x.com"]
        )
    assert client.messages.create.call_args.kwargs["tools"][0]["blocked_domains"] == ["x.com"]


async def test_no_key_raises():
    with pytest.raises(GatewayNotConfiguredError):
        await web_search("q", config=GatewayConfig(anthropic_api_key=""))


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
