"""0.20.1: token and cost accounting. Fakes only: no provider is ever called."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_gateway import (
    AllProvidersExhaustedError,
    EmptyCompletionError,
    GatewayConfig,
    InvalidOutputError,
    ModelPrice,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
)
from llm_gateway.providers import anthropic
from llm_gateway.providers.base import (
    ChatResult,
    parse_openai_style_response,
    provider_result_from_openai_style,
)

pytestmark = pytest.mark.asyncio

_PRICES = {"anthropic/m1": ModelPrice(input_per_mtok=1_000_000, output_per_mtok=2_000_000)}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _cfg(**kw) -> GatewayConfig:
    return GatewayConfig(
        anthropic_api_key="k",
        claude_model="m1",
        provider_order="anthropic",
        retry_attempts=1,
        model_prices=_PRICES,
        **kw,
    )


def _openai_resp(content, *, refusal=None, finish="length", prompt=10, completion=50, reasoning=50):
    message = SimpleNamespace(content=content, refusal=refusal, tool_calls=None)
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=4),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage
    )


# -- a reply that was billed and then rejected still carries its usage ---------


async def test_empty_openai_reply_carries_billed_usage():
    with pytest.raises(EmptyCompletionError) as e:
        parse_openai_style_response(_openai_resp(""), "m1")
    assert e.value.usage is not None
    assert (e.value.usage.input_tokens, e.value.usage.output_tokens) == (10, 50)
    assert e.value.usage.cache_read_input_tokens == 4


async def test_empty_openai_structured_reply_carries_billed_usage():
    with pytest.raises(EmptyCompletionError) as e:
        provider_result_from_openai_style(_openai_resp(None), "m1")
    assert e.value.usage is not None and e.value.usage.output_tokens == 50


async def test_openai_refusal_carries_billed_usage():
    with pytest.raises(InvalidOutputError) as e:
        provider_result_from_openai_style(_openai_resp(None, refusal="no"), "m1")
    assert e.value.usage is not None and e.value.usage.input_tokens == 10


async def test_anthropic_thinking_only_reply_carries_billed_usage():
    resp = MagicMock()
    resp.content = [SimpleNamespace(type="thinking", thinking="x")]
    resp.stop_reason = "max_tokens"
    resp.usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=50,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
        output_tokens_details=SimpleNamespace(thinking_tokens=50),
    )
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=resp)
    with (
        patch.object(anthropic, "_client", return_value=client),
        pytest.raises(EmptyCompletionError) as e,
    ):
        await anthropic.call(GatewayConfig(anthropic_api_key="k"), "s", "p", 50)
    assert e.value.usage is not None
    assert (e.value.usage.input_tokens, e.value.usage.output_tokens) == (10, 50)
    assert e.value.usage.reasoning_tokens == 50


async def test_failed_attempt_report_prices_the_billed_usage_of_an_empty_reply():
    reports = []
    with pytest.raises(EmptyCompletionError) as e:
        parse_openai_style_response(_openai_resp(""), "m1")
    failing = AsyncMock(side_effect=e.value)
    with patch("llm_gateway.router.CALLS", {"anthropic": failing}):
        with pytest.raises(AllProvidersExhaustedError):
            await complete_with_usage(
                system="s", prompt="p", config=_cfg(), on_attempt=reports.append
            )
    (report,) = reports
    assert report.outcome == "failed" and report.usage is not None
    assert report.usage.output_tokens == 50
    assert report.cost_usd == pytest.approx(10 + 100)


async def test_chat_failed_attempt_report_carries_billed_usage():
    reports = []
    with pytest.raises(EmptyCompletionError) as e:
        parse_openai_style_response(_openai_resp(""), "m1")
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(side_effect=e.value)}):
        with pytest.raises(AllProvidersExhaustedError):
            await chat(
                messages=[{"role": "user", "content": "p"}],
                config=_cfg(),
                on_attempt=reports.append,
            )
    assert reports[0].usage is not None and reports[0].cost_usd == pytest.approx(10 + 100)


async def test_chat_success_still_reports_cost():
    reports = []
    ok = ChatResult(content="hi", model="m1", input_tokens=3, output_tokens=4)
    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": AsyncMock(return_value=ok)}):
        await chat(
            messages=[{"role": "user", "content": "p"}], config=_cfg(), on_attempt=reports.append
        )
    assert reports[0].cost_usd == pytest.approx(3 + 8)


# -- Anthropic streaming usage is cumulative: it never goes down -----------------


def _event(type_: str, **kwargs) -> MagicMock:
    e = MagicMock()
    e.type = type_
    for k, v in kwargs.items():
        setattr(e, k, v)
    return e


class _FakeStream:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for e in self._events:
            yield e


async def test_stream_message_delta_zeros_do_not_erase_message_start_input_side():
    events = [
        _event(
            "message_start",
            message=MagicMock(
                usage=MagicMock(
                    input_tokens=7, cache_read_input_tokens=900, cache_creation_input_tokens=300
                )
            ),
        ),
        _event("content_block_start", index=0, content_block=MagicMock(type="text")),
        _event("content_block_delta", index=0, delta=MagicMock(type="text_delta", text="hi")),
        _event("content_block_stop", index=0),
        _event(
            "message_delta",
            delta=MagicMock(stop_reason="end_turn"),
            usage=MagicMock(
                input_tokens=0,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        ),
    ]
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeStream(events))
    with patch("llm_gateway.providers.anthropic._client", return_value=client):
        deltas = [
            d
            async for d in anthropic.stream_chat(
                GatewayConfig(anthropic_api_key="k"),
                [{"role": "user", "content": "hi"}],
                tools=None,
                max_tokens=100,
            )
        ]
    usage = deltas[-1].usage_details
    assert usage is not None
    assert usage.input_tokens == 7 + 900 + 300
    assert (usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (900, 300)
    assert usage.output_tokens == 5
