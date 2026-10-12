"""Claude 5.x request safety and the one-shot model fallback, at the SDK-call
boundary (fake client, no network).

Facts encoded here come from the Haiku 5.5 migration guide
(platform.claude.com/docs/en/models/haiku-5-5/migration-guide, checked
2026-10-12): temperature/top_p/top_k other than the defaults are a 400; thinking
is on by default and `thinking: {type: "disabled"}` is accepted at effort
`high` or below; effort goes in `output_config.effort`."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic import BadRequestError, NotFoundError

from llm_gateway import GatewayConfig
from llm_gateway.providers import anthropic

pytestmark = pytest.mark.asyncio

MSG = [{"role": "user", "content": "hi"}]


def _config(**overrides) -> GatewayConfig:
    return GatewayConfig(anthropic_api_key="test-key", **overrides)


def _ok(text: str = "hi"):
    resp = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=1, output_tokens=1)
    return resp


def _status_error(cls, status: int, message: str):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body={"error": {"message": message}})


def _client(create):
    client = MagicMock()
    client.messages.create = create
    return client


async def test_sampling_dropped_for_haiku_5_5_but_kept_for_haiku_4_5():
    create = AsyncMock(return_value=_ok())
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        await anthropic.chat(
            _config(claude_model="claude-haiku-5-5"),
            MSG,
            None,
            100,
            sampling={"temperature": 0.2, "top_p": 0.5, "stop": "STOP"},
        )
        sent = create.call_args.kwargs
        assert "extra_body" not in sent
        assert sent["stop_sequences"] == ["STOP"]

        await anthropic.chat(
            _config(),
            MSG,
            None,
            100,
            sampling={"temperature": 0.2},
        )
        assert create.call_args.kwargs["extra_body"] == {"temperature": 0.2}


async def test_default_sends_neither_thinking_nor_effort():
    create = AsyncMock(return_value=_ok())
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        await anthropic.chat(_config(claude_model="claude-haiku-5-5"), MSG, None, 100)
    sent = create.call_args.kwargs
    assert "output_config" not in sent
    assert "extra_body" not in sent


async def test_thinking_disabled_and_effort_low_for_haiku_5_5():
    create = AsyncMock(return_value=_ok())
    cfg = _config(
        claude_model="claude-haiku-5-5", anthropic_thinking="disabled", anthropic_effort="low"
    )
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        await anthropic.chat(cfg, MSG, None, 100)
    sent = create.call_args.kwargs
    assert sent["output_config"] == {"effort": "low"}
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_disabled_thinking_is_not_sent_with_xhigh_effort():
    create = AsyncMock(return_value=_ok())
    cfg = _config(
        claude_model="claude-haiku-5-5", anthropic_thinking="disabled", anthropic_effort="xhigh"
    )
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        await anthropic.chat(cfg, MSG, None, 100)
    sent = create.call_args.kwargs
    assert sent["output_config"] == {"effort": "xhigh"}
    assert "extra_body" not in sent


async def test_thinking_and_effort_are_ignored_for_haiku_4_5():
    create = AsyncMock(return_value=_ok())
    cfg = _config(anthropic_thinking="disabled", anthropic_effort="low")
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        await anthropic.chat(cfg, MSG, None, 100)
    sent = create.call_args.kwargs
    assert "output_config" not in sent
    assert "extra_body" not in sent


async def test_fallback_retries_once_on_model_not_found(caplog):
    create = AsyncMock(
        side_effect=[
            _status_error(NotFoundError, 404, "model: claude-haiku-5-5"),
            _ok("from fallback"),
        ]
    )
    cfg = _config(
        claude_model="claude-haiku-5-5",
        claude_model_fallback="claude-haiku-4-5-20251001",
        anthropic_thinking="disabled",
        anthropic_effort="low",
    )
    with (
        patch("llm_gateway.providers.anthropic._client", return_value=_client(create)),
        caplog.at_level("WARNING", logger="llm_gateway.providers.anthropic"),
    ):
        result = await anthropic.chat(cfg, MSG, None, 100)
    assert result.content == "from fallback"
    assert result.model == "claude-haiku-4-5-20251001"
    assert [c.kwargs["model"] for c in create.call_args_list] == [
        "claude-haiku-5-5",
        "claude-haiku-4-5-20251001",
    ]
    # The fallback is an older model: the 5.x-only parameters are not sent.
    assert "output_config" not in create.call_args_list[1].kwargs
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "claude-haiku-5-5" in warnings[0].getMessage()


async def test_fallback_on_unsupported_parameter_400():
    create = AsyncMock(
        side_effect=[
            _status_error(BadRequestError, 400, "`temperature` is not supported for this model"),
            _ok(),
        ]
    )
    cfg = _config(claude_model="claude-haiku-5-5", claude_model_fallback="claude-haiku-4-5")
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        await anthropic.chat(cfg, MSG, None, 100)
    assert create.await_count == 2


async def test_no_fallback_for_an_unrelated_400():
    err = _status_error(BadRequestError, 400, "prompt is too long: 250000 tokens > 200000 maximum")
    create = AsyncMock(side_effect=[err])
    cfg = _config(claude_model="claude-haiku-5-5", claude_model_fallback="claude-haiku-4-5")
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        with pytest.raises(BadRequestError):
            await anthropic.chat(cfg, MSG, None, 100)
    assert create.await_count == 1


async def test_fallback_is_attempted_only_once():
    create = AsyncMock(
        side_effect=[
            _status_error(NotFoundError, 404, "model: a"),
            _status_error(NotFoundError, 404, "model: b"),
        ]
    )
    cfg = _config(claude_model="claude-haiku-5-5", claude_model_fallback="claude-haiku-4-5")
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        with pytest.raises(NotFoundError):
            await anthropic.chat(cfg, MSG, None, 100)
    assert create.await_count == 2


async def test_no_fallback_when_unset_or_model_given_explicitly():
    create = AsyncMock(side_effect=_status_error(NotFoundError, 404, "model: x"))
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        with pytest.raises(NotFoundError):
            await anthropic.chat(_config(claude_model="claude-haiku-5-5"), MSG, None, 100)
        assert create.await_count == 1
        cfg = _config(claude_model="claude-haiku-5-5", claude_model_fallback="claude-haiku-4-5")
        with pytest.raises(NotFoundError):
            await anthropic.chat(cfg, MSG, None, 100, model="claude-sonnet-4-6")
        assert create.await_count == 2


async def test_call_path_applies_effort_inside_output_config_and_falls_back():
    create = AsyncMock(
        side_effect=[_status_error(NotFoundError, 404, "model: claude-haiku-5-5"), _ok("fb")]
    )
    cfg = _config(
        claude_model="claude-haiku-5-5",
        claude_model_fallback="claude-haiku-4-5",
        anthropic_effort="low",
    )
    with patch("llm_gateway.providers.anthropic._client", return_value=_client(create)):
        result = await anthropic.call(cfg, "sys", "prompt", 100)
    assert result.text == "fb"
    assert create.call_args_list[0].kwargs["output_config"] == {"effort": "low"}
    assert "output_config" not in create.call_args_list[1].kwargs


async def test_config_rejects_bad_thinking_and_effort():
    with pytest.raises(ValueError):
        _config(anthropic_thinking="enabled")
    with pytest.raises(ValueError):
        _config(anthropic_effort="turbo")
