"""0.9.0: Hugging Face (Inference Providers router or a dedicated Inference
Endpoint) through the shared OpenAI-compatible code, over an in-memory transport."""

import json
from unittest.mock import AsyncMock, patch

import httpx2
import openai as openai_sdk
import pytest
from pydantic import BaseModel

from llm_gateway import (
    GatewayConfig,
    LLMError,
    PolicyViolationError,
    ProviderPolicy,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
)
from llm_gateway.capabilities import capabilities_for, missing_capabilities
from llm_gateway.errors import ErrorKind, classify
from llm_gateway.pricing import DEFAULT_PRICES, lookup_price
from llm_gateway.providers import (
    CALLS,
    CHAT_CALLS,
    CONFIGURED,
    DEFAULT_MODEL,
    STREAM_CALLS,
    huggingface,
)
from llm_gateway.providers.base import ProviderResult
from tests.test_sdk_compat import _sse

ROUTER = "https://router.huggingface.co/v1"
ENDPOINT = "https://abc123.eu-west-1.aws.endpoints.huggingface.cloud/v1"
MODEL = "openai/gpt-oss-120b:cheapest"

_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"ok": true}'},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
}


class _Wire:
    def __init__(self, respond):
        self.respond = respond
        self.requests: list[httpx2.Request] = []
        self.bodies: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content or b"{}"))
        return self.respond(request)


def _ok(body: dict = _COMPLETION):
    return lambda req: httpx2.Response(200, request=req, json=body)


def _config(**overrides) -> GatewayConfig:
    settings = dict(
        huggingface_api_key="hf_test",
        huggingface_model=MODEL,
        provider_order="huggingface",
        retry_attempts=1,
        retry_base_delay_seconds=0,
        _env_file=None,
    )
    settings.update(overrides)
    return GatewayConfig(**settings)


def _sdk_client(config: GatewayConfig, wire: _Wire) -> openai_sdk.AsyncOpenAI:
    return openai_sdk.AsyncOpenAI(
        api_key=config.huggingface_api_key,
        base_url=config.huggingface_base_url,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(wire)),
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for name in (
        "HF_TOKEN",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_MODEL",
        "HUGGINGFACE_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    huggingface._clients.clear()
    reset_circuit_breakers()
    yield
    huggingface._clients.clear()
    reset_circuit_breakers()


# ─── Registry and configuration ────────────────────────────────────────────


@pytest.mark.parametrize("table", [CALLS, CHAT_CALLS, STREAM_CALLS, CONFIGURED, DEFAULT_MODEL])
def test_huggingface_is_registered_everywhere(table):
    assert "huggingface" in table


def test_off_by_default_and_needs_both_a_token_and_a_model():
    assert not huggingface.configured(GatewayConfig(_env_file=None))
    assert not huggingface.configured(GatewayConfig(_env_file=None, huggingface_api_key="t"))
    assert not huggingface.configured(_config(huggingface_model=""))
    assert not huggingface.configured(_config(huggingface_api_key=""))
    assert huggingface.configured(_config())


def test_token_is_read_from_either_environment_variable(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_a")
    assert GatewayConfig(_env_file=None).huggingface_api_key == "hf_a"
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_b")
    assert GatewayConfig(_env_file=None).huggingface_api_key == "hf_b"


def test_defaults_and_base_url_validation(monkeypatch):
    config = GatewayConfig(_env_file=None)
    assert config.huggingface_base_url == ROUTER and config.huggingface_model == ""
    assert _config(huggingface_base_url=f" {ENDPOINT}/ ").huggingface_base_url == ENDPOINT
    with pytest.raises(ValueError, match="huggingface_base_url"):
        _config(huggingface_base_url="router.huggingface.co/v1")
    monkeypatch.setenv("HUGGINGFACE_MODEL", " meta-llama/Llama-3.3-70B-Instruct ")
    assert huggingface.default_model(GatewayConfig(_env_file=None)) == (
        "meta-llama/Llama-3.3-70B-Instruct"
    )


def test_provider_order_policy_and_metadata_accept_huggingface():
    config = _config(
        provider_order="huggingface,anthropic",
        policy_only="huggingface",
        provider_metadata={"huggingface": {"region": "eu", "retention": "zero", "dpa": True}},
    )
    assert config.provider_order_list == ["huggingface", "anthropic"]


# ─── Request shape ─────────────────────────────────────────────────────────


def test_router_is_the_default_target_and_clients_are_cached_per_base_url():
    config = _config()
    client = huggingface._client(config)
    assert str(client.base_url) == f"{ROUTER}/"
    assert client is huggingface._client(config)
    dedicated = huggingface._client(_config(huggingface_base_url=ENDPOINT))
    assert dedicated is not client and str(dedicated.base_url) == f"{ENDPOINT}/"


async def test_bearer_token_model_and_url_on_the_router():
    config = _config()
    wire = _Wire(_ok())
    with patch.object(huggingface, "_client", return_value=_sdk_client(config, wire)):
        result = await huggingface.chat(
            config, [{"role": "user", "content": "hi"}], tools=None, max_tokens=20
        )
    (request,) = wire.requests
    assert request.headers["authorization"] == "Bearer hf_test"
    assert str(request.url) == f"{ROUTER}/chat/completions"
    assert wire.bodies[-1]["model"] == MODEL  # policy suffix passed through untouched
    assert (result.input_tokens, result.output_tokens) == (11, 3)


async def test_dedicated_endpoint_uses_its_base_url_and_model():
    config = _config(huggingface_base_url=ENDPOINT, huggingface_model="tgi")
    wire = _Wire(_ok())
    with patch.object(huggingface, "_client", return_value=_sdk_client(config, wire)):
        await huggingface.chat(config, [{"role": "user", "content": "hi"}], None, 5)
    assert str(wire.requests[-1].url) == f"{ENDPOINT}/chat/completions"
    assert wire.bodies[-1]["model"] == "tgi"


async def test_tools_pass_through():
    config = _config()
    wire = _Wire(_ok())
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with patch.object(huggingface, "_client", return_value=_sdk_client(config, wire)):
        await huggingface.chat(
            config, [{"role": "user", "content": "hi"}], tools, 5, tool_choice="auto"
        )
    assert wire.bodies[-1]["tools"] == tools and wire.bodies[-1]["tool_choice"] == "auto"


class _Answer(BaseModel):
    ok: bool


async def test_structured_output_is_json_mode_with_the_schema_in_the_prompt():
    from llm_gateway.structured import OutputSchema

    config = _config()
    wire = _Wire(_ok())
    with patch.object(huggingface, "_client", return_value=_sdk_client(config, wire)):
        result = await huggingface.call(
            config, "sys", "hi", 20, output_schema=OutputSchema.from_spec(_Answer)
        )
    assert wire.bodies[-1]["response_format"] == {"type": "json_object"}
    assert "JSON" in wire.bodies[-1]["messages"][0]["content"]
    assert result.text == '{"ok": true}'


async def test_streaming():
    def chunk(delta: dict, finish=None, **extra) -> dict:
        return {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "x",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if delta else [],
            **extra,
        }

    events = [
        chunk({"role": "assistant", "content": "Hel"}),
        chunk({"content": "lo"}, "stop"),
        chunk({}, usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}),
    ]
    wire = _Wire(
        lambda req: httpx2.Response(
            200,
            request=req,
            content=_sse(events, named=False),
            headers={"content-type": "text/event-stream"},
        )
    )
    config = _config()
    with patch.object(huggingface, "_client", return_value=_sdk_client(config, wire)):
        text = "".join(
            [
                d.content or ""
                async for d in huggingface.stream_chat(
                    config, [{"role": "user", "content": "hi"}], tools=None, max_tokens=5
                )
            ]
        )
    assert text == "Hello" and wire.bodies[-1]["stream"] is True


# ─── Errors ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, ErrorKind.AUTH),
        (429, ErrorKind.RATE_LIMITED),
        (503, ErrorKind.TRANSIENT),
        (400, ErrorKind.INVALID_REQUEST),
    ],
)
async def test_http_errors_are_classified(status, kind):
    config = _config()
    wire = _Wire(lambda req: httpx2.Response(status, request=req, json={"error": "nope"}))
    with patch.object(huggingface, "_client", return_value=_sdk_client(config, wire)):
        with pytest.raises(openai_sdk.APIStatusError) as excinfo:
            await huggingface.chat(config, [{"role": "user", "content": "hi"}], None, 5)
    assert classify(excinfo.value) is kind


async def test_gateway_raises_llm_error_when_the_only_provider_fails():
    config = _config()
    wire = _Wire(lambda req: httpx2.Response(401, request=req, json={"error": "bad token"}))
    with patch.object(huggingface, "_client", return_value=_sdk_client(config, wire)):
        with pytest.raises(LLMError):
            await chat(messages=[{"role": "user", "content": "hi"}], max_tokens=5, config=config)


# ─── Capabilities, pricing, policy ─────────────────────────────────────────


def test_capabilities_only_know_streaming_and_json_mode():
    capabilities = capabilities_for("huggingface", MODEL)
    assert capabilities.streaming is True and capabilities.structured_output == "json_mode"
    assert capabilities.tools is None and capabilities.images is None
    assert missing_capabilities("huggingface", MODEL, ["tools"], require_parameters=False) == []
    assert missing_capabilities("huggingface", MODEL, ["tools"], require_parameters=True)
    assert missing_capabilities(
        "huggingface", MODEL, ["structured_output"], require_parameters=True
    )


def test_no_default_price():
    assert not any(key.startswith("huggingface/") for key in DEFAULT_PRICES)
    assert lookup_price("huggingface", MODEL) is None
    priced = _config(
        model_prices={f"huggingface/{MODEL}": {"input_per_mtok": 0.5, "output_per_mtok": 1.0}}
    )
    assert lookup_price("huggingface", MODEL, priced.model_prices).output_per_mtok == 1.0


async def test_cost_is_unknown_without_a_price_and_computed_with_one():
    calls = {"huggingface": AsyncMock(return_value=ProviderResult("ok", MODEL, 1_000_000, 0))}
    with patch("llm_gateway.router.CALLS", calls):
        result = await complete_with_usage(system="s", prompt="p", config=_config())
        assert result.provider == "huggingface" and result.cost_usd is None
        priced = _config(
            model_prices={f"huggingface/{MODEL}": {"input_per_mtok": 0.5, "output_per_mtok": 1.0}}
        )
        result = await complete_with_usage(system="s", prompt="p", config=priced)
    assert result.cost_usd == pytest.approx(0.5)


async def test_policy_fails_closed_without_metadata_and_routes_with_it():
    calls = {"huggingface": AsyncMock(return_value=ProviderResult("ok", MODEL, 1, 1))}
    with patch("llm_gateway.router.CALLS", calls):
        with pytest.raises(PolicyViolationError) as info:
            await complete_with_usage(system="s", prompt="p", config=_config(policy_residency="eu"))
        assert "huggingface" in info.value.exclusions
        calls["huggingface"].assert_not_called()

        asserted = _config(
            provider_metadata={
                "huggingface": {
                    "region": "eu",
                    "retention": "zero",
                    "trains_on_data": False,
                    "dpa": True,
                }
            }
        )
        result = await complete_with_usage(
            system="s",
            prompt="p",
            config=asserted,
            policy=ProviderPolicy(
                residency="eu", require_zero_retention=True, forbid_training=True, require_dpa=True
            ),
        )
    assert result.provider == "huggingface"


async def test_require_parameters_skips_it_for_tools_it_cannot_confirm():
    calls = {"huggingface": AsyncMock(return_value=ProviderResult("ok", MODEL, 1, 1))}
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with patch("llm_gateway.chat.CHAT_CALLS", {"huggingface": AsyncMock()}):
        with pytest.raises(LLMError):
            await chat(
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
                max_tokens=5,
                config=_config(),
                policy=ProviderPolicy(require_parameters=True),
            )
    calls["huggingface"].assert_not_called()
