"""0.11.0: Claude on Vertex through the `eu` multi-region endpoint and the
Claude 5.5 model ids, with fake transports only (no network, no credentials).

Docs (checked 2026-10-19): platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
- multi-region host `https://aiplatform.eu.rep.googleapis.com`, location `eu` in the path;
- Claude 5.5 ids carry no `@date`: claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5;
- regional endpoints stop at Sonnet 4.6, newer models use `global` or `eu`/`us`.
"""

import json

import anthropic as anthropic_sdk
import httpx2
import pytest

from llm_gateway import GatewayConfig, chat, stream_chat
from llm_gateway.providers import vertex
from tests.test_vertex_provider import (
    MESSAGES,
    PROJECT,
    TOKEN,
    VERTEX_TEXT,
    _config,
    _isolate,  # noqa: F401 (autouse fixture)
    _message,
    _providers,
    _sse_response,
    _stream_events,
    _Wire,
)

EU_HOST = "https://aiplatform.eu.rep.googleapis.com/v1"


def _ok(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, request=request, json=_message())


def _path(location: str, model: str, verb: str) -> str:
    return f"/v1/projects/{PROJECT}/locations/{location}/publishers/anthropic/models/{model}:{verb}"


def test_eu_config_is_accepted_and_builds_the_multi_region_host():
    config = _config(vertex_location="EU", vertex_model="claude-haiku-5-5")
    assert config.vertex_location == "eu"
    assert config.vertex_model == "claude-haiku-5-5"
    assert vertex.base_url("eu") == EU_HOST
    assert vertex.base_url("us") == "https://aiplatform.us.rep.googleapis.com/v1"


async def test_real_client_for_eu_points_at_the_rep_host():
    config = _config(vertex_location="eu", vertex_model="claude-haiku-5-5")

    # Fake credentials: no ADC lookup and no network.
    async def fake_credentials(_config):
        return object()

    vertex._clients.clear()
    original = vertex._google_credentials
    vertex._google_credentials = fake_credentials
    try:
        client = await vertex._client(config)
    finally:
        vertex._google_credentials = original
    assert isinstance(client, anthropic_sdk.AsyncAnthropicVertex)
    assert client.region == "eu"
    assert str(client.base_url) == EU_HOST + "/"


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"])
async def test_eu_request_path_and_host_for_claude_5_5_ids(model):
    wire = _Wire(_ok)
    config = _config(vertex_location="eu", vertex_model=model)
    with _providers(vertex_wire=wire, location="eu"):
        result = await chat(messages=MESSAGES, config=config)

    assert result.content == VERTEX_TEXT
    request = wire.requests[0]
    assert request.url.host == "aiplatform.eu.rep.googleapis.com"
    assert request.url.path == _path("eu", model, "rawPredict")
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body = json.loads(request.content)
    assert body["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in body


async def test_eu_streaming_path_uses_stream_raw_predict_on_the_rep_host():
    wire = _Wire(lambda request: _sse_response(request, _stream_events(), named=True))
    config = _config(vertex_location="eu", vertex_model="claude-haiku-5-5")
    with _providers(vertex_wire=wire, location="eu"):
        deltas = [d async for d in stream_chat(messages=MESSAGES, config=config)]

    request = wire.requests[0]
    assert request.url.host == "aiplatform.eu.rep.googleapis.com"
    assert request.url.path == _path("eu", "claude-haiku-5-5", "streamRawPredict")
    assert json.loads(request.content)["stream"] is True
    assert "".join(d.content or "" for d in deltas) == VERTEX_TEXT


@pytest.mark.parametrize(
    "location, host",
    [
        ("global", "aiplatform.googleapis.com"),
        ("us", "aiplatform.us.rep.googleapis.com"),
        ("europe-west1", "europe-west1-aiplatform.googleapis.com"),
    ],
)
async def test_global_us_and_regional_locations_are_unchanged(location, host):
    wire = _Wire(_ok)
    with _providers(vertex_wire=wire, location=location):
        await chat(messages=MESSAGES, config=_config(vertex_location=location))
    request = wire.requests[0]
    assert request.url.host == host
    assert request.url.path == _path(location, "claude-haiku-4-5@20251001", "rawPredict")


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"])
async def test_5_5_ids_drop_sampling_and_take_effort_over_vertex(model):
    wire = _Wire(_ok)
    config = _config(
        vertex_location="eu",
        vertex_model=model,
        anthropic_thinking="disabled",
        anthropic_effort="low",
    )
    sampling = {"temperature": 0.2, "top_p": 0.5, "stop": ["END"]}
    with _providers(vertex_wire=wire, location="eu"):
        await chat(messages=MESSAGES, config=config, sampling=sampling)

    body = json.loads(wire.requests[0].content)
    assert "temperature" not in body and "top_p" not in body
    assert body["stop_sequences"] == ["END"]
    assert body["output_config"] == {"effort": "low"}
    # `thinking: disabled` is only valid on Haiku.
    if model == "claude-haiku-5-5":
        assert body["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in body


def test_sampling_is_kept_for_older_models_over_vertex():
    from llm_gateway.providers.anthropic import _REJECTS_SAMPLING

    assert not _REJECTS_SAMPLING.match("claude-haiku-4-5@20251001")
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"):
        assert _REJECTS_SAMPLING.match(model)


def test_5_5_ids_have_vertex_capabilities():
    from llm_gateway.capabilities import capabilities_for

    for model in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"):
        caps = capabilities_for("vertex", model, vertex_structured_outputs=True)
        assert (caps.tools, caps.images, caps.streaming) == (True, True, True)
        assert caps.structured_output == "strict"
        off = capabilities_for("vertex", model, vertex_structured_outputs=False)
        assert off.structured_output == "unsupported"


def test_unused_import_guard():
    assert GatewayConfig
