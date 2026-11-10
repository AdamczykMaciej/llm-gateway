"""Base URLs are https and public by default; model ids cannot escape the
URL path. No network."""

import pytest
from pydantic import ValidationError

from llm_gateway import GatewayConfig
from llm_gateway.errors import InvalidRequestError
from llm_gateway.providers import vertex

FIELDS = ["openai_compat_base_url", "azure_endpoint", "huggingface_base_url"]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize(
    "url",
    [
        "http://llm.example.com/v1",
        "https://localhost/v1",
        "https://127.0.0.1:8443/v1",
        "https://10.1.2.3/v1",
        "https://192.168.0.9/v1",
        "https://[::1]/v1",
        "https://[fd00::1]/v1",
        "https://0.0.0.0/v1",
        "https://[::ffff:127.0.0.1]/v1",
        "https://foo.localhost/v1",
    ],
)
def test_insecure_or_local_base_urls_refused_by_default(field, url):
    with pytest.raises(ValidationError) as exc:
        GatewayConfig(**{field: url})
    assert field in str(exc.value)
    if field == "openai_compat_base_url" or url.startswith("https"):
        assert "allow_insecure_base_urls" in str(exc.value)


@pytest.mark.parametrize("field", FIELDS)
def test_opt_in_allows_local_dev(field):
    url = "https://127.0.0.1:8443/v1"
    assert GatewayConfig(**{field: url}, allow_insecure_base_urls=True)


def test_public_https_still_fine():
    cfg = GatewayConfig(openai_compat_base_url="https://api.example.com/v1/")
    assert cfg.openai_compat_base_url == "https://api.example.com/v1"
    assert not cfg.allow_insecure_base_urls


@pytest.mark.parametrize(
    "model",
    ["../x", "a/b", "m?x=1", "m#f", "m%2e%2e", "m x", "..", "a\\b", "", " ", "m\n", "x" * 201],
)
def test_vertex_model_id_cannot_escape_path(model):
    config = GatewayConfig(
        vertex_project_id="my-project1", vertex_model="claude-haiku-4-5@20251001"
    )
    if model == "":
        assert vertex._model_id(config, model) == "claude-haiku-4-5@20251001"
        return
    with pytest.raises(InvalidRequestError):
        vertex._model_id(config, model)


def test_vertex_model_id_accepts_real_ids():
    config = GatewayConfig(vertex_project_id="my-project1")
    assert vertex._model_id(config, "claude-sonnet-4-5@20250929") == "claude-sonnet-4-5@20250929"
    assert vertex._model_id(config, None) == config.vertex_model
