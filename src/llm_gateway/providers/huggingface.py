"""Hugging Face: the Inference Providers router or a dedicated Inference Endpoint.

Both speak the OpenAI Chat Completions API
(https://huggingface.co/docs/inference-providers/index, checked 2026-09-29),
so this is the shared OpenAI-compatible request code with two settings:

- `HUGGINGFACE_BASE_URL`, default `https://router.huggingface.co/v1`, the
  Inference Providers router; or a dedicated Inference Endpoint's base URL
  (`https://<id>.<region>.<cloud>.endpoints.huggingface.cloud/v1`, served by
  TGI or vLLM);
- `HUGGINGFACE_API_KEY` (or `HF_TOKEN`): a Hugging Face token, sent as a
  Bearer token.

GDPR: the router is a proxy that forwards each request to one of several
third-party inference providers (Together, Groq, Cerebras, ...) that Hugging
Face selects (fastest, cheapest or your preference order). Where the request is
processed and what is retained therefore varies by provider and model, and the
gateway cannot know. A dedicated Inference Endpoint runs in the cloud and region
you choose, which makes it the option whose residency you can control. Either way
the gateway asserts nothing: state residency, retention, training and DPA for
"huggingface" in PROVIDER_METADATA, or a policy requiring any of them skips it.

Model ids are operator-chosen (`HUGGINGFACE_MODEL`, no default), so nothing is
known about their capabilities. Structured output is requested in JSON mode
with the schema in the system prompt and validated locally, because strict
`json_schema` enforcement depends on the model and the backend.
"""

from openai import AsyncOpenAI

from ..config import GatewayConfig
from .openai_compatible import OpenAICompatProvider, OpenAICompatSpec

_provider = OpenAICompatProvider(
    OpenAICompatSpec(
        provider="huggingface",
        api_key=lambda config: config.huggingface_api_key,
        base_url=lambda config: config.huggingface_base_url,
        default_model=lambda config: config.huggingface_model,
        strict_json_schema=lambda config: False,
        also_configured=lambda config: bool(config.huggingface_model),
    )
)

_clients = _provider._clients


def _client(config: GatewayConfig) -> AsyncOpenAI:
    return _provider._client(config)


# Looked up by name at call time, so `patch.object(<module>, "_client", ...)`
# works exactly as for the other providers.
_provider.client_factory = lambda config: _client(config)
configured = _provider.configured
default_model = _provider.default_model
call = _provider.call
chat = _provider.chat
stream_chat = _provider.stream_chat
