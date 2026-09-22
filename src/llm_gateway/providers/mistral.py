"""Mistral AI (https://api.mistral.ai/v1), an OpenAI-compatible host.

Mistral AI is a French company; its API is served from EU infrastructure.
The gateway records nothing beyond that: residency, retention and training
facts for policies come from the operator's PROVIDER_METADATA, as for every
provider (see README, "Provider policy and cost controls").

Chat Completions, streaming, tool calling and strict `json_schema` output
follow the OpenAI shapes (https://docs.mistral.ai/api/, checked 2026-09-22).
"""

from openai import AsyncOpenAI

from ..config import GatewayConfig
from .openai_compatible import OpenAICompatProvider, OpenAICompatSpec

BASE_URL = "https://api.mistral.ai/v1"

_provider = OpenAICompatProvider(
    OpenAICompatSpec(
        provider="mistral",
        api_key=lambda config: config.mistral_api_key,
        base_url=lambda config: BASE_URL,
        default_model=lambda config: config.mistral_model,
        strict_json_schema=lambda config: True,
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
