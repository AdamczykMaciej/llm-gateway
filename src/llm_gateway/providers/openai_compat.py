"""A generic OpenAI-compatible host chosen by the operator: DeepSeek,
Together, Fireworks, a self-hosted vLLM, and so on.

Registered as provider id `openai_compat` and usable once
`OPENAI_COMPAT_BASE_URL` and `OPENAI_COMPAT_API_KEY` are set (a local server
that needs no key takes any non-empty value). The gateway knows nothing
about the host: its model's capabilities are unknown to capabilities.py
unless `openai_compat_supports_tools` / `openai_compat_strict_json_schema`
say otherwise, and its residency, retention and training facts come from
PROVIDER_METADATA only.
"""

from openai import AsyncOpenAI

from ..config import GatewayConfig
from .openai_compatible import OpenAICompatProvider, OpenAICompatSpec

_provider = OpenAICompatProvider(
    OpenAICompatSpec(
        provider="openai_compat",
        api_key=lambda config: config.openai_compat_api_key,
        base_url=lambda config: config.openai_compat_base_url,
        default_model=lambda config: config.openai_compat_model,
        strict_json_schema=lambda config: config.openai_compat_strict_json_schema,
        also_configured=lambda config: bool(config.openai_compat_base_url),
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
