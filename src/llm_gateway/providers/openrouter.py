"""OpenRouter (https://openrouter.ai/api/v1), an OpenAI-compatible broker.

OpenRouter is a US company that forwards each request to one of many
third-party hosts, chosen by its own routing unless the request narrows it.
Where the data goes, and whether the host retains it or trains on it, is
therefore the caller's responsibility: set `openrouter_provider_allow` to
pin the upstream hosts, use OpenRouter's own data-policy settings, and assert
only what you have verified in PROVIDER_METADATA.

Model ids are `<vendor>/<model>` (e.g. `openai/gpt-oss-120b`). The optional
`HTTP-Referer` / `X-Title` headers are OpenRouter's app attribution
(https://openrouter.ai/docs/api-reference/overview, checked 2026-09-22).
Strict `json_schema` support depends on the upstream host, so structured
output is requested in JSON mode with the schema in the system prompt and
validated locally.
"""

from openai import AsyncOpenAI

from ..config import GatewayConfig
from .openai_compatible import OpenAICompatProvider, OpenAICompatSpec

BASE_URL = "https://openrouter.ai/api/v1"


def _headers(config: GatewayConfig) -> dict[str, str]:
    headers = {}
    if config.openrouter_app_url:
        headers["HTTP-Referer"] = config.openrouter_app_url
    if config.openrouter_app_name:
        headers["X-Title"] = config.openrouter_app_name
    return headers


def _extra_body(config: GatewayConfig) -> dict:
    allow = config.openrouter_provider_allow_list
    if not allow:
        return {}
    # https://openrouter.ai/docs/features/provider-routing: `only` restricts
    # routing to these hosts; `allow_fallbacks` off means fail rather than
    # route elsewhere when none of them can serve the request.
    return {"provider": {"only": list(allow), "allow_fallbacks": False}}


_provider = OpenAICompatProvider(
    OpenAICompatSpec(
        provider="openrouter",
        api_key=lambda config: config.openrouter_api_key,
        base_url=lambda config: BASE_URL,
        default_model=lambda config: config.openrouter_model,
        headers=_headers,
        extra_body=_extra_body,
        strict_json_schema=lambda config: False,
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
