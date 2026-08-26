from anthropic import AsyncAnthropic

from ..config import GatewayConfig
from .base import ProviderResult

_clients: dict[tuple[str, bool], AsyncAnthropic] = {}


def _client(config: GatewayConfig) -> AsyncAnthropic:
    key = (config.anthropic_api_key, config.ssl_verify)
    client = _clients.get(key)
    if client is None:
        import httpx

        http_client = httpx.AsyncClient(verify=False) if not config.ssl_verify else None
        client = AsyncAnthropic(api_key=config.anthropic_api_key, http_client=http_client)
        _clients[key] = client
    return client


def configured(config: GatewayConfig) -> bool:
    return bool(config.anthropic_api_key)


async def call(config: GatewayConfig, system: str, prompt: str, max_tokens: int) -> ProviderResult:
    resp = await _client(config).messages.create(
        model=config.claude_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    input_tokens = resp.usage.input_tokens if resp.usage else 0
    output_tokens = resp.usage.output_tokens if resp.usage else 0
    return ProviderResult(
        text=resp.content[0].text.strip(),
        model=config.claude_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
