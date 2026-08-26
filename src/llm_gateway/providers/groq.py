from openai import AsyncOpenAI

from ..config import GatewayConfig
from .base import ProviderResult

_clients: dict[tuple[str, bool], AsyncOpenAI] = {}


def _client(config: GatewayConfig) -> AsyncOpenAI:
    key = (config.groq_api_key, config.ssl_verify)
    client = _clients.get(key)
    if client is None:
        import httpx

        http_client = httpx.AsyncClient(verify=False) if not config.ssl_verify else None
        client = AsyncOpenAI(
            api_key=config.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            http_client=http_client,
        )
        _clients[key] = client
    return client


def configured(config: GatewayConfig) -> bool:
    return bool(config.groq_api_key)


def default_model(config: GatewayConfig) -> str:
    return config.groq_model


async def call(
    config: GatewayConfig,
    system: str,
    prompt: str,
    max_tokens: int,
    model: str | None = None,
) -> ProviderResult:
    model = model or default_model(config)
    resp = await _client(config).chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    input_tokens = resp.usage.prompt_tokens if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0
    return ProviderResult(
        text=(resp.choices[0].message.content or "").strip(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
