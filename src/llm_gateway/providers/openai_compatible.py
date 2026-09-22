"""One implementation for every host that speaks the OpenAI Chat Completions
API at its own base URL, parameterised per provider by an `OpenAICompatSpec`.

`mistral.py`, `openrouter.py` and `openai_compat.py` are thin modules that
build a spec and expose the registry surface (`configured`, `default_model`,
`call`, `chat`, `stream_chat`, `_client`, `_clients`) the way `groq.py` does,
so routing, the breaker, retries and the tests treat them like any other
provider.

Structured output: a spec says whether the host enforces `json_schema`
strictly (`strict_json_schema=True`, sent as-is) or only offers JSON mode; in
the latter case the schema is spelled out in the system prompt and
`response_format={"type": "json_object"}` is sent, and the result is validated
locally like every other provider.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field

from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from ..config import GatewayConfig
from ..structured import OutputSchema
from .base import (
    ChatResult,
    ProviderResult,
    StreamDelta,
    guard_empty_stream,
    openai_sampling_kwargs,
    parse_openai_style_chunk,
    parse_openai_style_response,
    provider_result_from_openai_style,
    sdk_client_cache_key,
    sdk_client_options,
    stream_request_options,
)


@dataclass(frozen=True)
class OpenAICompatSpec:
    """How to reach one OpenAI-compatible host."""

    provider: str
    api_key: Callable[[GatewayConfig], str]
    base_url: Callable[[GatewayConfig], str]
    default_model: Callable[[GatewayConfig], str]
    # Extra HTTP headers on every request (e.g. OpenRouter's app attribution).
    headers: Callable[[GatewayConfig], dict[str, str]] = field(default=lambda config: {})
    # Extra JSON body fields on every chat request (e.g. OpenRouter's
    # `provider` routing preference). Sent through the SDK's `extra_body`.
    extra_body: Callable[[GatewayConfig], dict] = field(default=lambda config: {})
    # Whether the host enforces `response_format={"type": "json_schema"}`; a
    # callable so an operator-configured host can say so via a setting.
    strict_json_schema: Callable[[GatewayConfig], bool] = field(default=lambda config: True)
    # A `configured` check beyond "the key is set", e.g. a base URL that must
    # also be present.
    also_configured: Callable[[GatewayConfig], bool] = field(default=lambda config: True)


class OpenAICompatProvider:
    """The registry surface for one spec. Instances are module-level
    singletons in mistral.py / openrouter.py / openai_compat.py."""

    def __init__(self, spec: OpenAICompatSpec) -> None:
        self.spec = spec
        self._clients: dict[tuple, AsyncOpenAI] = {}
        # The thin provider module sets this to its own `_client` name so the
        # tests can patch `module._client` exactly as they do for groq/openai.
        self.client_factory: Callable[[GatewayConfig], AsyncOpenAI] = self._client

    def _sdk(self, config: GatewayConfig) -> AsyncOpenAI:
        return self.client_factory(config)

    # ── Client ────────────────────────────────────────────────────────────

    def _client(self, config: GatewayConfig) -> AsyncOpenAI:
        spec = self.spec
        base_url = spec.base_url(config)
        headers = spec.headers(config)
        key = (
            *sdk_client_cache_key(spec.api_key(config), config),
            base_url,
            tuple(sorted(headers.items())),
        )
        client = self._clients.get(key)
        if client is None:
            # openai>=3.0 is httpx2-based; plain httpx clients are only a
            # temporary legacy escape hatch there, so build an httpx2 one.
            http_client = DefaultAsyncHttpx2Client(verify=False) if not config.ssl_verify else None
            client = AsyncOpenAI(
                api_key=spec.api_key(config),
                base_url=base_url,
                default_headers=headers or None,
                http_client=http_client,
                **sdk_client_options(config),
            )
            self._clients[key] = client
        return client

    # ── Registry surface ──────────────────────────────────────────────────

    def configured(self, config: GatewayConfig) -> bool:
        spec = self.spec
        return bool(spec.api_key(config)) and spec.also_configured(config)

    def default_model(self, config: GatewayConfig) -> str:
        return self.spec.default_model(config)

    def _structured_request(
        self, config: GatewayConfig, system: str, output_schema: OutputSchema
    ) -> tuple[str, dict]:
        """(system prompt, extra create() kwargs) for a structured-output call."""
        if self.spec.strict_json_schema(config):
            return system, {"response_format": output_schema.openai_response_format()}
        # JSON mode requires the word "JSON" in the messages; instructions() has it.
        system = "\n\n".join(part for part in (system, output_schema.instructions()) if part)
        return system, {"response_format": {"type": "json_object"}}

    def _extra(self, config: GatewayConfig) -> dict:
        body = self.spec.extra_body(config)
        return {"extra_body": body} if body else {}

    async def call(
        self,
        config: GatewayConfig,
        system: str,
        prompt: str,
        max_tokens: int,
        model: str | None = None,
        *,
        output_schema: OutputSchema | None = None,
        cache_system: bool = False,
    ) -> ProviderResult:
        # `cache_system` needs nothing here: hosts that cache prompts do so
        # automatically and report hits as usage.prompt_tokens_details.
        model = model or self.default_model(config)
        kwargs: dict = self._extra(config)
        if output_schema is not None:
            system, structured = self._structured_request(config, system, output_schema)
            kwargs.update(structured)
        resp = await self._sdk(config).chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
        return provider_result_from_openai_style(resp, model, provider=self.spec.provider)

    def _chat_kwargs(
        self,
        config: GatewayConfig,
        tools: list[dict] | None,
        tool_choice: object,
        sampling: dict | None,
        response_format: dict | None,
    ) -> dict:
        kwargs: dict = {**openai_sampling_kwargs(sampling), **self._extra(config)}
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    async def chat(
        self,
        config: GatewayConfig,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        *,
        model: str | None = None,
        tool_choice: object = None,
        sampling: dict | None = None,
        response_format: dict | None = None,
    ) -> ChatResult:
        model = model or self.default_model(config)
        resp = await self._sdk(config).chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            **self._chat_kwargs(config, tools, tool_choice, sampling, response_format),
        )
        return parse_openai_style_response(resp, model, provider=self.spec.provider)

    async def _stream_chat(
        self,
        config: GatewayConfig,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        *,
        model: str | None = None,
        tool_choice: object = None,
        sampling: dict | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[StreamDelta]:
        model = model or self.default_model(config)
        stream = await self._sdk(config).chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **self._chat_kwargs(config, tools, tool_choice, sampling, response_format),
            **stream_request_options(config),
        )
        async for chunk in stream:
            yield parse_openai_style_chunk(chunk, model)

    async def stream_chat(
        self,
        config: GatewayConfig,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        *,
        model: str | None = None,
        tool_choice: object = None,
        sampling: dict | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """`_stream_chat()` behind `guard_empty_stream()`: a stream with no
        text and no tool calls raises `EmptyCompletionError` before its first
        chunk."""
        deltas = self._stream_chat(
            config,
            messages,
            tools,
            max_tokens,
            model=model,
            tool_choice=tool_choice,
            sampling=sampling,
            response_format=response_format,
        )
        guarded = guard_empty_stream(
            deltas, provider=self.spec.provider, model=model or self.default_model(config)
        )
        async with aclosing(guarded):
            async for delta in guarded:
                yield delta
