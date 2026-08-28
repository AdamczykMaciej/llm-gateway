"""Configuration for the gateway, generic across any consuming application."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewayConfig(BaseSettings):
    """All gateway behavior is driven by this settings object.

    Build one from env vars with `GatewayConfig()`, or construct explicitly
    (e.g. in tests) with keyword args — nothing here reads the environment
    implicitly outside this class.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Provider credentials ─────────────────────────────────────────────
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""

    # ── Provider models ──────────────────────────────────────────────────
    claude_model: str = "claude-haiku-4-5-20251001"
    groq_model: str = "llama-3.3-70b-versatile"
    openai_model: str = "gpt-4o-mini"

    # ── Routing ───────────────────────────────────────────────────────────
    # Comma-separated provider names, tried in order. A provider is skipped
    # automatically when its API key is unset or its circuit breaker is open.
    provider_order: str = "anthropic,groq,openai"
    breaker_failure_threshold: int = 3
    breaker_cooldown_seconds: float = 60.0

    # ── Tracing ───────────────────────────────────────────────────────────
    # Prompts are never traced unless this is explicitly enabled, and even
    # then they're PII-masked first (see pii.mask_pii). Off by default.
    trace_include_prompts: bool = False
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_otlp_headers: str = ""  # comma-separated key=value pairs

    # ── HTTP service (llm_gateway.service) ──────────────────────────────
    # Comma-separated bearer keys accepted by the HTTP service. Irrelevant
    # when only the library is used in-process.
    gateway_api_keys: str = ""

    # Basic abuse guardrails for the HTTP service. Per-key, in-process (not
    # shared across Cloud Run replicas — see service/rate_limit.py). 0/negative
    # disables the corresponding check.
    rate_limit_per_minute: int = 60
    max_tokens_ceiling: int = 4000
    max_prompt_chars: int = 32000
    # Total decoded size of all data: URI images in one request. Unlike
    # max_prompt_chars (text only), image content had no ceiling at all
    # before this — a caller within their rate limit could still send
    # arbitrarily large base64 images every request. 10 MB comfortably
    # covers a real photo/screenshot while blocking abuse; both Anthropic's
    # (5 MB/image) and OpenAI's own per-image limits are well under it.
    max_image_bytes: int = 10_000_000

    # Set to false only behind a corporate SSL-inspection proxy.
    ssl_verify: bool = True

    @property
    def provider_order_list(self) -> list[str]:
        return [p.strip() for p in self.provider_order.split(",") if p.strip()]

    @property
    def gateway_api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.gateway_api_keys.split(",") if k.strip()]
