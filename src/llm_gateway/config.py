"""Configuration for the gateway, generic across any consuming application."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AZURE_AUTH_MODES = ("entra", "api_key")
AZURE_REASONING_EFFORTS = ("", "low", "medium", "high")


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
    # Only used when azure_auth == "api_key".
    azure_api_key: str = ""

    # ── Provider models ──────────────────────────────────────────────────
    claude_model: str = "claude-haiku-4-5-20251001"
    # Groq retired llama-3.3-70b-versatile on 2026-08-16. gpt-oss-120b is a
    # reasoning model: providers/groq.py sends it reasoning_effort="low".
    groq_model: str = "openai/gpt-oss-120b"
    openai_model: str = "gpt-4o-mini"
    # The Azure deployment name (not the model id), sent as `model`.
    azure_model: str = ""

    # ── Azure AI Foundry / Azure OpenAI (v1 API) ─────────────────────────
    # Resource endpoint: https://<resource>.openai.azure.com or
    # https://<resource>.services.ai.azure.com, with or without a trailing
    # /openai/v1; normalized to .../openai/v1/ (see providers/azure.py).
    azure_endpoint: str = ""
    # "entra": Microsoft Entra ID tokens from azure-identity (install the
    # `azure` extra), scope https://ai.azure.com/.default. "api_key": the
    # resource key in azure_api_key.
    azure_auth: str = "entra"
    # Entra only: a user-assigned managed identity's client id. Unset uses
    # DefaultAzureCredential (managed identity, workload identity, Azure CLI, ...).
    azure_managed_identity_client_id: str = ""
    # Sent as `reasoning_effort` on every azure request: "low" suits a
    # gpt-oss deployment. Set "" for a deployment of a non-reasoning model,
    # which may reject the parameter.
    azure_reasoning_effort: str = "low"

    # ── Routing ───────────────────────────────────────────────────────────
    # Comma-separated provider names, tried in order. A provider is skipped
    # automatically when its API key is unset or its circuit breaker is open.
    provider_order: str = "anthropic,groq,openai"
    breaker_failure_threshold: int = 3
    breaker_cooldown_seconds: float = 60.0

    # Short retry-with-backoff on a single provider before falling over to
    # the next one — see retry.py. Only transient errors (connection
    # errors, 5xx, overloaded) are retried; see errors.py for the full
    # classification. retry_attempts=1 means no retry; the breaker only sees
    # a failure once every attempt for that provider is exhausted, so this
    # doesn't change how quickly a genuinely-down provider trips its breaker.
    retry_attempts: int = 2
    retry_base_delay_seconds: float = 0.2

    # ── Timeouts ──────────────────────────────────────────────────────────
    # 0/negative disables the corresponding bound (the SDKs' own 600 s
    # request timeout then applies). Defaults are sized for single
    # completions of up to ~2000 output tokens, which usually finish in a
    # few seconds and occasionally take ~30 s.
    #
    # Per attempt on one provider, non-streaming: passed to the SDK client
    # and also enforced by the gateway around the whole attempt.
    request_timeout_seconds: float = 45.0
    # Streaming: the longest the SDK waits for the response to start or for
    # the next bytes of an open stream (an idle bound, not a total one).
    stream_idle_timeout_seconds: float = 30.0
    # The SDKs' own retry loop. 0 because the gateway owns retries: SDK
    # retries would multiply every gateway attempt (the previous default of
    # 2 meant up to 6 HTTP requests per provider), retry 429s and 4xx-ish
    # 408/409s blindly, sleep on retry-after headers instead of failing
    # over, and stay invisible to tracing and the circuit breaker.
    sdk_max_retries: int = 0
    # Wall-clock budget for one complete()/chat()/stream_chat() call across
    # every attempt, retry and failover — including a stream that has
    # already started. On expiry the call raises LLMDeadlineExceeded (an
    # LLMError). Default fits one hung provider (45 s) plus a full attempt
    # on the next one.
    call_deadline_seconds: float = 90.0

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

    @field_validator("azure_auth")
    @classmethod
    def _check_azure_auth(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in AZURE_AUTH_MODES:
            raise ValueError(f"azure_auth must be one of: {', '.join(AZURE_AUTH_MODES)}")
        return normalized

    @field_validator("azure_reasoning_effort")
    @classmethod
    def _check_azure_reasoning_effort(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in AZURE_REASONING_EFFORTS:
            raise ValueError('azure_reasoning_effort must be "" (omit it), low, medium or high')
        return normalized

    @property
    def provider_order_list(self) -> list[str]:
        return [p.strip() for p in self.provider_order.split(",") if p.strip()]

    @property
    def gateway_api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.gateway_api_keys.split(",") if k.strip()]
