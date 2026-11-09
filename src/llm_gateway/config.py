"""Configuration for the gateway, generic across any consuming application."""

import ipaddress
import json
import re
import socket
import unicodedata
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .policy import (
    ProviderMetadata,
    ProviderPolicy,
    normalize_region,
    provider_ids,
    validate_provider_metadata,
)
from .pricing import ModelPrice, validate_price_table

AZURE_AUTH_MODES = ("entra", "api_key")
AZURE_REASONING_EFFORTS = ("", "low", "medium", "high")
AZURE_MAX_TOKENS_PARAMS = ("max_completion_tokens", "max_tokens")

# Vertex AI locations: the global endpoint, a multi-region ("us", "eu") or a
# region such as europe-west1.
VERTEX_MULTI_REGIONS = ("us", "eu")
_VERTEX_LOCATION = re.compile(r"global|us|eu|[a-z]+-[a-z]+[0-9]+")
# Google Cloud project ids, optionally domain-scoped ("example.com:my-project").
_GCP_PROJECT_ID = re.compile(r"(?:[a-z0-9.-]+:)?[a-z][a-z0-9-]{4,28}[a-z0-9]")
_GCP_SERVICE_ACCOUNT = re.compile(r"[a-z0-9._-]+@[a-z0-9.-]+\.gserviceaccount\.com")
# Credential configuration types accepted in vertex_credentials_file. A
# "service_account" key file is refused: it is a long-lived secret.
VERTEX_CREDENTIAL_TYPES = (
    "external_account",
    "external_account_authorized_user",
    "authorized_user",
    "impersonated_service_account",
)


def read_vertex_credentials_file(path: str) -> dict[str, Any]:
    """The parsed credential configuration at `path`. Raises `ValueError`
    for an unreadable file, invalid JSON, a service-account key or another
    unsupported type. Messages never include the file's content."""
    try:
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    except OSError as e:
        raise ValueError(
            f"vertex_credentials_file {path!r} can't be read ({type(e).__name__})"
        ) from None
    except ValueError:
        raise ValueError(f"vertex_credentials_file {path!r} is not valid JSON") from None
    if not isinstance(info, dict):
        raise ValueError(f"vertex_credentials_file {path!r} must hold a JSON object")
    kind = info.get("type")
    source = info.get("source_credentials")
    if kind == "service_account" or (
        kind == "impersonated_service_account"
        and isinstance(source, dict)
        and source.get("type") == "service_account"
    ):
        raise ValueError(
            "vertex_credentials_file is a service account key, which is not accepted. Use "
            "Workload Identity Federation (an external_account credential configuration "
            "from `gcloud iam workload-identity-pools create-cred-config`) or leave it "
            "unset for Application Default Credentials."
        )
    if kind not in VERTEX_CREDENTIAL_TYPES:
        raise ValueError(
            f"vertex_credentials_file type must be one of {', '.join(VERTEX_CREDENTIAL_TYPES)}"
        )
    return info


def service_account_impersonation_url(email: str) -> str:
    return (
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
        f"{email}:generateAccessToken"
    )


def azure_app_id_uri_scope(app_id_uri: str) -> str:
    """The Microsoft Entra scope for an application ID URI."""
    return f"{app_id_uri.rstrip('/')}/.default"


_METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata.goog", "metadata"})
_METADATA_NETWORKS = (
    ipaddress.ip_network("fd00:ec2::/32"),  # AWS IMDS over IPv6
    ipaddress.ip_network("100.100.100.200/32"),  # Alibaba Cloud metadata
)
# inet_aton also reads 2852039166, 0xa9fea9fe, 0251.0376.0251.0376 and 169.254.43518.
_DOT_LOOKALIKES = str.maketrans({"\u3002": ".", "\uff0e": ".", "\uff61": "."})
_NUMERIC_HOST = re.compile(r"(0x[0-9a-f]+|\d+)(\.(0x[0-9a-f]+|\d+)){0,3}", re.IGNORECASE)


def _is_metadata_host(host: str) -> bool:
    """A cloud instance-metadata endpoint or any link-local address: a
    configured base URL there would turn the gateway into an SSRF proxy for
    instance credentials."""
    # HTTP clients map full-width digits and ideographic full stops (U+3002,
    # U+FF0E, U+FF61) to ASCII (IDNA), so "169。254。169。254" reaches
    # 169.254.169.254; judge the host the way the client will read it.
    host = unicodedata.normalize("NFKC", host).translate(_DOT_LOOKALIKES).lower().rstrip(".")
    if host in _METADATA_HOSTS:
        return True
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if _NUMERIC_HOST.fullmatch(host):
            try:
                ip = ipaddress.IPv4Address(socket.inet_aton(host))
            except (OSError, ValueError):
                ip = None
    if ip is None:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_link_local or any(
        ip in net for net in _METADATA_NETWORKS if net.version == ip.version
    )


def _check_http_url(value: str, name: str, *, https_only: bool = False) -> None:
    """An http(s) URL without embedded credentials. The message never repeats
    the value: it may carry a password. `https_only` is for endpoints that
    receive a bearer token."""
    if https_only and not value.startswith("https://"):
        raise ValueError(f"{name} must start with https:// (a token is sent to it)")
    if not value.startswith(("http://", "https://")):
        raise ValueError(f"{name} must start with http:// or https://")
    try:
        parts = urlsplit(value)
        has_userinfo = (
            parts.username is not None or parts.password is not None or "@" in (parts.netloc)
        )
    except ValueError:
        raise ValueError(f"{name} is not a valid URL") from None
    if has_userinfo:
        raise ValueError(
            f"{name} must not contain credentials (user:password@); "
            "put the key in the matching *_api_key setting"
        )
    if not parts.hostname:
        raise ValueError(f"{name} has no host")
    if _is_metadata_host(parts.hostname):
        raise ValueError(f"{name} must not point at a link-local or cloud metadata address")


class GatewayConfig(BaseSettings):
    """All gateway behavior is driven by this settings object.

    Build one from env vars with `GatewayConfig()`, or construct explicitly
    (e.g. in tests) with keyword args — nothing here reads the environment
    implicitly outside this class.
    """

    # hide_input_in_errors: a ValidationError must not repeat the rejected
    # value (a URL with a password, a malformed key) in its message.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    # ── Provider credentials ─────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", repr=False)
    groq_api_key: str = Field(default="", repr=False)
    openai_api_key: str = Field(default="", repr=False)
    # Only used when azure_auth == "api_key".
    azure_api_key: str = Field(default="", repr=False)
    # OpenAI-compatible hosts (providers/openai_compatible.py).
    mistral_api_key: str = Field(default="", repr=False)
    openrouter_api_key: str = Field(default="", repr=False)
    # The generic escape hatch: any host speaking the Chat Completions API.
    # Registered as provider id "openai_compat" once base_url and key are set;
    # a local server that needs no key takes any non-empty value.
    openai_compat_api_key: str = Field(default="", repr=False)
    openai_compat_base_url: str = ""
    # Hugging Face (providers/huggingface.py): a token for the Inference
    # Providers router or for a dedicated Inference Endpoint. Read from
    # HUGGINGFACE_API_KEY or HF_TOKEN.
    huggingface_api_key: str = Field(
        default="", repr=False, validation_alias=AliasChoices("huggingface_api_key", "hf_token")
    )

    # ── Provider models ──────────────────────────────────────────────────
    claude_model: str = "claude-haiku-4-5-20251001"
    # Retried once, before the provider chain moves on, when `claude_model` is
    # rejected as unknown or because it does not accept the request (404, or a
    # 400 about the model/a parameter). "" = no model fallback.
    claude_model_fallback: str = ""
    # Claude 5.x only (a no-op for older models). `anthropic_thinking`:
    # "" (the model's default; Haiku 5.5 thinks adaptively by default, billed
    # as output), "disabled" (Haiku 5.5 only) or "adaptive". `anthropic_effort`:
    # "" or low|medium|high|xhigh|max, sent as output_config.effort.
    anthropic_thinking: str = ""
    anthropic_effort: str = ""
    # Groq retired llama-3.3-70b-versatile on 2026-08-16. gpt-oss-120b is a
    # reasoning model: providers/groq.py sends it reasoning_effort="low".
    groq_model: str = "openai/gpt-oss-120b"
    openai_model: str = "gpt-4o-mini"
    # The Azure deployment name (not the model id), sent as `model`.
    azure_model: str = ""
    # https://docs.mistral.ai/getting-started/models/ — the small tier.
    mistral_model: str = "mistral-small-latest"
    # OpenRouter ids are "<vendor>/<model>".
    openrouter_model: str = "openai/gpt-oss-120b"
    openai_compat_model: str = ""
    # Hugging Face model ids are operator-chosen, so there is no default: the
    # provider stays unconfigured until this is set. On the router an id may
    # carry a policy or provider suffix (openai/gpt-oss-120b:cheapest).
    huggingface_model: str = ""
    # The Inference Providers router by default; set to a dedicated Inference
    # Endpoint's OpenAI-compatible base URL (https://<id>.<region>.<cloud>.endpoints
    # .huggingface.cloud/v1) for a deployment whose region you control.
    huggingface_base_url: str = "https://router.huggingface.co/v1"

    # ── OpenRouter routing and attribution ───────────────────────────────
    # Comma-separated upstream host names (https://openrouter.ai/docs/features/provider-routing);
    # when set, requests are pinned to these hosts and fail rather than route
    # elsewhere. Empty leaves routing to OpenRouter.
    openrouter_provider_allow: str = ""
    # Sent as HTTP-Referer / X-Title, OpenRouter's app attribution. Optional.
    openrouter_app_url: str = ""
    openrouter_app_name: str = ""

    # ── Generic OpenAI-compatible host capabilities ──────────────────────
    # What the operator's host supports; the gateway can't know. Tools are
    # passed through when True; structured output uses strict json_schema
    # when True and JSON mode with the schema in the prompt otherwise.
    openai_compat_supports_tools: bool = True
    openai_compat_strict_json_schema: bool = False

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
    # The request field that carries the token budget. max_completion_tokens
    # suits reasoning deployments (gpt-oss, o-series); set "max_tokens" for a
    # deployment that rejects it. Per-deployment acceptance isn't verified.
    azure_max_tokens_param: str = "max_completion_tokens"
    # Where the deployment processes prompts, asserted by the operator (the
    # endpoint alone doesn't say): "eu" for a Data Zone EU deployment (SKU
    # DataZoneStandard), "us" for the US Data Zone, "global" for GlobalStandard.
    # "" = not stated. Feeds the azure provider's `residency` (residency.py).
    azure_data_zone: str = ""
    # Alternative to azure_data_zone for a regional deployment, e.g. "swedencentral".
    azure_region: str = ""

    # ── Anthropic Claude on Google Cloud Vertex AI ───────────────────────
    # Configured when vertex_project_id and vertex_model are set. Needs the
    # `vertex` extra. See providers/vertex.py and the README.
    vertex_project_id: str = ""
    # "global", a multi-region ("eu", "us") or a region. Regional endpoints
    # serve Claude up to Sonnet 4.6 (Haiku 4.5 is in europe-west1); Claude 5.5
    # (claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5, no @date) is on
    # "global" or the "eu"/"us" multi-region endpoints
    # (https://aiplatform.eu.rep.googleapis.com), which cost 10% more.
    vertex_location: str = "europe-west1"
    vertex_model: str = "claude-haiku-4-5@20251001"
    # A credential configuration file (e.g. an external_account config for
    # Workload Identity Federation; it holds no key). Empty = Application
    # Default Credentials. Service-account key files are rejected.
    vertex_credentials_file: str = ""
    # A Google service account to impersonate, e.g. llm@proj.iam.gserviceaccount.com.
    vertex_impersonate_service_account: str = ""
    # Workload Identity Federation from an Azure managed identity: the Entra
    # application ID URI the pool provider accepts as audience. Needs an
    # external_account vertex_credentials_file and the `azure` extra.
    vertex_azure_app_id_uri: str = ""
    # With vertex_azure_app_id_uri: a user-assigned managed identity's client
    # id. Empty = the system-assigned identity.
    vertex_azure_managed_identity_client_id: str = ""
    # Send output_schema calls (Vertex `output_config` structured outputs) to
    # vertex. Off by default: organizations deny the feature until the policy
    # constraints/vertexai.allowedPartnerModelFeatures allows
    # publishers/anthropic/models/<model>:structured_outputs, and a denied call
    # is a 400. While off, routing skips vertex for those calls.
    vertex_structured_outputs: bool = False

    # ── Gemini on Google Cloud Vertex AI ─────────────────────────────────
    # Configured when gemini_project_id is set (deliberately not defaulted from
    # vertex_project_id, so setting up vertex never turns gemini on). It uses
    # the vertex_* credential settings (ADC, vertex_credentials_file, Workload
    # Identity Federation, impersonation) and the `vertex` extra. See
    # providers/gemini.py and the README.
    gemini_project_id: str = ""
    # "global", a multi-region ("eu", "us") or a region; which models are
    # served where is in the README.
    gemini_location: str = "eu"
    # Vertex's OpenAI-compatible endpoint names Gemini models google/<model>.
    gemini_model: str = "google/gemini-3.5-flash"

    # ── Routing ───────────────────────────────────────────────────────────
    # Comma-separated provider names, tried in order. A provider is skipped
    # automatically when its API key is unset or its circuit breaker is open.
    provider_order: str = "anthropic,groq,openai"
    breaker_failure_threshold: int = 3
    breaker_cooldown_seconds: float = 60.0
    # Longer cooldowns for conditions that do not heal on their own: an empty
    # balance (billing) and a retired model. 0 = use breaker_cooldown_seconds.
    breaker_billing_cooldown_seconds: float = 300.0
    breaker_model_missing_cooldown_seconds: float = 1800.0

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
    # stream_chat() only: fail an attempt that sends no first chunk within this
    # many seconds as a provider TimeoutError, so the chain moves to the next
    # provider (safe: nothing was yielded yet). None (default) adds no bound.
    # It only tightens the pre-first-chunk wait: that wait is already capped by
    # stream_idle_timeout_seconds and call_deadline_seconds (whichever is
    # smallest wins), and once a chunk arrived only the idle bound and the
    # deadline apply. A per-call `first_token_timeout_seconds=` wins.
    first_token_timeout_seconds: float | None = None
    # stream_chat() only: when a provider fails AFTER chunks were yielded
    # (transport error, idle timeout, provider 5xx), move to the next provider
    # and yield a `StreamDelta(restart=True)` marker first; the consumer must
    # discard everything received so far. Off by default (a mid-stream failure
    # then ends the stream with the error, as before). Per-call
    # `restart_on_failure=` wins. `stream_max_restarts` bounds the restarts.
    stream_restart_on_failure: bool = False
    stream_max_restarts: int = 1
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

    # Tool results (role "tool" messages) longer than this many characters are
    # truncated, with a marker, before chat()/stream_chat() send them on, so
    # one giant tool output can't blow the context window or the bill. 0 (the
    # default) disables the guard. A per-call `max_tool_result_chars` wins.
    tool_result_max_chars: int = 0

    # Streaming size caps (characters). A hostile or buggy provider could
    # stream unbounded text or tool-call arguments into the caller's memory;
    # past either cap stream_chat() ends with StreamTooLargeError (an
    # InvalidOutputError), never a silent truncation. 0 disables a cap.
    # Per tool call's arguments:
    stream_max_tool_arguments_chars: int = 1_000_000
    # Whole stream (text plus all tool-call arguments):
    stream_max_total_chars: int = 16_000_000

    # ── Tracing ───────────────────────────────────────────────────────────
    # Prompts are never traced unless this is explicitly enabled, and even
    # then they're PII-masked first (see pii.mask_pii). Off by default.
    trace_include_prompts: bool = False
    otel_exporter_otlp_endpoint: str = Field(default="", repr=False)
    # Comma-separated key=value pairs; usually carries a token, so never in repr.
    otel_exporter_otlp_headers: str = Field(default="", repr=False)

    # ── HTTP service (llm_gateway.service) ──────────────────────────────
    # Comma-separated bearer keys accepted by the HTTP service. Irrelevant
    # when only the library is used in-process.
    gateway_api_keys: str = Field(default="", repr=False)

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
    # Ceiling on the raw HTTP request body, enforced before it is parsed (a
    # caller can no longer make the service buffer an arbitrarily large JSON
    # document that the later max_prompt_chars check would only reject after
    # the fact). Default covers max_image_bytes as base64 plus text. 0 disables.
    max_request_bytes: int = 16_777_216

    # SSE resilience (service/app.py). A client that stops reading must not pin
    # an upstream provider stream: a single write blocked longer than
    # stream_write_timeout_seconds closes the stream (and the upstream request).
    # max_concurrent_streams bounds open streams per process; over it, new
    # streaming requests get 503. stream_heartbeat_seconds sends an SSE comment
    # when the stream is idle that long, so proxies keep the connection.
    # shutdown_grace_seconds is how long lifespan shutdown lets in-flight
    # non-streaming calls finish before open streams get a final error event.
    # 0 disables each of the four.
    stream_write_timeout_seconds: float = 30.0
    max_concurrent_streams: int = 256
    stream_heartbeat_seconds: float = 15.0
    shutdown_grace_seconds: float = 20.0

    # ── Provider policy and cost controls (policy.py, routing.py) ────────
    # Operator-asserted compliance facts per provider id, as JSON:
    #   PROVIDER_METADATA='{"anthropic": {"region": "us", "retention": "zero",
    #                       "trains_on_data": false, "dpa": true}}'
    # A provider without an entry gets the conservative defaults (region and
    # retention "unknown", trains_on_data true, dpa false), which fail every
    # requirement below. The library never asserts these facts for a vendor.
    provider_metadata: dict[str, ProviderMetadata] = {}
    # The global policy. A per-call `policy=` (or the HTTP request's
    # `policy` object) can only narrow it. All off by default.
    policy_residency: str = ""
    # Strict EU mode: when set (e.g. "eu"), routing refuses to call any provider
    # whose `residency` (residency.py, derived from its configuration) differs,
    # and an "unknown" residency is never allowed. Raises ResidencyViolationError
    # before any network call and never falls through to another provider.
    # None = off (the default; behaviour unchanged).
    require_residency: str | None = None
    policy_require_zero_retention: bool = False
    policy_forbid_training: bool = False
    policy_require_dpa: bool = False
    policy_only: str = ""  # comma-separated provider ids; empty = no restriction
    policy_ignore: str = ""  # comma-separated provider ids
    policy_require_parameters: bool = False
    policy_max_cost_usd: float = 0.0  # estimated worst case per call; 0 disables
    policy_sort: Literal["order", "price"] = "order"
    # USD per 1M tokens by "provider/model", merged over pricing.DEFAULT_PRICES:
    #   MODEL_PRICES='{"groq/openai/gpt-oss-120b": {"input_per_mtok": 0.15,
    #                  "output_per_mtok": 0.6}}'
    model_prices: dict[str, ModelPrice] = {}

    # Per-attempt usage/cost hook (attempts.py): called with an `AttemptReport`
    # once per provider attempt, sync or async; exceptions are logged and
    # ignored. A per-call `on_attempt=` replaces it. Code-only: not read from
    # the environment.
    on_attempt: Callable[[Any], Any] | None = Field(default=None, exclude=True, repr=False)

    # Set to false only behind a corporate SSL-inspection proxy.
    ssl_verify: bool = True

    @field_validator("anthropic_thinking")
    @classmethod
    def _check_anthropic_thinking(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ("", "disabled", "adaptive"):
            raise ValueError("anthropic_thinking must be one of: '', disabled, adaptive")
        return normalized

    @field_validator("anthropic_effort")
    @classmethod
    def _check_anthropic_effort(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ("", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("anthropic_effort must be one of: '', low, medium, high, xhigh, max")
        return normalized

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

    @field_validator("azure_data_zone", "azure_region")
    @classmethod
    def _check_azure_residency_hint(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized and not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", normalized):
            raise ValueError("must be a short lowercase code such as 'eu', 'us' or 'swedencentral'")
        return normalized

    @field_validator("require_residency")
    @classmethod
    def _check_require_residency(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return normalize_region(value, field_name="require_residency")

    @field_validator("azure_max_tokens_param")
    @classmethod
    def _check_azure_max_tokens_param(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in AZURE_MAX_TOKENS_PARAMS:
            raise ValueError(
                f"azure_max_tokens_param must be one of: {', '.join(AZURE_MAX_TOKENS_PARAMS)}"
            )
        return normalized

    @field_validator(
        "vertex_project_id",
        "vertex_model",
        "vertex_credentials_file",
        "vertex_impersonate_service_account",
        "vertex_azure_app_id_uri",
        "vertex_azure_managed_identity_client_id",
    )
    @classmethod
    def _strip_vertex_setting(cls, value: str) -> str:
        return value.strip()

    @field_validator("vertex_project_id")
    @classmethod
    def _check_vertex_project_id(cls, value: str) -> str:
        if value and not _GCP_PROJECT_ID.fullmatch(value):
            raise ValueError("vertex_project_id must be a Google Cloud project id, e.g. my-project")
        return value

    @field_validator("vertex_location")
    @classmethod
    def _check_vertex_location(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _VERTEX_LOCATION.fullmatch(normalized):
            raise ValueError(
                "vertex_location must be 'global', a multi-region ('eu', 'us') or a region "
                "such as 'europe-west1'"
            )
        return normalized

    @field_validator("gemini_project_id", "gemini_model", "huggingface_model")
    @classmethod
    def _strip_setting(cls, value: str) -> str:
        return value.strip()

    @field_validator("gemini_project_id")
    @classmethod
    def _check_gemini_project_id(cls, value: str) -> str:
        if value and not _GCP_PROJECT_ID.fullmatch(value):
            raise ValueError("gemini_project_id must be a Google Cloud project id, e.g. my-project")
        return value

    @field_validator("gemini_location")
    @classmethod
    def _check_gemini_location(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _VERTEX_LOCATION.fullmatch(normalized):
            raise ValueError(
                "gemini_location must be 'global', a multi-region ('eu', 'us') or a region "
                "such as 'europe-west4'"
            )
        return normalized

    @field_validator("vertex_impersonate_service_account")
    @classmethod
    def _check_vertex_impersonation(cls, value: str) -> str:
        if value and not _GCP_SERVICE_ACCOUNT.fullmatch(value.lower()):
            raise ValueError(
                "vertex_impersonate_service_account must be a service account email "
                "(...@<project>.iam.gserviceaccount.com)"
            )
        return value

    @model_validator(mode="after")
    def _check_vertex_auth(self) -> "GatewayConfig":
        # Fail at startup, not on the first call, for unusable Vertex auth settings.
        info = (
            read_vertex_credentials_file(self.vertex_credentials_file)
            if self.vertex_credentials_file
            else None
        )
        kind = info.get("type") if info else None
        if self.vertex_azure_app_id_uri:
            if kind != "external_account":
                raise ValueError(
                    "vertex_azure_app_id_uri needs vertex_credentials_file to be an "
                    "external_account credential configuration"
                )
            if "environment_id" in ((info or {}).get("credential_source") or {}):
                raise ValueError(
                    "vertex_azure_app_id_uri can't be used with an AWS credential configuration"
                )
        if self.vertex_azure_managed_identity_client_id and not self.vertex_azure_app_id_uri:
            raise ValueError(
                "vertex_azure_managed_identity_client_id needs vertex_azure_app_id_uri"
            )
        impersonate = self.vertex_impersonate_service_account
        if impersonate and info:
            if kind == "impersonated_service_account":
                raise ValueError(
                    "vertex_impersonate_service_account can't be combined with an "
                    "impersonated_service_account credential file, which already impersonates"
                )
            existing = info.get("service_account_impersonation_url")
            if existing and existing != service_account_impersonation_url(impersonate):
                raise ValueError(
                    "vertex_impersonate_service_account differs from the service account "
                    "in vertex_credentials_file's service_account_impersonation_url"
                )
        return self

    @property
    def provider_order_list(self) -> list[str]:
        return [p.strip() for p in self.provider_order.split(",") if p.strip()]

    @property
    def openrouter_provider_allow_list(self) -> tuple[str, ...]:
        return tuple(p.strip() for p in self.openrouter_provider_allow.split(",") if p.strip())

    @property
    def secret_values(self) -> tuple[str, ...]:
        """Every configured credential, for scrubbing error text (see
        errors.redact_secrets)."""
        values = [
            self.anthropic_api_key,
            self.groq_api_key,
            self.openai_api_key,
            self.azure_api_key,
            self.mistral_api_key,
            self.openrouter_api_key,
            self.openai_compat_api_key,
            self.huggingface_api_key,
            *self.gateway_api_keys_list,
            *(
                pair.partition("=")[2].strip().removeprefix("Bearer ").strip()
                for pair in self.otel_exporter_otlp_headers.split(",")
            ),
        ]
        return tuple(v for v in values if len(v) >= 4)

    @property
    def gateway_api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.gateway_api_keys.split(",") if k.strip()]

    @property
    def policy(self) -> ProviderPolicy:
        """The global routing policy built from the `policy_*` settings."""
        return ProviderPolicy(
            residency=self.policy_residency.strip() or None,
            require_zero_retention=self.policy_require_zero_retention,
            forbid_training=self.policy_forbid_training,
            require_dpa=self.policy_require_dpa,
            only=(
                provider_ids(self.policy_only, field_name="only")
                if self.policy_only.strip()
                else None
            ),
            ignore=provider_ids(self.policy_ignore, field_name="ignore"),
            require_parameters=self.policy_require_parameters,
            max_cost_usd=self.policy_max_cost_usd if self.policy_max_cost_usd > 0 else None,
            sort=self.policy_sort,
        )

    @field_validator("openai_compat_base_url", "azure_endpoint")
    @classmethod
    def _check_optional_url(cls, value: str, info: ValidationInfo) -> str:
        value = value.strip()
        if value:
            _check_http_url(
                value, info.field_name or "url", https_only=info.field_name == "azure_endpoint"
            )
        return value.rstrip("/") if info.field_name == "openai_compat_base_url" else value

    @field_validator("huggingface_base_url")
    @classmethod
    def _check_huggingface_base_url(cls, value: str) -> str:
        value = value.strip()
        _check_http_url(value, "huggingface_base_url", https_only=True)
        return value.rstrip("/")

    @field_validator("provider_metadata")
    @classmethod
    def _check_provider_metadata(
        cls, value: dict[str, ProviderMetadata]
    ) -> dict[str, ProviderMetadata]:
        return validate_provider_metadata(value)

    @field_validator("model_prices")
    @classmethod
    def _check_model_prices(cls, value: dict[str, ModelPrice]) -> dict[str, ModelPrice]:
        return validate_price_table(value)

    @model_validator(mode="after")
    def _check_policy(self) -> "GatewayConfig":
        # Fail at startup, not on the first call, for a bad POLICY_* value.
        try:
            _ = self.policy
        except ValueError as exc:
            raise ValueError(f"invalid POLICY_* setting: {exc}") from exc
        return self
