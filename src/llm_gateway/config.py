"""Configuration for the gateway, generic across any consuming application."""

import json
import re
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .policy import ProviderMetadata, ProviderPolicy, validate_provider_metadata
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


def read_vertex_credentials_file(path: str) -> dict:
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
    # The request field that carries the token budget. max_completion_tokens
    # suits reasoning deployments (gpt-oss, o-series); set "max_tokens" for a
    # deployment that rejects it. Per-deployment acceptance isn't verified.
    azure_max_tokens_param: str = "max_completion_tokens"

    # ── Anthropic Claude on Google Cloud Vertex AI ───────────────────────
    # Configured when vertex_project_id and vertex_model are set. Needs the
    # `vertex` extra. See providers/vertex.py and the README.
    vertex_project_id: str = ""
    # "global", a multi-region ("eu", "us") or a region. Claude Haiku 4.5 is
    # offered regionally in europe-west1 (ML processing inside the EU) but
    # not on the eu multi-region endpoint.
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
            if "environment_id" in (info.get("credential_source") or {}):
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
            only=self.policy_only if self.policy_only.strip() else None,
            ignore=self.policy_ignore,
            require_parameters=self.policy_require_parameters,
            max_cost_usd=self.policy_max_cost_usd if self.policy_max_cost_usd > 0 else None,
            sort=self.policy_sort,
        )

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
