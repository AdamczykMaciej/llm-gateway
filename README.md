# llm-gateway

A small, self-hosted multi-provider LLM gateway: a fallback chain across
Anthropic / Groq / OpenAI, a per-provider circuit breaker, PII-masked OTel
tracing, and an OpenAI-compatible HTTP API — so any app can point at one
provider and quietly keep working when that provider is down, rate-limited,
or missing a key.

Extracted from the LLM-router hardening work in the
[InterviewAI](https://github.com/AdamczykMaciej/interviewer) backend, and
generalized to have zero product-specific coupling.

Two ways to use it:

## 1. As a Python library (in-process, no network hop)

```bash
pip install "llm-gateway @ git+https://github.com/AdamczykMaciej/llm-gateway.git"
```

```python
from llm_gateway import complete, GatewayConfig

config = GatewayConfig(
    anthropic_api_key="sk-ant-...",
    groq_api_key="gsk-...",  # optional fallback
    openai_api_key="sk-...",  # optional fallback
)

text = await complete(system="You are a helpful assistant.", prompt="Hi!", config=config)
```

Providers are tried in `config.provider_order` (default
`"anthropic,groq,openai"`); a provider is skipped automatically when its key
is unset or its circuit breaker is open after repeated recent failures.
`GatewayConfig()` with no args reads from environment variables / a `.env`
file (see `.env.example`).

## 2. As an HTTP service (OpenAI-compatible)

```bash
pip install "llm-gateway[service] @ git+https://github.com/AdamczykMaciej/llm-gateway.git"
llm-gateway   # serves on :8080
```

or with Docker: `docker build -t llm-gateway . && docker run -p 8080:8080 --env-file .env llm-gateway`.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi!"}]}'
```

Response is OpenAI-shaped, so any OpenAI-SDK-compatible client works by just
changing `base_url`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key=GATEWAY_API_KEY)
client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hi!"}])
```

`model`:
- `"auto"` (default) — runs the configured provider fallback chain.
- `"<provider>/<model>"`, e.g. `"anthropic/claude-sonnet-4-6"` — calls that
  provider directly, no fallback.

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health` (no auth).
Note: `/healthz` is deliberately not used — it's reserved platform-wide on
Cloud Run and 404s for external callers regardless of app routing.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Primary provider |
| `GROQ_API_KEY` | — | Fallback provider (free tier available) |
| `OPENAI_API_KEY` | — | Fallback provider |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | |
| `OPENAI_MODEL` | `gpt-4o-mini` | |
| `PROVIDER_ORDER` | `anthropic,groq,openai` | Comma-separated, tried in order |
| `BREAKER_FAILURE_THRESHOLD` | `3` | Consecutive failures before a provider is skipped |
| `BREAKER_COOLDOWN_SECONDS` | `60` | How long a tripped provider is skipped |
| `TRACE_INCLUDE_PROMPTS` | `false` | Include (PII-masked) prompt text in traces |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | Any OTLP collector (Langfuse, Grafana Cloud, ...) |
| `OTEL_EXPORTER_OTLP_HEADERS` | — | Comma-separated `key=value` pairs |
| `GATEWAY_API_KEYS` | — | Comma-separated bearer keys the HTTP service accepts. **Unset = open, no auth** — set this before deploying. |

## PII masking

`llm_gateway.pii.mask_pii()` redacts emails, phone numbers, IBANs, card
numbers, and PESEL-shaped national IDs from text before it's traced. It's a
best-effort regex-based redactor, not a certified PII detector.

## Known v1 limitations

- No streaming responses.
- No per-key usage metering/rate limiting on the HTTP service — a valid
  `GATEWAY_API_KEYS` entry has unlimited access.
- `model` selection is provider-level only (`"auto"` or `"<provider>/<model>"`)
  — no per-request temperature/other sampling params yet.

## Development

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest -q
```

## Deployment

See `infra/terraform/` for the Cloud Run + Artifact Registry + Secret
Manager + Workload Identity Federation + Cloud KMS setup this repo deploys
with via `.github/workflows/ci.yml` on every push to `main`.

Provider API keys and `GATEWAY_API_KEYS` are managed with
[SOPS](https://github.com/getsops/sops), encrypted against a Cloud KMS key
(`kms.tf`) — the ciphertext (`infra/terraform/secrets.enc.yaml`) is safe to
commit; only principals with `roles/cloudkms.cryptoKeyEncrypterDecrypter` on
that key can decrypt it. To set or rotate a value:

```bash
cd infra/terraform
sops secrets.enc.yaml   # opens decrypted in $EDITOR, re-encrypts on save
git commit -am "rotate secrets" && git push   # CI applies the new values
```

See `secrets.yaml.example` for the full key list and first-time setup.
