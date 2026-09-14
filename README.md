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

Requires Python 3.11+ and the current provider SDK majors: `anthropic>=1.5,<2`
and `openai>=3.13,<4` (the OpenAI SDK also drives the Groq provider). Both are
built on [`httpx2`](https://pypi.org/project/httpx2/) rather than `httpx` — an
HTTP client you hand to either SDK yourself must be an `httpx2` client.

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

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/usage` (all
bearer-key auth when `GATEWAY_API_KEYS` is set); `GET /health` (no auth, always).
Note: `/healthz` is deliberately not used — it's reserved platform-wide on
Cloud Run and 404s for external callers regardless of app routing.

### Tool calling / ReAct-style agents

`tools`/`tool_choice`/`tool_calls` follow OpenAI's wire format exactly, so
LangGraph's prebuilt `create_react_agent` (or any LangChain `ChatOpenAI`
consumer) works with zero custom integration code — just point `base_url`
at the gateway. See `examples/langgraph_react_agent.py` for a full working
example (weather-lookup tool, real multi-turn tool-call round trip).

Anthropic has no native tool format compatible with OpenAI's — the gateway
translates both directions internally (`providers/_anthropic_translate.py`),
so tool-calling works identically regardless of which provider actually
serves the request. `tool_choice: "none"` is enforced by omitting tools
from that call entirely (Anthropic has no direct equivalent otherwise).

### Structured output

`response_format` (`{"type": "json_schema", "json_schema": {...}}` or
`{"type": "json_object"}`) is passed straight through for OpenAI/Groq —
they support it natively. Anthropic has no equivalent feature, so it's
emulated with a forced single tool call matching the schema, transparently
unwrapped back into plain `content` — the caller never sees a tool call,
just the same JSON-schema response shape as any other provider.

### Sampling parameters

`temperature`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`
are accepted and passed through to whichever provider serves the request.
Groq/OpenAI accept all of them natively; Anthropic supports `temperature`,
`top_p`, and `stop` (translated to `stop_sequences`) — `seed` and the two
penalty params have no Anthropic equivalent and are silently dropped rather
than erroring.

### Errors

Error responses are OpenAI-shaped (`{"error": {"message", "type", "code"}}`),
not FastAPI's default `{"detail": "..."}` — so the openai-python SDK (and
therefore LangChain) can parse them the way it expects to.

### Retries, failover and timeouts

Every provider failure is classified once, in `llm_gateway/errors.py`, from
the anthropic/openai SDK exception hierarchies (Groq goes through the openai
SDK). The classification decides three things: whether the same provider is
retried, whether the error counts toward that provider's circuit breaker,
and — always — failing over to the next provider in `PROVIDER_ORDER`.

| Kind | Errors | Retry same provider | Counts for breaker | Fails over |
|---|---|---|---|---|
| Transient | connection errors, 408, 409, 5xx, Anthropic 529 `overloaded` | yes | yes | yes |
| Timeout | SDK `APITimeoutError`, gateway per-attempt timeout | no | yes | yes |
| Rate limited | 429 | no | yes | yes |
| Auth | 401, 403 | no | yes | yes |
| Invalid request | 400, 404, 413, 422, any other 4xx | no | **no** | yes |
| Unknown | anything else (e.g. a response-parsing bug) | no | yes | yes |

Errors that arrive *inside* an already-open SSE stream carry no useful HTTP
status; they are classified by their `error.type` (`overloaded_error`,
`invalid_request_error`, ...).

Why these choices:

- **Only transient errors are retried** (`RETRY_ATTEMPTS`, default `2` =
  one retry, after `RETRY_BASE_DELAY_SECONDS`, default `0.2`s, doubling).
  An auth failure or a malformed request fails identically on the next try.
- **Timeouts and 429s fail over immediately.** Retrying a hung provider
  spends a whole second timeout the next provider could use; retrying a
  throttled provider 0.2 s later almost always gets another 429. (The
  gateway has no provider-side rate-limit handling beyond this — the
  `RATE_LIMIT_PER_MINUTE` limiter only applies to inbound callers.)
- **Auth errors fail over and count for the breaker**, so one provider's
  revoked key doesn't take down the chain, and a persistently broken key is
  skipped for `BREAKER_COOLDOWN_SECONDS` instead of costing a call every time.
- **Invalid-request errors still fail over, but never trip the breaker.**
  The gateway translates each request per provider (sampling params, image
  blocks, tools, structured output) and providers differ in context window,
  model names and content policy, so "rejected here" doesn't reliably mean
  "rejected everywhere". The extra call is cheap: 4xx responses are fast and
  never retried. It must not count for the breaker, though — a caller's bad
  request says nothing about the provider's health, and counting it would
  let one caller take a healthy provider out of rotation for everyone.
- **The SDKs' own retries are off** (`SDK_MAX_RETRIES=0`). The gateway owns
  retries now; SDK retries (default 2) multiplied every gateway attempt — up
  to 6 HTTP requests per provider — retried 429s blindly, slept on
  `retry-after` instead of failing over, and were invisible to tracing and
  the breaker.

Streaming uses the same rules for its pre-flight (everything before the
first chunk reaches the caller, including a retry); after that, a failure
ends the stream — see [Streaming](#streaming).

**Time bounds** (`0` disables any of them):

| Setting | Default | Bounds |
|---|---|---|
| `REQUEST_TIMEOUT_SECONDS` | `45` | One non-streaming attempt on one provider. Passed to the SDK client (5 s connect) and enforced by the gateway around the whole attempt. |
| `STREAM_IDLE_TIMEOUT_SECONDS` | `30` | Streaming: the wait for the response to start and every gap between chunks (SDK read timeout). Not a cap on a stream that keeps flowing. |
| `CALL_DEADLINE_SECONDS` | `90` | One whole `complete()`/`chat()`/`stream_chat()` call — every attempt, retry, backoff and failover, including a stream that has already started. |

When the deadline runs out the call raises `LLMDeadlineExceeded`, a subclass
of `LLMError`, so existing `except LLMError` handling (and the HTTP service's
503 / SSE error event) needs no change. A timeout that fired only because
the deadline ran out isn't counted against that provider's breaker.

The defaults are sized for single completions of up to ~2000 output tokens,
which usually finish in a few seconds and occasionally take ~30 s: 45 s
gives that 50% headroom, and 90 s fits one hung provider plus a full attempt
on the next. Raise both for long generations (large `max_tokens` on a slow
model, or long streams).

Worst case for one call with the default three-provider chain:

| | Before (0.2.0) | After (0.3.0) |
|---|---|---|
| Non-streaming | 2 gateway × 3 SDK attempts × 600 s timeout per provider, plus backoff: ~1 h per provider, ~3 h for the chain — and no bound at all for a provider that trickles bytes | 90 s |
| Streaming | 3 SDK attempts × 600 s to open, per provider (~90 min for the chain); no bound once the stream started | 90 s, start to last chunk |

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
| `RETRY_ATTEMPTS` | `2` | Total attempts on one provider for transient errors before failing over. `1` disables retries. |
| `RETRY_BASE_DELAY_SECONDS` | `0.2` | Delay before a retry; doubles each attempt. |
| `REQUEST_TIMEOUT_SECONDS` | `45` | Per-attempt timeout for non-streaming calls. `0` falls back to the SDK default (600 s). |
| `STREAM_IDLE_TIMEOUT_SECONDS` | `30` | Streaming: max wait for the stream to start and between chunks. `0` falls back to the SDK default. |
| `SDK_MAX_RETRIES` | `0` | The provider SDKs' own retry count. The gateway owns retries; leave at `0`. |
| `CALL_DEADLINE_SECONDS` | `90` | Wall-clock budget for one call across all retries and failovers. `0` disables it. |
| `TRACE_INCLUDE_PROMPTS` | `false` | Include (PII-masked) prompt text in traces |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | Any OTLP collector (Langfuse, Grafana Cloud, ...) |
| `OTEL_EXPORTER_OTLP_HEADERS` | — | Comma-separated `key=value` pairs |
| `GATEWAY_API_KEYS` | — | Comma-separated bearer keys the HTTP service accepts. **Unset = open, no auth** — set this before deploying. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-key request cap, in-process. `0` disables it. |
| `MAX_TOKENS_CEILING` | `4000` | Reject a request if `max_tokens` exceeds this. `0` disables it. |
| `MAX_PROMPT_CHARS` | `32000` | Reject a request if total message content exceeds this many characters. `0` disables it. |
| `MAX_IMAGE_BYTES` | `10000000` | Reject a request if the total decoded size of all `data:` URI images exceeds this. `0` disables it. |

## PII masking

`llm_gateway.pii.mask_pii()` redacts emails, phone numbers, IBANs, card
numbers, and PESEL-shaped national IDs from text before it's traced. It's a
best-effort regex-based redactor, not a certified PII detector.

## Guardrails, and what's deliberately not solved here

What the gateway does:
- **Per-key rate limiting** (`RATE_LIMIT_PER_MINUTE`) — in-process sliding
  window, keyed by the caller's bearer key. Per-instance only: under Cloud
  Run horizontal scale-out the effective ceiling is up to
  `max_instances × RATE_LIMIT_PER_MINUTE`, not an exact global limit. Good
  enough for basic abuse protection on a single-tenant gateway; a shared
  store (Redis) would be needed for an exact cross-replica limit.
- **Request size ceilings** (`MAX_TOKENS_CEILING`, `MAX_PROMPT_CHARS`,
  `MAX_IMAGE_BYTES`) — reject obviously-abusive payloads before they reach
  a provider.
- **PII-masked tracing** — see above.
- **`GET /v1/models` availability** reflects real state: whether a
  provider's key is configured and whether its circuit breaker is
  currently open, not just a static list. Requires the same bearer auth as
  every other endpoint when `GATEWAY_API_KEYS` is set — it used to be
  reachable without a key even then, which leaked provider-configuration
  state to unauthenticated callers; fixed.

What it does *not* do, on purpose:
- **Prompt injection defense.** There is no technical fix for this at a
  gateway layer — the gateway relays `system`/`prompt` text and has no way
  to distinguish malicious content from legitimate content (true of every
  LLM gateway, not a gap specific to this one). That defense belongs in the
  calling application: how it constructs prompts, scopes tool use, and
  validates model output.
- **Volumetric DDoS protection.** The rate limiting above is abuse
  protection, not network-layer DDoS mitigation. That needs Cloud Armor (or
  equivalent) in front of Cloud Run — deliberately not added while this is
  a single-tenant gateway for internal use, not a public-facing product.

### Streaming

`"stream": true` returns Server-Sent Events, chunk-shaped like OpenAI's own
streaming (`chat.completion.chunk`, ending with `data: [DONE]`) — works with
`ChatOpenAI(streaming=True)`/`.stream()` the same way non-streaming does.

Fallback across providers only happens **before the first chunk** is sent —
once a byte has reached the client, there's no way to un-send it, so a
mid-stream failure ends the stream (as an OpenAI-shaped error event, not a
silently truncated connection) rather than silently retrying elsewhere. A
failure before anything is sent (auth, connection, immediate rate limit)
falls back exactly like the non-streaming path.

Tool-call arguments stream as fragments the same way OpenAI's own API does
— the first fragment carries `id`/`name`, later ones carry only argument
text — so client-side aggregation code (e.g. LangChain's) works unmodified
regardless of which provider actually served the request.

### Multi-modal (images)

OpenAI's multi-part content format works in messages —
`"content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]`
— for both `data:` URIs and plain `https://` URLs. Passed through as-is for
OpenAI/Groq; translated to Anthropic's `image`/`source` block format
internally. Images aren't counted against `MAX_PROMPT_CHARS` (only text
parts are); the decoded size of `data:` URI images is instead bounded
separately by `MAX_IMAGE_BYTES` — `https://` URLs aren't counted there
either, since the gateway never fetches them itself (the provider does), so
there's no local payload to bound.

### Usage metering

`GET /v1/usage` (authenticated) returns the *calling key's own* cumulative
usage — request count, input/output tokens, and when tracking started. A
key can only ever see its own usage. In-process and ephemeral (resets on
restart, not shared across Cloud Run replicas — same tradeoff as rate
limiting); this is cost *visibility*, not a billing system, and a valid key
still has unlimited spend within its rate-limit window.

## Known v1 limitations

- Usage/rate-limit state doesn't survive a restart or scale-out beyond one
  Cloud Run replica.
- `GATEWAY_API_KEYS` is a flat, static list — no per-key labels, expiry, or
  revocation short of editing the secret and redeploying. Fine for a
  handful of consuming apps; would need real key-management (a store +
  admin API) beyond that.

## Development

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install pip-audit && .venv/bin/pip-audit
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

### Changing provider order / models

`provider_order`, `claude_model`, `groq_model`, `openai_model` are Terraform
variables (`infra/terraform/variables.tf`) wired straight to Cloud Run env
vars — that's the versioned, reviewable path (edit, commit, push, CI
redeploys in ~2 min).

For an instant change with no rebuild/redeploy, update the live Cloud Run
revision's env vars directly:

```bash
gcloud run services update llm-gateway --region=us-central1 \
  --update-env-vars PROVIDER_ORDER=groq,anthropic,CLAUDE_MODEL=claude-opus-4-7
```

This takes effect in seconds, but it's a manual override, not a config
change: the next `terraform apply` (i.e. the next push to `main`) resets
Cloud Run's env vars back to whatever `variables.tf` says. Treat it as a
temporary/emergency lever — reflect anything you want to keep back into
Terraform.

See `secrets.yaml.example` for the full key list and first-time setup.
