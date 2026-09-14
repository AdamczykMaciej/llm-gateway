# llm-gateway

A small, self-hosted multi-provider LLM gateway: a fallback chain across
Anthropic / Azure AI Foundry / Groq / OpenAI, a per-provider circuit breaker, PII-masked OTel
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
and `openai>=3.13,<4` (the OpenAI SDK also drives the Groq and Azure providers). Both are
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

### Token usage: `complete_with_usage()`

`complete()` returns a string, and keeps doing exactly that. To also learn
which provider served the call and what it cost, call
`complete_with_usage()`. It takes the same arguments and returns a
`Completion`:

```python
from llm_gateway import complete_with_usage

result = await complete_with_usage(system="...", prompt="...", config=config)
result.text  # the string complete() would have returned
result.provider  # the provider that actually served, e.g. "groq" after a failover
result.model  # that provider's model
result.usage  # Usage(input_tokens, output_tokens, cache_read_input_tokens, ...)
result.stop_reason  # provider-native: "end_turn", "stop", "max_tokens", ...
```

It's a separate function rather than a `return_usage=True` flag because a
flag would make the return type depend on a runtime value: every caller
would get `str | Completion` and need a cast or `typing.overload`s. Two
functions keep both signatures exact. `chat()` already works this way, and
`complete()` is now just `(await complete_with_usage(...)).text`.

`Usage` means the same thing for every provider:

- `input_tokens` counts **all** prompt tokens, cached or not. Anthropic
  reports uncached, cache-read and cache-write tokens as three separate
  numbers, so the gateway sums them. OpenAI and Groq's `prompt_tokens`
  already include cached tokens.
- `cache_read_input_tokens` and `cache_creation_input_tokens` are the
  subsets of `input_tokens` read from or written to a prompt cache. OpenAI
  and Groq report cache reads as `prompt_tokens_details.cached_tokens` and
  have no cache-write count. `usage.uncached_input_tokens` is the rest.

For Anthropic, that gives the cost of a call as
`uncached_input_tokens × input price + cache_creation_input_tokens × 1.25 ×
input price + cache_read_input_tokens × 0.1 × input price + output_tokens ×
output price` (5-minute cache writes; see Prompt caching below).

**Streaming.** `stream_chat()`'s final chunk carries `usage` (the
`(input, output)` tuple, unchanged), `usage_details` (the same `Usage`) and
`provider`. Anthropic's usage comes from `message_start` and `message_delta`.
OpenAI and Groq streams are always requested with
`stream_options={"include_usage": True}`.

**Tracing.** The `llm_gateway.complete` / `.chat` / `.stream_chat` spans
record `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
`gen_ai.usage.cache_read.input_tokens` and
`gen_ai.usage.cache_creation.input_tokens`: counts only, never prompt text.

### Reasoning models and empty replies (0.4.1)

**Empty replies are provider failures.** A reply with no usable text (None,
empty or whitespace content) and no tool calls raises `EmptyCompletionError`
inside that provider's attempt. The error kind is `EMPTY_RESPONSE`: the call
fails over to the next provider, isn't retried on the same one, and **does**
count toward that provider's circuit breaker. An empty reply comes from a
provider/model configuration, not from one caller's input, so the same budget
fails the same way for everyone.

- The error message and the warning log carry provider, model, finish reason
  and reasoning tokens, never content.
- Tool-call replies with null content are unaffected.
- Streams are held back until the first chunk with text or a tool call. A
  stream that ends without either raises before anything reaches the caller,
  so `stream_chat()` can still fail over.
- Before 0.4.1, `complete()` returned `""` for an OpenAI/Groq reply with empty
  content.

**Groq reasoning models get `reasoning_effort="low"`.** Groq's reasoning
models return their reasoning in a separate `reasoning` field. The reasoning
tokens are billed as output and count against the completion budget, so at
Groq's default (medium) effort a small `max_tokens` can be spent entirely on
reasoning. The reply then has empty content and `finish_reason="length"`.

The gateway sends `reasoning_effort: "low"` in `complete()`, `chat()` and
`stream_chat()` to models whose id starts with:

- `openai/gpt-oss-` (and bare `gpt-oss-`), which accept low/medium/high;
- `qwen/qwen3.8-27b`, which accepts none/default/low/medium/high.

Azure deployments get `AZURE_REASONING_EFFORT` (default `low`) instead;
see [Azure AI Foundry](#azure-ai-foundry-042).

Both are listed in [Groq's reasoning docs](https://console.groq.com/docs/reasoning),
checked 2026-09-14. Other models never get the parameter, including Qwen 3.6
27B (none/default only) and MiniMax M2.7 (no effort values documented).

**Reasoning tokens in usage.** `usage.reasoning_tokens` reports the reasoning
share of `output_tokens` and is already included in it. It comes from
OpenAI/Groq `completion_tokens_details.reasoning_tokens` or Anthropic
`output_tokens_details.thinking_tokens`. Spans record it as
`llm_gateway.usage.reasoning_tokens`, and each served call logs it at debug
level on the `llm_gateway` logger.

### Structured output: `output_schema=`

Pass a pydantic model class (or a JSON Schema dict) and get validated data
back:

```python
from pydantic import BaseModel, Field


class Verdict(BaseModel):
    score: int = Field(ge=0, le=100)
    summary: str


result = await complete_with_usage(
    system="Score the answer.", prompt=answer, config=config, output_schema=Verdict
)
result.parsed  # Verdict(score=..., summary=...)
result.text  # the raw JSON
```

Each provider is asked through its native mechanism:

| Provider  | Request                                                                                          |
|-----------|--------------------------------------------------------------------------------------------------|
| Anthropic | `output_config.format = {"type": "json_schema", "schema": ...}`: [structured outputs][ant-so], generally available, constrained decoding, supported on `claude-haiku-4-5` and newer. The JSON comes back as a text block. |
| OpenAI    | `response_format = {"type": "json_schema", "json_schema": {"name", "strict": true, "schema"}}`  |
| Groq      | the same strict `json_schema` on `openai/gpt-oss-20b` / `openai/gpt-oss-120b`, the models [Groq documents][groq-so] with strict support. Every other model, such as `llama-3.1-8b-instant`, gets JSON mode (`{"type": "json_object"}`) with the schema spelled out in the system prompt. |
| Azure     | the same strict `json_schema` as OpenAI, for every deployment (Azure lists structured outputs for `gpt-oss-120b`). |

[ant-so]: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
[groq-so]: https://console.groq.com/docs/structured-outputs

Anthropic's docs recommend native structured outputs over forcing a single
tool call. Forced tool use is also rejected when manual extended thinking is
on. So this path doesn't use the forced-tool emulation that `chat()`'s
`response_format` still uses (that path is unchanged in 0.4).

Strict modes accept only part of JSON Schema: every object needs
`additionalProperties: false`, OpenAI also needs every property listed in
`required`, and numeric/length/pattern constraints aren't supported. The
gateway rewrites the schema it *sends* to fit those rules. Your pydantic
model still enforces every constraint when the reply is validated, so a
`le=100` violation is caught locally.

**When the reply is unusable** (not JSON, fails validation, a refusal,
Groq's `400 json_validate_failed`), that provider's attempt fails with
`InvalidOutputError` and the call **fails over to the next provider**. It is
classified `INVALID_OUTPUT`: not retried on the same provider, and **not
counted by the circuit breaker**, because the provider answered and one
caller's hard schema must not take it out of rotation for everyone. If
every provider fails, you get `LLMError` as usual. Error messages name the
failing fields but never include the model's output.

A dict schema gets only a shallow check (top-level type and `required`
keys), since the gateway has no JSON Schema validator dependency. Use a
pydantic model for full validation.

### Prompt caching: `cache_system=True`

```python
result = await complete_with_usage(
    system=LONG_STATIC_INSTRUCTIONS, prompt=user_input, config=config, cache_system=True
)
result.usage.cache_creation_input_tokens  # > 0 on the first call
result.usage.cache_read_input_tokens  # > 0 on later calls within the TTL
```

On Anthropic, `system` is sent as one text block with
`cache_control: {"type": "ephemeral"}` (the default 5-minute TTL), **but only
when it can actually be cached**. Minimum cacheable prompt lengths from
[Anthropic's prompt-caching docs][ant-pc] (checked 2026-09-14):

| Model                                                              | Minimum tokens |
|--------------------------------------------------------------------|---------------:|
| Claude Opus 5, Fable 5 / 5.1, Mythos 5 / 5.1                       | 512            |
| Claude Opus 4.8, Sonnet 5, Sonnet 4.6, Sonnet 4.5, Opus 4.1, Opus 4, Sonnet 4 | 1,024 |
| Claude Opus 4.7, Mythos Preview, Haiku 3.5                         | 2,048          |
| **Claude Haiku 4.5** (the default `CLAUDE_MODEL`), Opus 4.6, Opus 4.5 | **4,096**   |

[ant-pc]: https://platform.claude.com/docs/en/build-with-claude/prompt-caching

The docs: *"Shorter prompts cannot be cached, even if marked with
`cache_control`. Any requests to cache fewer than this number of tokens will
be processed without caching, and no error is returned."* The gateway checks
a local estimate (3 characters per token, deliberately generous) against the
table and leaves the request unchanged below it. An overestimate only sends
a marker the API ignores. Counting exactly with `messages.count_tokens`
would add a separate API request, and its latency, to every call, which
costs more than the check saves. Unknown models get the largest minimum
(4,096).

**When it saves money.** From the same docs, a 5-minute cache write costs
**1.25×** the base input price (a 1-hour write costs 2×; the gateway uses
5 minutes), and a cache read costs **0.1×**. For a cacheable prefix of *N*
tokens:

- one call with no reuse inside 5 minutes: 1.25*N* instead of *N*, so
  **25% more** on that prefix;
- two calls inside 5 minutes: 1.25*N* + 0.1*N* = 1.35*N* instead of 2*N*,
  **32.5% less**;
- *k* calls: 1.25*N* + 0.1(*k*−1)*N* instead of *kN*, approaching a 90%
  saving on the prefix.

So turn it on for a long, byte-identical system prompt that repeats within
minutes (for example, several answers scored in one practice session). Leave
it off for prompts that change per request or are rarely repeated. Anything
that varies (user input, per-request data) belongs in `prompt`, not
`system`.

OpenAI and Groq cache long prompt prefixes automatically, as Azure does for
the deployments that support prompt caching. `cache_system` changes nothing
in their requests, and their hits show up as
`usage.cache_read_input_tokens`.

### Azure AI Foundry (0.4.2)

The `azure` provider calls a model deployment through the
[Azure OpenAI v1 API][az-v1] with the plain openai SDK and Azure's base URL:
no `api-version` parameter and no `AzureOpenAI` client. It was added for
`gpt-oss-120b`, which Azure lists with the Chat Completions API, streaming,
function calling, structured outputs and reasoning (Preview; deploying it
needs a Foundry project) in [Foundry Models sold by Azure][az-models].

```bash
pip install "llm-gateway[azure] @ git+https://github.com/AdamczykMaciej/llm-gateway.git"
```

```python
config = GatewayConfig(
    azure_endpoint="https://my-resource.services.ai.azure.com",
    azure_model="gpt-oss-120b",  # the deployment name
    provider_order="anthropic,azure,groq,openai",
)
```

`azure` is not in the default `provider_order`; add it where you want it.

| Setting (env var = upper case) | Default | |
|---|---|---|
| `azure_endpoint` | — | `https://<resource>.openai.azure.com` or `https://<resource>.services.ai.azure.com`, with or without a trailing `/openai/v1` and slash. Normalized to `…/openai/v1/`. A Foundry project endpoint (`…/api/projects/<project>`) gets `/openai/v1/` appended the same way. |
| `azure_model` | — | The **deployment name**, sent as `model` and reported as `Completion.model`. |
| `azure_auth` | `entra` | `entra` or `api_key`. Any other value fails config validation. |
| `azure_api_key` | — | The resource key. Used only with `azure_auth=api_key`. |
| `azure_managed_identity_client_id` | — | Entra only: the client id of a user-assigned managed identity. Unset means `DefaultAzureCredential`. |
| `azure_reasoning_effort` | `low` | Sent as `reasoning_effort` on every azure request. `low`, `medium`, `high`, or `""` to omit the parameter. |
| `azure_max_tokens_param` | `max_completion_tokens` | The request field that carries the token budget: `max_completion_tokens` or `max_tokens`. Any other value fails config validation. |

The provider counts as configured when `azure_endpoint` and `azure_model` are
set, and, for `api_key` auth, `azure_api_key`. With Entra auth nothing is
checked at startup: a missing or broken identity fails the call instead (see
below), and the chain moves on.

**Authentication.**

- `entra` (default): Microsoft Entra ID tokens for the scope
  `https://ai.azure.com/.default`, from azure-identity's async
  `get_bearer_token_provider`. `AsyncOpenAI` awaits a callable `api_key`
  before every request, and the token provider caches the token and
  refreshes it before it expires. The credential is
  `ManagedIdentityCredential(client_id=...)` when
  `azure_managed_identity_client_id` is set, else `DefaultAzureCredential()`.
  Give that identity the **Cognitive Services OpenAI User** role on the
  resource. This needs the `[azure]` extra (azure-identity, plus aiohttp,
  the HTTP transport its async credentials use). Without it, each azure
  call fails with `ProviderAuthError` (logged once) and fails over; nothing
  else in the gateway imports azure-identity.
- `api_key`: the SDK sends the key as `Authorization: Bearer <key>`, like
  Microsoft's own `OpenAI(api_key=..., base_url=".../openai/v1/")` example.
  The v1 API accepts a key in either `api-key` or `Authorization`
  ([v1 OpenAPI spec][az-spec]).

A token that can't be acquired (`ClientAuthenticationError`,
`CredentialUnavailableError`) raises `ProviderAuthError`, classified `AUTH`:
not retried, fails over, counts toward the breaker. Its message names the
exception type and never includes a token or key. Cached credentials are
closed by `await llm_gateway.providers.azure.aclose()`, which the HTTP
service calls on shutdown.

**Requests.** Everything else works like the OpenAI provider: the same
`REQUEST_TIMEOUT_SECONDS`, `STREAM_IDLE_TIMEOUT_SECONDS` and
`SDK_MAX_RETRIES`, and `SSL_VERIFY=false` (which also turns off certificate
checks for the Entra credential's token requests). Structured output uses
strict `json_schema` ([Azure structured outputs][az-so]). Streams request
`stream_options={"include_usage": true}`. Usage comes from
`prompt_tokens`, `completion_tokens`,
`completion_tokens_details.reasoning_tokens` and
`prompt_tokens_details.cached_tokens`, under provider `azure`. The token
budget goes in the field named by `azure_max_tokens_param`. The default,
`max_completion_tokens`, suits reasoning deployments: gpt-oss and the o-series
need it, and it covers reasoning plus visible tokens. Set `max_tokens` if a
deployment rejects it. The gateway doesn't verify which field a given
deployment accepts.

**`reasoning_effort`.** Same reason as Groq: at a reasoning model's default
effort, a small `max_tokens` can go entirely to reasoning and leave empty
content. Deployment names are arbitrary, so the gateway can't recognise a
reasoning model by name the way it does on Groq; the setting applies to every
azure request instead. The v1 chat-completions request schema defines
`reasoning_effort`, and Azure lists `gpt-oss-120b` with reasoning. Azure's
[reasoning models page][az-reasoning] gives accepted values only for Azure
OpenAI GPT and o-series models, not gpt-oss; `low`/`medium`/`high` are the
levels gpt-oss itself defines. Set `AZURE_REASONING_EFFORT=""` for a
deployment of a non-reasoning model, which may reject the parameter.

**Empty replies and content filtering** ([Azure content filtering][az-cf]):

- A prompt blocked by the content filter returns HTTP 400 with
  `error.code = "content_filter"`. That is `INVALID_REQUEST`: it fails over,
  isn't retried, and doesn't count toward the breaker, since it's about the
  caller's prompt, not the provider's health. The same applies when a stream
  is opened.
- A completion blocked by the filter returns 200 with
  `finish_reason: "content_filter"` and no content. That is an empty reply
  (`EmptyCompletionError`, `EMPTY_RESPONSE`): it fails over and counts toward
  the breaker, like any other empty reply. A stream that ends that way before
  any text fails over too. A stream the filter cuts off after text has
  already reached the caller just ends with `finish_reason: "content_filter"`.

**Troubleshooting: repeated 400 failovers.** If the logs show repeated `azure`
`INVALID_REQUEST` (400) failovers right after you enable the provider, the
likely cause is `azure_reasoning_effort` or `azure_max_tokens_param` not
matching the deployment's model. Set `AZURE_REASONING_EFFORT=""` and/or
`AZURE_MAX_TOKENS_PARAM=max_tokens`. A 400 never trips the breaker, so a
mismatch doesn't take azure out of rotation: every call pays a failed Azure
round trip before failing over.

**Data residency.** The deployment's SKU decides where inference is
processed. gpt-oss-120b is currently offered only as GlobalStandard
(processed in any Azure region). For EU-only processing, use a Data Zone
Standard deployment of a model that supports it (e.g. Mistral or DeepSeek
models listed in Microsoft's Data Zone tables) and set
`azure_reasoning_effort` to `""` for non-reasoning models. Choosing the
deployment type and region is the operator's responsibility; the gateway
only calls the endpoint it's given.

[az-v1]: https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle
[az-models]: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
[az-spec]: https://github.com/Azure/azure-rest-api-specs/blob/main/specification/ai/data-plane/OpenAI.v1/azure-v1-v1-generated.json
[az-so]: https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs
[az-reasoning]: https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning
[az-cf]: https://learn.microsoft.com/en-us/azure/foundry-classic/foundry-models/concepts/content-filter

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

In-process callers who want validated output with failover should use
`complete_with_usage(output_schema=...)` instead (see "Structured output"
under the library section), which uses Anthropic's native structured outputs.

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
| Auth | 401, 403; `ProviderAuthError` (e.g. no Entra token for Azure) | no | yes | yes |
| Invalid request | 400 (including Azure's `content_filter`), 404, 413, 422, any other 4xx | no | **no** | yes |
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
| `AZURE_ENDPOINT` | — | Azure AI Foundry / Azure OpenAI resource endpoint; see [Azure AI Foundry](#azure-ai-foundry-042) |
| `AZURE_MODEL` | — | Azure deployment name |
| `AZURE_AUTH` | `entra` | `entra` (needs the `[azure]` extra) or `api_key` |
| `AZURE_API_KEY` | — | Azure resource key, for `AZURE_AUTH=api_key` |
| `AZURE_MANAGED_IDENTITY_CLIENT_ID` | — | User-assigned managed identity for Entra auth; unset uses `DefaultAzureCredential` |
| `AZURE_REASONING_EFFORT` | `low` | `reasoning_effort` on every azure request; `""` omits it |
| `AZURE_MAX_TOKENS_PARAM` | `max_completion_tokens` | Token-budget field for azure requests: `max_completion_tokens` or `max_tokens` |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq retired `llama-3.3-70b-versatile` on 2026-08-16 |
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
OpenAI/Groq/Azure; translated to Anthropic's `image`/`source` block format
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
