# llm-gateway

A small, self-hosted multi-provider LLM gateway: a fallback chain across
Anthropic / Claude on Google Vertex AI / Azure AI Foundry / Groq / OpenAI, a per-provider circuit breaker, PII-masked OTel
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

### Claude on Google Vertex AI (0.6.0)

The `vertex` provider serves Anthropic's Claude through Google Cloud Vertex AI
(Google's docs now call it *Gemini Enterprise Agent Platform*). It is the same
model family as the `anthropic` provider, billed through Google Cloud instead of
Anthropic, so an Anthropic spend limit or outage doesn't take it down with it.
Requests go through the Anthropic SDK's `AsyncAnthropicVertex` and the
`anthropic` provider's own request code: the request body is identical apart
from `model` (which moves into the URL) and `anthropic_version:
vertex-2023-10-16` ([Claude on Vertex AI][vx-anthropic]). Prompt caching,
structured output, usage, `cost_usd` and the empty-reply guard all behave as
they do for `anthropic`.

```bash
pip install "llm-gateway[vertex] @ git+https://github.com/AdamczykMaciej/llm-gateway.git"
# Workload Identity Federation from an Azure managed identity also needs azure-identity:
pip install "llm-gateway[vertex,azure] @ git+https://github.com/AdamczykMaciej/llm-gateway.git"
```

```python
config = GatewayConfig(
    vertex_project_id="my-vertex-project",
    provider_order="anthropic,vertex,groq",  # vertex isn't in the default order
)
```

| Setting (env var = upper case) | Default | |
|---|---|---|
| `vertex_project_id` | — | The Google Cloud project id. |
| `vertex_location` | `europe-west1` | `global`, a multi-region (`eu`, `us`) or a region. Anything else fails config validation. |
| `vertex_model` | `claude-haiku-4-5@20251001` | The Vertex model id. Google's model card also lists it as `claude-haiku-4-5`. |
| `vertex_credentials_file` | — | Path to a credential configuration file, e.g. an `external_account` config for Workload Identity Federation. It holds no key. Empty = Application Default Credentials. |
| `vertex_impersonate_service_account` | — | A service account email to act as. |
| `vertex_azure_app_id_uri` | — | Workload Identity Federation from an Azure managed identity: the Entra application ID URI the pool provider accepts as audience. Needs an `external_account` `vertex_credentials_file` and the `[azure]` extra. |
| `vertex_azure_managed_identity_client_id` | — | With `vertex_azure_app_id_uri`: the client id of a user-assigned managed identity. Empty = the system-assigned identity. |
| `vertex_structured_outputs` | `false` | Send `output_schema` calls to Vertex. Enable **only after** the organization policy `constraints/vertexai.allowedPartnerModelFeatures` allows `publishers/anthropic/models/claude-haiku-4-5:structured_outputs` on the project (see [Features](#features)). While `false`, routing skips `vertex` for `output_schema` calls. |

The provider counts as configured when `vertex_project_id` and `vertex_model`
are set. Bad values fail at startup: a malformed location, project id or
service account, an unreadable or non-JSON credentials file, a service-account
key file, and inconsistent Azure or impersonation settings.

#### EU endpoints: what is actually guaranteed

Facts from Google's docs as of 2026-09-14:

- **Claude Haiku 4.5 is GA** (released 2025-10-15, retirement "not sooner than"
  2026-10-15). Its regions are **`us-east5`, `europe-west1` and the global
  endpoint** ([model card][vx-haiku], updated 2026-09-03).
- **There is no `eu` multi-region endpoint for Haiku 4.5.** Google's locations
  tables tick only `europe-west1` in Europe for it and nothing under the US/EU
  multi-region columns ([locations][vx-locations]). The pricing page has no
  Haiku 4.5 rows under its "EU Multi-Region" tab ([pricing][vx-pricing]). The
  quota table lists only `us-east5`, `europe-west1` and global for it
  ([quotas][vx-quotas]). Anthropic's page shows the multi-region endpoints for
  newer models ([Claude on Vertex AI][vx-anthropic]). That is why
  `vertex_location` defaults to `europe-west1`. `eu` is still accepted for
  newer models; with Haiku 4.5 expect a 4xx, which fails over without tripping
  the breaker.
- **Processing location.** A regional endpoint "ensure[s] that ML processing
  remains entirely within the broader multi-regional or country jurisdiction
  associated with that region" ([data residency][vx-residency], updated
  2026-09-09). For Haiku 4.5, the model card gives ML processing for
  `europe-west1` as "Europe: Multi-region". The quota page says "ML processing
  for all available Anthropic models occurs … within the EU when requests are
  made to regionally-available APIs in Europe". So the commitment is **EU
  processing, not in-country (Belgium) processing**. The EU multi-region
  boundary covers EU member states only and excludes the UK and Switzerland.
- **The global endpoint** routes and processes "anywhere globally" and gives
  "no regional isolation or data residency guarantees".

The library asserts none of this for you: `vertex` starts with
`region=unknown` like every provider (see the EU-only example below).

#### Pricing

USD per 1M tokens for Claude Haiku 4.5 ([pricing][vx-pricing]):

| | Input | Output | 5-min cache write | 1-h cache write | Cache hit |
|---|---|---|---|---|---|
| Global endpoint | 1.00 | 5.00 | 1.25 | 2.00 | 0.10 |
| `europe-west1` / `us-east5` | 1.10 | 5.50 | 1.375 | 2.20 | 0.11 |

Regional and multi-region endpoints carry a 10% premium over global for Claude
Sonnet 4.5 and newer models, Haiku 4.5 included ([Claude on Vertex AI][vx-anthropic]).
`DEFAULT_PRICES` holds the global rates under `vertex/claude-haiku-4-5@20251001`
and `vertex/claude-haiku-4-5`, and `lookup_price()` multiplies them by
`pricing.VERTEX_REGIONAL_PREMIUM` (1.1) whenever `vertex_location` isn't
`global`. Worst-case estimates, `max_cost_usd` and `cost_usd` all use that
price. A `MODEL_PRICES` entry for a `vertex/...` key is used as given, with no
premium added. 1-hour cache writes aren't modelled, as for `anthropic`.

#### Features

Google lists function calling, prompt caching (5-minute and 1-hour),
extended thinking, batch predictions and token counting for Haiku 4.5
([model card][vx-haiku]), and structured outputs for every Claude 4.5 and
later model ([structured outputs][vx-so], updated 2026-09-11). Anthropic's
feature table marks the same features as available on Google Cloud
([features overview][claude-features]). Not supported on Vertex: URL sources
for images and documents, the Files API, server-side tools (code execution,
web fetch), the MCP connector and server-side `fallbacks`. The gateway uses
none of these, with one exception: an `image_url` with an `https://` URL
becomes a URL image source, which Vertex rejects (a 400 that fails over).
Send images as `data:` URIs.

Two operator-side switches:

- **Enable the model**: open the Claude Haiku 4.5 model card in Model Garden
  and click **Enable** ([use Claude][vx-use]).
- **Structured outputs are off twice by default.** For projects in an
  organization, Google denies the `structured_outputs` feature until the
  policy `constraints/vertexai.allowedPartnerModelFeatures` allows it
  ([structured outputs][vx-so], [controlling model access][vx-policy]). The
  gateway's own `VERTEX_STRUCTURED_OUTPUTS` also defaults to `false`: vertex
  then reports structured output as unsupported, and routing skips it for
  every `output_schema` call before any request. It doesn't pay a 400, doesn't
  touch the breaker, and needs no `POLICY_REQUIRE_PARAMETERS`. Set
  `VERTEX_STRUCTURED_OUTPUTS=true` only **after** the policy on the project
  allows `publishers/anthropic/models/claude-haiku-4-5:structured_outputs` (or
  a broader `publishers/anthropic/models/claude-haiku-4-5` or
  `publishers/anthropic`):

  ```yaml
  name: projects/PROJECT_ID/policies/vertexai.allowedPartnerModelFeatures
  spec:
    rules:
    - values:
        allowedValues:
        - publishers/anthropic/models/claude-haiku-4-5:structured_outputs
  ```

  `chat()`'s `response_format` uses a forced tool call rather than structured
  outputs, so it is served by Vertex either way.

Capabilities mirror `anthropic` for the same model id (`@version` suffixes
included): tools, images and streaming for Haiku 4.5, and strict structured
output only with `VERTEX_STRUCTURED_OUTPUTS=true`.

#### Authentication

No API key and no service-account key. The SDK sends a Google OAuth access
token, asks google-auth before every request whether it has expired, and
refreshes it in a worker thread when it has. Credentials are built on first
use, also in a worker thread:

- **Application Default Credentials** (no `vertex_credentials_file`): an
  attached service account on Google Cloud, or `gcloud auth
  application-default login` locally. ADC that resolves to a service-account
  key (`GOOGLE_APPLICATION_CREDENTIALS` pointing at a key file) is refused.
- **`vertex_credentials_file`**: `external_account` (Workload Identity
  Federation), `external_account_authorized_user`, `authorized_user` or
  `impersonated_service_account` (unless its source is a key). Each is loaded
  with its type-specific google-auth loader, never
  `load_credentials_from_dict()`. That function is deprecated upstream, and
  for `external_account` it also resolves a project id over the network.
- **`vertex_impersonate_service_account`**: for an `external_account` config it
  becomes the config's `service_account_impersonation_url` (the STS flow
  `gcloud ... --service-account` writes, which needs `roles/iam.workloadIdentityUser`
  on the service account). Any other source is wrapped in
  `impersonated_credentials.Credentials`, which needs
  `roles/iam.serviceAccountTokenCreator`.

Failures to load or refresh credentials (`DefaultCredentialsError`,
`RefreshError`, `TransportError`), a failed Azure managed-identity token, and a
missing `[vertex]` or `[azure]` extra all raise `ProviderAuthError`: AUTH, not
retried, fails over, counts toward the breaker. The message names the exception
type and never includes a token or the STS/Entra error text. The missing-extra
error is logged once. Nothing else in the gateway imports google-auth.

#### Workload Identity Federation from Azure Container Apps

A container app with a managed identity reaches Vertex without any Google
key. The identity gets a Microsoft Entra token for an Entra application,
Google STS exchanges it for a federated token, and that token impersonates a
service account allowed to call Vertex
([WIF with AWS or Azure][wif-azure], updated 2026-09-10).

Why the library supplies the Azure token itself: `gcloud ... create-cred-config
--azure` writes a URL-sourced config that reads the VM Instance Metadata
Service (`http://169.254.169.254/metadata/identity/oauth2/token` with a static
`Metadata: True` header, [AIP-4117][aip-4117]). Container Apps documents a
different identity endpoint, `IDENTITY_ENDPOINT`, with an `X-IDENTITY-HEADER`
whose value "is rotated by the platform" ([Container Apps managed
identity][aca-mi]). A static config can't send that header. With
`vertex_azure_app_id_uri` set, the gateway drops the config's
`credential_source` and hands google-auth a subject-token supplier (google-auth
2.29.0+) that calls azure-identity's `ManagedIdentityCredential` for
`<app id uri>/.default`, so the token comes from whichever managed identity
endpoint the host exposes.

One-time setup (documentation only; the gateway runs none of this):

1. **Entra ID**: create an application, set its Application ID URI (e.g.
   `api://gcp-vertex-wif`), and note the tenant id. Assign the container app a
   managed identity and note its **object (principal) id**; for a user-assigned
   identity, also note its **client id**. By default any identity in the tenant
   can get tokens for the application: require app role assignment and assign
   the managed identity if you want to restrict that.
2. **Google Cloud**:

```bash
PROJECT_ID=my-vertex-project
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
TENANT_ID=00000000-0000-0000-0000-000000000000   # Entra tenant
APP_ID_URI=api://gcp-vertex-wif
MI_OBJECT_ID=11111111-1111-1111-1111-111111111111  # the managed identity's object id
SA=llm-gateway-vertex@$PROJECT_ID.iam.gserviceaccount.com

gcloud services enable aiplatform.googleapis.com iam.googleapis.com \
  cloudresourcemanager.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
  --project "$PROJECT_ID"

gcloud iam workload-identity-pools create azure-pool \
  --project "$PROJECT_ID" --location=global --display-name="Azure workloads"

# The issuer must match the `iss` claim of the managed identity's token
# (decode one to check; v1 tokens use https://sts.windows.net/<tenant>/).
gcloud iam workload-identity-pools providers create-oidc azure-aca \
  --project "$PROJECT_ID" --location=global --workload-identity-pool=azure-pool \
  --issuer-uri="https://sts.windows.net/$TENANT_ID/" \
  --allowed-audiences="$APP_ID_URI" \
  --attribute-mapping="google.subject=assertion.sub" \
  --attribute-condition="assertion.sub=='$MI_OBJECT_ID'"

gcloud iam service-accounts create llm-gateway-vertex --project "$PROJECT_ID"
# Call Claude on Vertex (Agent Platform User, formerly Vertex AI User).
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA" --role=roles/aiplatform.user
# Let the federated managed identity impersonate the service account.
gcloud iam service-accounts add-iam-policy-binding "$SA" --project "$PROJECT_ID" \
  --role=roles/iam.workloadIdentityUser \
  --member="principal://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/azure-pool/subject/$MI_OBJECT_ID"

gcloud iam workload-identity-pools create-cred-config \
  "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/azure-pool/providers/azure-aca" \
  --service-account="$SA" --azure --app-id-uri="$APP_ID_URI" \
  --output-file=vertex-wif.json
```

Then enable Claude Haiku 4.5 in Model Garden and check the quota (below).
`roles/aiplatform.user` is "Agent Platform User" in the [access control
reference][vx-iam].

`vertex-wif.json` has this shape ([AIP-4117][aip-4117]). It is not a secret:
it names the pool, the audience and the service account, and contains no key.

```json
{
  "type": "external_account",
  "audience": "//iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/azure-pool/providers/azure-aca",
  "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
  "token_url": "https://sts.googleapis.com/v1/token",
  "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/llm-gateway-vertex@my-vertex-project.iam.gserviceaccount.com:generateAccessToken",
  "credential_source": {
    "url": "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=api://gcp-vertex-wif",
    "headers": {"Metadata": "True"},
    "format": {"type": "json", "subject_token_field_name": "access_token"}
  }
}
```

3. **The app** provides the file, the settings and the extras:

```bash
pip install "llm-gateway[vertex,azure] @ git+https://github.com/AdamczykMaciej/llm-gateway.git"
PROVIDER_ORDER=anthropic,vertex,groq
VERTEX_PROJECT_ID=my-vertex-project
VERTEX_LOCATION=europe-west1
VERTEX_CREDENTIALS_FILE=/app/config/vertex-wif.json
VERTEX_AZURE_APP_ID_URI=api://gcp-vertex-wif
VERTEX_AZURE_MANAGED_IDENTITY_CLIENT_ID=<client id of the user-assigned identity>  # omit for system-assigned
```

`VERTEX_IMPERSONATE_SERVICE_ACCOUNT` isn't needed here because the config
already names the service account. If you set it, it must match. Prefer
`VERTEX_CREDENTIALS_FILE` over `GOOGLE_APPLICATION_CREDENTIALS`: ADC loads an
`external_account` config through the generic loader, which also looks up a
project id over the network.

#### Quota and troubleshooting

Claude on Vertex has per-project, per-location quotas (requests per minute,
input and output tokens per minute) for the base model `anthropic-claude-haiku`
([quotas][vx-quotas]). A fallback is only useful if its quota can absorb your
primary's traffic: check the project's actual limits for `europe-west1` on
the Quotas page, and request an increase there before relying on it. New
projects often start low.

- **429 `RESOURCE_EXHAUSTED`**: quota. RATE_LIMITED, fails over immediately,
  counts toward the breaker.
- **404 `NOT_FOUND`**: the model isn't enabled in Model Garden, or isn't offered
  in `vertex_location` (e.g. `eu` for Haiku 4.5). **400 `INVALID_ARGUMENT` /
  `FAILED_PRECONDITION`**: a request Vertex rejects, e.g.
  `VERTEX_STRUCTURED_OUTPUTS=true` before the organization policy allows the
  feature, or a URL image. Both are INVALID_REQUEST:
  they fail over and never trip the breaker, so every call pays the round trip.
- **401/403**: the service account lacks `roles/aiplatform.user`, or an
  organization policy blocks the model. AUTH, counts toward the breaker.
- **`ProviderAuthError`**: credentials couldn't be loaded or refreshed
  (pool/provider audience, issuer, attribute condition, `roles/iam.workloadIdentityUser`,
  the Entra app's audience), or an extra is missing.

Timeouts, retries (`SDK_MAX_RETRIES=0`; the gateway owns retries) and the
breaker work as for every provider. `SSL_VERIFY=false` affects only the Vertex
API connection, not google-auth's or azure-identity's token requests. Cached
clients and managed-identity credentials are closed by
`await llm_gateway.providers.vertex.aclose()`, which the HTTP service calls on
shutdown. `GET /v1/models` lists `vertex/<vertex_model>` with its
configured/available state.

[vx-anthropic]: https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
[vx-haiku]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/haiku-4-5
[vx-locations]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations
[vx-residency]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/data-residency
[vx-quotas]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/quotas
[vx-pricing]: https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
[vx-so]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/structured-outputs
[vx-use]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/use-claude
[vx-policy]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/control-model-access
[vx-iam]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/general/access-control
[claude-features]: https://platform.claude.com/docs/en/build-with-claude/overview
[wif-azure]: https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-other-clouds
[aip-4117]: https://google.aip.dev/auth/4117
[aca-mi]: https://learn.microsoft.com/en-us/azure/container-apps/managed-identity

### Provider policy and cost controls (0.5.0)

The gateway can restrict *which* providers may process a call, based on
where and how they process prompts, what the request needs, and what it may
cost. The controls are modelled on OpenRouter's provider routing (`zdr`,
`data_collection`, `only`/`ignore`, `require_parameters`, `max_price`,
`sort`) and run in-process. They **fail closed**: when no provider
qualifies, the call raises before any network request, and failover (streams
included) never falls back to an excluded provider.

> **Provider metadata is operator-asserted.** The library does not know or
> verify any vendor's processing region, retention, training use or
> contracts. Every fact comes from your configuration and should come from
> your signed agreements (DPA, zero-data-retention terms, the region of the
> deployment you actually use), not from a vendor's marketing page. A
> provider without metadata is `region=unknown`, `retention=unknown`,
> `trains_on_data=true`, `dpa=false`, and fails every requirement below.

**1. Assert what your contracts say**, per provider id (`anthropic`, `groq`,
`openai`, `azure`), as JSON in `PROVIDER_METADATA`, or as
`provider_metadata={...}` (dicts or `ProviderMetadata`) in code:

| Field | Values | Default |
|---|---|---|
| `region` | a short lowercase code: `eu`, `us`, `global`, ... | `unknown` |
| `retention` | `zero`, `abuse_monitoring_30d`, `unknown` | `unknown` |
| `trains_on_data` | `true` / `false` | `true` |
| `dpa` | `true` / `false` | `false` |
| `notes` | free text, e.g. which agreement or deployment backs the entry | `""` |

Unknown provider ids, unknown fields and invalid values fail at startup
(`GatewayConfig()` raises a `ValidationError`).

**2. Set the policy.** The global policy comes from `POLICY_*` settings.
Every call can pass `policy=ProviderPolicy(...)` to `complete()`,
`complete_with_usage()`, `chat()` and `stream_chat()`:

| Requirement | A provider qualifies when |
|---|---|
| `residency="eu"` | its asserted `region` is exactly `eu` (`global` does not match) |
| `require_zero_retention` | `retention` is `zero` |
| `forbid_training` | `trains_on_data` is `false` |
| `require_dpa` | `dpa` is `true` |
| `only` / `ignore` | its id is in `only` (when set) and not in `ignore` |
| `require_parameters` | its model is known to support every feature the request uses (below) |
| `max_cost_usd` | its estimated worst-case cost for the call is within the cap (below) |

**A per-call policy can only narrow the global one.** Flags are OR-ed, `only`
lists intersect (disjoint lists allow nothing), `ignore` lists union, the
lower cost cap wins, and both budget checks must allow. A per-call residency
that differs from the global one raises `PolicyViolationError`. `sort` only
orders providers that already qualify, so a per-call `sort` wins. The policy
also applies to `force_provider` (`model="<provider>/<model>"` over HTTP).

#### EU-only example

Only an Azure deployment in an EU Data Zone (here `mistral-medium-3-5`) may
see prompts. Assert only what your Azure agreement says; the retention value
below is a placeholder to replace with yours:

```bash
PROVIDER_ORDER=azure,anthropic,groq,openai
AZURE_ENDPOINT=https://my-resource.services.ai.azure.com
AZURE_MODEL=mistral-medium-3-5  # an EU Data Zone Standard deployment
AZURE_REASONING_EFFORT=""        # not a reasoning model
POLICY_RESIDENCY=eu
POLICY_REQUIRE_DPA=true
POLICY_FORBID_TRAINING=true
PROVIDER_METADATA='{"azure": {"region": "eu", "retention": "abuse_monitoring_30d",
  "trains_on_data": false, "dpa": true,
  "notes": "mistral-medium-3-5, EU Data Zone deployment, Microsoft DPA"}}'
```

`anthropic`, `groq` and `openai` have no metadata, so they are excluded
before any call. If the Azure deployment fails, the call raises `LLMError`
instead of failing over to them. A `gpt-oss-120b` deployment on Azure is
GlobalStandard only (no EU Data Zone), so its honest metadata is
`"region": "global"`, which `POLICY_RESIDENCY=eu` excludes. The library
doesn't assert that either: `azure` defaults to `region=unknown` like every
other provider.

The same in code, with a per-call cost cap on top:

```python
from llm_gateway import GatewayConfig, PolicyViolationError, ProviderPolicy, complete_with_usage

config = GatewayConfig(
    provider_metadata={
        "azure": {
            "region": "eu",
            "retention": "abuse_monitoring_30d",
            "trains_on_data": False,
            "dpa": True,
        }
    },
    policy_residency="eu",
    policy_require_dpa=True,
    policy_forbid_training=True,
)
try:
    result = await complete_with_usage(
        system="Score the answer.",
        prompt=answer,
        config=config,
        policy=ProviderPolicy(max_cost_usd=0.02),
    )
except PolicyViolationError as e:
    e.exclusions  # {"anthropic": ("region=unknown does not match residency=eu", ...), ...}
```

#### EU-only example with Vertex

Claude on Vertex's `europe-west1` endpoint processes prompts inside the EU
(see [Claude on Google Vertex AI](#claude-on-google-vertex-ai-060)). Assert that,
and only what your Google Cloud agreement says for the rest:

```bash
PROVIDER_ORDER=anthropic,vertex,groq
VERTEX_PROJECT_ID=my-vertex-project
VERTEX_LOCATION=europe-west1
POLICY_RESIDENCY=eu
POLICY_REQUIRE_DPA=true
PROVIDER_METADATA='{"vertex": {"region": "eu", "dpa": true,
  "notes": "Claude Haiku 4.5 on Vertex AI europe-west1, Google Cloud DPA"}}'
```

`anthropic` and `groq` have no metadata, so every call goes to Vertex only, and
a Vertex failure raises `LLMError` instead of leaving the EU. Assert
`"region": "eu"` only for an EU regional endpoint (or `eu` for a model that
offers it). With `VERTEX_LOCATION=global`, the honest metadata is
`"region": "global"`, which `POLICY_RESIDENCY=eu` excludes. Without a residency
policy, the same `PROVIDER_ORDER` makes Vertex a plain, separately billed
fallback for `anthropic`.

#### Capability filter

Before calling, the gateway skips a provider whose model is **known** not to
support a feature the request uses: `output_schema`, `tools` (unless
`tool_choice="none"`), image input, or streaming. Before 0.5 such a request
went out, got a 400, and failed over. With `require_parameters`, it also
skips models whose support is **unknown**, and Groq models that only have
JSON mode (anything but `openai/gpt-oss-20b` / `-120b`, from
`providers/groq.py`) for `output_schema`. The table lives in
`llm_gateway/capabilities.py`. It records only facts from the vendors' docs.
Azure deployment names are the operator's choice, so an `azure` deployment
is recognized only when named after gpt-oss (`gpt-oss-*`: strict structured
output, tools, streaming, no images); any other deployment is unknown. When capabilities alone empty the
chain, the call raises `UnsupportedCapabilityError`, a subclass of
`PolicyViolationError`, naming the missing capability per provider.

#### Cost: `cost_usd`, `max_cost_usd`, budget hook, `sort`

- **`Completion.cost_usd`** is computed from the served call's actual usage:
  uncached input, cache reads, cache writes and output, each at its own
  rate. It is `None` when the model has no price, never a guessed `0`. It is
  excluded from `Completion` equality.
- **Prices** (USD per 1M tokens) live in `llm_gateway/pricing.py`, keyed
  `"provider/model"`. The defaults cover only the default models, at list
  price, checked 2026-09-14: `claude-haiku-4-5` ($1 input, $0.10 cache read,
  $1.25 cache write, $5 output), `gpt-4o-mini` ($0.15 / $0.075 cached /
  $0.60) and Groq's `openai/gpt-oss-120b` ($0.15 / $0.60). Azure deployments
  have no default (the price depends on the deployment's model and SKU), and
  neither does Groq's `llama-3.3-70b-versatile` ("Contact sales"): their
  cost is `None` until you set one. Add or replace entries with
  `MODEL_PRICES`, e.g.
  `{"azure/mistral-medium-3-5": {"input_per_mtok": ..., "output_per_mtok": ...}}`.
  Negotiated discounts, batch pricing and regional premiums are not applied.
- **`max_cost_usd`** skips providers whose *estimated worst case* exceeds the
  cap: prompt characters / 4 as input tokens, at the higher of the input and
  cache-write rates, plus the full `max_tokens` at the output rate. Output is
  the dominant term and is charged in full, so the estimate is conservative
  for text. Chars/4 can undercount non-English text and code, and image
  tokens aren't estimated. A provider whose model has no price is skipped,
  because the cap can't be checked.
- **`budget_check`** (`ProviderPolicy(budget_check=fn)`): the host app's
  hook, sync or async, called right before each provider attempt as
  `fn(provider, model, estimated_cost_usd)`. Return falsy to deny. A denial
  skips that provider like any other exclusion and **never counts toward
  its circuit breaker**. If every provider is denied, the call raises
  `PolicyViolationError`. An exception from the hook propagates. The library
  keeps no per-user state or spend ledger; the hook is where yours plugs in.
- **`sort="price"`** tries the cheapest estimated provider first (unpriced
  last, ties in `provider_order`). The default `order` keeps
  `provider_order`. There is no latency sort.

#### Errors and observability

`PolicyViolationError` (a subclass of `LLMError`, `ErrorKind.POLICY_VIOLATION`:
not retried, never counted by the breaker) names the effective policy and
each excluded provider with its reasons, in the message and as
`.exclusions`. It never includes prompt content. The same exclusions are
logged at DEBUG on the `llm_gateway` logger and recorded on the call's span
as `llm_gateway.policy`, `llm_gateway.policy.eligible_providers`,
`llm_gateway.policy.excluded_providers` and
`llm_gateway.policy.exclusion_reasons`. `complete_with_usage()` spans also
record `llm_gateway.cost_usd`.

#### Over HTTP

`POST /v1/chat/completions` accepts the same fields (minus `budget_check`)
as a `policy` object:

```json
{"messages": [...], "policy": {"residency": "eu", "require_dpa": true, "only": ["azure"]}}
```

It is merged with the server's `POLICY_*` settings under the same rules, so
a client can narrow the server policy but never loosen it. Unknown fields
are rejected with 422, an invalid provider id or residency with 400, and a
request no provider may serve with 400 (`invalid_request_error`) before any
provider is called. For `"stream": true` that last case is an SSE error
event with code 400.

#### Settings

| Variable | Default | Purpose |
|---|---|---|
| `PROVIDER_METADATA` | `{}` | JSON object: provider id → `region` / `retention` / `trains_on_data` / `dpa` / `notes`. Operator-asserted. |
| `POLICY_RESIDENCY` | — | Only providers whose `region` equals this, e.g. `eu`. |
| `POLICY_REQUIRE_ZERO_RETENTION` | `false` | Only `retention=zero`. |
| `POLICY_FORBID_TRAINING` | `false` | Only `trains_on_data=false`. |
| `POLICY_REQUIRE_DPA` | `false` | Only `dpa=true`. |
| `POLICY_ONLY` | — | Comma-separated provider ids allowed. |
| `POLICY_IGNORE` | — | Comma-separated provider ids never used. |
| `POLICY_REQUIRE_PARAMETERS` | `false` | Also skip models with unknown support for a requested feature, and JSON-mode-only models for `output_schema`. |
| `POLICY_MAX_COST_USD` | `0` | Cap on a call's estimated worst-case cost. `0` disables it. |
| `POLICY_SORT` | `order` | `order` or `price`. |
| `MODEL_PRICES` | `{}` | JSON object: `"provider/model"` → `input_per_mtok`, `output_per_mtok`, optional `cached_input_per_mtok`, `cache_write_per_mtok`. Merged over the defaults. |

Not covered yet: `chat()` / `stream_chat()` results don't carry `cost_usd`;
`GET /v1/models` availability ignores the policy; the HTTP service has no
budget hook.

### OpenAI-compatible hosts: Mistral, OpenRouter, any (0.7.0)

Three more provider ids share one implementation (`providers/openai_compatible.py`)
for hosts that speak the OpenAI Chat Completions API at their own base URL:

| id | Host | Notes |
|---|---|---|
| `mistral` | `https://api.mistral.ai/v1` | Mistral AI, a French company with an EU-hosted API. Tools, streaming and strict `json_schema` output. |
| `openrouter` | `https://openrouter.ai/api/v1` | A US broker that forwards to many third-party hosts. `OPENROUTER_PROVIDER_ALLOW` pins the upstream hosts. Structured output uses JSON mode + local validation, since strictness depends on the upstream host. |
| `openai_compat` | `OPENAI_COMPAT_BASE_URL` | Whatever you point it at. Declare what it supports with `OPENAI_COMPAT_SUPPORTS_TOOLS` / `OPENAI_COMPAT_STRICT_JSON_SCHEMA`. |

Add them to `PROVIDER_ORDER` like any other id; the breaker, retries, failover,
policies and the price table treat them the same way. The library asserts no
residency, retention or training facts for any of them: a `residency=eu` policy
excludes `mistral` until your `PROVIDER_METADATA` says `{"mistral": {"region": "eu"}}`,
and `openrouter` is only as EU as the hosts you pin. Default prices cover
`mistral/mistral-small-latest` and `openrouter/openai/gpt-oss-120b`; set
`MODEL_PRICES` for anything else (an `openai_compat` model has no default).

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
| `VERTEX_PROJECT_ID` | — | Google Cloud project for Claude on Vertex AI; see [Claude on Google Vertex AI](#claude-on-google-vertex-ai-060). Needs the `[vertex]` extra |
| `VERTEX_LOCATION` | `europe-west1` | `global`, `eu`/`us` multi-region, or a region |
| `VERTEX_MODEL` | `claude-haiku-4-5@20251001` | Vertex model id |
| `VERTEX_CREDENTIALS_FILE` | — | Credential configuration file (e.g. Workload Identity Federation `external_account`); unset uses ADC. Key files are rejected |
| `VERTEX_IMPERSONATE_SERVICE_ACCOUNT` | — | Service account email to impersonate |
| `VERTEX_AZURE_APP_ID_URI` | — | WIF from an Azure managed identity: the Entra application ID URI (needs the `[azure]` extra) |
| `VERTEX_AZURE_MANAGED_IDENTITY_CLIENT_ID` | — | User-assigned managed identity for `VERTEX_AZURE_APP_ID_URI`; unset uses the system-assigned identity |
| `VERTEX_STRUCTURED_OUTPUTS` | `false` | Route `output_schema` calls to Vertex. Enable only after `constraints/vertexai.allowedPartnerModelFeatures` allows `publishers/anthropic/models/claude-haiku-4-5:structured_outputs` |
| `MISTRAL_API_KEY` | — | Mistral AI (EU company, EU-hosted API); see [OpenAI-compatible hosts](#openai-compatible-hosts-mistral-openrouter-any-070) |
| `MISTRAL_MODEL` | `mistral-small-latest` | |
| `OPENROUTER_API_KEY` | — | OpenRouter, a broker that forwards to third-party hosts |
| `OPENROUTER_MODEL` | `openai/gpt-oss-120b` | OpenRouter ids are `<vendor>/<model>` |
| `OPENROUTER_PROVIDER_ALLOW` | — | Comma-separated upstream hosts to pin routing to; requests fail rather than route elsewhere |
| `OPENROUTER_APP_URL` / `OPENROUTER_APP_NAME` | — | Sent as `HTTP-Referer` / `X-Title` (OpenRouter app attribution) |
| `OPENAI_COMPAT_BASE_URL` | — | Any host speaking the Chat Completions API (DeepSeek, Together, Fireworks, a local vLLM…). Registers provider id `openai_compat` |
| `OPENAI_COMPAT_API_KEY` | — | Its key; a keyless local server takes any non-empty value |
| `OPENAI_COMPAT_MODEL` | — | Its model id |
| `OPENAI_COMPAT_SUPPORTS_TOOLS` | `true` | Whether that host takes `tools` |
| `OPENAI_COMPAT_STRICT_JSON_SCHEMA` | `false` | Whether it enforces `json_schema`; otherwise JSON mode with the schema in the prompt |
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
