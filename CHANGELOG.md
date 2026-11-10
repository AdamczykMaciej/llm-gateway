# Changelog

## 0.42.7

Second coverage pass (98.85% to 99.82%, floor unchanged at 98.8 because 99.8 would leave under
0.05 headroom). Tests only, no behaviour change and no bug found; the branches now covered are the
risky ones: privacy-layer default entry points, stream tail flush and closer-less streams; Azure
Entra credential and token failures; adaptive thinking; the Anthropic streaming model fallback
(retried once before the first chunk, never after); tool-id remapping collisions; retired fallback
models; service shutdown grace and guarded streaming responses; rate-limit window ageing; the
single-flight cancel-with-last-waiter rule; numeric host forms in the SSRF checks; scrubbing of SDK
requests without cached content.

- Internal: one provably unreachable defensive return in the think-tag stripper is marked
  `# pragma: no cover` with its reason.

## 0.42.6

API ergonomics pass. Probed and found sound, no change: `__all__` (unique, resolvable, documented,
no private names), `py.typed` in the wheel, fully annotated public functions, overloads of
`complete_with_usage` (typed `Completion[Model]` for `output_schema=` / `validate=`), README
quickstart and every fenced block executed against fakes, every error class in the README
taxonomy with when it is raised and whether it is retryable, `__version__`-style strings (the HTTP
service reads the installed package version). No deprecated public names exist, so there is
nothing to warn about. Provider names stay `str`: the set is open (any OpenAI-compatible host).

- Types: new `FailureClass` `Literal` (exported) for the closed set of `failure_class()` labels;
  `failure_class()` and `ProviderFailure.kind` are typed with it, `FAILURE_CLASSES` is derived from
  it, and `ResponseTooLargeError.kind` is `Literal["body", "sse_event", "stream"]`. Runtime values
  are unchanged.
- Docs: the README classification table lists `POLICY_VIOLATION`; this changelog gains the missing
  0.35.2 entry. `tests/test_ergonomics_pass_10.py` fails if a version from 0.29.0 to the current
  one has no changelog entry.

## 0.35.2

Version bump only; no code or documentation change (0.35.3 carries the exception-chain scrubbing).

## 0.42.5

Agentic pass 9. Probed with fakes and found sound, no change: tool-call translation across
providers (parallel calls, ids kept and matched, results grouped per Anthropic turn), invalid
JSON / schema-violating / undeclared tool calls (typed `ToolCallIssue` or `ToolCallInvalidError`,
never a crash, values never echoed), streamed tool-call fragments (split JSON, interleaved
indexes, same-index hosts), structured output (nullable types, enums, nested arrays, bounded and
accounted repair, truncation), message-history pairing validation and system placement. Tracing
records only a tool-call count and no content; the gateway does not execute tools, so it emits no
`execute_tool` spans. `idempotency_key` is in-flight de-duplication only, by design: there is no
result cache, so no TTL.

- Fix: `chat()` and `complete_with_usage()` with an `idempotency_key` treated two concurrent calls
  that differed only in `policy` (or the config's policy) as the same request and handed the
  second caller the first one's result, which may have come from a provider the second policy
  forbids. The policy is now part of the request fingerprint, so such a pair raises
  `InvalidRequestError` like any other mismatch.

## 0.42.4

Accounting pass 8. Probed with fakes and found sound, no change: a stream cut after its usage chunk
is reported once on its own failed attempt (`partial=True`) and the restart's usage on its own
attempt, never merged or doubled; usage from the final chunk only, early close and cancellation
(the attempt keeps whatever usage had arrived); Anthropic cache read/write added to
`input_tokens` once, OpenAI `cached_tokens` kept as a subset (no double count), reasoning tokens
already inside output; price lookup is exact-key, so `-mini`, dated or unknown ids never fall back
to another model's price (unknown gives `None`, not 0); prices are USD per 1M tokens throughout.
Limitation, unchanged: an Anthropic stream cut before its final event reports no usage (it is
only emitted at the end), so `partial` costs are a lower bound. No prices touched or re-verified.

- Fix: a negative token count from a provider (a buggy host or proxy) was kept, so usage could be
  negative and `cost_usd` could subtract. Counts are now clamped at 0 on parse and in `cost_usd`.
- Fix: the cost summed over structured-output repair attempts carried float drift
  (`0.1 + 0.2 = 0.30000000000000004`); it is now rounded to 10 places like a single cost.

## 0.42.3

Reliability pass 7. Probed with fakes and found sound, no change: cancellation mid-call, during a
backoff sleep and mid-stream (`CancelledError` re-raised, provider stream closed, half-open probe
released, no failover or breaker count); an outer `wait_for` timeout around a stream; no retry,
backoff or fallback past the call deadline; no silent provider switch after streamed tokens
(mid-stream restart stays opt-in and emits a restart marker); half-open admits exactly one probe
under 32 racing threads (one lock); malformed `Retry-After` values ignored.

- Fix: `Retry-After` in HTTP-date form (RFC 9110, the other legal form) was ignored, so the
  provider's requested wait was lost. It is now honoured, capped at 60 s like the seconds form; a
  date already past counts as 0.

## 0.42.2

Security pass 6. Dependency audit (`pip-audit`, all extras): only `pip` itself (PYSEC-2026-3721,
tooling, not a runtime dependency); no minimum bumped. Checked and found sound, no change: SDK
clients all built via `sdk_http_client` (redirects off), prompt text in spans opt-in
(`trace_include_prompts`) and responses never recorded, `request_id`/`idempotency_key` validated
with `fullmatch` (no CR/LF reaches a header), no eval/exec/yaml/pickle on input.

- Fix: `GatewayConfig.model_dump()` / `model_dump_json()` included every API key, the gateway keys
  and the OTLP endpoint/headers (only `repr` hid them), so logging a dump leaked credentials. Those
  fields are now `exclude=True`. Attribute access, `model_copy` and env loading are unchanged;
  rebuilding a config from a dump no longer carries the credentials.

## 0.42.1

Tests only; no behaviour change. Coverage 98.60% to 98.82% (floor raised to 98.8). New
`tests/test_coverage_gaps_5.py` covers the non-streaming `chat()` / `complete_with_usage()`
admission branches (policy and budget denial, lost breaker admission, unregistered provider),
tool-definition validation edges (reserved name, non-JSON extras, size cap), structured-output
limits (non-array reply, recursion, cyclic `$ref`), best-effort scrubbing of SDK errors that carry
HTTP objects, and the request body cap. No bug was exposed.

## 0.42.0

Error-taxonomy audit (`classify`, retry/breaker policy) against the providers' current error docs, read
2026-11-10. Mappings that already matched were left alone: Anthropic 400/413/429/500/504/529,
OpenAI `insufficient_quota` vs `rate_limit_exceeded`, `context_length_exceeded` and Azure
`content_filter` (400, fail fast, never counts against the breaker), Gemini/Vertex 429
`RESOURCE_EXHAUSTED`, 503 `UNAVAILABLE`, OpenRouter 402/408/429/502/503, Groq 413/422/429/5xx.
Pages: platform.claude.com/docs/en/api/errors, developers.openai.com/api/docs/guides/error-codes,
console.groq.com/docs/errors, openrouter.ai/docs/api-reference/errors,
aistudio.google.com/docs/troubleshooting, learn.microsoft.com (Foundry content filtering). The
Mistral and Hugging Face pages documented no error shapes or could not be fetched, so their
mappings are unchanged.

- Fix: Groq HTTP 498 (flex tier at capacity) was `invalid_request`; it is now `rate_limited`
  (fail over, no same-provider retry).
- Fix: OpenRouter HTTP 403 is shared by permissions, moderation flags, guardrail blocks and budget
  blocks. A moderation/guardrail 403 (`metadata.flagged_input`/`reasons`/`patterns`) is now
  `invalid_request` (no breaker trip) instead of `auth`; a budget 403 (`metadata.limit_source`)
  is `billing` (breaker opens for the billing cooldown). Other 403s stay `auth`.
- Fix: Anthropic `billing_error` (402) arriving inside a 200 stream is now `billing`; it was `unknown`.
- Fix: OpenAI `credit_balance_exhausted` on a 429 and Gemini's depleted "prepayment credits" 400
  are now `billing` (they were `rate_limited` / `invalid_request`, so every request kept calling).
- Documented policy, unchanged: a content-filter block fails over to the next provider in the
  chain (another provider's filter may accept it) without counting against the breaker; a
  context-length error is `invalid_request`, fails fast on the same provider and does not trip the breaker.

## 0.41.0

Structured-output audit (`output_schema=`, `validate=`, `repair_attempts=`). Every provider already
sent its native mechanism (see the README table) and validated the reply before returning it; the
gaps were in how odd replies were handled.

- New: `OutputTruncatedError` (subclass of `InvalidOutputError`). A structured reply that stopped on
  a token limit (`length`, `max_tokens`) and is not complete JSON now raises it instead of a generic
  parse error. It fails over as before, and `repair_attempts` no longer spends a paid call on it
  (the repair would hit the same limit): raise `max_tokens` instead.
- Fix: a cut-off reply is no longer parsed leniently, so a complete object embedded in a truncated
  one (`{"a": {"b": 1}, "c": [1,`) cannot be returned as the answer.
- Fix: a leading BOM is ignored (it used to break scalar and fenced replies).
- Fix: data nested deeper than the validator's depth limit (64) is reported as a validation error
  at that path; it was silently accepted. Deeply nested JSON handled by a pydantic model also
  raises `InvalidOutputError` instead of `RecursionError`.
- Fix: shallow dict-schema failures (not an object/array, missing required key) now carry `paths`
  like every other validation failure.
- The repair prompt echoes at most 8000 characters of the previous reply.
- Unchanged and now documented: a pydantic model coerces `"3"` to `3`; a dict schema rejects it.

## 0.40.0

Early-close reliability audit of `stream_chat`. The engine already closed the provider generator,
released the breaker probe, freed the idempotency slot, reported one `cancelled`/partial attempt and
ended spans without an error status on `break`, exceptions in the loop body, task cancellation and
client disconnect; these are now pinned by tests (1000 early closes leave no open upstream
connection, also under asyncio debug mode, and the service stops the upstream call on disconnect).

- Fix: the OpenAI-SDK-backed providers (OpenAI, Groq, Azure, and the OpenAI-compatible family:
  Mistral, OpenRouter, Hugging Face, ...) iterated the SDK stream without closing it, so a consumer
  that stopped early left the HTTP response (and its pooled connection) open until garbage
  collection. The stream is now closed when the provider generator exits. No API change.

## 0.39.0

Model catalogue freshness (audit against the vendors' pages, read 2026-11-10). Existing entries,
default models and default prices were re-checked and are unchanged; additions only.

- Retired-model block list, Anthropic
  (https://platform.claude.com/docs/en/about-claude/model-deprecations): `claude-3-sonnet-20240229`,
  `claude-2.1`, `claude-2.0` (retired 2025-07-21).
- Groq (https://console.groq.com/docs/deprecations): `meta-llama/llama-guard-4-12b` (2026-03-05),
  `moonshotai/kimi-k2-instruct` (2025-10-10), `gemma2-9b-it` (2025-10-08),
  `deepseek-r1-distill-llama-70b` (2025-10-02), `llama3-70b-8192` and `llama3-8b-8192` (2025-08-30),
  `mistral-saba-24b` (2025-07-30), `qwen-qwq-32b` (2025-07-14).
- OpenAI (https://developers.openai.com/api/docs/deprecations), all 2026-10-23: the `gpt-4.1-nano`
  alias (only the dated id was listed), `gpt-4`, `gpt-4-0613`, `gpt-4-turbo`,
  `gpt-4-turbo-2024-04-09`, `o1`, `o1-2024-12-17`, `o3-mini`, `o3-mini-2025-01-31`,
  `gpt-3.5-turbo`, `gpt-3.5-turbo-0125`, `o4-mini`, `o4-mini-2025-04-16`.
- Gemini (https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions):
  `gemini-1.5-pro-001/002` and `gemini-1.5-flash-001/002` (retired).
- New default prices: `anthropic/claude-fable-5-2` ($8 in / $40 out, cache 10 / 16 / 0.20;
  https://platform.claude.com/docs/en/about-claude/pricing) and `openai/gpt-6-luna` ($0.10 / $0.50,
  cached $0.01, batch half; https://developers.openai.com/api/docs/pricing).
- Not verifiable: the Vertex pricing page did not render its tables, so the Vertex Claude Haiku 4.5
  price stays marked unverified. Defaults checked and still live: `claude-haiku-4-5-20251001` (docs say
  not retiring sooner than 2026-10-15; still Active), `gpt-4o-mini` (listed on the pricing page; only
  its dated snapshot retires 2027-01-14, so the alias is not blocked), `gemini-3.5-flash`,
  `openai/gpt-oss-120b`.

## 0.38.0

Prompt-injection boundary hardening (audit of `untrusted.py`), backward compatible.

- Fix: `wrap_untrusted` removed `<<<`/`>>>` in one pass, so `<<>>><` reassembled into a live `<<<`;
  removal now runs to a fixpoint. Fullwidth and angle-quote lookalike runs (`＜＜＜`, `«««`, ...) are
  removed too.
- Fix: `sanitize_untrusted` stripped HTML comments before hidden characters, so `<!<ZWSP>--` survived
  as a comment; hidden characters go first and comments are removed to a fixpoint.
- Fix: `ApprovalDenied.message()` echoed the model-chosen tool name verbatim; it is now sanitised.
- New `tool_result_message()` wraps a tool result as a `role: "tool"` message.
- Docs: clarified what is and is not protected (fencing is not a security boundary).
- Behaviour change: content containing the lookalike runs above loses them (previously kept).

## 0.37.1

Fix: the service's OpenAPI `info.version` was hard-coded to `0.9.0`; it now reports the installed
package version. New edge-path tests (message-shape validation, error scrubbing, think-stripping,
privacy streams).

## 0.37.0

Retry amplification under an outage: a per-call request cap and a faster breaker. Defaults unchanged
except where noted.

- Worst case per call was `(1 + repair_attempts) x providers x retry_attempts` requests (9 with the
  defaults and three providers, 54 with `repair_attempts=5`); SDK-level retries were already 0.
  New `max_attempts_per_call` (`MAX_ATTEMPTS_PER_CALL`, default 0 = no cap) bounds every request of
  one call, including repair calls.
- New `breaker_count_retries` (`BREAKER_COUNT_RETRIES`, default false): retried failures count toward
  the breaker, so N concurrent callers stop after about `breaker_failure_threshold` requests per provider.
- Always on: a call mid-retry stops retrying a provider whose breaker other callers just opened
  (a half-open probe is unaffected; `force_provider` still retries).
- Behaviour change: a provider's `Retry-After` is now jittered upwards by up to 10% (never below the
  requested wait) so callers do not return in the same instant.
- New `breaker.tripped()`; `call_with_retry(on_retry_failure=)`.

## 0.36.0

Observability: more OpenTelemetry GenAI attributes, nothing removed or renamed.

- Call spans (`llm_gateway.complete` / `.chat` / `.stream_chat`) add `gen_ai.request.max_tokens`.
- `llm_gateway.complete` now sets `gen_ai.response.finish_reasons` (the provider stop reason,
  with `llm_gateway.finish_reason` as alias), as `chat` already did.
- Successful `llm_gateway.attempt` spans add `gen_ai.response.model` and, when known,
  `gen_ai.response.finish_reasons`.
- Prompt content stays off by default (`TRACE_INCLUDE_PROMPTS`, PII-masked); responses are never recorded.

## 0.35.3

Security: SDK error attributes are scrubbed, not just their text.

- `errors.scrub_exception_chain` (run by every `AllProvidersExhaustedError` /
  `LLMDeadlineExceeded` the gateway builds) now also scrubs the chained SDK error's `body`,
  `param`, `code`, `response` and `request` (openai, anthropic, httpx/httpx2 `HTTPStatusError`,
  and any exception with string/dict/list or `httpx`-style attributes). Error trackers such as
  Sentry serialise these, and they carried request headers (`Authorization`, `x-api-key`,
  `api-key`), URLs with `?key=` and echoed bodies.
- Exception types are unchanged. Still available on `__cause__`: `status_code`,
  `response.status_code`, response headers (`Retry-After`), request method and host, and the
  shape of `body` (error `type` / `code`). Credential headers read `[REDACTED]`, credentials in
  URLs and body text are redacted, and the request body (the prompt) is dropped.
- Not covered: exceptions the gateway never chains (raised to your own code directly) and
  secrets you copy out of frame locals yourself.

## 0.35.1

Documentation accuracy. No behaviour change in `src` (docstrings only).

- Docs: every fenced `python` block in `README.md` and `MIGRATION.md` is now executed by
  `tests/test_docs_blocks.py` against fake providers; fragments that lacked imports or used `...`
  as live code were completed. `tests/test_docs_coverage.py` fails if a public name or a
  `GatewayConfig` field is missing from the README, or an environment-variable default disagrees
  with the code.
- Docs: new README "Public API reference" and the previously missing environment variables
  (`BREAKER_BILLING_COOLDOWN_SECONDS`, `BREAKER_MODEL_MISSING_COOLDOWN_SECONDS`,
  `STREAM_MAX_RESTARTS`, `ALLOW_INSECURE_BASE_URLS`, `STREAM_MAX_*`, `MAX_RESPONSE_BODY_BYTES`,
  `MAX_SSE_EVENT_BYTES`, `MAX_STREAM_BYTES`, `MAX_REQUEST_BYTES`).
- Docs: README and MIGRATION "Upgrading from 0.28.0 to 0.35.0". Changelog cross-references added
  to 0.31.0 (size caps, fixed in 0.32.0), 0.33.0 (flagged as a behaviour change) and 0.34.0
  (`stream_chat` validation, added in 0.35.0).

## 0.35.0

`stream_chat` gets `validate_tool_calls`. Additive; the default changes nothing.

- New: `stream_chat(validate_tool_calls="off"|"warn"|"error")`, same checks as `chat()` (unknown
  tool, arguments vs schema) plus arguments that are not a JSON object (`arguments_not_json`),
  run once the streamed fragments are complete (the provider stream ended). `warn` streams as
  usual and logs names and paths only. `error` withholds tool-call fragments (and everything after
  the first) until the check passes, then yields them in order; text before them streams at once.
  On failure it fails over only if nothing was yielded (no text, no restart marker); otherwise
  `ToolCallInvalidError` is raised to the consumer with no failover. Rejected attempts are reported
  to `on_attempt` as failed, with their usage. No `tools` means no check; bad values are
  `InvalidRequestError`.
- Parity audit `chat` vs `stream_chat`: `deadline_seconds`, `idempotency_key` (header, spans,
  reports; stream has no single-flight by design), `policy`, size caps (`stream_max_*`; `chat`
  gets whole objects) and pricing (shared `lookup_price`, serving model) already match.
  Remaining differences, left as is: `chat()` has no `cache_system`; `stream_chat` has no
  single-flight; the HTTP service exposes none of these per-call options.

## 0.34.0

Opt-in validation of the tool calls a model returns. Additive; the default changes nothing.

- New: `chat(validate_tool_calls="off"|"warn"|"error")`. Each reply's tool calls go through
  `check_tool_calls` inside the attempt (unknown tool, arguments vs the tool's JSON Schema,
  arguments over `stream_max_tool_arguments_chars`). `warn` logs names and paths only; `error`
  raises the new `ToolCallInvalidError` (an `InvalidOutputError`, with `issues`), so the engine
  fails over. Default off: strict checking could break callers whose tools have schemas the
  model does not follow. Any other value is `InvalidRequestError`.
- Audit, no change needed: pre-flight checks (0.14), truncated tool calls (0.16), duplicate/empty
  ids (`unique_tool_call_ids`), tool-result pairing and malformed-JSON handling are consistent.
  Not done: an automatic repair retry (a failover or the `tool_error_message` loop does this);
  `stream_chat` does not validate (callers assemble fragments; added in 0.35.0).

## 0.33.0

Cost accounting audit: unknown usage is no longer priced as free. **Behaviour change**:
`cost_usd` can now be `None` where it was `0.0`; handle it in any ledger or budget code.

- Fixed: a reply with no usage block (some `openai_compat` / `huggingface` hosts, or any host that
  omits it) gave `Usage()` all zeros and `Completion.cost_usd == 0.0`, and a 0.0 `AttemptReport.cost_usd`,
  so a ledger under-reported spend. `cost_usd()` now returns `None` when a call reports neither input nor
  output tokens. New `Completion.usage_reported` is False in that case.
- Audit, no change needed: Anthropic cache reads/writes are added back into `input_tokens` and priced
  separately (docs: platform.claude.com prompt-caching); OpenAI-style `cached_tokens` stay inside
  `prompt_tokens`; reasoning tokens are a subset of output; stream usage is taken from the last usage
  chunk once per attempt and each failed attempt is reported on its own; cost uses the serving model.
  Not verified: whether Gemini's OpenAI endpoint counts thinking tokens in `completion_tokens`.

## 0.32.0

Bounded response reading. Additive; the defaults never affect normal LLM replies.

- New: `max_response_body_bytes` (32 MiB), `max_sse_event_bytes` (8 MiB) and `max_stream_bytes`
  (128 MiB) on `GatewayConfig` (env `MAX_RESPONSE_BODY_BYTES`, ...; 0 disables). A response-event
  guard on every SDK client counts decoded bytes (compression bombs included), refuses an
  oversized Content-Length before reading, closes the connection at once and raises the new
  `ResponseTooLargeError` (`kind`, `limit`). It subclasses `StreamTooLargeError`, so it is an
  `InvalidOutputError`: failover, no same-provider retry and no circuit-breaker count (the
  provider answered; size says nothing about availability). `classify` looks through the SDK's
  connection-error wrapper.
- Covers openai, azure, groq, openai_compat, huggingface, mistral, openrouter, gemini, anthropic
  and vertex. Not covered: credential fetches by `google-auth` / `azure-identity`. A decoder step
  can expand one wire chunk (~64 KiB, ~1000x for gzip) past a cap before it is checked. With
  `sdk_max_retries > 0` the SDK may re-request after an oversized reply.
- Changed: the three limits are part of the SDK client cache key.

## 0.31.0

Base URL and model-id hardening. **Behaviour change** (safe defaults, one opt-out).

- Changed: `openai_compat_base_url`, `huggingface_base_url` and `azure_endpoint` must now be
  `https://` and must not name `localhost` or a loopback, private, unspecified or unique-local IP
  literal. Set `allow_insecure_base_urls=True` (`ALLOW_INSECURE_BASE_URLS=true`) for local
  development, e.g. `http://localhost:11434/v1`. Metadata/link-local hosts stay refused either way.
- Fixed: the vertex provider wrote the model id (config or per call) into the request URL path
  unchecked; ids with `..`, `/`, `?`, `#`, `%` or whitespace now raise `InvalidRequestError`.
- Known limit: non-streamed response bodies and SSE lines have no size cap yet (fixed in 0.32.0).

## 0.30.0

Pre-call cost estimates now cover the Claude model fallback. Additive; the shape of
`RoutableModel` / `CostEstimate` is unchanged.

- Fixed: `routable_models()` / `estimate_max_cost()` ignored `claude_model_fallback`, which the
  anthropic provider retries once with on a model rejection (`complete`, `chat` and `stream_chat`).
  A budget reserved for a cheap primary with a dearer fallback was too low. The fallback is now an
  extra `RoutableModel` right after the anthropic primary, when anthropic is routable, the fallback
  differs from `claude_model`, is not retired (with `block_retired_models`) and no explicit `model`
  is passed. An unpriced fallback makes `max_cost_usd` `None`, never 0.
- Fixed: `ProviderPolicy.max_cost_usd` and `sort="price"` priced anthropic by the primary model
  only; they now use the larger of primary and fallback (an unpriced fallback counts as unpriced).

## 0.29.0

Per-call end-to-end deadline. Additive; the default keeps today's behaviour.

- New: `deadline_seconds=` on `complete()`, `complete_with_usage()`, `chat()` and `stream_chat()`
  sets one total time budget for that call, replacing `config.call_deadline_seconds`. It bounds
  attempts, backoff and `Retry-After` waits, failover and streams already under way; the time
  left caps each attempt's timeout. Non-positive, non-finite or non-numeric values raise
  `InvalidRequestError`. It is not part of the idempotency fingerprint.
- Fixed: `repair_attempts` calls each started a fresh deadline, so a call could run several times
  past its budget. They now share the first call's budget.
- Fixed: the `LLMDeadlineExceeded` message reports the budget actually in force.

## 0.28.0

Correct pre-call cost estimates. Additive; no behaviour change to existing calls.

- New: `estimate_max_cost(config, input_tokens=, max_output_tokens=, policy=, provider=, model=,
  chat=)` returns a `CostEstimate` priced over only the models the call can route to, found with
  the router's own `plan_chain` (provider chain, credentials, retired models, config and per-call
  policy). An unrelated price row, such as a new premium model, no longer changes it.
- New: `routable_models(config, ...)`, plus `CostEstimate` and `RoutableModel`, all exported.
- An unpriced routable model gives `max_cost_usd=None` and is listed in `unpriced`, never 0;
  an unroutable call returns an empty estimate with `excluded` reasons instead of raising.
- Docs: README "Reserving budget before a call" and a MIGRATION entry.

## 0.27.1

Supply-chain and release hygiene. No runtime behaviour change.

- Changed: upper bounds added to `typing-extensions` (<5) and to the dev tools (pytest-asyncio,
  pytest-cov, ruff, pyright, build).
- New: `constraints/dev.txt` (pinned test/dev environment) and `scripts/lock.sh` to regenerate it;
  CI installs with it and all extras.
- New: `scripts/audit.sh` (pip-audit, osv-scanner if installed) with an ignore list,
  `scripts/audit-ignore.txt`, one `<ID> <expiry> <reason>` per line, expired entries fail.
  CI runs it. pip-audit: no known vulnerabilities.
- New: `scripts/sbom.sh` writes a CycloneDX SBOM; `pip-audit` and `cyclonedx-bom` join the dev extra.
- New: `tests/test_dist_contents.py` builds the wheel and sdist and asserts their file lists.
- Docs: SECURITY.md supply-chain section. LICENSE (MIT) and `py.typed` were already present.

## 0.27.0

API ergonomics. Typing and message text only; no runtime behaviour change.

- New: precise `TypedDict`s, exported from `llm_gateway` (module `llm_gateway.types`): `Message`
  (`SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolMessage`), content parts (`TextPart`,
  `ImagePart`, `ContentPart`), `ToolCallMessagePart`, `ToolSpec`, `FunctionSpec`, `ToolChoice`
  and `NamedToolChoice`. `chat()` / `stream_chat()` now take `Sequence[Message | dict]`,
  `Sequence[ToolSpec | dict]` and `ToolChoice | dict | str | None`, so plain `list[dict]` still
  type-checks and nothing is validated or converted at runtime. `tests/typing/consumer.py` is a
  pyright-checked consumer example.
- Changed: every `GatewayConfig` validation error now reads `<field> <problem>. Got <value>.
  Fix: <how>.` Secret-like fields (`*_key`, `*_token`, ...) show `<redacted, N chars>`, URL
  credentials are never echoed and long values are truncated. `POLICY_*` errors name the
  `policy_*` field, and `policy_sort` is checked with the same format. Existing wording such as
  "must be one of" is kept, so matching on it still works.
- Fixed: docstrings on the public symbols that lacked one (`ToolCall`, `ChatResult`, `Span`,
  `TokenFormat` and the public methods and properties of `Vault`, `Usage`, `GatewayConfig`,
  `Privacy`); `Privacy.passthrough`'s docstring sat after code and was not a docstring.
  `tests/test_public_docstrings.py` enumerates `__all__` and its classes' public members.

## 0.26.0

Idempotency keys everywhere, plus two more attempt-span attributes. Opt-in and backward
compatible: nothing changes without `idempotency_key=`.

- New: `stream_chat`, `complete` and `complete_with_usage` take `idempotency_key=` with the same
  rules as `chat()` (1-128 chars of letters, digits, `. _ : -`; `InvalidRequestError` otherwise).
  The key is on the call span, every attempt span and every `AttemptReport`, and is sent as the
  `Idempotency-Key` header to OpenAI only (`call` and streaming); no other provider receives it.
- New: concurrent `complete()` / `complete_with_usage()` calls with the same key and the same
  request share one provider call and result; same key, different request raises
  `InvalidRequestError`. A key is scoped per operation (the same key on `chat()` and `complete()`
  does not collide). `repair_attempts` repair calls keep the key on spans and reports but do not
  send the OpenAI header (their body differs, OpenAI would reject the reuse).
- Decision for streams: no fan-out. A stream is consumed, not replayed, and each consumer drives its
  own upstream request, so sharing one would need buffering and replay. A second `stream_chat()`
  with the same key while the first is still open raises `InvalidRequestError` on first iteration,
  before any provider call. The key is free again when the first stream ends, fails or is closed.
- New: attempt spans carry `llm_gateway.fallback_reason` (the `ErrorKind` of the previous failure,
  or `skipped` when earlier providers were passed over without a call; only on attempts that are a
  fallback or a later attempt) and `llm_gateway.breaker_state` (`closed`, `probe` or `open`, read
  just before the attempt was admitted) on `chat`, `stream_chat` and `complete*`.
- New: `breaker.state(provider, model=None)`, a pure read for diagnostics.

## 0.25.0

Idempotency keys. Opt-in and backward compatible: nothing changes without `idempotency_key=`.

- New: `chat(idempotency_key=)` (1-128 chars of letters, digits, `. _ : -`). The key is on the call
  span, every attempt span (`llm_gateway.idempotency_key`) and every `AttemptReport`, so retries
  and fallbacks of one logical call correlate with the caller's own id.
- New: the key is sent as the `Idempotency-Key` header to OpenAI chat only; no other provider
  receives it.
- New: concurrent `chat()` calls in one process with the same key and the same request share one
  provider call and result (followers fire no `on_attempt`). Same key, different request raises
  `InvalidRequestError`. Nothing is cached after the call ends; one waiter cancelling does not
  cancel the others.
- Not done: Anthropic documents no idempotency header, so it gets none.

## 0.24.0

Untrusted-content helpers. Opt-in and backward compatible: nothing changes unless you call them.

- New: `wrap_untrusted(text, source=, max_chars=, sanitize=)` fences external text with a fresh
  random 128-bit fence per call, removes `<<<`/`>>>` look-alikes (so content cannot forge a
  closing fence), optionally truncates with a marker, and labels it as data.
- New: `sanitize_untrusted(text)` strips zero-width, bidi-override, soft-hyphen and Unicode tag
  characters and HTML comments.
- New: `gate_tool_calls(calls, sensitive=, approve=)` splits model-requested tool calls into
  allowed and denied. A sensitive tool runs only if the host callback (sync or async) returns
  `True`; no callback or a raising callback denies (fail closed). `ApprovalDenied.message()`
  gives the tool message to send back.
- Docs: README "Untrusted content and prompt injection" states the threat model.
- Already present, unchanged: `tool_result_max_chars` cap, PII-masked tracing.

## 0.23.0

Model catalogue correctness, checked against the vendors' docs on 2026-11-09. **Behaviour change**:
backward compatible except that a model whose announced shutdown date has passed is now skipped (see below).

- New: `llm_gateway/lifecycle.py` records announced retirements (Anthropic, Groq, OpenAI, Gemini on
  Vertex), each with its source URL and check date. Routing skips a provider whose model is
  retired, so a fallback chain never reaches it, and raises `ModelRetiredError` (a
  `PolicyViolationError`) when nothing else remains. An explicit per-call model is checked too.
  `claude_model_fallback` is not used when retired. One WARNING per model skipped, and one per model
  retiring within `RETIREMENT_WARNING_DAYS` (default 90). `BLOCK_RETIRED_MODELS=false` restores the
  old behaviour (for example an enterprise Groq contract that outlives the public shutdown).
  Sources: platform.claude.com/docs/en/about-claude/model-deprecations,
  console.groq.com/docs/deprecations, developers.openai.com/api/docs/deprecations,
  docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions.
- Fix: strict structured output now covers every model on Anthropic's list (Opus 4.6/4.7, Sonnet
  4.5/4.6, Mythos Preview added) and no longer claims retired Opus 4.1
  (platform.claude.com/docs/en/build-with-claude/structured-outputs).
- New: `anthropic/claude-fable-5-1` price ($10/$50, cache 12.50/20/0.25;
  platform.claude.com/docs/en/about-claude/pricing).
- Fix: Switzerland regions are no longer treated as EU residency for `azure_region`.
- Still unverified: Azure deployments, Claude on Vertex dates, Mistral ids, OpenAI bare aliases
  such as `gpt-4o-mini` (the dated snapshot retires 2027-01-14).

## 0.22.0

Outage resilience. Backward compatible (new attributes, config fields and an endpoint).

- New: `AllProvidersExhaustedError.failures` and `LLMDeadlineExceeded.failures` list one
  `ProviderFailure` (provider, model, `kind`, status, scrubbed detail) per failed attempt, and the
  message ends with `Causes: anthropic=billing, groq=rate_limited, openai=model_retired.`.
  Kinds come from `errors.failure_class`: billing, rate_limited, auth, model_retired, overloaded,
  timeout, network, invalid_request, ...
- Fix: Groq's 400 `model_decommissioned` (and "has been retired/decommissioned" wording) was a plain
  invalid request, so a retired model never opened its breaker. It is now model-missing.
- Fix: HTTP 402 and billing-shaped 403s (OpenAI `billing_not_active`, GCP "billing account ...
  disabled") were classified as auth/invalid request; they are now billing.
- New: `breaker_billing_cooldown_seconds` (default 300) and `breaker_model_missing_cooldown_seconds`
  (default 1800): conditions that need a config change no longer re-probe every 60 s.
  Never shorter than `breaker_cooldown_seconds`.
- New: `circuit_breaker_status()` and `GET /v1/providers/health` report per-provider and
  per-model breaker state, remaining cooldown and last error class (no error text).
- Test: all providers failing at once, deadline bound with hanging providers, half-open admits one
  concurrent probe, classification shapes.

## 0.21.2

Coverage pass 3. Backward compatible.

- Fix: a malformed tool definition or `tool_choice` (missing `function`, missing or non-string
  `name`) sent to Anthropic or Vertex Claude crashed with a bare `KeyError`/`TypeError`, which the
  engine treated as a provider failure and failed over on. It is now `InvalidRequestError`.
- Test: error/fallback/stream/translation gap tests and Hypothesis tests over Anthropic and
  OpenAI response shapes and tool definitions.

## 0.21.1

Types and docs. No behaviour change; fully backward compatible.

- New: `Completion` is generic in `parsed`. `complete_with_usage(output_schema=Model)` or
  `validate=Model` is now typed `Completion[Model]` (so `result.parsed` is `Model | None`);
  other calls stay `Completion[Any]`, and a bare `Completion` annotation still works.
- New: `typing-extensions` is a declared dependency (it was already installed through pydantic).
- Docs: `MIGRATION.md` covers the 0.17-0.21 behaviour changes (stricter tool-argument and
  `validate=` checks, https-only token endpoints, partial streams and `AttemptReport.partial`).
- Test: a typed consumer file (`tests/typing/consumer.py`) is checked with pyright in CI.

## 0.21.0

Agentic validation pass. Stricter checks; no API changes.

- Fix: tool-argument checks (`check_tool_calls`) and `validate=` ignored most JSON Schema limits,
  so `minimum`, `maxLength`, `maxItems` and similar passed silently. Now enforced: `minimum`,
  `maximum`, `exclusiveMinimum/Maximum`, `minLength/maxLength`, `minItems/maxItems/uniqueItems`,
  `minProperties/maxProperties`, `allOf`, `not`, `additionalProperties` as a schema, and `oneOf`
  now requires exactly one match. `pattern` and `format` stay unchecked (ReDoS surface).
- Fix: a dict `output_schema` only had its top-level type and `required` keys checked; the reply
  is now validated against the whole schema subset (pydantic models were already fully checked).
- Fix: provider tool-call arguments with a duplicate key or `NaN`/`Infinity` are rejected as
  `InvalidOutputError` (they fail over) instead of letting the last duplicate win silently.

## 0.20.2

Security pass 4. No API changes.

- Fix: `redact_secrets` now scrubs `?key=` query parameters (Gemini), bare Google API keys
  (`AIza...`) and OAuth tokens (`ya29.`), and prefixed names such as `access_token`,
  `refresh_token`, `client_secret` and `Ocp-Apim-Subscription-Key`.
- Fix: `GatewayConfig` repr/str no longer shows `otel_exporter_otlp_endpoint` or
  `otel_exporter_otlp_headers` (usually a token); header values are scrubbed from error text.
- Hardening: `azure_endpoint` and `huggingface_base_url` must be `https://` (a bearer token is sent
  there). `openai_compat_base_url` still accepts `http://` for local servers.
- The HTTP service logs a warning at startup when `GATEWAY_API_KEYS` is unset.

## 0.20.1

Token and cost accounting fixes. No API changes.

- Fix: a reply the provider billed and the gateway then rejected (empty completion such as a
  reasoning model that spent its budget, a refusal, a tool call cut off by `max_tokens`) was
  reported to `on_attempt` with `usage=None` and `cost_usd=None`, so a spend ledger saw it as free.
  `EmptyCompletionError` and `InvalidOutputError` now carry `usage` (default `None`) and failed
  attempt reports use it.
- Fix: Anthropic streams no longer lose input or cache tokens when a later `message_delta`
  reports `0` for them; the input side of the usage is cumulative and now only grows.
- Price table re-checked 2026-11-09 against the Anthropic, OpenAI and Groq pricing pages; no
  rate changed. Vertex, Mistral and OpenRouter rows are still unverified (flagged in
  `PRICE_SOURCES`).

## 0.20.0

Streaming cancellation and metering. Additive, backward compatible.

- Fix: when a stream ended early (consumer `aclose()`, task cancellation, client disconnect) the
  provider stream was closed only after the `on_attempt` hook ran, so a slow hook held the upstream
  request open and a hook that was itself cancelled left it open. The provider stream is now
  closed first.
- New: `AttemptReport.partial` (true for a stream cut short after it started: cancelled, closed
  early or failed mid-stream; `usage` may be `None` or lag the billed amount, so treat the cost as
  a lower bound) and `AttemptReport.streamed_chars` (text and tool-argument characters forwarded),
  so a ledger can mark or estimate unreported output. Both default to `False`/`0`.

## 0.19.0

Tracing aligned with the OpenTelemetry GenAI semantic conventions (Development status; spec in
https://github.com/open-telemetry/semantic-conventions-genai, built on semantic-conventions
v1.45.0, checked 2026-11-09). Additive: every existing attribute is still set.

| Old (deprecated alias, still set) | New (semconv) |
| --- | --- |
| `gen_ai.system` | `gen_ai.provider.name` |
| `llm_gateway.error_code` | `error.type` |
| `llm_gateway.finish_reason` | `gen_ai.response.finish_reasons` (array) |
| `gen_ai.prompt` (opt-in) | `gen_ai.input.messages` (opt-in, JSON) |
| (none) | `gen_ai.operation.name` (`chat`), `gen_ai.response.model` |

- `gen_ai.request.model` on the call span is now the model requested from the provider; the model
  the provider reported is `gen_ai.response.model` (before, the response model overwrote it).
- Call and attempt spans have kind `CLIENT`. Span names are unchanged (`llm_gateway.complete`,
  `.chat`, `.stream_chat`, `.attempt`).
- New `llm_gateway.fallback` span event when the chain moves on to a later provider;
  `llm_gateway.stream_restart` now also carries the provider and `error.type`.
- Cost stays the custom `llm_gateway.cost_usd`; no semconv attribute exists for it.
- Prompt content is recorded only with `trace_include_prompts=True` (default off, PII-masked,
  truncated); responses are never recorded.

## 0.18.7

Multi-step tool-loop tests with a mid-loop failover across Anthropic, Vertex Claude, OpenAI-style
and Gemini fakes (`tests/test_tool_loop.py`). No API changes.

- Fix: a model that leaves tool call ids empty or repeated every turn (Gemini's OpenAI endpoint)
  got the same repaired id (`call_0`) on each turn, so the second request failed pre-flight with
  "duplicate tool call id". Repaired ids now avoid every id already in `messages`
  (`unique_tool_call_ids(..., taken=)`).
- Fix: tool call ids that Anthropic rejects (anything outside `[a-zA-Z0-9_-]{1,64}`, e.g.
  `functions.add:0` from OpenAI-compatible hosts) are renamed consistently in the tool call and its
  result when the history is sent to Claude, instead of a 400 after failing over to anthropic or
  vertex.

## 0.18.6

Property-based tests for the Anthropic stream, the OpenAI to Anthropic translation and the Vertex
Claude path (`tests/test_properties_anthropic.py`). No API changes.

- Fix: structurally broken message history (no role, a tool message without `tool_call_id`, a
  malformed tool call, non-text system content, a non-dict message) raised `KeyError`/`TypeError`
  from inside `to_anthropic_messages()` for the anthropic and vertex providers; it now raises
  `InvalidRequestError` naming the message index.

## 0.18.5

Property-based robustness tests (Hypothesis, `tests/test_properties.py`). No API changes.

- Fix: the base-URL metadata guard now reads the host the way HTTP clients do. A host written with
  ideographic full stops (`http://169。254。169。254/`) or full-width digits reached 169.254.169.254
  past the link-local check; hosts are NFKC-normalized and the dot look-alikes mapped before the check.

## 0.18.4

Supply-chain and dependency audit. No API changes.

- Raised dependency floors that admitted versions with known advisories (pip-audit on the
  lowest-allowed set of every extra was dirty, on the latest-resolved set clean): `aiohttp>=3.14.4`
  (azure), `opentelemetry-sdk` / `opentelemetry-exporter-otlp-proto-http>=1.45.1`, and for the
  service extra `starlette>=1.3.1,<2` with `fastapi>=0.133` (the old `fastapi>=0.110` floor could
  not even co-install with `starlette>=0.49.1`).
- sdist no longer ships `infra/`, `.github/`, `Dockerfile` or `.env.example`; the wheel was already
  source-only and includes `py.typed`. Builds are byte-reproducible with `SOURCE_DATE_EPOCH`.
- CI: workflow-level `permissions: contents: read` (the deploy job keeps its own grants) and
  `persist-credentials: false` on checkout. Actions were already pinned by full SHA.

## 0.18.3

Strict typing for the provider adapters, `structured.py` and `router.py`. No behaviour changes.

- `[tool.pyright]` now runs strict on `src/llm_gateway/providers`, `structured.py` and `router.py`
  (the rest stays standard). Findings in those modules: 1,430 before, 0 after; whole `src` is 0.
- New `providers/_wire.py`: TypedDicts for the OpenAI-shaped message / tool / tool-call / stream
  delta dicts (`OpenAIMessage`, `OpenAITool`, `OpenAIToolCall`, `ToolCallDelta`) and the Anthropic
  request blocks (`AnthropicTextBlock`, `AnthropicImageBlock`, `AnthropicToolUseBlock`,
  `AnthropicToolResultBlock`, `AnthropicMessage`, `AnthropicTool`). At runtime they are plain dicts.
  `StreamDelta.tool_call_deltas` is now `list[ToolCallDelta]`; public call signatures still take
  `list[dict[str, Any]]`, so existing callers type-check unchanged.
- No real bug surfaced while typing, so no behaviour fix is included.

## 0.18.2

Documentation, typing and API audit. No behaviour changes.

- README: a five-minute quick start that runs with a fake provider, and a 0.18 feature guide
  (failover and policy, streaming restart markers, the `check_tool_calls` loop, structured output,
  `on_attempt` metering, budgets, the first-token / idle / deadline table, pricing, security notes).
  Every snippet is executed by `tests/test_readme_snippets.py` against fakes (no network).
- Error reference table gains a "Fails over?" column.
- `AttemptHook` is exported from `llm_gateway`. Docstrings added where missing (`OutputPolicy`,
  `Redactor`) and completed (`mask_pii`, `tool_error_message`, `reset_circuit_breakers`,
  `stream_chat` yields). `tests/test_public_api.py` pins `__all__`, docstrings, `py.typed` and the
  backward-compatible call shapes below.
- No public name was renamed in 0.16.5-0.18.1, so there is nothing to deprecate: the one-argument
  forms `redact_secrets(text, config)` and `cost_usd(price, usage)` still work unchanged.
- Typing: standard `pyright` on `src` is clean. Strict mode is not yet: about 1,900 findings, almost
  all bare `dict`/untyped `[]` inside the provider adapters (`providers/*`, `structured.py`,
  `router.py`); tracked as follow-up work.

### Upgrading to 0.18 (hosts)

Nothing is required: every change since 0.16.5 is additive and off by default. Adoption steps, in
the order that usually pays off:

1. Pin `llm-gateway>=0.18.2` and run your tests; no code change should be needed.
2. Metering: pass `on_attempt=` (or set `GatewayConfig(on_attempt=...)`) and write the ledger from
   the `AttemptReport`, so failed attempts that were billed are counted, not only the winner.
3. Streams: set `first_token_timeout_seconds` to fail over from a silent provider, and replace a
   hand-written "retry the stream on the next provider" loop with `restart_on_failure=True`; on a
   `delta.restart` marker drop the text, tool-call fragments and usage collected so far.
4. Agents: after each `chat()` run `check_tool_calls(tools, reply.tool_calls)` and answer bad calls
   with `tool_error_message(issue)` instead of failing the turn; catch `ToolCallTruncatedError`
   (raise `max_tokens`) and `StreamTooLargeError` where you stream.
5. Budgets and prices: use `ProviderPolicy(max_cost_usd=..., budget_check=...)`; override
   `MODEL_PRICES` for models you negotiate; check the table with `python -m llm_gateway.pricing --list`.
6. Config: base URLs with embedded credentials or metadata hosts are now rejected at startup, and
   provider clients no longer follow redirects; fix any such configuration before upgrading.

## 0.18.1

Tiered price catalog and a re-verified default table. Backward compatible: every new `ModelPrice` field is optional.

- Feature: `ModelPrice` gains `long_context` (`LongContextTier`: threshold input tokens, a `multiplier` and/or explicit rates, applied to the whole request by its own input tokens, cache tokens included), `batch_multiplier`, `cache_write_1h_per_mtok`, and `source`/`checked` audit fields.
- `cost_usd(..., batch=False, cache_write_1h_tokens=0)`; `worst_case_cost_usd` applies the tier to its estimated prompt size.
- `python -m llm_gateway.pricing --list` prints the table (no network).
- Defaults: added `anthropic/claude-sonnet-5-5` and `claude-opus-5-5`; `claude-haiku-5-5` now carries its over-100,000-token 5x tier; 1-hour cache-write and batch rates added for Claude rows; `groq/openai/gpt-oss-120b` gains the 0.075 cached rate; `openai/gpt-4o-mini` batch 0.5x. Rows that could not be confirmed on 2026-11-08 (Vertex Claude, Mistral Small, OpenRouter) are marked "unverified" and unchanged.

## 0.18.0

Opt-in mid-stream restart for `stream_chat()`. Off by default; existing streams are unchanged.

- Feature: `stream_chat(restart_on_failure=True)` (`GatewayConfig.stream_restart_on_failure`,
  `stream_max_restarts`, default 1): a provider failing after chunks were yielded (transport error,
  idle timeout, provider 5xx) hands over to the next provider. The new attempt starts with a
  `StreamDelta(restart=True, restart_attempt, failed_provider, failed_error, provider, model)`
  marker; the consumer discards everything before it, partial tool-call fragments included.
  Never on cancellation, content filter, `StreamTooLargeError`/invalid output, invalid request or
  deadline. Every attempt is reported to `on_attempt` and traced. No provider left: the original
  error is raised. The HTTP service does not restart.
- `StreamDelta` gains `restart`, `restart_attempt`, `failed_provider`, `failed_error` (defaults
  keep it backward compatible).

## 0.17.0

Per-attempt usage/cost hook and a first-token timeout for streams. New optional parameters only.

- Feature: `on_attempt` (`GatewayConfig.on_attempt`, or per call on `chat`, `stream_chat`,
  `complete`, `complete_with_usage`; sync or async) receives an `AttemptReport` once per provider
  attempt: provider, model, request id, attempt number, outcome (succeeded, failed, cancelled),
  error type name, usage (partial when the provider reported it before failing, else `None`) and
  `cost_usd` from the price table. No prompt, reply or error message in it. Hook exceptions are
  logged by type and never affect the call.
- Feature: `first_token_timeout_seconds` (config field and `stream_chat()` argument, default
  `None`): no first chunk in time raises `FirstTokenTimeoutError` (a `TimeoutError`) for that
  attempt and fails over before anything is yielded. Interaction with
  `stream_idle_timeout_seconds` and `call_deadline_seconds` is documented in the README.

## 0.16.9

Stream size caps, budget-hook contract, provider-shape fixtures.

- Fix (medium): streamed tool-call arguments and text were buffered without a bound, so a hostile
  or buggy provider could stream unlimited data into memory. `stream_chat()` now ends the stream
  with the new `StreamTooLargeError` (an `InvalidOutputError` subclass: failover and breaker
  handling as before, never a silent truncation) past `stream_max_tool_arguments_chars` (default
  1,000,000 per tool call) or `stream_max_total_chars` (default 16,000,000 for the whole stream).
  0 disables a cap.
- Docs: `BudgetCheck` is a gate, not a reservation. The gateway holds and charges nothing before a
  provider answers, so there is nothing to release on cancel or failure; a host that reserves in
  its hook settles or releases it itself. No code change.
- Fix (low): an OpenAI-style reply with `content: ""` next to tool calls (Mistral) returned
  `""`; it is now `None`, like OpenAI and Anthropic.
- Tests: `tests/fixtures/providers/*.json` hold documentation example payloads (OpenAI, Anthropic,
  Mistral, Gemini via its OpenAI-compatible endpoint, Groq error bodies), run through the real
  parsers by `tests/test_provider_fixtures.py`. No network.

## 0.16.8

Third security pass (redirects, metadata URLs, error and span text).

- Fix (medium): the SDK clients followed redirects, and httpx only strips `Authorization` across
  origins, so a hostile or compromised endpoint could bounce `x-api-key` / `api-key` headers to
  another host. Every provider client now has `follow_redirects=False` (and still honors
  `ssl_verify`).
- Fix (medium): `openai_compat_base_url`, `huggingface_base_url` and `azure_endpoint` accept
  link-local and cloud metadata hosts (169.254.0.0/16 including numeric/hex and IPv4-mapped forms,
  `fe80::/10`, `fd00:ec2::/32`, `100.100.100.200`, `metadata.google.internal`); they are now
  rejected at config time. Loopback and private ranges stay allowed.
- Fix (medium): the call spans (`chat`, `stream_chat`, `complete`) recorded the escaping exception,
  including the chained provider error text, as a span event and used the message as status
  description. They now record no exception event and the status carries the exception type only.
- Fix (low): provider error text in `AllProvidersExhaustedError`, the deadline error and mid-stream
  log lines was unbounded; it is capped at 1000 characters (`redact_secrets(..., max_chars=)`,
  new `errors.error_detail`).
- Checked and already correct: config reprs hide keys, no pickle/eval, privacy and tool JSON depth
  limits, `pip-audit` clean for the dependencies (only pip itself, a tooling finding).

## 0.16.7

Agentic pass 3: tool calling in agent loops.

- A reply cut off by the token budget inside a tool call raises the new
  `ToolCallTruncatedError` (an `InvalidOutputError` subclass: failover is unchanged) with
  `finish_reason == "length"` and the `tool_names`, on OpenAI-style and Anthropic alike; before,
  OpenAI-style hosts reported a generic "not valid JSON". Arguments are never passed on half-parsed.
- New `check_tool_calls(tools, tool_calls)` returns typed `ToolCallIssue`s (unknown tool, arguments
  failing the tool's JSON schema, with paths, never values) and `tool_error_message(issue)` builds
  the `role: "tool"` reply to feed back to the model. Never raises.
- `chat()` makes tool-call ids in a reply unique and non-empty (hosts repeat or omit them), so
  the results can be paired on the next request.
- Anthropic: a tool result carrying image parts (screenshots) is translated to image blocks
  instead of being sent as raw OpenAI parts; tool-call arguments in history must be a JSON object.
- Pre-flight: more than 128 tools, or tool definitions over 1,000,000 characters of JSON, are an
  `InvalidRequestError` before any provider is called.
- `Usage` supports `+` and `Usage.total(usages)` for exact token totals across a tool loop.

## 0.16.6

Reliability pass 4: cancellation and resource cleanup.

- A call cancelled (`task.cancel()`, consumer `aclose()`, client disconnect) while it held a
  half-open circuit-breaker probe now hands the probe back at once, in `complete`, `chat` and
  `stream_chat`; before, the provider stayed closed to everyone for a full cooldown.
- Attempt spans now carry `llm_gateway.outcome="cancelled"` for a cancelled call or a stream the
  consumer abandoned (the latter used to read `ok`), and `complete`/`chat` emit one on cancel
  (previously none). Cancellation is still never a breaker failure, never fails over, and
  `CancelledError` is never swallowed.
- `stream_chat`: a provider generator whose `aclose()` raises no longer masks the consumer's own
  exit (a close, a cancellation) with that error.
- Timeout classes, documented: no chunk within `stream_idle_timeout_seconds` (before the first
  token or mid-stream) is a provider `TimeoutError` and counts toward the breaker;
  `call_deadline_seconds` running out raises `LLMDeadlineExceeded` and is not counted.

## 0.16.5

Correctness pass 3: provider adapters' less-travelled paths.

- Streaming tool calls: hosts that number every parallel call 0 (Gemini's OpenAI endpoint) or omit
  the index no longer have their calls glued together; a new call gets its own `index`.
  Well-numbered streams are unchanged.
- Error bodies that carry the upstream HTTP status as a numeric `code` (OpenRouter, in-stream) are
  classified by it (502 transient, 429 rate limited, ...) instead of `UNKNOWN`.
- Model-not-found spelled as a 400 (Mistral `invalid_model`, OpenRouter "is not a valid model ID")
  now feeds the per-model breaker like a 404 does.
- Claude on Vertex AI raises `InvalidRequestError` for image URLs (base64 only) before any call.
- `response_format={"type": "json_object"}` on anthropic and vertex was silently dropped; it now
  adds a JSON-only system instruction.

## 0.16.4

Reliability pass 3: the HTTP service's SSE path. Defaults are generous; each knob below disables
with `0`.

- Slow readers: a body write blocked longer than `STREAM_WRITE_TIMEOUT_SECONDS` (30) closes the
  stream and, through `aclosing`, the upstream provider request. The body generator is now always
  closed explicitly, not left to the garbage collector.
- `MAX_CONCURRENT_STREAMS` (256): over it, new streaming requests get `503` with `Retry-After`.
- Graceful shutdown: on lifespan shutdown the service answers new calls `503`, waits up to
  `SHUTDOWN_GRACE_SECONDS` (20) for in-flight non-streaming calls, ends open streams with a final
  SSE error event and `[DONE]`, then closes the providers.
- `STREAM_HEARTBEAT_SECONDS` (15): an idle stream gets an SSE `: keepalive` comment so proxies do
  not cut the connection.
- Backpressure: the upstream stream feeds a bounded queue (8 chunks), so a slow client stalls the
  provider read instead of buffering the response in memory.

## 0.16.3

Repo hygiene. No behaviour change.

- `.gitignore` covers coverage data, caches, `.venv*/` and build output.
- Upper bounds on the OpenTelemetry, azure-identity and aiohttp dependencies; `pyright` and
  `build` added to the `dev` extra. `pip-audit` on the resolved set is clean.
- CI: pyright and a coverage floor run with the tests, the package build is checked, and every
  action is pinned by commit SHA.
- `CONTRIBUTING.md` added.

## 0.16.2

API ergonomics, types and docs pass 2. No behaviour change.

- Types: `pyright src` 29 errors -> 0. Streaming provider
  functions are `AsyncGenerator[StreamDelta]` (subtype of the old `AsyncIterator`, so `aclosing`
  type-checks); OpenAI-format messages go through a typed `sdk_messages()` helper instead of
  `list[dict]` overload noise; two narrowing fixes; one targeted `pyright: ignore` for the SDK's
  lazy `AsyncAnthropicVertex` export.
- Docs: README examples for `validate=`, `request_id`, `parallel_tool_calls` (with support matrix),
  the tool result cap, the privacy layer (with limits) and `aclose()` in a FastAPI lifespan; new
  `MIGRATION.md` (0.13 -> 0.16). Docstrings of the public entry points list what they raise.
- Tests: tagged README blocks are executed as written; `__all__` is checked against documented
  names and against accidental private or missing exports.

## 0.16.1

Test-coverage pass 2 (96.30% -> 96.81%, floor 95.8%) and two fixes it exposed.

- Fix: `stream_chat()` wrote no `llm_gateway.attempt` span for a stream that ended empty, failed
  after its first chunk, hit the call deadline, or was abandoned by the consumer; the span is now
  written in every case (error outcome on failure).
- Fix: the `validate=` JSON Schema subset treated `true`/`false` as equal to `1`/`0` in `enum` and
  `const` (Python `==`), so `true` passed `{"enum": [1]}`; comparison is now JSON-exact.
- Tests only otherwise: streaming fallback/cancellation, per-model breaker bounds (property test),
  privacy streaming of text and tool-call arguments (chunking-invariance property tests), request
  ids across retries/fallbacks/repairs, tool-result cap properties, `service/main.py`.

## 0.16.0

Agentic pass 2 (additive; every new argument defaults to the old behaviour).

- Tracing: `request_id=` on `chat()`, `stream_chat()`, `complete()` and `complete_with_usage()`
  (generated when omitted), returned as `ChatResult.request_id`, `Completion.request_id` and
  `StreamDelta.request_id` (usage chunk). The call span gains `llm_gateway.request_id`; every
  provider attempt emits a child `llm_gateway.attempt` span (request id, attempt number, provider,
  model, outcome class, latency, token counts; never content, error text or keys). Behaviour note:
  traces now hold one extra span per attempt under the call span.
- Parallel tool calls: `parallel_tool_calls=` on `chat()`/`stream_chat()`. `False` sends OpenAI-style
  `parallel_tool_calls: false` (OpenAI, Azure, Groq, Mistral, OpenRouter) or Anthropic
  `disable_parallel_tool_use` (Anthropic, Vertex Claude); Gemini, Hugging Face and `openai_compat`
  have no documented switch and ignore it. `providers.PARALLEL_TOOL_CALLS_SUPPORTED` lists support.
- Structured output: opt-in `validate=` (pydantic model or JSON Schema subset) on
  `complete_with_usage()` returns a typed `parsed` or raises `InvalidOutputError` with `.paths`
  (field paths, never values). No automatic retry; `repair_attempts=` (default 0) is the explicit,
  paid knob. `InvalidOutputError` gains a `paths` attribute (empty where unknown).
- Tool result size guard: `TOOL_RESULT_MAX_CHARS` / `max_tool_result_chars=` truncate oversized
  `role: "tool"` text with a marker before it is resent. Off by default.

## 0.15.4

Second correctness pass (finish reasons, filtered replies, reasoning models, usage).

- Fix (high): a completion ended by a content filter or refusal with no text counted toward the
  circuit breaker (`EMPTY_RESPONSE`). Whether the filter fires depends on the caller's content, so
  one user's trigger text could open the breaker for everyone; it is now `INVALID_OUTPUT` (fails
  over, never counts). Applies to `content_filter`, Gemini `SAFETY`, Anthropic `refusal` and a 200
  with no choices. Budget-exhausted empties (`length`) still count. The exception type is unchanged.
- Fix (high): OpenAI o-series / gpt-5 models got `max_tokens` and non-default sampling, a 400 on
  every call. They now get `max_completion_tokens` and `temperature`, `top_p`, penalties and `stop`
  are dropped. Azure does the same for sampling while `azure_reasoning_effort` is set.
- Fix (medium): `chat()` and streams lost truncation: `finish_reason` was always `stop` or
  `tool_calls` in chat, and Anthropic `max_tokens` streamed as `stop`. One vocabulary now:
  `stop`, `length`, `content_filter`, `tool_calls` (new values only where `stop` was wrong).
- Fix (medium): a reply with no `choices` (Gemini blocking a prompt) raised IndexError (`UNKNOWN`,
  counted); it is now an `EmptyCompletionError`.
- Fix (medium): inline `<think>...</think>` text from DeepSeek-R1/Qwen3-style hosts leaked into
  `content` (chat and streams); a leading block is stripped, a think-only reply is empty.
- Fix (medium): Anthropic rejected OpenAI-range `temperature` above 1 and `temperature` together
  with `top_p` on Opus 4.1-4.6, Sonnet 4.5+ and Haiku 4.5; temperature is clamped to 1 and `top_p`
  is dropped when both are set on those models.
- Fix (medium): a `max_tokens` cut inside an Anthropic tool call returned a half-built call; it is
  now `InvalidOutputError`.
- Fix (medium): OpenRouter `prompt_tokens_details.cache_write_tokens` was ignored, so cache-write
  premiums were billed at the base rate; it now fills `cache_creation_input_tokens`.
- Fix (low): Anthropic multi-part content: empty text parts (rejected upstream) are dropped, and
  unsupported part types (`input_audio`, `file`) or an image without a url raise a clear
  `InvalidRequestError`, now classified `INVALID_REQUEST` (it was `UNKNOWN`, counted); invalid tool
  arguments in message history are the same.
- Docs: the README and `errors.py` table state why an Azure output content filter does not count.
- Verified, no change needed: Vertex Claude shares the Anthropic path; Groq, Mistral, OpenRouter,
  Hugging Face and OpenAI-compatible hosts use the shared OpenAI-style parsers.

## 0.15.3

Second reliability pass (breaker scope, client lifecycle, bounded state).

- Fix (high): the circuit breaker was keyed per provider only, so one retired or overloaded model
  (a 404 `model_not_found`, a 529 `overloaded_error`) took the whole provider out of rotation, or,
  for a 404, was never parked and hit on every call. Those failures now open only the
  (provider, model) key (`errors.is_model_scoped`); auth, billing, rate-limit, timeout and
  connection failures still open the whole provider. `breaker.is_open/allow/record_success/
  record_failure/record_error` take an optional `model=`; without it nothing changes. All three
  engines pass the resolved model.
- Fix (high): the Anthropic, OpenAI, Groq, Mistral, OpenRouter, Hugging Face and generic
  OpenAI-compatible SDK clients were never closed on shutdown (only Azure, Gemini and Vertex were).
  Every provider module now has `aclose()`; new `llm_gateway.aclose()` closes them all (one failing
  close does not skip the rest) and the HTTP service calls it on shutdown.
- Fix (medium): opening a stream had no attempt bound, so a hang before the first byte (credential
  fetch, connect) lasted until the 90 s call deadline; it is now bounded by
  `stream_idle_timeout_seconds`.
- Fix (medium): the per-credential SDK client caches and the model-scoped breaker state were
  unbounded (rotating keys, caller-chosen model ids). Client caches are capped at 16 (oldest
  evicted and closed after a grace period) and model-scoped breaker keys at 256. A success no longer
  leaves a zeroed breaker entry behind.
- Fix (low): two concurrent first calls to Vertex both built an SDK client and one connection pool
  leaked; the loser is now closed.
- Verified, no change needed: every adapter (Anthropic, OpenAI, Groq, Azure, Vertex, Gemini,
  Mistral, OpenRouter, Hugging Face, OpenAI-compatible) passes the connect/read/write/pool
  deadlines to its SDK client. Per-key usage and rate-limit maps are keyed by authenticated keys
  only, so they are bounded by the configured key list.
- Tests: engine-level half-open concurrency (25 `complete` and 15 `chat` calls admit exactly one
  probe, the rest fail over), model-scoped breaker behaviour, per-adapter timeouts, client reuse,
  shutdown, cancellation safety and cache bounds.

## 0.15.2

Second security pass (privacy layer, service, URLs, tracing).

- Fix (medium): `mask_pii` email pattern was quadratic on long runs of address characters with no
  `@` (a 60k-char string took seconds); local part and domain are now length-bounded.
- Fix (medium): the HTTP service had no request-body ceiling before parsing. New
  `GatewayConfig.max_request_bytes` (default 16 MiB, 0 disables) rejects an oversized body with
  413 by Content-Length and by counting chunked bytes, before it is buffered.
- Fix (medium): 422 validation errors echoed the rejected request values (prompt text); they now
  carry only the field path and message, in the OpenAI error shape.
- Fix (low): privacy layer raised a raw `RecursionError` on tool-call arguments nested 100k deep;
  they are redacted as plain text, and the walker has a depth limit.
- Fix (low): `openai_compat_base_url`, `huggingface_base_url` and `azure_endpoint` reject embedded
  `user:password@` credentials and non-http(s) schemes; `GatewayConfig` validation errors no longer
  repeat the rejected input; `redact_secrets` scrubs `scheme://user:pass@` in error text.
- Fix (low): `/docs`, `/redoc` and `/openapi.json` are disabled when `GATEWAY_API_KEYS` is set.
- Deps: `starlette>=0.49.1` floor for the service extra (known DoS CVEs in older releases).
- Tests: tracing spans carry no message content or keys; no CORS headers by default; placeholder
  injection through a tool result is not restored; thousands of entities stay fast.

## 0.15.1

Test-coverage pass (94% -> 97% statement+branch) and two request-validation fixes.

- Fix: a tool `type` that is a list or object (e.g. `{"type": [["x"]]}`) crashed pre-flight
  validation with `TypeError` (a 500 from the service) instead of `InvalidRequestError`.
- Fix: a tool name with a trailing newline (`"search\n"`) passed validation because the name
  pattern used `$`; it now uses `fullmatch`.
- Tests: tool-validation rejection branches with hypothesis properties, Groq/OpenAI request
  options, breaker-open paths, tracing never-raise, privacy message mapping, Vertex credential
  dispatch. `pytest-cov` and `hypothesis` are dev dependencies; coverage floor 95%.

## 0.15.0

Privacy layer (`llm_gateway.privacy`, opt-in; existing calls are unchanged).

- `PrivacyLayer` wraps `complete_with_usage`, `chat` and `stream_chat` with reversible redaction
  (per-call in-memory `Vault`), per-call output policies (`RESTORE`, `STRIP`, `PASSTHROUGH`),
  strict mode (`PrivacyPolicyMissingError`), structured-output and tool-argument handling, and a
  counts-only audit hook.
- Hardening over the first draft: a placeholder-shaped literal in the input is reserved so it can
  never restore to another value; kinds with `_`/digits no longer leak unmatched tokens and
  invalid kinds raise `ValueError`; restored values in streamed tool-call arguments are
  JSON-escaped; held-back text is flushed before the finishing delta; multi-character
  delimiters split across chunks are held; the vault is cleared even if closing the stream fails.

## 0.14.1

Docs and typing pass; no behavior change.

- Ship `py.typed`, so type checkers use the package's inline annotations.
- Typing fixes: `stream_chat()` is annotated `AsyncGenerator[StreamDelta, None]`; `GatewayConfig.policy`
  passes validated provider-id tuples; price scaling and the `tool_choice` check no longer rely on
  `None`/`object` narrowing.
- README: error reference (class, when raised, retried?), the missing policy/price/`SSL_VERIFY`
  environment variables, and `SECURITY.md` (secret handling, how to report a vulnerability).
- Tests run the README's library examples against fake providers and check that every public
  error is documented.

## 0.14.0

Agentic pass.

- New `InvalidRequestError` (an `LLMError` and `ValueError`; the service answers 400). `chat()` and
  `stream_chat()` now validate before any provider call: tool names and uniqueness, JSON Schema
  structure of `parameters` (type, enum, required, properties, anyOf), `tool_choice` naming a
  known tool, and tool-call/tool-result pairing (orphan or duplicate result, missing
  `tool_call_id`, unanswered parallel call). Previously these became a provider 400 that failed
  over and counted against the breaker, or a bare `KeyError` on Anthropic.
- Anthropic and Vertex Claude: `tool_choice="none"` on a history that already holds tool calls
  keeps the tool definitions and sends `{"type": "none"}`; omitting them made Anthropic reject
  the request.

## 0.13.3

Correctness pass.

- Streaming (OpenAI-compatible providers): usage sent on the last chunk that also carries
  `choices` (Mistral and other hosts) is no longer dropped, so tokens, cost and quota are counted.
- Streaming (Anthropic, Vertex Claude): tool-call `index` is now the tool-call ordinal from 0,
  not the content-block index, so text before a tool call no longer shifts indexes.
- Non-streaming OpenAI-style providers: malformed or non-object tool-call arguments raise
  `InvalidOutputError` (fails over, no breaker count) instead of an unclassified `UNKNOWN` error.
- Structured output: a reply with a sentence before or after the JSON, or a fence inside prose,
  now parses; the strict parse still runs first and unusable text is still rejected.
- Anthropic: system messages given as a list of text parts no longer raise `TypeError`.

## 0.13.2

Reliability pass.

- Streaming: closing or cancelling the consumer (client disconnect, `break`) now closes the
  provider's generator, so the upstream request and connection are released instead of leaking.
- Streaming: a failure after the first chunk now counts toward the provider's circuit breaker
  (it was invisible before), then propagates unchanged.
- Service: a raw (non-`LLMError`) error mid-stream ends the SSE response with a generic error
  event and `[DONE]` instead of dropping the connection; the engine is closed via `aclosing`.
- Retry: backoff is jittered (50-100% of the doubling delay) and honors a provider's
  `Retry-After` (seconds, capped at 60 s) when longer; a wait that doesn't fit the call deadline
  gives up instead of sleeping.
- Circuit breaker: a real half-open state. After the cooldown exactly one probe call is admitted
  (`breaker.allow`); the rest stay out until it succeeds (closes) or fails (re-opens). State
  changes are lock-protected. A lost probe's lease expires after one cooldown.

## 0.13.1

Security pass.

- Provider API keys and `GATEWAY_API_KEYS` no longer appear in `repr()`/`str()` of `GatewayConfig`.
- Provider error text is scrubbed (`errors.redact_secrets`) before it enters `LLMError` messages,
  span statuses and the service's 503 body: configured keys, bearer tokens, `api-key: ...` pairs and
  common key shapes.
- Service auth: a non-ASCII bearer token returns 401 instead of a 500.
- Service: tool-call arguments and tool definitions count toward `MAX_PROMPT_CHARS`; `max_tokens` must be >= 1.
- Hostile JSON (deeply nested model output or tool-call arguments) raises `InvalidOutputError` /
  `ValueError` instead of `RecursionError`.
