# Changelog

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
