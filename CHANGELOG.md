# Changelog

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
