# Changelog

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
