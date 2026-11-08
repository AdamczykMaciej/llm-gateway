# Migrating from 0.13 to 0.16

No release in this range removes a public name or changes a function signature
incompatibly; every new argument defaults to the old behaviour. This page lists
the behaviour changes a caller might notice. Details are in `CHANGELOG.md`.

## Requests are validated before any provider call (0.14.0)

`chat()` and `stream_chat()` raise `InvalidRequestError` (an `LLMError` and a
`ValueError`; the HTTP service answers 400) for: bad tool names or duplicates, an
invalid `parameters` JSON Schema, a `tool_choice` naming an unknown tool, and
tool results that do not pair with a tool call. Before, these reached the
provider, came back as a 400, failed over and counted against the breaker.

Do: catch `InvalidRequestError` (or `ValueError`) where you build messages; do
not retry it, the request is wrong. Later releases apply the same error to a bad
`request_id`, `repair_attempts`, `max_tool_result_chars` and unsupported Anthropic
content parts.

## Content-filter and refusal empties no longer trip the breaker (0.15.4)

A completion ended by a content filter or refusal with no text is now
`InvalidOutputError`-class (`INVALID_OUTPUT`): the call fails over to the next
provider and the breaker is never charged. Before, it counted as
`EMPTY_RESPONSE`, so one user's trigger text could take a provider out of
rotation for everyone. Budget-exhausted empties (`finish_reason="length"`) still
count. The exception types you catch are unchanged. A reply with no `choices`
is now `EmptyCompletionError` instead of `IndexError`.

## The circuit breaker is per model for model-scoped failures (0.15.3)

A retired model (404) or an overloaded one (529) now opens only that
(provider, model) pair; auth, billing, rate-limit, timeout and connection
failures still open the whole provider. Expect a provider's other models to keep
serving where the whole provider used to go dark. `breaker.*` functions take an
optional `model=`; without it nothing changes. Call `await llm_gateway.aclose()`
on shutdown (new in 0.15.3).

## Reasoning models and sampling parameters (0.15.4)

- OpenAI o-series and gpt-5 get `max_completion_tokens` instead of `max_tokens`,
  and `temperature`, `top_p`, penalties and `stop` are dropped (they returned
  400 before). Azure drops sampling while `azure_reasoning_effort` is set.
- Anthropic: `temperature` above 1 is clamped to 1, and `top_p` is dropped when
  both are set on Opus 4.1-4.6, Sonnet 4.5+ and Haiku 4.5.
- Inline `<think>...</think>` blocks are stripped from `content` (chat and
  streams); a think-only reply counts as empty.

If you relied on a sampling value reaching a reasoning model, it no longer does.

## Finish reasons and streams

- `finish_reason` is one vocabulary: `stop`, `length`, `content_filter`,
  `tool_calls`. `chat()` and Anthropic streams now report `length` for a
  truncated reply, where they said `stop`. Check for `length` explicitly.
- A `max_tokens` cut inside an Anthropic tool call raises `InvalidOutputError`
  instead of returning a half-built call.
- Streaming tool-call `index` is the tool-call ordinal from 0, not the content
  block index (Anthropic, Vertex Claude).
- Each stream and attempt now writes an `llm_gateway.attempt` child span (0.16.0):
  traces hold one extra span per attempt.

## Service changes

- Request bodies over `MAX_REQUEST_BYTES` (default 16 MiB) get 413; set `0` to
  disable (0.15.2).
- 422 errors no longer echo the rejected values; `/docs` and `/openapi.json` are
  off when `GATEWAY_API_KEYS` is set.
- `openai_compat_base_url`, `huggingface_base_url` and `azure_endpoint` reject
  embedded credentials and non-http(s) schemes.

## New and opt-in (nothing to do)

`PrivacyLayer` (0.15.0), `request_id=`, `validate=` / `repair_attempts=`,
`parallel_tool_calls=` and `max_tool_result_chars=` (0.16.0). See the README.

## Typing (0.16.2)

Streaming provider functions are annotated `AsyncGenerator[StreamDelta]` (a
subtype of `AsyncIterator`), so existing annotations keep working.
