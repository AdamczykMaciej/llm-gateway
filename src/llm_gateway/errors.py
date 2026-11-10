"""Gateway error types and the one place provider failures are classified.

Every engine (`router.complete`, `chat.chat`, `streaming.stream_chat`), the
retry helper and the circuit breaker ask `policy_for(exc)` what to do with a
failed provider call, so the rules below are the whole policy:

| Kind              | Examples                                       | Retry | Breaker |
|-------------------|------------------------------------------------|-------|---------|
| `TRANSIENT`       | connection error, 408, 409, 5xx, Anthropic 529 | yes   | counts  |
| `TIMEOUT`         | SDK `APITimeoutError`, per-attempt timeout     | no    | counts  |
| `RATE_LIMITED`    | 429                                            | no    | counts  |
| `AUTH`            | 401, 403; `ProviderAuthError` (e.g. no Entra   | no    | counts  |
|                   | token for Azure, a google-auth refresh error   |       |         |
|                   | or a missing extra for Vertex)                 |       |         |
| `BILLING`         | 400/402/429 whose message says the account is  | no    | counts, |
|                   | out of credit or over a spend limit (see       |       | opens   |
|                   | `_BILLING_MARKERS`)                            |       | at once |
| `INVALID_REQUEST` | 400, 404, 413, 422 and every other 4xx,        | no    | ignored |
|                   | including Azure's 400 `content_filter`         |       |         |
| `INVALID_OUTPUT`  | no text, a refusal, or structured output that  | no    | ignored |
|                   | fails to parse/validate; Groq's 400            |       |         |
|                   | `json_validate_failed`                         |       |         |
| `EMPTY_RESPONSE`  | `EmptyCompletionError`: no text (None, empty   | no    | counts  |
|                   | or whitespace) and no tool calls, e.g. a       |       |         |
|                   | reasoning model that spent its token budget    |       |         |
| `UNKNOWN`         | anything else (e.g. a response-parsing bug)    | no    | counts  |
| `POLICY_VIOLATION`| `PolicyViolationError`: no provider satisfies  | no    | ignored |
|                   | the routing policy (raised by the gateway      |       |         |
|                   | before a call, never by a provider)            |       |         |

Vertex AI errors arrive as ordinary anthropic SDK status errors (429
`RESOURCE_EXHAUSTED` is `RATE_LIMITED`, 5xx `TRANSIENT`, 400/404
`INVALID_REQUEST`). A Google-style error body inside a 200 stream is
classified by its `status` (see `_GOOGLE_RPC_STATUSES`).

Every provider-failure kind fails over to the next provider in the chain.
`POLICY_VIOLATION` is not a provider failure: it ends the call, because
every remaining provider has already been excluded (see routing.py). See the README's
"Retries, failover and timeouts" section for why timeouts and 429s are not
retried on the same provider, and why a 400/422 still fails over.

`INVALID_OUTPUT` never trips the breaker. The provider was reachable and
answered, and whether a reply matches a caller's schema depends mostly on
that schema and prompt: counting it would let one caller's hard schema take
a healthy provider out of rotation for everyone. It is not retried on the
same provider either. The next provider is a better bet than re-rolling the
same model, and a retry would double the cost of a call that already failed.

An empty reply that ended in a safety filter or refusal (`content_filter`,
Gemini `SAFETY`, Anthropic `refusal`, or a 200 with no choices) is
`INVALID_OUTPUT` instead: whether the filter fires depends on the caller's
content, so, like Azure's 400 `content_filter` on the prompt, it must not let
one user's trigger text open the breaker for everyone.

`EMPTY_RESPONSE` is also not retried, but it *does* count toward the breaker.
An empty reply is not about one caller's schema: it is what a provider/model
configuration produces for everyone, typically a reasoning model that spends
its whole `max_tokens` budget on reasoning (`finish_reason="length"`). The same
budget gives the same result on a retry, and taking the provider out of
rotation saves every later caller the latency and the billed reasoning tokens.

When *no* provider serves the call at all, `complete()`/`chat()`/
`stream_chat()` raise one of two `LLMError` subclasses so a caller can tell
"fix my deployment" apart from "wait and retry":

- `GatewayNotConfiguredError`: not one provider in `provider_order` has
  credentials set. A deployment-time problem, not an outage.
- `AllProvidersExhaustedError`: at least one provider is configured, but
  every one currently failed, or every one's circuit breaker is open (or the
  call deadline ran out first) — e.g. every configured provider is down at
  once (an Anthropic spend limit, an OpenAI billing lapse and a Groq rate
  limit, all at the same time). This is the transient case: the same
  providers are expected to recover within `BREAKER_COOLDOWN_SECONDS`.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, get_args

import anthropic
import openai

if TYPE_CHECKING:
    from .config import GatewayConfig
    from .providers.base import Usage


class LLMError(Exception):
    """Raised when no provider could serve a call."""


class LLMDeadlineExceeded(LLMError):
    """The call's overall deadline (`GatewayConfig.call_deadline_seconds`)
    ran out across retries and failovers. A subclass of `LLMError`, so
    callers that only catch `LLMError` keep working unchanged.

    `failures` lists what each attempted provider failed with (see
    `ProviderFailure`); empty when none had failed yet."""

    failures: tuple["ProviderFailure", ...] = ()


class GatewayNotConfiguredError(LLMError):
    """No provider in `provider_order` has credentials/configuration set (or
    is even registered), so none was attempted. A permanent, deployment-time
    problem: the caller is missing an API key or an env var, not experiencing
    an outage. Distinguish this from `AllProvidersExhaustedError` (every
    *configured* provider is unavailable right now) so the calling app can
    tell "fix my configuration" apart from "retry shortly". A subclass of
    `LLMError`, so existing `except LLMError` handling is unaffected."""


class AllProvidersExhaustedError(LLMError):
    """At least one provider was configured, but the call could not be
    served: either every attempt failed (e.g. Anthropic over its spend
    limit, OpenAI out of credits and Groq rate-limited, all at once), or
    every configured provider's circuit breaker is currently open (or the
    call deadline ran out before any could be attempted). This is the
    "temporarily exhausted, retry shortly" case — the providers are known
    good configuration-wise, just unavailable right now, typically for no
    longer than `BREAKER_COOLDOWN_SECONDS`. A subclass of `LLMError`.

    `failures` has one `ProviderFailure` per failed attempt (provider, model,
    classified `kind`, status code, redacted detail), so an operator can tell
    "Anthropic over its spend limit, Groq rate limited, OpenAI model retired"
    apart without parsing the message. Empty when every breaker was open."""

    failures: tuple["ProviderFailure", ...] = ()


class PolicyViolationError(LLMError):
    """No provider in the chain satisfies the call's routing policy
    (residency, retention, training, DPA, only/ignore, capabilities, cost cap
    or the budget hook). Raised *before* any network call to an excluded
    provider: the gateway never falls back to a non-compliant one.

    `exclusions` maps each excluded provider id to the reasons it was
    excluded. Neither the message nor the reasons contain prompt content."""

    def __init__(self, message: str, exclusions: dict[str, tuple[str, ...]] | None = None):
        super().__init__(message)
        self.exclusions: dict[str, tuple[str, ...]] = dict(exclusions or {})


class ResidencyViolationError(PolicyViolationError):
    """`GatewayConfig.require_residency` is set and no provider in the chain has
    that residency (an "unknown" residency never counts). Raised before any
    network call; the gateway never falls back to a provider outside the
    required residency. A subclass of `PolicyViolationError`."""


class UnsupportedCapabilityError(PolicyViolationError):
    """Every provider was excluded only because its model can't serve a
    feature the request uses (structured output, tools, images, streaming).
    A subclass of `PolicyViolationError`, so one `except` covers both."""


class ModelRetiredError(PolicyViolationError):
    """Every provider was excluded only because its model is retired (the
    vendor's announced shutdown date has passed; see lifecycle.py). Raised
    before any network call. Turn the check off with
    `GatewayConfig.block_retired_models=False`. A subclass of
    `PolicyViolationError`."""


class InvalidRequestError(LLMError, ValueError):
    """The request itself is malformed, so no provider was tried: a bad tool
    definition or JSON Schema, a `tool_choice` naming an unknown tool, or a
    tool result with no matching tool call. Raised before any network call
    (a provider would answer 400, which the engine would count as a provider
    failure and fail over on). The message names the offending tool or
    message index, never message content. A subclass of `LLMError` (and
    `ValueError`)."""


class InvalidOutputError(Exception):
    """A provider answered, but not with something this call can use: no text
    content, a refusal, or structured output that is not valid JSON or fails
    schema validation. Raised inside one provider attempt, so the engine fails
    over to the next provider. The message never includes the model's output,
    which can echo prompt content, including personal data.

    `paths` lists where validation failed (dotted, e.g. `items.1.qty`;
    `<root>` for the top level) when the failure came from a schema check,
    else it is empty. Paths are field names, never values.

    `usage` is the token usage the provider billed for the rejected reply
    when the adapter knew it, else `None`; attempt reports use it so a paid
    but unusable reply is not metered as free."""

    usage: "Usage | None" = None

    def __init__(self, message: str = "", paths: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.paths = paths


class ToolCallTruncatedError(InvalidOutputError):
    """The reply hit the token budget (`finish_reason` "length") inside a tool
    call, so its arguments are incomplete. It is never passed on as a
    half-parsed dict. A subclass of `InvalidOutputError`, so the engine fails
    over exactly as before; catch this one (it is also what an exhausted chain
    chains from) to retry with a larger `max_tokens`. `tool_names` lists the
    calls in the cut-off reply; no arguments are included."""

    def __init__(self, message: str = "", tool_names: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.finish_reason = "length"
        self.tool_names = tool_names


class OutputTruncatedError(InvalidOutputError):
    """A structured-output reply hit the token budget (`finish_reason`
    "length" / "max_tokens") and is not complete JSON. It is never repaired by
    guessing at an embedded fragment. A subclass of `InvalidOutputError`, so
    the engine fails over exactly as before; catch it to retry with a larger
    `max_tokens`. No output is included."""

    def __init__(self, message: str = "", stop_reason: str = "length") -> None:
        super().__init__(message, ("<root>",))
        self.finish_reason = stop_reason


class ToolCallInvalidError(InvalidOutputError):
    """With `chat(validate_tool_calls="error")`: the model named a tool the
    caller never offered, or its arguments fail the tool's JSON Schema (or are
    larger than `stream_max_tool_arguments_chars`). A subclass of
    `InvalidOutputError`, so the engine fails over to the next provider and an
    exhausted chain chains from it. `issues` are `ToolCallIssue`s (call id, tool
    name, kind, failing argument paths; never values)."""

    def __init__(self, message: str = "", issues: tuple[Any, ...] = ()) -> None:
        paths = tuple(dict.fromkeys(p for issue in issues for p in issue.paths))
        super().__init__(message, paths)
        self.issues = issues


class FirstTokenTimeoutError(TimeoutError):
    """A streaming attempt sent no first chunk within
    `first_token_timeout_seconds`. A builtin `TimeoutError`, so it is
    classified as a provider timeout (counts for the breaker, not retried on
    the same provider) and the stream fails over before anything was yielded."""


class StreamTooLargeError(InvalidOutputError):
    """A streamed reply grew past a configured size cap
    (`stream_max_tool_arguments_chars` for one tool call's arguments,
    `stream_max_total_chars` for the whole stream's text and arguments). The
    stream ends with this error instead of being truncated silently. A subclass
    of `InvalidOutputError`, so failover and breaker handling are unchanged.
    The message carries sizes and limits, never the output."""

    def __init__(self, message: str = "", limit: int = 0) -> None:
        super().__init__(message)
        self.limit = limit


class ResponseTooLargeError(StreamTooLargeError):
    """A provider's HTTP response passed a transport-level size cap while it
    was being read: a non-streamed body over `max_response_body_bytes`, one SSE
    event (or line) over `max_sse_event_bytes`, or a whole stream over
    `max_stream_bytes`. The connection is closed at once and nothing past the
    cap is buffered. A subclass of `StreamTooLargeError` (so of
    `InvalidOutputError`): classified `INVALID_OUTPUT`, never retried on the
    same provider, does not count toward the circuit breaker (the provider
    answered; an oversized reply says nothing about its availability), and the
    chain fails over. The SDKs wrap errors raised while a non-streamed body is
    read in their own connection error; `classify` looks through that
    `__cause__`. `kind` is `"body"`, `"sse_event"` or `"stream"`; `limit` is
    the cap in bytes. The message never carries response content.
    """

    def __init__(
        self,
        message: str = "",
        limit: int = 0,
        kind: Literal["body", "sse_event", "stream"] = "body",
    ) -> None:
        super().__init__(message, limit=limit)
        self.kind = kind


class ProviderAuthError(Exception):
    """A provider could not authenticate before sending its request, e.g.
    Azure with `azure_auth="entra"` when no Microsoft Entra token can be
    acquired or azure-identity isn't installed. Classified `AUTH`: fails over
    and counts toward the breaker, like a 401. The message names the provider
    and the cause's exception type, never a token or key."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class EmptyCompletionError(Exception):
    """A provider answered with no usable text (None, empty or whitespace
    content) and no tool calls.

    The usual cause is a reasoning model whose reasoning used up the whole
    completion budget before any answer text, which the provider reports as
    `finish_reason="length"` with the tokens counted as reasoning. The message
    carries provider, model, finish reason and reasoning tokens, never content.
    `usage` is what the provider billed for the reply when known, else `None`.
    """

    usage: "Usage | None" = None

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        finish_reason: str | None,
        reasoning_tokens: int | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens
        super().__init__(
            f"{provider} returned empty content for model {model} "
            f"(finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens})."
        )


class ErrorKind(StrEnum):
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    AUTH = "auth"
    BILLING = "billing"
    INVALID_REQUEST = "invalid_request"
    INVALID_OUTPUT = "invalid_output"
    EMPTY_RESPONSE = "empty_response"
    UNKNOWN = "unknown"
    POLICY_VIOLATION = "policy_violation"


@dataclass(frozen=True)
class ErrorPolicy:
    retry: bool
    """Worth another attempt on the same provider after a short backoff."""
    trips_breaker: bool
    """Counts toward that provider's circuit breaker."""
    opens_immediately: bool = False
    """One occurrence opens the breaker for the cooldown, regardless of the
    failure threshold: the condition (an empty balance) cannot clear by itself
    within a few requests, so waiting for N more failing round trips only
    adds latency and noise."""


POLICIES: dict[ErrorKind, ErrorPolicy] = {
    ErrorKind.TRANSIENT: ErrorPolicy(retry=True, trips_breaker=True),
    ErrorKind.TIMEOUT: ErrorPolicy(retry=False, trips_breaker=True),
    ErrorKind.RATE_LIMITED: ErrorPolicy(retry=False, trips_breaker=True),
    ErrorKind.AUTH: ErrorPolicy(retry=False, trips_breaker=True),
    ErrorKind.BILLING: ErrorPolicy(retry=False, trips_breaker=True, opens_immediately=True),
    ErrorKind.INVALID_REQUEST: ErrorPolicy(retry=False, trips_breaker=False),
    ErrorKind.INVALID_OUTPUT: ErrorPolicy(retry=False, trips_breaker=False),
    ErrorKind.EMPTY_RESPONSE: ErrorPolicy(retry=False, trips_breaker=True),
    ErrorKind.UNKNOWN: ErrorPolicy(retry=False, trips_breaker=True),
    # A policy exclusion (including a budget_check denial) says nothing about
    # the provider's health.
    ErrorKind.POLICY_VIOLATION: ErrorPolicy(retry=False, trips_breaker=False),
}

# APITimeoutError subclasses APIConnectionError in both SDKs, so timeouts are
# checked first. The builtin TimeoutError is what asyncio.timeout() raises
# when the gateway's own per-attempt bound fires.
_TIMEOUT_ERRORS = (anthropic.APITimeoutError, openai.APITimeoutError, TimeoutError)
_CONNECTION_ERRORS = (anthropic.APIConnectionError, openai.APIConnectionError)
_STATUS_ERRORS = (anthropic.APIStatusError, openai.APIStatusError)
_SDK_ERRORS = (anthropic.APIError, openai.APIError)

# Error `type`s carried in the body of an error that arrives *inside* an SSE
# stream, where there is no meaningful HTTP status: Anthropic raises an
# APIStatusError with the stream's 200 status, OpenAI a bare APIError.
_BODY_ERROR_TYPES: dict[str, ErrorKind] = {
    "api_error": ErrorKind.TRANSIENT,
    "overloaded_error": ErrorKind.TRANSIENT,
    "server_error": ErrorKind.TRANSIENT,
    "timeout_error": ErrorKind.TRANSIENT,
    "rate_limit_error": ErrorKind.RATE_LIMITED,
    "rate_limit_exceeded": ErrorKind.RATE_LIMITED,
    "authentication_error": ErrorKind.AUTH,
    "permission_error": ErrorKind.AUTH,
    "invalid_request_error": ErrorKind.INVALID_REQUEST,
    "not_found_error": ErrorKind.INVALID_REQUEST,
    "request_too_large": ErrorKind.INVALID_REQUEST,
    # Anthropic's 402 type; inside a 200 stream there is no status to read.
    "billing_error": ErrorKind.BILLING,
}


def _error_metadata(body: object) -> dict:
    if not isinstance(body, dict):
        return {}
    nested = body.get("error")
    source = nested if isinstance(nested, dict) else body
    metadata = source.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _classify_forbidden(body: object) -> ErrorKind:
    """OpenRouter shares 403 between permissions, a moderation flag on the
    input, a guardrail block and a budget block
    (https://openrouter.ai/docs/api-reference/errors). Only the first is an
    auth problem: a flagged input is about this request's content (it must not
    trip the provider's breaker) and a budget block is a spend limit."""
    metadata = _error_metadata(body)
    if "limit_source" in metadata or "limit_usd" in metadata:
        return ErrorKind.BILLING
    if "flagged_input" in metadata or "reasons" in metadata or "patterns" in metadata:
        return ErrorKind.INVALID_REQUEST
    return ErrorKind.AUTH


def _classify_status(status: int, body: object = None) -> ErrorKind:
    if status in (408, 409) or status >= 500:
        return ErrorKind.TRANSIENT
    if status == 429:
        return ErrorKind.RATE_LIMITED
    if status == 498:  # Groq: flex tier at capacity, request not processed
        return ErrorKind.RATE_LIMITED
    if status == 403:
        return _classify_forbidden(body)
    if status == 401:
        return ErrorKind.AUTH
    if status == 402:  # Payment Required: always an account problem
        return ErrorKind.BILLING
    return ErrorKind.INVALID_REQUEST


# Google API error `status` values (google.rpc.Code names), for a Vertex AI
# error body without an Anthropic `type`: `{"error": {"code": 429,
# "status": "RESOURCE_EXHAUSTED", ...}}`. Only consulted when there is no
# usable HTTP status, i.e. inside a 200 stream.
_GOOGLE_RPC_STATUSES: dict[str, ErrorKind] = {
    "RESOURCE_EXHAUSTED": ErrorKind.RATE_LIMITED,
    "UNAVAILABLE": ErrorKind.TRANSIENT,
    "INTERNAL": ErrorKind.TRANSIENT,
    "DEADLINE_EXCEEDED": ErrorKind.TRANSIENT,
    "ABORTED": ErrorKind.TRANSIENT,
    "UNAUTHENTICATED": ErrorKind.AUTH,
    "PERMISSION_DENIED": ErrorKind.AUTH,
    "INVALID_ARGUMENT": ErrorKind.INVALID_REQUEST,
    "FAILED_PRECONDITION": ErrorKind.INVALID_REQUEST,
    "NOT_FOUND": ErrorKind.INVALID_REQUEST,
    "OUT_OF_RANGE": ErrorKind.INVALID_REQUEST,
}


def _classify_body(body: object) -> ErrorKind:
    if not isinstance(body, dict):
        return ErrorKind.UNKNOWN
    nested = body.get("error")
    error_type = nested.get("type") if isinstance(nested, dict) else body.get("type")
    if error_type:
        return _BODY_ERROR_TYPES.get(error_type, ErrorKind.UNKNOWN)
    source = nested if isinstance(nested, dict) else body
    status = source.get("status")
    if isinstance(status, str):
        return _GOOGLE_RPC_STATUSES.get(status, ErrorKind.UNKNOWN)
    # OpenRouter (and some other brokers) put the upstream HTTP status in the
    # body's numeric `code`: `{"error": {"code": 502, "message": "Provider
    # returned error", "metadata": {"provider_name": ...}}}`.
    code = source.get("code")
    if isinstance(code, int) and not isinstance(code, bool) and 400 <= code <= 599:
        return _classify_status(code, body)
    return ErrorKind.UNKNOWN


# Error codes sent with an HTTP 400 that mean "the model's output failed the
# requested format" rather than "the request was malformed". Groq returns
# `json_validate_failed` when JSON mode or json_schema generation fails.
_OUTPUT_ERROR_CODES = frozenset({"json_validate_failed"})


def _body_error_code(body: object) -> object:
    if not isinstance(body, dict):
        return None
    nested = body.get("error")
    return nested.get("code") if isinstance(nested, dict) else body.get("code")


# Stable, lower-case fragments of provider billing/quota messages. Matched as
# substrings of the lower-cased message/code/type, so wording changes around
# them do not matter. Deliberately specific: a bare "billing" or "quota" would
# also match genuine request errors.
_BILLING_MARKERS = (
    "credit balance is too low",  # Anthropic 400
    "reached your specified api usage limits",  # Anthropic spend limit 400
    "api usage limits",
    "insufficient_quota",  # OpenAI code/type, 429 or 400
    "exceeded your current quota",  # OpenAI message
    "plans and billing",
    "billing hard limit",
    "spend limit",
    "spending limit",
    "insufficient credits",
    "insufficient balance",
    "out of credits",
    "billing_not_active",  # OpenAI code
    "account is not active",  # OpenAI: "...check your billing details"
    "billing account",  # Vertex/GCP 403: billing account disabled/closed
    "billing is disabled",
    "billing has not been enabled",
    "payment required",
    "monthly spend",
    "usage limit has been reached",
    "credit_balance_exhausted",  # OpenAI error code (developers.openai.com error-codes)
    "prepayment credits",  # Gemini 400: "Your prepayment credits are depleted"
)
# Only these statuses (or none, inside a 200 stream) can mean "billing".
_BILLING_STATUSES = frozenset({400, 402, 403, 429})


def _error_text(exc: BaseException) -> str:
    parts: list[str] = [str(getattr(exc, "message", "") or "")]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        source = nested if isinstance(nested, dict) else body
        parts += [str(source.get(k) or "") for k in ("message", "code", "type")]
    return " ".join(parts).lower()


def is_billing_error(exc: BaseException) -> bool:
    """True for a provider error saying the account is out of credit, over a
    spend limit or out of quota. Reads only the error's own message/code/type,
    never request content."""
    if not isinstance(exc, _SDK_ERRORS):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 400 and status not in _BILLING_STATUSES:
        return False
    text = _error_text(exc)
    return any(marker in text for marker in _BILLING_MARKERS)


# An empty reply that ended in a safety filter or refusal depends on the
# caller's content, not on the provider/model configuration.
_FILTERED_FINISH_REASONS = frozenset(
    {
        "content_filter",
        "safety",
        "refusal",
        "recitation",
        "blocklist",
        "prohibited_content",
        "spii",
        "no_choices",
    }
)


def _caused_by_oversized_response(exc: BaseException) -> bool:
    """The SDKs wrap whatever the transport raises while reading a body in
    their own connection error; look through the explicit cause chain."""
    seen = 0
    cause = exc.__cause__
    while cause is not None and seen < 5:
        if isinstance(cause, ResponseTooLargeError):
            return True
        cause = cause.__cause__
        seen += 1
    return False


def classify(exc: BaseException) -> ErrorKind:
    if _caused_by_oversized_response(exc):
        return ErrorKind.INVALID_OUTPUT
    if isinstance(exc, PolicyViolationError):
        return ErrorKind.POLICY_VIOLATION
    if isinstance(exc, InvalidOutputError):
        return ErrorKind.INVALID_OUTPUT
    if isinstance(exc, InvalidRequestError):
        return ErrorKind.INVALID_REQUEST
    if isinstance(exc, EmptyCompletionError):
        if str(exc.finish_reason or "").lower() in _FILTERED_FINISH_REASONS:
            return ErrorKind.INVALID_OUTPUT
        return ErrorKind.EMPTY_RESPONSE
    if isinstance(exc, ProviderAuthError):
        return ErrorKind.AUTH
    if isinstance(exc, _TIMEOUT_ERRORS):
        return ErrorKind.TIMEOUT
    if isinstance(exc, _CONNECTION_ERRORS):
        return ErrorKind.TRANSIENT
    if is_billing_error(exc):
        return ErrorKind.BILLING
    if isinstance(exc, _STATUS_ERRORS) and exc.status_code >= 400:
        if exc.status_code == 400 and _body_error_code(exc.body) in _OUTPUT_ERROR_CODES:
            return ErrorKind.INVALID_OUTPUT
        return _classify_status(exc.status_code, exc.body)
    if isinstance(exc, _SDK_ERRORS):
        return _classify_body(exc.body)
    return ErrorKind.UNKNOWN


_MODEL_MISSING_CODES = frozenset(
    {
        "model_not_found",
        "deployment_not_found",
        "deploymentnotfound",
        "invalid_model",
        "model_decommissioned",  # Groq: 400 for a retired model
        "model_deprecated",
        "model_retired",
    }
)
# Wording of a bad model id on hosts that answer 400 instead of 404 (Mistral's
# `invalid_model`, OpenRouter's "<id> is not a valid model ID").
_MODEL_MISSING_PHRASES = (
    "not a valid model",
    "invalid model:",
    "has been decommissioned",
    "has been retired",
    "model has been deprecated",
    "is no longer supported",
)


def is_model_scoped(exc: BaseException) -> str | None:
    """Whether a provider error is specific to the model that was called, so a
    breaker may open that (provider, model) instead of the whole provider.

    `"missing"`: the model (or Azure deployment) does not exist for this
    account, a 404 naming the model or a `model_not_found`-style code; it
    cannot heal by itself. `"overloaded"`: the model is overloaded (HTTP 529
    or an `overloaded_error` body). `None` otherwise: auth, billing, rate
    limits, timeouts, connection and ordinary 5xx failures stay provider-wide.
    Reads only the error's own status, code and message."""
    if not isinstance(exc, _SDK_ERRORS):
        return None
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status == 529:
        return "overloaded"
    body = getattr(exc, "body", None)
    code = str(_body_error_code(body) or "").lower()
    nested = body.get("error") if isinstance(body, dict) else None
    source = nested if isinstance(nested, dict) else body if isinstance(body, dict) else {}
    if str(source.get("type") or "") == "overloaded_error":
        return "overloaded"
    if code in _MODEL_MISSING_CODES or str(source.get("type") or "") == "invalid_model":
        return "missing"
    if status == 400 and any(p in _error_text(exc) for p in _MODEL_MISSING_PHRASES):
        return "missing"
    if status == 404 and "model" in _error_text(exc):
        return "missing"
    return None


FailureClass = Literal[
    "network",
    "timeout",
    "rate_limited",
    "auth",
    "billing",
    "invalid_request",
    "invalid_output",
    "empty_response",
    "unknown",
    "policy_violation",
    "model_retired",
    "overloaded",
]
"""The closed set of `failure_class()` labels (also `ProviderFailure.kind`)."""

_FAILURE_CLASSES: dict[ErrorKind, FailureClass] = {
    ErrorKind.TRANSIENT: "network",
    ErrorKind.TIMEOUT: "timeout",
    ErrorKind.RATE_LIMITED: "rate_limited",
    ErrorKind.AUTH: "auth",
    ErrorKind.BILLING: "billing",
    ErrorKind.INVALID_REQUEST: "invalid_request",
    ErrorKind.INVALID_OUTPUT: "invalid_output",
    ErrorKind.EMPTY_RESPONSE: "empty_response",
    ErrorKind.UNKNOWN: "unknown",
    ErrorKind.POLICY_VIOLATION: "policy_violation",
}
FAILURE_CLASSES: frozenset[str] = frozenset(get_args(FailureClass))


def failure_class(exc: BaseException) -> FailureClass:
    """A stable, alertable label for why a provider failed: `billing`
    (out of credit, spend limit, quota), `rate_limited`, `auth`,
    `model_retired` (the model or deployment no longer exists: needs a config
    change), `overloaded`, `timeout`, `network` (connection errors, 5xx),
    `invalid_request`, `invalid_output`, `empty_response`, `policy_violation`
    or `unknown`. Billing and auth win over the model check."""
    kind = classify(exc)
    if kind in (ErrorKind.BILLING, ErrorKind.AUTH):
        return _FAILURE_CLASSES[kind]
    scope = is_model_scoped(exc)
    if scope == "missing":
        return "model_retired"
    if scope == "overloaded":
        return "overloaded"
    return _FAILURE_CLASSES[kind]


@dataclass(frozen=True)
class ProviderFailure:
    """One failed provider attempt, as listed in `AllProvidersExhaustedError.failures`."""

    provider: str
    model: str | None
    kind: FailureClass
    """One of `FAILURE_CLASSES` (see `failure_class`)."""
    status_code: int | None
    detail: str
    """Provider error text, credential-scrubbed and capped to 300 chars."""


def provider_failure(
    provider: str,
    model: str | None,
    exc: BaseException,
    config: "GatewayConfig | None" = None,
) -> ProviderFailure:
    status = getattr(exc, "status_code", None)
    return ProviderFailure(
        provider=provider,
        model=model,
        kind=failure_class(exc),
        status_code=status if isinstance(status, int) else None,
        detail=redact_secrets(str(exc), config, max_chars=300),
    )


def failure_summary(failures: Sequence[ProviderFailure]) -> str:
    """`anthropic=billing, groq=rate_limited` (one entry per provider, latest wins)."""
    latest = {f.provider: f.kind for f in failures}
    return ", ".join(f"{p}={k}" for p, k in latest.items())


def policy_for(exc: BaseException) -> ErrorPolicy:
    return POLICIES[classify(exc)]


def retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds a provider asked us to wait (`Retry-After`, delta-seconds or
    HTTP-date form) on an SDK status error, or `None`. Capped at 60 s so a
    hostile or broken header can't park a call; a date in the past is 0. The
    retry loop still drops the retry if it doesn't fit the call deadline."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
        try:
            value = float(raw)
        except ValueError:
            when = parsedate_to_datetime(raw)
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            # A date already past means "now".
            value = max((when - datetime.now(UTC)).total_seconds(), 0.0)
    except (TypeError, ValueError, AttributeError, IndexError, OverflowError):
        return None
    if not value >= 0:  # negative delta or NaN
        return None
    return min(value, 60.0)


def is_retryable(exc: BaseException) -> bool:
    return policy_for(exc).retry


_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{4,}"),
    re.compile(
        # Also access_token, client_secret, Ocp-Apim-Subscription-Key, ...: up to
        # three word-and-separator prefixes before the credential name.
        r"(?i)(?<![A-Za-z0-9])((?:[a-z]{1,20}[-_]){0,3}"
        r"(?:authorization|x-api-key|api[-_ ]?key|apikey|token|secret|password"
        r"|subscription[-_]key))"
        r"(\"?\s*[:=]\s*\"?)(?:bearer\s+)?[^\s\",;&'}]{4,}"
    ),
    # A `key=` query parameter (Gemini's `?key=AIza...`).
    re.compile(r"(?i)([?&]key)(=)[^\s&\"'#]{4,}"),
    re.compile(r"\b(?:sk|gsk|hf|xai|pk|rk)[-_][A-Za-z0-9_-]{8,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    re.compile(r"\bya29\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"),
)


# scheme://user:password@host — credentials embedded in a URL.
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.-]{1,15}://)[^\s/@:]{1,256}:[^\s/@]{1,256}@")


def redact_secrets(
    text: str, config: "GatewayConfig | None" = None, *, max_chars: int | None = None
) -> str:
    """Scrub credentials from provider error text before it goes into an
    exception message, a span status or an HTTP response: configured key
    values verbatim, plus bearer tokens, `api-key: ...` pairs, and the common
    provider key shapes (providers sometimes echo the rejected key).
    With `max_chars`, longer text is cut (before scrubbing, so the work stays
    bounded) and marked `... [truncated]`."""
    truncated = max_chars is not None and len(text) > max_chars
    if truncated:
        text = text[: max_chars or 0]
    if config is not None:
        for value in sorted(config.secret_values, key=len, reverse=True):
            text = text.replace(value, "[REDACTED]")
    text = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", text)
    text = _SECRET_PATTERNS[0].sub(r"\1 [REDACTED]", text)
    for pattern in _SECRET_PATTERNS[1:3]:
        text = pattern.sub(r"\1\2[REDACTED]", text)
    for pattern in _SECRET_PATTERNS[3:]:
        text = pattern.sub("[REDACTED]", text)
    return f"{text}... [truncated]" if truncated else text


ERROR_DETAIL_MAX_CHARS = 1000

# Header and mapping-key names whose value is a credential whatever it looks like.
_SENSITIVE_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "ocp-apim-subscription-key",
        "cookie",
        "set-cookie",
        "api_key",
        "apikey",
        "key",
        "token",
        "access_token",
        "secret",
        "password",
        "client_secret",
    }
)
_MAX_SCRUB_NODES = 20000


def _scrub_value(value: object, config: "GatewayConfig | None", budget: list[int], depth: int = 0):
    """A credential-scrubbed copy of a str / dict / list / tuple (other values
    are returned as they are). Mapping keys that name a credential are blanked
    whatever their value looks like. Bounded in depth and node count."""
    budget[0] -= 1
    if budget[0] < 0 or depth > 8:
        return value
    if isinstance(value, str):
        return redact_secrets(value, config)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() in _SENSITIVE_NAMES and isinstance(v, str):
                out[k] = "[REDACTED]"
            else:
                out[k] = _scrub_value(v, config, budget, depth + 1)
        return out
    if isinstance(value, list):
        return [_scrub_value(v, config, budget, depth + 1) for v in value]
    if isinstance(value, tuple) and type(value) is tuple:
        return tuple(_scrub_value(v, config, budget, depth + 1) for v in value)
    return value


def _scrub_http_object(obj: object, config: "GatewayConfig | None", seen: set[int]) -> None:
    """Scrub an `httpx`-style request or response in place: credential headers
    blanked, other header values and the URL redacted, a buffered response body
    redacted, the request body dropped (it holds the prompt, not credentials).
    In place, so the frames an error tracker captures hold the scrubbed object
    too. Duck-typed (`httpx`, `httpx2`, `requests`-like); best effort."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    headers = getattr(obj, "headers", None)
    if headers is not None and hasattr(headers, "items"):
        try:
            for name, val in list(headers.items()):
                new = (
                    "[REDACTED]"
                    if str(name).lower() in _SENSITIVE_NAMES
                    else redact_secrets(str(val), config)
                )
                if new != val:
                    headers[name] = new
        except Exception:  # noqa: BLE001, S110
            pass
    try:
        url = getattr(obj, "url", None)
        if url is not None and not isinstance(url, str) and hasattr(obj, "method"):
            scrubbed = redact_secrets(str(url), config)
            if scrubbed != str(url):
                obj.url = type(url)(scrubbed)  # type: ignore[attr-defined]
        if hasattr(obj, "method") and hasattr(obj, "stream"):
            if hasattr(obj, "_content"):
                obj._content = b""  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        content = getattr(obj, "_content", None)
        if isinstance(content, bytes) and content and hasattr(obj, "status_code"):
            text = content.decode("utf-8", "replace")
            scrubbed = redact_secrets(text, config)
            if scrubbed != text:
                obj._content = scrubbed.encode()  # type: ignore[attr-defined]
                for cached in ("_text", "_decoded_content"):
                    obj.__dict__.pop(cached, None)
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        request = obj.__dict__.get("_request") or getattr(obj, "request", None)
        if request is not None and request is not obj:
            _scrub_http_object(request, config, seen)
    except Exception:  # noqa: BLE001, S110
        pass


def _scrub_exception_attributes(exc: BaseException, config: "GatewayConfig | None") -> None:
    """Scrub `body`, `param`, `code`, `response`, `request` and any other
    string / dict / list / HTTP-object attribute of an SDK exception."""
    budget = [_MAX_SCRUB_NODES]
    seen: set[int] = set()
    for name, value in list(getattr(exc, "__dict__", {}).items()):
        try:
            if isinstance(value, str | dict | list | tuple):
                scrubbed = _scrub_value(value, config, budget)
                if scrubbed != value:
                    setattr(exc, name, scrubbed)
            elif hasattr(value, "headers") or hasattr(value, "_content"):
                _scrub_http_object(value, config, seen)
        except Exception:  # noqa: BLE001, S110 - scrubbing must never mask the real error
            pass


def scrub_exception_chain(
    error: BaseException | None, config: "GatewayConfig | None" = None
) -> None:
    """Scrub credentials from the text of `error` and its `__cause__` /
    `__context__` chain, in place. The gateway chains its aggregate errors from
    the last provider error, and a traceback (logged by the caller, sent to an
    error tracker) prints that error's message, which a provider can fill with
    the rejected key or an `Authorization` header. Only string `args` (and a
    string `message`) are rewritten, and only when scrubbing changes them. SDK
    errors also carry `body`, `response` and `request`, which error trackers
    serialise: strings, dicts and lists among their attributes are replaced by
    scrubbed copies, and an attached HTTP request/response is scrubbed in
    place (credential headers such as `Authorization` / `x-api-key` blanked,
    URL query keys redacted, response body redacted, request body dropped).
    The exception type, `status_code`, `response.status_code`, response
    headers (e.g. `Retry-After`), the request method and host, and the shape
    of `body` (error type and code) stay available. Never raises."""
    seen: set[int] = set()
    todo: list[BaseException | None] = [error]
    while todo and len(seen) < 16:
        exc = todo.pop()
        if exc is None or id(exc) in seen:
            continue
        seen.add(id(exc))
        todo += [exc.__cause__, exc.__context__]
        try:
            args = tuple(redact_secrets(a, config) if isinstance(a, str) else a for a in exc.args)
            if args != exc.args:
                exc.args = args
            message = getattr(exc, "message", None)
            if isinstance(message, str):
                scrubbed = redact_secrets(message, config)
                if scrubbed != message:
                    exc.message = scrubbed  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001, S110 - scrubbing must never mask the real error
            pass
        _scrub_exception_attributes(exc, config)


def error_detail(error: BaseException | None, config: "GatewayConfig | None" = None) -> str:
    """A provider error's text for a gateway exception message or span status:
    scrubbed of credentials and capped (a provider body can be megabytes and
    can echo request content)."""
    return redact_secrets(str(error), config, max_chars=ERROR_DETAIL_MAX_CHARS)


def deadline_exceeded(
    seconds: float,
    last_error: Exception | None,
    config: "GatewayConfig | None" = None,
    failures: Sequence[ProviderFailure] = (),
) -> LLMDeadlineExceeded:
    scrub_exception_chain(last_error, config)
    detail = f" Last error: {error_detail(last_error, config)}" if last_error else ""
    causes = f" Causes: {failure_summary(failures)}." if failures else ""
    error = LLMDeadlineExceeded(f"Call deadline of {seconds:g}s exceeded.{detail}{causes}")
    error.failures = tuple(failures)
    return error


def all_failed(
    last_error: Exception | None,
    config: "GatewayConfig | None",
    failures: Sequence[ProviderFailure],
    note: str = "",
) -> AllProvidersExhaustedError:
    """The single aggregate error for a chain where every attempt failed: the
    last error's text (as before) plus each provider's classified cause."""
    scrub_exception_chain(last_error, config)
    causes = f" Causes: {failure_summary(failures)}." if failures else ""
    error = AllProvidersExhaustedError(
        f"All configured providers failed. Last error: {error_detail(last_error, config)}"
        f"{note}{causes}"
    )
    error.failures = tuple(failures)
    return error
