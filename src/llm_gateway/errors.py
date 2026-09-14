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
|                   | token for Azure)                               |       |         |
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

`EMPTY_RESPONSE` is also not retried, but it *does* count toward the breaker.
An empty reply is not about one caller's schema: it is what a provider/model
configuration produces for everyone, typically a reasoning model that spends
its whole `max_tokens` budget on reasoning (`finish_reason="length"`). The same
budget gives the same result on a retry, and taking the provider out of
rotation saves every later caller the latency and the billed reasoning tokens.
"""

from dataclasses import dataclass
from enum import StrEnum

import anthropic
import openai


class LLMError(Exception):
    """Raised when no provider could serve a call."""


class LLMDeadlineExceeded(LLMError):
    """The call's overall deadline (`GatewayConfig.call_deadline_seconds`)
    ran out across retries and failovers. A subclass of `LLMError`, so
    callers that only catch `LLMError` keep working unchanged."""


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


class UnsupportedCapabilityError(PolicyViolationError):
    """Every provider was excluded only because its model can't serve a
    feature the request uses (structured output, tools, images, streaming).
    A subclass of `PolicyViolationError`, so one `except` covers both."""


class InvalidOutputError(Exception):
    """A provider answered, but not with something this call can use: no text
    content, a refusal, or structured output that is not valid JSON or fails
    schema validation. Raised inside one provider attempt, so the engine fails
    over to the next provider. The message never includes the model's output,
    which can echo prompt content, including personal data."""


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
    """

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


POLICIES: dict[ErrorKind, ErrorPolicy] = {
    ErrorKind.TRANSIENT: ErrorPolicy(retry=True, trips_breaker=True),
    ErrorKind.TIMEOUT: ErrorPolicy(retry=False, trips_breaker=True),
    ErrorKind.RATE_LIMITED: ErrorPolicy(retry=False, trips_breaker=True),
    ErrorKind.AUTH: ErrorPolicy(retry=False, trips_breaker=True),
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
}


def _classify_status(status: int) -> ErrorKind:
    if status in (408, 409) or status >= 500:
        return ErrorKind.TRANSIENT
    if status == 429:
        return ErrorKind.RATE_LIMITED
    if status in (401, 403):
        return ErrorKind.AUTH
    return ErrorKind.INVALID_REQUEST


def _classify_body(body: object) -> ErrorKind:
    if not isinstance(body, dict):
        return ErrorKind.UNKNOWN
    nested = body.get("error")
    error_type = nested.get("type") if isinstance(nested, dict) else body.get("type")
    return _BODY_ERROR_TYPES.get(error_type, ErrorKind.UNKNOWN) if error_type else ErrorKind.UNKNOWN


# Error codes sent with an HTTP 400 that mean "the model's output failed the
# requested format" rather than "the request was malformed". Groq returns
# `json_validate_failed` when JSON mode or json_schema generation fails.
_OUTPUT_ERROR_CODES = frozenset({"json_validate_failed"})


def _body_error_code(body: object) -> object:
    if not isinstance(body, dict):
        return None
    nested = body.get("error")
    return nested.get("code") if isinstance(nested, dict) else body.get("code")


def classify(exc: BaseException) -> ErrorKind:
    if isinstance(exc, PolicyViolationError):
        return ErrorKind.POLICY_VIOLATION
    if isinstance(exc, InvalidOutputError):
        return ErrorKind.INVALID_OUTPUT
    if isinstance(exc, EmptyCompletionError):
        return ErrorKind.EMPTY_RESPONSE
    if isinstance(exc, ProviderAuthError):
        return ErrorKind.AUTH
    if isinstance(exc, _TIMEOUT_ERRORS):
        return ErrorKind.TIMEOUT
    if isinstance(exc, _CONNECTION_ERRORS):
        return ErrorKind.TRANSIENT
    if isinstance(exc, _STATUS_ERRORS) and exc.status_code >= 400:
        if exc.status_code == 400 and _body_error_code(exc.body) in _OUTPUT_ERROR_CODES:
            return ErrorKind.INVALID_OUTPUT
        return _classify_status(exc.status_code)
    if isinstance(exc, _SDK_ERRORS):
        return _classify_body(exc.body)
    return ErrorKind.UNKNOWN


def policy_for(exc: BaseException) -> ErrorPolicy:
    return POLICIES[classify(exc)]


def is_retryable(exc: BaseException) -> bool:
    return policy_for(exc).retry


def deadline_exceeded(seconds: float, last_error: Exception | None) -> LLMDeadlineExceeded:
    detail = f" Last error: {last_error}" if last_error else ""
    return LLMDeadlineExceeded(f"Call deadline of {seconds:g}s exceeded.{detail}")
