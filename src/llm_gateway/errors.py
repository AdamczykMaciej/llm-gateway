"""Gateway error types and the one place provider failures are classified.

Every engine (`router.complete`, `chat.chat`, `streaming.stream_chat`), the
retry helper and the circuit breaker ask `policy_for(exc)` what to do with a
failed provider call, so the rules below are the whole policy:

| Kind              | Examples                                       | Retry | Breaker |
|-------------------|------------------------------------------------|-------|---------|
| `TRANSIENT`       | connection error, 408, 409, 5xx, Anthropic 529 | yes   | counts  |
| `TIMEOUT`         | SDK `APITimeoutError`, per-attempt timeout     | no    | counts  |
| `RATE_LIMITED`    | 429                                            | no    | counts  |
| `AUTH`            | 401, 403                                       | no    | counts  |
| `INVALID_REQUEST` | 400, 404, 413, 422 and every other 4xx         | no    | ignored |
| `UNKNOWN`         | anything else (e.g. a response-parsing bug)    | no    | counts  |

Every kind fails over to the next provider in the chain. See the README's
"Retries, failover and timeouts" section for why timeouts and 429s are not
retried on the same provider, and why a 400/422 still fails over.
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


class ErrorKind(StrEnum):
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN = "unknown"


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
    ErrorKind.UNKNOWN: ErrorPolicy(retry=False, trips_breaker=True),
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


def classify(exc: BaseException) -> ErrorKind:
    if isinstance(exc, _TIMEOUT_ERRORS):
        return ErrorKind.TIMEOUT
    if isinstance(exc, _CONNECTION_ERRORS):
        return ErrorKind.TRANSIENT
    if isinstance(exc, _STATUS_ERRORS) and exc.status_code >= 400:
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
