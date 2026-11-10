"""0.42.0 error-taxonomy audit: recorded fake payloads per provider doc.

Sources (read 2026-11-10): platform.claude.com/docs/en/api/errors,
openrouter.ai/docs/api-reference/errors, console.groq.com/docs/errors,
developers.openai.com/api/docs/guides/error-codes,
aistudio.google.com/docs/troubleshooting.
"""

import anthropic
import httpx
import openai
import pytest

from llm_gateway import breaker
from llm_gateway.errors import ErrorKind, classify, failure_class, is_retryable


def _oai(status: int, body: dict) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://x.test/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return openai.APIStatusError("error", response=response, body=body.get("error", body))


def _stream_error(body: dict) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(200, request=request)
    return anthropic.APIStatusError("error", response=response, body=body)


@pytest.mark.parametrize(
    "status, body, kind",
    [
        # Groq 498: flex tier at capacity, not a malformed request.
        (
            498,
            {"error": {"message": "Flex tier capacity exceeded", "type": "capacity"}},
            ErrorKind.RATE_LIMITED,
        ),
        # OpenRouter 403 moderation flag: about the input, not the credentials.
        (
            403,
            {
                "error": {
                    "code": 403,
                    "message": "Input flagged",
                    "metadata": {
                        "reasons": ["violence"],
                        "flagged_input": "x",
                        "provider_name": "p",
                        "model_slug": "m",
                    },
                }
            },
            ErrorKind.INVALID_REQUEST,
        ),
        (
            403,
            {"error": {"code": 403, "message": "Blocked", "metadata": {"patterns": ["ssn"]}}},
            ErrorKind.INVALID_REQUEST,
        ),
        # OpenRouter 403 budget block: a spend limit.
        (
            403,
            {
                "error": {
                    "code": 403,
                    "message": "Key limit",
                    "metadata": {"limit_source": "key", "limit_usd": 5, "spend_usd": 5},
                }
            },
            ErrorKind.BILLING,
        ),
        # Plain 403 stays auth.
        (403, {"error": {"code": 403, "message": "Forbidden"}}, ErrorKind.AUTH),
        # OpenAI: documented billing code on a 429 must not look like throttling.
        (
            429,
            {
                "error": {
                    "message": "Out of credit",
                    "type": "invalid_request_error",
                    "code": "credit_balance_exhausted",
                }
            },
            ErrorKind.BILLING,
        ),
        (
            429,
            {
                "error": {
                    "message": "Rate limit reached for requests",
                    "type": "requests",
                    "code": "rate_limit_exceeded",
                }
            },
            ErrorKind.RATE_LIMITED,
        ),
        # Gemini 400: depleted prepay credits is an account problem.
        (
            400,
            {
                "error": {
                    "code": 400,
                    "status": "FAILED_PRECONDITION",
                    "message": "Your prepayment credits are depleted.",
                }
            },
            ErrorKind.BILLING,
        ),
        # Context length stays a fail-fast request error.
        (
            400,
            {
                "error": {
                    "message": "maximum context length",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                }
            },
            ErrorKind.INVALID_REQUEST,
        ),
        # Azure content filter: prompt-specific, never counts against the provider.
        (
            400,
            {
                "error": {
                    "message": "The response was filtered",
                    "code": "content_filter",
                    "status": 400,
                }
            },
            ErrorKind.INVALID_REQUEST,
        ),
        # 503 overloaded stays transient.
        (
            503,
            {"error": {"message": "Model temporarily overloaded", "code": "server_is_overloaded"}},
            ErrorKind.TRANSIENT,
        ),
    ],
)
def test_openai_style_status_errors(status, body, kind):
    assert classify(_oai(status, body)) is kind


def test_groq_498_is_not_retried_on_the_same_provider():
    assert not is_retryable(_oai(498, {"error": {"message": "capacity"}}))


def test_anthropic_billing_error_inside_a_200_stream():
    exc = _stream_error(
        {"type": "error", "error": {"type": "billing_error", "message": "Billing issue"}}
    )
    assert classify(exc) is ErrorKind.BILLING
    assert failure_class(exc) == "billing"
    assert not is_retryable(exc)


def test_billing_error_opens_the_breaker_at_once():
    breaker.reset()
    exc = _stream_error({"type": "error", "error": {"type": "billing_error", "message": "x"}})
    breaker.record_error("anthropic", exc, threshold=5, cooldown_seconds=30)
    assert breaker.tripped("anthropic")
    breaker.reset()


def test_moderation_403_does_not_trip_the_breaker():
    breaker.reset()
    exc = _oai(
        403,
        {
            "error": {
                "code": 403,
                "message": "flagged",
                "metadata": {"flagged_input": "x", "reasons": ["a"]},
            }
        },
    )
    breaker.record_error("openrouter", exc, threshold=1, cooldown_seconds=30)
    assert not breaker.tripped("openrouter")
    breaker.reset()
