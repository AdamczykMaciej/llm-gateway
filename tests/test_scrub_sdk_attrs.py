"""SDK exceptions carry `body`, `response`, `request` (headers, URL) besides
their message. Error trackers serialise those, so the gateway scrubs them on
the chained error too, without changing the exception type. Fakes only."""

import traceback

import anthropic
import httpx
import httpx2
import openai
import pytest

from llm_gateway import GatewayConfig
from llm_gateway.errors import (
    AllProvidersExhaustedError,
    all_failed,
    classify,
    deadline_exceeded,
    retry_after_seconds,
    scrub_exception_chain,
)

CANARY = "CANARY-plain-credential-0123456789"  # no recognisable key shape
SHAPED = "sk-ant-api03-CANARYSHAPED0123456789abcdef"
ALL = (CANARY, SHAPED)


def _cfg():
    return GatewayConfig(anthropic_api_key=CANARY, provider_order="anthropic", retry_attempts=1)


def _parts(mod):
    request = mod.Request(
        "POST",
        f"https://api.example.com/v1/models/m:generate?key={CANARY}&alt=json",
        headers={"Authorization": f"Bearer {SHAPED}", "x-api-key": CANARY, "api-key": CANARY},
        content=b'{"prompt": "hi"}',
    )
    body = {
        "error": {
            "message": f"bad key {CANARY}",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
            "param": CANARY,
        }
    }
    import json

    response = mod.Response(
        401,
        headers={
            "retry-after": "7",
            "x-echo-key": f"key={CANARY}",
            "content-type": "application/json",
        },
        content=json.dumps(body).encode(),
        request=request,
    )
    return request, response, body


def _openai():
    request, response, body = _parts(httpx)
    exc = openai.AuthenticationError(f"Error code: 401 - {body}", response=response, body=body)
    return exc


def _anthropic():
    request, response, body = _parts(httpx)
    return anthropic.AuthenticationError(f"401 {body}", response=response, body=body)


def _httpx(mod):
    def make():
        request, response, _ = _parts(mod)
        return mod.HTTPStatusError("401 Unauthorized", request=request, response=response)

    return make


FACTORIES = {
    "openai": _openai,
    "anthropic": _anthropic,
    "httpx": _httpx(httpx),
    "httpx2": _httpx(httpx2),
}


def _everything(exc: BaseException) -> str:
    """repr, traceback, every attribute (nested), headers, URL and body text."""
    seen: set[int] = set()
    out: list[str] = []

    def walk(value, depth=0):
        if depth > 6 or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, str | bytes):
            out.append(value if isinstance(value, str) else value.decode("latin-1"))
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(k, depth + 1)
                walk(v, depth + 1)
        elif isinstance(value, list | tuple):
            for v in value:
                walk(v, depth + 1)
        elif isinstance(
            value, BaseException | httpx.Response | httpx.Request | httpx2.Response | httpx2.Request
        ):
            out.append(repr(value))
            for v in vars(value).values():
                walk(v, depth + 1)
            for name in ("headers", "url"):
                h = getattr(value, name, None)
                if h is not None:
                    out.append(str(dict(h)) if hasattr(h, "items") else str(h))
            if isinstance(value, httpx.Response | httpx2.Response):
                out.append(value.text)
                walk(value.request, depth + 1)

    walk(exc)
    out.append(repr(exc))
    out.append("".join(traceback.format_exception(exc)))
    return "\n".join(out)


@pytest.mark.parametrize("name", FACTORIES)
def test_canaries_present_before_scrub(name):
    text = _everything(FACTORIES[name]())
    assert CANARY in text  # the test would be vacuous otherwise


@pytest.mark.parametrize("name", FACTORIES)
def test_scrub_removes_canaries_from_every_attribute(name):
    exc = FACTORIES[name]()
    original_type = type(exc)
    scrub_exception_chain(exc, _cfg())
    text = _everything(exc)
    for canary in ALL:
        assert canary not in text, (name, canary)
    assert type(exc) is original_type


@pytest.mark.parametrize("name", ["openai", "anthropic"])
def test_status_code_body_shape_and_retry_after_survive(name):
    exc = FACTORIES[name]()
    scrub_exception_chain(exc, _cfg())
    assert exc.status_code == 401
    assert exc.response.status_code == 401
    assert exc.response.headers["content-type"] == "application/json"
    assert exc.request.method == "POST"
    assert exc.request.url.host == "api.example.com"
    assert exc.body["error"]["type"] == "invalid_request_error"
    assert exc.body["error"]["code"] == "invalid_api_key"
    assert retry_after_seconds(exc) == 7.0
    assert classify(exc).value == "auth"


def test_httpx_status_error_keeps_status():
    exc = FACTORIES["httpx"]()
    scrub_exception_chain(exc, _cfg())
    assert exc.response.status_code == 401
    assert exc.response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("name", FACTORIES)
def test_gateway_errors_and_chain_never_contain_canaries(name):
    exc = FACTORIES[name]()
    failed = all_failed(exc, _cfg(), failures=())
    failed.__cause__ = exc
    late = deadline_exceeded(5, exc, _cfg())
    late.__cause__ = exc
    for gateway_error in (failed, late):
        assert isinstance(gateway_error, (AllProvidersExhaustedError, Exception))
        text = _everything(gateway_error) + _everything(exc)
        for canary in ALL:
            assert canary not in text


def test_scrub_is_idempotent_and_survives_odd_attributes():
    exc = _openai()
    exc.extra = object()
    exc.nested = {"a": [("x", CANARY)], "authorization": CANARY, 3: None}
    exc.headers_like = type("H", (), {"headers": 5})()
    scrub_exception_chain(exc, _cfg())
    scrub_exception_chain(exc, _cfg())
    assert CANARY not in str(exc.nested)
    assert exc.nested["a"][0][0] == "x"


def test_response_without_content_is_handled():
    response = httpx.Response(500, headers={"x-api-key": CANARY})
    err = httpx.HTTPStatusError("boom", request=httpx.Request("GET", "http://a"), response=response)
    scrub_exception_chain(err, _cfg())
    assert CANARY not in str(dict(response.headers))
