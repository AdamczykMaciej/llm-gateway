"""Coverage pass 5 (0.42.1): the non-streaming orchestrators' admission
branches (policy / breaker), tool-definition validation edges, HTTP-object
scrubbing of SDK errors, structured-output limits and the request body cap.
Fakes only; no provider is ever called."""

import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from llm_gateway import (
    GatewayConfig,
    InvalidOutputError,
    InvalidRequestError,
    PolicyViolationError,
    ProviderPolicy,
    chat,
    complete_with_usage,
    reset_circuit_breakers,
)
from llm_gateway.errors import scrub_exception_chain
from llm_gateway.providers.base import ChatResult, ProviderResult
from llm_gateway.structured import OutputSchema, _schema_errors
from llm_gateway.tool_validation import check_tool_calls, validate_chat_request

_USER = {"role": "user", "content": "hi"}


@pytest.fixture(autouse=True)
def _reset():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def _config(**kw):
    base = {
        "anthropic_api_key": "k1",
        "groq_api_key": "k2",
        "provider_order": "anthropic,groq",
        "retry_attempts": 1,
    }
    return GatewayConfig(**{**base, **kw})


def _chat_fake(text):
    async def fake(*a, **kw):
        return ChatResult(content=text, model="m", input_tokens=1, output_tokens=1)

    return fake


def _complete_fake(text):
    async def fake(*a, **kw):
        return ProviderResult(text=text, model="m", input_tokens=1, output_tokens=1)

    return fake


# ── chat(): admission branches ──────────────────────────────────────────


async def test_chat_policy_denial_skips_first_provider_and_never_calls_it():
    calls = []

    async def first(*a, **kw):
        calls.append("a")
        return await _chat_fake("a")()

    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": first, "groq": _chat_fake("g")}):
        result = await chat(
            messages=[_USER], config=_config(), policy=ProviderPolicy(ignore=("anthropic",))
        )
    assert result.content == "g" and calls == []


async def test_chat_policy_excluding_everything_raises_violation_without_a_call():
    calls = []

    async def fake(*a, **kw):
        calls.append(1)
        return await _chat_fake("x")()

    with patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": fake}):
        with pytest.raises(PolicyViolationError):
            await chat(
                messages=[_USER],
                config=_config(provider_order="anthropic"),
                policy=ProviderPolicy(ignore=("anthropic",)),
            )
    assert calls == []


async def test_chat_breaker_admission_lost_after_check_skips_the_provider():
    calls = []

    async def first(*a, **kw):
        calls.append("a")
        return await _chat_fake("a")()

    with (
        patch("llm_gateway.chat.breaker.allow", side_effect=lambda p, m=None: p == "groq"),
        patch("llm_gateway.chat.CHAT_CALLS", {"anthropic": first, "groq": _chat_fake("g")}),
    ):
        result = await chat(messages=[_USER], config=_config())
    assert result.content == "g" and calls == []


async def test_chat_budget_check_denying_first_provider_falls_to_second_and_all_raises():
    async def deny_anthropic(provider, model, cost):
        return provider != "anthropic"

    async def deny_all(provider, model, cost):
        return False

    calls = []

    async def first(*a, **kw):
        calls.append("a")
        return await _chat_fake("a")()

    fakes = {"anthropic": first, "groq": _chat_fake("g")}
    with patch("llm_gateway.chat.CHAT_CALLS", fakes):
        result = await chat(
            messages=[_USER],
            config=_config(),
            policy=ProviderPolicy(budget_check=deny_anthropic),
        )
        assert result.content == "g" and calls == []
        with pytest.raises(PolicyViolationError):
            await chat(
                messages=[_USER], config=_config(), policy=ProviderPolicy(budget_check=deny_all)
            )
    assert calls == []


async def test_complete_budget_check_denying_first_provider_falls_to_second():
    async def deny_anthropic(provider, model, cost):
        return provider != "anthropic"

    with patch(
        "llm_gateway.router.CALLS", {"anthropic": _complete_fake("a"), "groq": _complete_fake("g")}
    ):
        out = await complete_with_usage(
            system="s",
            prompt="hi",
            config=_config(),
            policy=ProviderPolicy(budget_check=deny_anthropic),
        )
    assert out.text == "g"


# ── complete_with_usage(): the same branches on the text path ───────────


async def test_complete_policy_denial_falls_through_to_second_provider():
    with patch(
        "llm_gateway.router.CALLS", {"anthropic": _complete_fake("a"), "groq": _complete_fake("g")}
    ):
        out = await complete_with_usage(
            system="s", prompt="hi", config=_config(), policy=ProviderPolicy(ignore=("anthropic",))
        )
    assert out.text == "g"


async def test_complete_breaker_admission_lost_after_check_skips_the_provider():
    calls = []

    async def first(*a, **kw):
        calls.append("a")
        return await _complete_fake("a")()

    with (
        patch("llm_gateway.router.breaker.allow", side_effect=lambda p, m=None: p == "groq"),
        patch("llm_gateway.router.CALLS", {"anthropic": first, "groq": _complete_fake("g")}),
    ):
        out = await complete_with_usage(system="s", prompt="hi", config=_config())
    assert out.text == "g" and calls == []


# ── tool definition validation ──────────────────────────────────────────


def _tool(name="t", parameters=None):
    fn = {"name": name}
    if parameters is not None:
        fn["parameters"] = parameters
    return {"type": "function", "function": fn}


def test_gateway_reserved_tool_name_is_rejected():
    with pytest.raises(InvalidRequestError, match="reserved"):
        validate_chat_request([_USER], [_tool("__llm_gateway_structured_output")], None)


def test_tool_parameters_that_are_not_json_are_rejected_not_crashed():
    params = {"type": "object", "properties": {"x": {"type": "string", "default": {1, 2}}}}
    with pytest.raises(InvalidRequestError, match="not JSON"):
        validate_chat_request([_USER], [_tool("t", params)], None)


def test_tool_definition_with_non_json_extras_is_rejected_not_crashed():
    tool = _tool("t")
    tool["function"]["description"] = {1, 2}  # a set is not JSON
    with pytest.raises(InvalidRequestError, match="not JSON"):
        validate_chat_request([_USER], [tool], None)


def test_oversized_tool_definitions_are_rejected():
    from llm_gateway.tool_validation import MAX_TOOLS_JSON_CHARS

    tool = _tool("t")
    tool["function"]["description"] = "x" * (MAX_TOOLS_JSON_CHARS + 1)
    with pytest.raises(InvalidRequestError, match="limit"):
        validate_chat_request([_USER], [tool], None)


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(InvalidRequestError, match="duplicate"):
        validate_chat_request([_USER], [_tool("t"), _tool("t")], None)


def test_check_tool_calls_ignores_malformed_tool_entries_but_still_flags_unknown_calls():
    from llm_gateway.providers.base import ToolCall

    tools = ["junk", {"function": "nope"}, {"function": {"name": 5}}, _tool("ok")]
    calls = [
        ToolCall(id="1", name="ok", arguments={}),
        ToolCall(id="2", name="ghost", arguments={}),
    ]
    issues = check_tool_calls(tools, calls)  # type: ignore[arg-type]
    assert [(i.call_id, i.kind) for i in issues] == [("2", "unknown_tool")]


# ── structured output ───────────────────────────────────────────────────


def test_array_schema_rejects_a_non_array_reply():
    so = OutputSchema.from_spec({"type": "array", "items": {"type": "string"}})
    with pytest.raises(InvalidOutputError, match="not a JSON array"):
        so.parse('{"a": 1}')


def test_pydantic_model_recursion_error_is_an_invalid_output_not_a_crash():
    pydantic = pytest.importorskip("pydantic")

    class M(pydantic.BaseModel):
        a: int

    so = OutputSchema.from_spec(M)
    with patch.object(M, "model_validate", side_effect=RecursionError):
        with pytest.raises(InvalidOutputError, match="nested too deeply"):
            so.parse('{"a": 1}')


def test_schema_nested_past_the_validator_depth_is_not_accepted():
    schema: dict = {"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"}  # a cycle
    out: list[str] = []
    _schema_errors("x", schema, schema, "", out)
    assert out, "a schema deeper than the validator looks must not silently pass"


# ── scrubbing SDK errors that carry HTTP objects ────────────────────────

CANARY = "CANARY-plain-credential-0123456789"


class _Req:
    def __init__(self):
        self.method = "POST"
        self.url = f"https://h.example/v1?key={CANARY}"
        self.headers = {"x-api-key": CANARY, "x-echo": f"k={CANARY}"}
        self.stream = object()
        self._content = b"payload"


class _Resp:
    def __init__(self, request):
        self.status_code = 401
        self.headers = {"authorization": "Bearer abc", "x-echo": f"k={CANARY}"}
        self._content = f"bad key {CANARY}".encode()
        self._text = "cached"
        self.request = request


def test_scrub_removes_secrets_from_response_headers_body_and_cached_text():
    cfg = _config(anthropic_api_key=CANARY)
    req = _Req()
    resp = _Resp(req)
    err = RuntimeError("boom")
    err.response = resp  # type: ignore[attr-defined]
    scrub_exception_chain(err, cfg)
    assert CANARY not in resp._content.decode()
    assert resp.headers["authorization"] == "[REDACTED]"
    assert CANARY not in resp.headers["x-echo"]
    assert "_text" not in resp.__dict__  # the cached decoded text is dropped
    assert req.headers["x-api-key"] == "[REDACTED]"
    assert req._content == b""  # a request body is never kept


def test_scrub_survives_objects_whose_headers_or_url_raise():
    class Hostile:
        method = "GET"
        stream = object()

        @property
        def headers(self):
            raise RuntimeError("no")

        url = NS()

    err = RuntimeError("boom")
    err.response = Hostile()  # type: ignore[attr-defined]
    scrub_exception_chain(err, _config())  # must not raise or mask the error
    assert str(err) == "boom"


def test_scrub_is_best_effort_when_url_rebuild_or_body_write_fails():
    class Url:
        def __str__(self):
            return f"https://h.example/?key={CANARY}"

        def __init__(self, *a):
            if a:
                raise TypeError("cannot rebuild")

    class Req:
        method = "GET"
        stream = object()
        headers = {"accept": "application/json"}  # nothing to redact: left alone
        url = Url()

    class Resp:
        status_code = 500
        headers: dict = {}
        request = Req()

        @property
        def _content(self):
            return CANARY.encode()

        @_content.setter
        def _content(self, value):
            raise AttributeError("read-only")

    err = RuntimeError("boom")
    err.response = Resp()  # type: ignore[attr-defined]
    scrub_exception_chain(err, _config(anthropic_api_key=CANARY))
    assert str(err) == "boom"
    assert err.response.request.headers == {"accept": "application/json"}  # type: ignore[attr-defined]


# ── request body cap (service) ──────────────────────────────────────────


async def test_body_limit_garbage_content_length_does_not_reject_and_streamed_overflow_does():
    from llm_gateway.service.body_limit import BodyLimitMiddleware

    seen: dict = {"app_body": 0}

    async def app(scope, receive, send):
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                break
            seen["app_body"] += len(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = BodyLimitMiddleware(app, max_bytes=10)

    async def run(headers, chunks):
        sent = []
        queue = [
            {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
            for i, c in enumerate(chunks)
        ]

        async def receive():
            return queue.pop(0) if queue else {"type": "http.disconnect"}

        async def send(m):
            sent.append(m)

        await mw({"type": "http", "headers": headers}, receive, send)
        return sent

    ok = await run([(b"content-length", b"abc")], [b"12345"])
    assert ok[0]["status"] == 200
    over = await run([], [b"123456", b"123456"])
    assert over[0]["status"] == 413
    assert json.loads(over[1]["body"])  # a JSON error body, not the app's reply


async def test_body_limit_answers_413_once_even_if_the_app_keeps_reading():
    from llm_gateway.service.body_limit import BodyLimitMiddleware

    async def app(scope, receive, send):
        first = await receive()
        second = await receive()  # after the overflow: told the client is gone
        assert first["type"] == second["type"] == "http.disconnect"

    sent = []
    queue = [{"type": "http.request", "body": b"x" * 50, "more_body": True}]

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(m):
        sent.append(m)

    await BodyLimitMiddleware(app, max_bytes=10)({"type": "http", "headers": []}, receive, send)
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [413]


async def test_complete_skips_a_provider_with_no_registered_call():
    with patch("llm_gateway.router.CALLS", {"groq": _complete_fake("g")}):
        out = await complete_with_usage(system="s", prompt="hi", config=_config())
    assert out.text == "g"


def test_schema_errors_ignores_non_object_subschemas_and_caps_the_error_list():
    out: list[str] = []
    _schema_errors("x", "not a schema", {}, "", out)
    assert out == []
    many = {"type": "array", "items": {"type": "string"}}
    _schema_errors(list(range(500)), many, many, "", out)
    assert 0 < len(out) < 500  # bounded, so a hostile reply cannot blow up the message
