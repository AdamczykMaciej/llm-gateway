"""Property-based robustness tests (Hypothesis).

Bounded and fast: examples are capped, there is no per-example deadline, and
under CI (`CI` set) the seed is fixed so a failure is reproducible. No network,
no provider SDK calls: chunks are plain namespaces and error bodies are data.
"""

import ipaddress
import json
import os
import string
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from openai.types.chat import ChatCompletionChunk

from llm_gateway.config import _check_http_url
from llm_gateway.errors import (
    ErrorKind,
    InvalidOutputError,
    classify,
    is_billing_error,
    is_model_scoped,
    is_retryable,
    redact_secrets,
)
from llm_gateway.pricing import DEFAULT_PRICES, LongContextTier, ModelPrice, cost_usd
from llm_gateway.providers.base import (
    StreamDelta,
    ToolIndexNormalizer,
    Usage,
    _ThinkStripper,  # pyright: ignore[reportPrivateUsage]
    _tool_arguments,  # pyright: ignore[reportPrivateUsage]
    parse_openai_style_chunk,
    strip_think,
)

PROFILE = settings(
    max_examples=int(os.environ.get("PROPERTY_EXAMPLES", "60")),
    deadline=None,
    derandomize=bool(os.environ.get("CI")),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,
)

_REQUEST = httpx2.Request("POST", "https://api.example.test")

# --- strategies ----------------------------------------------------------

_text = st.text(max_size=40)
_json_leaf = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | _text
_json = st.recursive(
    _json_leaf,
    lambda c: st.lists(c, max_size=4) | st.dictionaries(_text, c, max_size=4),
    max_leaves=12,
)
_json_object = st.dictionaries(_text, _json, max_size=4)


@st.composite
def _splits(draw: st.DrawFn, text: str) -> list[str]:
    """`text` cut into consecutive pieces (some possibly empty)."""
    cuts = sorted(draw(st.lists(st.integers(0, len(text)), max_size=8)))
    bounds = [0, *cuts, len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]


# --- streaming chunk parsing ---------------------------------------------


def _chunk(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish: str | None = None,
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


def _fragment(index: int | None, call_id: str | None, name: str | None, args: str) -> Any:
    function = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(index=index, id=call_id, function=function)


def _aggregate(deltas: list[StreamDelta]) -> tuple[str, dict[int, dict[str, str]]]:
    """What a client aggregating by `index` reassembles."""
    text = ""
    calls: dict[int, dict[str, str]] = {}
    for delta in deltas:
        text += delta.content or ""
        for frag in delta.tool_call_deltas or ():
            entry = calls.setdefault(frag["index"], {"id": "", "name": "", "arguments": ""})
            entry["id"] += frag.get("id", "")
            function = frag.get("function", {})
            entry["name"] += function.get("name", "")
            entry["arguments"] += function.get("arguments", "")
    return text, calls


@st.composite
def _stream_plan(draw: st.DrawFn):
    text = draw(st.text(alphabet=string.ascii_letters + " <>\n", max_size=60))
    n_calls = draw(st.integers(0, 3))
    calls = [
        (
            f"call_{i}",
            draw(st.sampled_from(["get_weather", "search", "f"])),
            json.dumps(draw(_json_object)),
        )
        for i in range(n_calls)
    ]
    shape = draw(st.sampled_from(["openai", "same_index", "no_index"]))
    return text, calls, shape


@given(plan=_stream_plan(), data=st.data())
@PROFILE
def test_random_splits_reassemble_to_the_same_text_and_tool_calls(plan: Any, data: st.DataObject):
    text, calls, shape = plan
    chunks: list[Any] = [_chunk(content=piece) for piece in data.draw(_splits(text))]
    if shape == "openai":
        # Fragments keyed by index, interleaved across parallel calls.
        pending = []
        for i, (call_id, name, args) in enumerate(calls):
            pieces = data.draw(_splits(args)) or [""]
            pending.append(
                [_fragment(i, call_id, name, pieces[0])]
                + [_fragment(i, None, None, p) for p in pieces[1:]]
            )
        while any(pending):
            live = [q for q in pending if q]
            queue = data.draw(st.sampled_from(live))
            chunks.append(_chunk(tool_calls=[queue.pop(0)]))
    else:
        # Gemini-style: each call arrives whole, all at index 0 (or no index).
        for call_id, name, args in calls:
            chunks.append(
                _chunk(
                    tool_calls=[
                        _fragment(0 if shape == "same_index" else None, call_id, name, args)
                    ]
                )
            )
    chunks.append(_chunk(finish="tool_calls" if calls else "stop"))

    normalizer = ToolIndexNormalizer()
    deltas = [normalizer.apply(parse_openai_style_chunk(c, "m")) for c in chunks]
    got_text, got_calls = _aggregate(deltas)

    assert got_text == text
    assert [got_calls[i] for i in sorted(got_calls)] == [
        {"id": cid, "name": name, "arguments": args} for cid, name, args in calls
    ]


_chunk_json = st.fixed_dictionaries(
    {},
    optional={
        "id": _json,
        "object": _json,
        "created": _json,
        "model": _json,
        "usage": _json,
        "choices": st.lists(
            st.dictionaries(
                st.sampled_from(["index", "delta", "finish_reason", "logprobs"]),
                st.one_of(
                    _json,
                    st.dictionaries(
                        st.sampled_from(["content", "role", "tool_calls"]),
                        st.one_of(
                            _json,
                            st.lists(
                                st.dictionaries(
                                    st.sampled_from(["index", "id", "type", "function"]), _json
                                ),
                                max_size=3,
                            ),
                        ),
                    ),
                ),
                max_size=4,
            ),
            max_size=2,
        ),
    },
)


@given(body=_chunk_json)
@PROFILE
def test_garbage_chunks_fail_typed_or_parse(body: dict[str, Any]):
    try:
        chunk = ChatCompletionChunk.model_validate(body)
    except ValueError:  # pydantic.ValidationError: the SDK's typed rejection
        return
    delta = parse_openai_style_chunk(chunk, "m")
    ToolIndexNormalizer().apply(delta)


@given(args=_json_object, cut=st.floats(0, 1))
@PROFILE
def test_tool_arguments_roundtrip_and_truncation(args: dict[str, Any], cut: float):
    raw = json.dumps(args)
    assert _tool_arguments(raw) == json.loads(raw)
    if len(raw) > 1:
        # A proper prefix of a JSON object is never a complete object.
        with pytest.raises(InvalidOutputError):
            _tool_arguments(raw[: min(int(cut * len(raw)), len(raw) - 1)] or "{")


@given(raw=st.text(max_size=80) | st.binary(max_size=40).map(lambda b: b.decode("latin-1")))
@PROFILE
def test_tool_arguments_garbage_is_typed(raw: str):
    try:
        out = _tool_arguments(raw)
    except InvalidOutputError:
        return
    assert isinstance(out, dict)


def test_tool_arguments_pathological_json_is_typed():
    for raw in ("[" * 100_000, "1" * 10_000, '{"a":' * 50_000, "[1,", "null", "[]", '"x"', "nan"):
        try:
            _tool_arguments(raw)
        except InvalidOutputError:
            pass


@given(text=st.text(alphabet="<think>/ \nab", max_size=60), data=st.data())
@PROFILE
def test_think_stripper_is_split_invariant(text: str, data: st.DataObject):
    stripper = _ThinkStripper()
    streamed = "".join(stripper.feed(p) for p in data.draw(_splits(text))) + stripper.flush()
    assert streamed == (strip_think(text) or "")


# --- ToolIndexNormalizer --------------------------------------------------

_frag = st.fixed_dictionaries(
    {},
    optional={
        "index": st.none() | st.integers(-3, 6),
        "id": st.sampled_from(["", "a", "b", "c", "d"]),
        "function": st.just({"arguments": "x"}),
    },
)


@given(frames=st.lists(st.lists(_frag, max_size=3), max_size=12))
@PROFILE
def test_normalizer_arbitrary_fragments_never_crash_and_never_merge_ids(frames: list[Any]):
    normalizer = ToolIndexNormalizer()
    owner: dict[int, str] = {}
    for frame in frames:
        out = normalizer.apply(StreamDelta(tool_call_deltas=frame))
        for src, dst in zip(frame, out.tool_call_deltas or [], strict=True):
            assert isinstance(dst["index"], int)
            if src.get("id"):
                # A call id never shares an index with a different id.
                assert owner.setdefault(dst["index"], src["id"]) == src["id"]


@given(n=st.integers(1, 5), data=st.data())
@PROFILE
def test_normalizer_passes_correctly_numbered_streams_through(n: int, data: st.DataObject):
    frames = [
        [{"index": i, "id": f"c{i}", "function": {"name": "f", "arguments": ""}}] for i in range(n)
    ] + [[{"index": data.draw(st.integers(0, n - 1)), "function": {"arguments": "x"}}]]
    normalizer = ToolIndexNormalizer()
    for frame in frames:
        out = normalizer.apply(StreamDelta(tool_call_deltas=frame))
        assert out.tool_call_deltas == frame


# --- error classification -------------------------------------------------


def _sdk_errors(status: int | None, body: object, message: str) -> list[Exception]:
    out: list[Exception] = []
    if status is not None:
        response = httpx2.Response(status, request=_REQUEST)
        out.append(openai.APIStatusError(message, response=response, body=body))
        out.append(openai.RateLimitError(message, response=response, body=body))
    out.append(openai.APIError(message, _REQUEST, body=body))
    return out


@given(
    status=st.none() | st.integers(100, 599),
    body=_json,
    message=st.text(max_size=60),
)
@PROFILE
def test_any_provider_error_shape_classifies_to_a_valid_kind(
    status: int | None, body: object, message: str
):
    for exc in _sdk_errors(status, body, message):
        assert isinstance(classify(exc), ErrorKind)
        assert isinstance(is_retryable(exc), bool)
        assert isinstance(is_billing_error(exc), bool)
        assert is_model_scoped(exc) in (None, "missing", "overloaded")


@given(
    exc=st.sampled_from([ValueError("x"), KeyError("k"), RuntimeError(), OSError(), Exception()])
)
def test_foreign_exceptions_classify_to_a_valid_kind(exc: Exception):
    assert isinstance(classify(exc), ErrorKind)


# --- redact_secrets -------------------------------------------------------

_tail = st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=8, max_size=40)
_key = st.builds(
    lambda p, t: f"{p}{t}", st.sampled_from(["sk-", "gsk_", "hf_", "xai-", "pk_", "rk-"]), _tail
)
_noise = st.text(alphabet=string.ascii_letters + " :=,\n\"'{}", max_size=30)
_KEY_SHAPE = ("sk-", "gsk_", "hf_", "xai-", "pk_", "rk-")


@given(key=_key, before=_noise, after=_noise)
@PROFILE
def test_redact_removes_key_shaped_substrings(key: str, before: str, after: str):
    out = redact_secrets(f"{before} {key} {after}")
    assert key not in out
    if key[3:] not in before + after:
        assert key[3:] not in out


@given(token=_tail, scheme=st.sampled_from(["Bearer", "bearer", "BASIC", "Basic"]))
@PROFILE
def test_redact_removes_bearer_tokens_and_key_pairs(token: str, scheme: str):
    assert token not in redact_secrets(f"{scheme} {token}")
    for name in ("api_key", "x-api-key", "password", "Authorization", "token"):
        assert token not in redact_secrets(f'{name}: "{token}"')
        assert token not in redact_secrets(f"{name}={token}&x=1")


@given(text=st.text(max_size=200) | _noise.flatmap(lambda n: _key.map(lambda k: f"{n} {k}")))
@PROFILE
def test_redact_is_idempotent(text: str):
    once = redact_secrets(text)
    assert redact_secrets(once) == once


@given(text=st.text(max_size=400), limit=st.integers(0, 120))
@PROFILE
def test_redact_is_length_capped(text: str, limit: int):
    out = redact_secrets(text, max_chars=limit)
    assert out.endswith("... [truncated]") == (len(text) > limit)
    # Scrubbing can swap a short secret for "[REDACTED]", never blow the text up.
    assert len(out) <= 3 * min(len(text), limit) + len("... [truncated]")


# --- base URL guard -------------------------------------------------------


def _ipv4_forms(ip: ipaddress.IPv4Address) -> list[str]:
    n = int(ip)
    a, b, c, d = ip.packed
    return [
        str(ip),
        str(n),
        hex(n),
        f"0x{a:x}.0x{b:x}.0x{c:x}.0x{d:x}",
        f"0{a:o}.0{b:o}.0{c:o}.0{d:o}",
        f"{a}.{b}.{(c << 8) | d}",
        f"{a}.{(b << 16) | (c << 8) | d}",
        f"[::ffff:{ip}]",
        f"[::ffff:{n >> 16:x}:{n & 0xFFFF:x}]",
        f"{ip}.",
    ]


_link_local = st.integers(0xA9FE0000, 0xA9FEFFFF).map(ipaddress.IPv4Address)


@given(ip=_link_local, data=st.data(), scheme=st.sampled_from(["http", "https"]))
@PROFILE
def test_link_local_never_passes_in_any_encoding(ip: Any, data: st.DataObject, scheme: str):
    host = (
        data.draw(st.sampled_from(_ipv4_forms(ip))).upper()
        if data.draw(st.booleans())
        else (data.draw(st.sampled_from(_ipv4_forms(ip))))
    )
    with pytest.raises(ValueError):
        _check_http_url(f"{scheme}://{host}:80/v1", "base_url")


@given(host=st.sampled_from(["fe80::1", "fe80::abcd%eth0", "[fd00:ec2::254]", "100.100.100.200"]))
def test_other_metadata_addresses_blocked(host: str):
    host = host if host[0] == "[" or "." in host else f"[{host}]"
    with pytest.raises(ValueError):
        _check_http_url(f"http://{host}/", "base_url")


@given(url=st.text(max_size=60).map(lambda t: "http://" + t))
@PROFILE
def test_arbitrary_urls_only_raise_value_error(url: str):
    try:
        _check_http_url(url, "base_url")
    except ValueError:
        pass


_FULLWIDTH = {ord(c): chr(ord(c) + 0xFEE0) for c in string.digits}


@given(ip=_link_local, dot=st.sampled_from([".", "。", "．", "｡"]))
@PROFILE
def test_unicode_lookalike_link_local_hosts_blocked(ip: Any, dot: str):
    host = str(ip).translate(_FULLWIDTH).replace(".", dot)
    with pytest.raises(ValueError):
        _check_http_url(f"http://{host}/", "base_url")


# --- Usage and pricing ----------------------------------------------------

_n = st.integers(0, 10**9)
_usage = st.builds(Usage, _n, _n, _n, _n, _n)


@given(a=_usage, b=_usage, c=_usage)
@PROFILE
def test_usage_addition_associative_commutative_with_identity(a: Usage, b: Usage, c: Usage):
    assert (a + b) + c == a + (b + c)
    assert a + b == b + a
    assert a + Usage() == a
    assert Usage.total([a, b, c]) == a + b + c
    assert Usage.total([]) == Usage()


_price = st.builds(
    ModelPrice,
    input_per_mtok=st.floats(0, 100),
    output_per_mtok=st.floats(0, 500),
    cached_input_per_mtok=st.none() | st.floats(0, 100),
    cache_write_per_mtok=st.none() | st.floats(0, 100),
    batch_multiplier=st.none() | st.floats(0.01, 2),
    long_context=st.none()
    | st.builds(
        LongContextTier,
        threshold_input_tokens=st.integers(1, 1_000_000),
        multiplier=st.floats(1, 10),
    ),
)


@given(
    price=_price,
    base=st.integers(0, 2_000_000),
    step=st.integers(0, 2_000_000),
    batch=st.booleans(),
)
@PROFILE
def test_cost_is_nonnegative_and_monotonic_across_tier_boundaries(
    price: ModelPrice, base: int, step: int, batch: bool
):
    lo = Usage(input_tokens=base, output_tokens=1000, cache_read_input_tokens=base // 3)
    hi = Usage(input_tokens=base + step, output_tokens=1000, cache_read_input_tokens=base // 3)
    cost_lo, cost_hi = cost_usd(price, lo, batch=batch), cost_usd(price, hi, batch=batch)
    assert cost_lo is not None and cost_hi is not None
    assert 0 <= cost_lo <= cost_hi + 1e-9
    if price.long_context is not None:
        t = price.long_context.threshold_input_tokens
        at = cost_usd(price, Usage(input_tokens=t), batch=batch)
        over = cost_usd(price, Usage(input_tokens=t + 1), batch=batch)
        assert at is not None and over is not None and at <= over + 1e-9


@given(key=st.sampled_from(sorted(DEFAULT_PRICES)), n=st.integers(0, 1_000_000))
@PROFILE
def test_builtin_prices_monotonic_in_input_tokens(key: str, n: int):
    price = DEFAULT_PRICES[key]
    a = cost_usd(price, Usage(input_tokens=n))
    b = cost_usd(price, Usage(input_tokens=n + 1))
    assert a is not None and b is not None and a <= b + 1e-9


def test_ideographic_dot_metadata_host_regression():
    with pytest.raises(ValueError):
        _check_http_url("http://169。254。169。254/latest", "base_url")
