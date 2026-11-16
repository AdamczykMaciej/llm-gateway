"""AI-output provenance: record rendering, builders and the Marker contract."""

import hashlib
from datetime import UTC, datetime, timedelta, timezone

import pytest

from llm_gateway import (
    TRAINED_ALGORITHMIC_MEDIA,
    AppendMarker,
    Completion,
    Marker,
    NoopMarker,
    Provenance,
    Usage,
    provenance_of,
)
from llm_gateway.providers.base import ChatResult, StreamDelta

WHEN = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _prov(**kw):
    return Provenance(provider="p", model="m", created_at=WHEN, **kw)


_DETAIL = dict(request_id="req-secret", surface="surf-x", prompt_version="pv-9")


def _flat(obj) -> str:
    return repr(obj)


def test_default_output_discloses_only_the_marking_essentials():
    prov = Provenance(provider="prov-x", model="model-y", created_at=WHEN, **_DETAIL)
    for out in (
        prov.to_dict(),
        prov.c2pa_assertion(),
        prov.http_headers(),
        prov.pdf_metadata(),
    ):
        for secret in ("prov-x", "model-y", "req-secret", "surf-x", "pv-9"):
            assert secret not in _flat(out), (secret, out)
    action = prov.c2pa_assertion()["data"]["actions"][0]
    assert action == {
        "action": "c2pa.created",
        "when": "2026-01-02T03:04:05Z",
        "digitalSourceType": TRAINED_ALGORITHMIC_MEDIA,
    }
    assert prov.to_dict() == {"ai_generated": True, "created_at": "2026-01-02T03:04:05Z"}
    assert prov.http_headers() == {
        "X-AI-Generated": "true",
        "X-AI-Created": "2026-01-02T03:04:05Z",
    }


def test_app_is_the_only_optional_default_field():
    prov = _prov(app="my-app", **_DETAIL)
    assert prov.to_dict()["app"] == "my-app"
    assert prov.http_headers()["X-AI-App"] == "my-app"
    assert prov.pdf_metadata()["Producer"] == "my-app"
    assert prov.c2pa_assertion()["data"]["actions"][0]["parameters"] == {"app": "my-app"}
    assert "Producer" not in _prov().pdf_metadata()


def test_include_opts_in_to_detail():
    prov = _prov(app="a", **_DETAIL)
    part = prov.to_dict(include=("provider", "surface"))
    assert part["provider"] == "p" and part["surface"] == "surf-x" and "model" not in part
    full = prov.to_dict(include="all")
    assert {"provider", "model", "request_id", "surface", "prompt_version"} <= set(full)
    action = prov.c2pa_assertion(include="all")["data"]["actions"][0]
    assert action["softwareAgent"] == "p/m"
    assert action["parameters"]["prompt_version"] == "pv-9"
    only_one = prov.c2pa_assertion(include=("model",))["data"]["actions"][0]
    assert "softwareAgent" not in only_one and only_one["parameters"]["model"] == "m"
    headers = prov.http_headers(include="all")
    assert headers["X-AI-Provider"] == "p" and headers["X-AI-Prompt-Version"] == "pv-9"
    meta = prov.pdf_metadata(include="all")
    assert meta["Creator"] == "AI-generated (p/m)" and meta["AIPromptVersion"] == "pv-9"
    assert _prov().to_dict(include=("surface",)) == _prov().to_dict()  # unset stays absent
    with pytest.raises(ValueError):
        prov.to_dict(include=("secret_key",))


def test_naive_and_foreign_timezones_are_normalised_to_utc():
    naive = Provenance(provider="p", model="m", created_at=datetime(2026, 1, 2, 3, 4, 5))
    assert naive.created_at == WHEN
    plus2 = Provenance(
        provider="p", model="m", created_at=WHEN.astimezone(timezone(timedelta(hours=2)))
    )
    assert plus2.to_dict()["created_at"] == "2026-01-02T03:04:05Z"


def test_http_headers_cannot_split_a_header():
    headers = _prov(request_id="a\r\nSet-Cookie: x=1", app="  ").http_headers(include="all")
    assert "\r" not in headers["X-AI-Request-Id"] and "\n" not in headers["X-AI-Request-Id"]
    assert "X-AI-App" not in headers
    assert Provenance(provider="\n", model="m").http_headers(include="all")["X-AI-Provider"] == "?"
    blank = Provenance(provider="  ", model="m").http_headers(include="all")
    assert "X-AI-Provider" not in blank


def test_completion_provenance_and_equality_unaffected():
    kw = dict(text="t", provider="prov", model="mod", usage=Usage(1, 1))
    c = Completion(**kw, request_id="r", created_at=WHEN)
    assert c == Completion(**kw)
    p = c.provenance(app="x", surface="y", prompt_version="z")
    assert (p.provider, p.model, p.request_id, p.created_at) == ("prov", "mod", "r", WHEN)
    assert p.app == "x"
    # No timestamp on the result: falls back to now.
    assert Completion(**kw).provenance().created_at.tzinfo is not None


def test_provenance_of_chat_result_and_stream_chunk():
    chat = ChatResult(content="x", model="m", input_tokens=1, output_tokens=1, request_id="r")
    assert provenance_of(chat, provider="p").provider == "p"
    with pytest.raises(ValueError):
        provenance_of(chat)
    delta = StreamDelta(provider="p", model="m", request_id="r2")
    assert provenance_of(delta).request_id == "r2"
    with pytest.raises(ValueError):
        provenance_of(StreamDelta(provider="p"))


class _Signer:
    def sign(self, payload: bytes) -> bytes:
        return hashlib.sha256(payload).digest()


def _verifier(payload: bytes, sig: bytes) -> bool:
    return sig == hashlib.sha256(payload).digest()


def test_noop_marker_is_a_marker_and_changes_nothing():
    m = NoopMarker()
    assert isinstance(m, Marker)
    assert m.mark(b"abc", "audio/mpeg", _prov()) == b"abc"
    assert m.verify(b"abc", "audio/mpeg") is None


def _trailer(body: bytes, sig: bytes = b"") -> bytes:
    return b"x\nAIPROV1:" + body.hex().encode() + b":" + sig.hex().encode()


def test_append_marker_round_trip_and_tamper():
    m = AppendMarker(_Signer(), _verifier)
    assert isinstance(m, Marker)
    marked = m.mark(b"\x00audio", "audio/wav", _prov(app="a"))
    assert marked.startswith(b"\x00audio")
    record = m.verify(marked, "audio/wav")
    assert record is not None and record["app"] == "a" and record["ai_generated"] is True
    assert "provider" not in record and "model" not in record
    full = AppendMarker(_Signer(), _verifier, include="all")
    rec = full.verify(full.mark(b"a", "audio/wav", _prov()), "audio/wav")
    assert rec is not None and rec["provider"] == "p"
    assert m.verify(b"plain", "audio/wav") is None
    assert m.verify(marked.replace(b"\x00audio", b"\x00AUDIO", 1), "audio/wav") is None
    assert AppendMarker(_Signer()).verify(marked, "audio/wav") is None  # no verifier: untrusted
    assert m.verify(b"x\nAIPROV1:zz:zz", "audio/wav") is None
    # Validly signed but not a JSON object, or not JSON at all.
    for body in (b"[1]", b"nope"):
        signed = _trailer(body, hashlib.sha256(b"x" + body).digest())
        assert m.verify(signed, "audio/wav") is None
