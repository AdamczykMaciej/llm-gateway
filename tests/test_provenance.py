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


def test_assertion_and_dict_omit_unset_fields():
    prov = _prov(request_id="r1", app="a", surface="s", prompt_version="v")
    action = prov.c2pa_assertion()["data"]["actions"][0]
    assert action["digitalSourceType"] == TRAINED_ALGORITHMIC_MEDIA
    assert action["when"] == "2026-01-02T03:04:05Z"
    assert action["parameters"]["surface"] == "s"
    assert _prov().to_dict() == {
        "ai_generated": True,
        "provider": "p",
        "model": "m",
        "created_at": "2026-01-02T03:04:05Z",
    }
    assert "app" not in _prov().c2pa_assertion()["data"]["actions"][0]["parameters"]


def test_naive_and_foreign_timezones_are_normalised_to_utc():
    naive = Provenance(provider="p", model="m", created_at=datetime(2026, 1, 2, 3, 4, 5))
    assert naive.created_at == WHEN
    plus2 = Provenance(
        provider="p", model="m", created_at=WHEN.astimezone(timezone(timedelta(hours=2)))
    )
    assert plus2.to_dict()["created_at"] == "2026-01-02T03:04:05Z"


def test_http_headers_cannot_split_a_header():
    headers = _prov(request_id="a\r\nSet-Cookie: x=1", app="  ").http_headers()
    assert headers["X-AI-Generated"] == "true"
    assert "\r" not in headers["X-AI-Request-Id"] and "\n" not in headers["X-AI-Request-Id"]
    assert "X-AI-App" not in headers
    assert Provenance(provider="\n", model="m").http_headers()["X-AI-Provider"] == "?"
    assert Provenance(provider="  ", model="m").http_headers()["X-AI-Provider"] == "unknown"


def test_pdf_metadata_producer_is_the_app():
    assert "Producer" not in _prov().pdf_metadata()
    meta = _prov(app="my-app").pdf_metadata()
    assert meta["Producer"] == "my-app" and meta["DigitalSourceType"] == TRAINED_ALGORITHMIC_MEDIA


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
    assert m.verify(b"plain", "audio/wav") is None
    assert m.verify(marked.replace(b"\x00audio", b"\x00AUDIO", 1), "audio/wav") is None
    assert AppendMarker(_Signer()).verify(marked, "audio/wav") is None  # no verifier: untrusted
    assert m.verify(b"x\nAIPROV1:zz:zz", "audio/wav") is None
    # Validly signed but not a JSON object, or not JSON at all.
    for body in (b"[1]", b"nope"):
        signed = _trailer(body, hashlib.sha256(b"x" + body).digest())
        assert m.verify(signed, "audio/wav") is None
