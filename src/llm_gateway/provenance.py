"""AI-output provenance: machine-readable "this was AI-generated" marking.

Supports transparency duties such as EU AI Act Art. 50(2) (outputs marked in a
machine-readable format). It is app-agnostic and opt-in: the library holds no
keys and makes no network calls. Text results carry a `Provenance` record that
renders as a C2PA-style assertion, HTTP headers or PDF/XMP metadata. Output is
minimal by default (detail such as provider and model is opt-in). Audio, images
and video are marked through a `Marker` that the application supplies, backed by
a `Signer` it owns (keys, signing service). Not legal advice.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

# IPTC digital source type for media created by a trained algorithmic model.
TRAINED_ALGORITHMIC_MEDIA = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
_NOT_HEADER_SAFE = re.compile(r"[^\x20-\x7e]")
# Detail that is NOT exposed unless the caller opts in (minimal disclosure).
DETAIL_FIELDS = ("provider", "model", "request_id", "surface", "prompt_version")
_HEADER_NAMES = {
    "provider": "X-AI-Provider",
    "model": "X-AI-Model",
    "request_id": "X-AI-Request-Id",
    "surface": "X-AI-Surface",
    "prompt_version": "X-AI-Prompt-Version",
}
_PDF_NAMES = {k: "AI" + "".join(w.title() for w in k.split("_")) for k in DETAIL_FIELDS}


def _clean(value: str | None) -> str | None:
    """Single-line printable ASCII, so a value can never split an HTTP header."""
    if value is None:
        return None
    return _NOT_HEADER_SAFE.sub("?", value).strip()[:200] or None


@dataclass(frozen=True)
class Provenance:
    """Who generated an output, and when. `ai_generated` is always True here.

    `app`, `surface` and `prompt_version` are caller-supplied free text (the
    library gives them no meaning). Holds no secrets and no content."""

    provider: str
    model: str
    request_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    app: str | None = None
    surface: str | None = None
    prompt_version: str | None = None
    ai_generated: bool = True

    def __post_init__(self) -> None:
        tz = self.created_at.tzinfo
        when = self.created_at.replace(tzinfo=UTC) if tz is None else self.created_at
        object.__setattr__(self, "created_at", when.astimezone(UTC))

    def _when(self) -> str:
        return self.created_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _detail(self, include: Iterable[str]) -> dict[str, str]:
        """The opt-in detail fields named in `include` (`"all"` for every one), when set."""
        names = {include} if isinstance(include, str) else set(include)
        unknown = names - set(DETAIL_FIELDS) - {"all"}
        if unknown:
            raise ValueError(f"unknown provenance detail field(s): {sorted(unknown)}")
        wanted = DETAIL_FIELDS if "all" in names else tuple(f for f in DETAIL_FIELDS if f in names)
        return {k: v for k in wanted if (v := getattr(self, k)) is not None}

    def to_dict(self, include: Iterable[str] = ()) -> dict[str, Any]:
        """Plain JSON-safe dict. By default only the marking essentials (the
        AI-generated flag, creation time and `app` when set); provider, model,
        request id, surface and prompt version only when named in `include`
        (or `"all"`) for internal or audit use."""
        out: dict[str, Any] = {"ai_generated": self.ai_generated, "created_at": self._when()}
        if self.app is not None:
            out["app"] = self.app
        return out | self._detail(include)

    def c2pa_assertion(self, include: Iterable[str] = ()) -> dict[str, Any]:
        """A C2PA-style `c2pa.actions` assertion declaring `trainedAlgorithmicMedia`.

        By default it exposes only the source type, creation time and `app`
        (when set); see `to_dict` for `include`. It is the assertion payload
        only: it becomes a signed manifest when an application-owned `Signer`
        embeds it (see `Marker`)."""
        action: dict[str, Any] = {
            "action": "c2pa.created",
            "when": self._when(),
            "digitalSourceType": TRAINED_ALGORITHMIC_MEDIA,
        }
        detail = self._detail(include)
        if "provider" in detail and "model" in detail:
            action["softwareAgent"] = f"{self.provider}/{self.model}"
        params = ({"app": self.app} if self.app is not None else {}) | detail
        if params:
            action["parameters"] = params
        return {"label": "c2pa.actions", "data": {"actions": [action]}}

    def http_headers(self, include: Iterable[str] = ()) -> dict[str, str]:
        """Response headers marking the body as AI-generated (ASCII, single-line).
        Default: `X-AI-Generated`, `X-AI-Created` and `X-AI-App` (when set);
        see `to_dict` for `include`."""
        headers = {"X-AI-Generated": "true", "X-AI-Created": self._when()}
        if (app := _clean(self.app)) is not None:
            headers["X-AI-App"] = app
        for key, value in self._detail(include).items():
            if (cleaned := _clean(value)) is not None:
                headers[_HEADER_NAMES[key]] = cleaned
        return headers

    def pdf_metadata(self, include: Iterable[str] = ()) -> dict[str, str]:
        """Metadata for PDF/XMP exports (Info-dictionary style keys). Default: no
        provider or model; `Producer` is the app when set; see `to_dict` for `include`."""
        meta = {
            "Creator": "AI-generated",
            "Keywords": "AI-generated",
            "AIGenerated": "true",
            "DigitalSourceType": TRAINED_ALGORITHMIC_MEDIA,
            "CreationDate": self._when(),
            "Subject": "Content generated by an AI system",
        }
        if self.app is not None:
            meta["Producer"] = self.app
        detail = self._detail(include)
        if "provider" in detail and "model" in detail:
            meta["Creator"] = f"AI-generated ({self.provider}/{self.model})"
        for key, value in detail.items():
            meta[_PDF_NAMES[key]] = value
        return meta


def provenance_of(
    result: object,
    *,
    provider: str | None = None,
    app: str | None = None,
    surface: str | None = None,
    prompt_version: str | None = None,
) -> Provenance:
    """Build a `Provenance` from a `Completion`, `ChatResult` or the usage chunk of
    `stream_chat()` (a `StreamDelta` with `provider` set). `ChatResult` does not
    name its provider, so pass `provider=` for it. Raises `ValueError` when the
    provider or model is unknown."""
    prov = getattr(result, "provider", None) or provider
    model = getattr(result, "model", None)
    if not prov or not model:
        raise ValueError("provenance needs a provider and a model (pass provider=)")
    created = getattr(result, "created_at", None)
    return Provenance(
        provider=prov,
        model=model,
        request_id=getattr(result, "request_id", None),
        created_at=created if created is not None else datetime.now(UTC),
        app=app,
        surface=surface,
        prompt_version=prompt_version,
    )


class Signer(Protocol):
    """Application-owned signing. The library never holds keys: implement this
    over your KMS/HSM or signing service (locally or via your own network call)."""

    def sign(self, payload: bytes) -> bytes:
        """Return a detached signature over `payload`."""
        ...


@runtime_checkable
class Marker(Protocol):
    """Embeds provenance in a media payload (audio, image, video, ...) and reads it back.

    A C2PA implementation lives in the application (it needs the c2pa tooling and
    certificates); this library only defines the contract."""

    def mark(self, data: bytes, mime: str, provenance: Provenance) -> bytes:
        """Return `data` with provenance embedded."""
        ...

    def verify(self, data: bytes, mime: str) -> Mapping[str, Any] | None:
        """Return the embedded provenance record (a `Provenance.to_dict()`-shaped
        mapping) when present and valid, else None."""
        ...


class NoopMarker:
    """Development marker: returns data unchanged and verifies nothing. It does NOT
    mark; use it only where marking is not required (tests, local dev)."""

    def mark(self, data: bytes, mime: str, provenance: Provenance) -> bytes:
        """Return `data` unchanged."""
        return data

    def verify(self, data: bytes, mime: str) -> Mapping[str, Any] | None:
        """Always None: nothing is embedded."""
        return None


class AppendMarker:
    """Reference `Marker`: appends a signed JSON trailer after the payload.

    Illustrates the Marker/Signer contract; it is NOT C2PA and many parsers ignore
    trailing bytes, so do not rely on it for compliance. `verify` returns a record
    only when your `verifier(payload, signature) -> bool` accepts the signature."""

    _MAGIC = b"\nAIPROV1:"

    def __init__(
        self,
        signer: Signer,
        verifier: Callable[[bytes, bytes], bool] | None = None,
        include: Iterable[str] = (),
    ):
        self._include = tuple([include] if isinstance(include, str) else include)
        self._signer = signer
        self._verifier = verifier

    def mark(self, data: bytes, mime: str, provenance: Provenance) -> bytes:
        """Append the signed provenance trailer to `data`."""
        body = json.dumps(provenance.to_dict(self._include), sort_keys=True).encode()
        sig = self._signer.sign(data + body)
        return data + self._MAGIC + body.hex().encode() + b":" + sig.hex().encode()

    def verify(self, data: bytes, mime: str) -> Mapping[str, Any] | None:
        """The embedded record if the trailer parses and `verifier` accepts it, else None."""
        head, sep, tail = data.rpartition(self._MAGIC)
        if not sep:
            return None
        try:
            body_hex, _, sig_hex = tail.partition(b":")
            body, sig = bytes.fromhex(body_hex.decode()), bytes.fromhex(sig_hex.decode())
            record = json.loads(body)
        except ValueError:
            return None
        if self._verifier is None or not self._verifier(head + body, sig):
            return None
        return record if isinstance(record, dict) else None
