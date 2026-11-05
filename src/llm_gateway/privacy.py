"""A provider-agnostic privacy layer in front of the gateway.

Text is redacted BEFORE it reaches a provider and what comes back is dealt
with by an explicit, per-call output policy:

    from llm_gateway.privacy import PrivacyLayer, Privacy, OutputPolicy

    layer = PrivacyLayer(strict=True)            # a call without a policy raises
    privacy = Privacy(redactor=my_redactor, output=OutputPolicy.RESTORE)
    completion = await layer.complete_with_usage(
        system="...", prompt="Write to anna@example.com", privacy=privacy
    )

Pieces
- `Redactor`: a protocol, `detect(text) -> spans`. The layer owns replacement,
  so one detector works for every call shape (system, user, tool arguments).
- `Vault`: per-request, in memory only. Maps an issued token to the original
  value. `repr` shows counts, never values; it cannot be pickled or copied
  into a log by accident; `clear()` runs when the call ends.
- `OutputPolicy`: RESTORE (tokens this call issued become the original again:
  for text only the data owner sees), STRIP (they are removed: for anything
  that may describe a third party or is stored), PASSTHROUGH (no change).
  Chosen explicitly on every call.
- Strict mode: with `PrivacyLayer(strict=True)` a call with `privacy=None`
  raises `PrivacyPolicyMissingError`. Opting out is explicit and carries a
  reason: `Privacy.passthrough("why")`.
- Streaming: tokens split across chunks are held back until complete, in text
  and in tool-call arguments.
- Audit hook: `on_audit(AuditEvent)` gets COUNTS per kind, never values.

Tokens default to `[kind_N]` (`[person_1]`, `[email_2]`), numbered per kind so
two different values never share a token and restore is exact. Set
`TokenFormat(numbered=False)` for `[kind]`; restore then only works for the
first value of each kind and the second is refused a token of its own.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from .errors import InvalidOutputError


class PrivacyPolicyMissingError(RuntimeError):
    """Strict mode: a call was made without a privacy policy."""


class OutputPolicy(StrEnum):
    RESTORE = "restore"
    STRIP = "strip"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str


class Redactor(Protocol):
    def detect(self, text: str) -> Iterable[Span]:
        """Spans of personal data in `text`. May overlap or be unsorted."""
        ...


@dataclass(frozen=True)
class TokenFormat:
    numbered: bool = True
    open: str = "["
    close: str = "]"

    def render(self, kind: str, n: int) -> str:
        return (
            f"{self.open}{kind}_{n}{self.close}"
            if self.numbered
            else f"{self.open}{kind}{self.close}"
        )

    @property
    def pattern(self) -> re.Pattern[str]:
        o, c = re.escape(self.open), re.escape(self.close)
        suffix = r"_\d+" if self.numbered else ""
        return re.compile(rf"{o}[A-Za-z][A-Za-z -]{{0,30}}?{suffix}{c}")


DEFAULT_FORMAT = TokenFormat()
_MAX_TOKEN_LEN = 48  # longest token the streamer holds back waiting for its close


class Vault:
    """Token -> original, for one request. Never persisted, never logged."""

    def __init__(self, fmt: TokenFormat = DEFAULT_FORMAT) -> None:
        self._fmt = fmt
        self._by_token: dict[str, str] = {}
        self._by_value: dict[tuple[str, str], str] = {}
        self._counts: Counter[str] = Counter()

    def token_for(self, kind: str, original: str) -> str | None:
        """The token for `original`; the same value always gets the same token.
        None when the format is unnumbered and `kind` already stands for
        another value (the caller then masks without a restorable mapping)."""
        key = (kind, original)
        known = self._by_value.get(key)
        if known is not None:
            return known
        self._counts[kind] += 1
        token = self._fmt.render(kind, self._counts[kind])
        if token in self._by_token:
            return None
        self._by_token[token] = original
        self._by_value[key] = token
        return token

    def original(self, token: str) -> str | None:
        return self._by_token.get(token)

    def tokens(self) -> frozenset[str]:
        return frozenset(self._by_token)

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def __len__(self) -> int:
        return len(self._by_token)

    def clear(self) -> None:
        self._by_token.clear()
        self._by_value.clear()
        self._counts.clear()

    def __repr__(self) -> str:  # counts only
        return f"<Vault entries={len(self._by_token)}>"

    def __reduce__(self) -> Any:
        raise TypeError("a Vault is never serialised")

    def __copy__(self) -> Any:
        raise TypeError("a Vault is never copied")

    def __deepcopy__(self, memo: Any) -> Any:
        raise TypeError("a Vault is never copied")


@dataclass(frozen=True)
class AuditEvent:
    """What happened in one call. Counts only: no value ever appears here."""

    call: str
    policy: str
    redacted: dict[str, int]
    restored: int
    stripped: int
    reason: str | None = None


@dataclass(frozen=True)
class Privacy:
    """The privacy policy of one call."""

    redactor: Redactor | None
    output: OutputPolicy
    fmt: TokenFormat = DEFAULT_FORMAT
    # Extra placeholders to remove under STRIP (e.g. an app's older mask
    # vocabulary that a model may echo), besides this call's own tokens.
    strip_pattern: re.Pattern[str] | None = None
    reason: str | None = None
    label: str = ""
    # Redactor for system messages when it must differ from the user-side one
    # (a prompt template holds instructions and example names, not user text).
    # None: the same `redactor`.
    system_redactor: Redactor | None = None
    # Tidies a finished (non-streamed) text after placeholders were removed from
    # it: closes the gap a removed token leaves. Not applied to streamed chunks.
    tidy: Callable[[str], str] | None = None

    @classmethod
    def passthrough(cls, reason: str, label: str = "") -> Privacy:
        if not reason or not reason.strip():
            raise ValueError("a passthrough policy needs a reason")
        return cls(None, OutputPolicy.PASSTHROUGH, reason=reason, label=label)


# ── redaction ────────────────────────────────────────────────────────────────


def _merge(spans: Iterable[Span], length: int) -> list[Span]:
    ordered = sorted(
        (s for s in spans if 0 <= s.start < s.end <= length),
        key=lambda s: (s.start, -(s.end - s.start)),
    )
    merged: list[Span] = []
    for s in ordered:
        if merged and s.start < merged[-1].end:
            continue  # inside or overlapping an earlier (longer-first) span
        merged.append(s)
    return merged


def redact_text(text: str, redactor: Redactor | None, vault: Vault) -> str:
    if not text or redactor is None:
        return text
    out: list[str] = []
    pos = 0
    for s in _merge(redactor.detect(text), len(text)):
        original = text[s.start : s.end]
        token = vault.token_for(s.kind, original)
        out.append(text[pos : s.start])
        out.append(token if token is not None else "")
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map_strings(v, fn) for k, v in value.items()}
    return value


def _map_json_string(raw: str, fn: Callable[[str], str]) -> str:
    """Apply `fn` to every string value of a JSON document; a string that is not
    JSON is treated as plain text."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return fn(raw)
    return json.dumps(_map_strings(data, fn), ensure_ascii=False)


def _map_message(msg: dict, fn: Callable[[str], str]) -> dict:
    out = dict(msg)
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = fn(content)
    elif isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append({**p, "text": fn(p["text"])})
            else:
                parts.append(p)
        out["content"] = parts
    calls = out.get("tool_calls")
    if isinstance(calls, list):
        new_calls = []
        for c in calls:
            if isinstance(c, dict) and isinstance(c.get("function"), dict):
                fn_spec = dict(c["function"])
                args = fn_spec.get("arguments")
                if isinstance(args, str):
                    fn_spec["arguments"] = _map_json_string(args, fn)
                elif isinstance(args, dict):
                    fn_spec["arguments"] = _map_strings(args, fn)
                new_calls.append({**c, "function": fn_spec})
            else:
                new_calls.append(c)
        out["tool_calls"] = new_calls
    return out


def redact_messages(
    messages: list[dict],
    redactor: Redactor | None,
    vault: Vault,
    system_redactor: Redactor | None = None,
) -> list[dict]:
    """Messages with system, user, assistant, tool results AND tool-call
    arguments redacted (a copy; the caller's list is untouched). System
    messages use `system_redactor` when given."""
    sys_r = system_redactor or redactor
    out = []
    for m in messages:
        r = sys_r if m.get("role") == "system" else redactor
        out.append(_map_message(m, lambda t, r=r: redact_text(t, r, vault)))
    return out


# ── output ───────────────────────────────────────────────────────────────────


class _Output:
    """Applies the output policy to text; counts what it did."""

    def __init__(self, privacy: Privacy, vault: Vault) -> None:
        self.privacy = privacy
        self.vault = vault
        self.restored = 0
        self.stripped = 0
        self._re = privacy.fmt.pattern

    def apply(self, text: str) -> str:
        pol = self.privacy.output
        if pol is OutputPolicy.PASSTHROUGH or not text:
            return text

        # Placeholders this call did not issue (an app's older vocabulary, or a
        # token the model made up) are removed under RESTORE and STRIP alike:
        # nothing can restore them and they must not reach a reader.
        pattern = self.privacy.strip_pattern
        if pattern is not None:

            def foreign(m: re.Match[str]) -> str:
                if self.vault.original(m.group(0)) is not None:
                    return m.group(0)
                self.stripped += 1
                return ""

            text = pattern.sub(foreign, text)

        def sub(m: re.Match[str]) -> str:
            token = m.group(0)
            original = self.vault.original(token)
            if original is None:
                return token
            if pol is OutputPolicy.RESTORE:
                self.restored += 1
                return original
            self.stripped += 1
            return ""

        return self._re.sub(sub, text)

    def apply_final(self, text: str) -> str:
        """`apply` for a complete text: tidied when something was removed."""
        before = self.stripped
        out = self.apply(text)
        if self.stripped != before and self.privacy.tidy is not None:
            out = self.privacy.tidy(out)
        return out

    def apply_json(self, value: Any) -> Any:
        return _map_strings(value, self.apply_final)


class StreamFilter:
    """The output policy over a stream of chunks. A token split across chunks
    is held until it is complete (or cannot be a token any more)."""

    def __init__(self, out: _Output) -> None:
        self._out = out
        self._held = ""
        self._open = out.privacy.fmt.open

    def feed(self, chunk: str) -> str:
        if self._out.privacy.output is OutputPolicy.PASSTHROUGH:
            return chunk
        buf = self._held + chunk
        cut = len(buf)
        i = buf.rfind(self._open)
        if i != -1 and self._out.privacy.fmt.close not in buf[i:] and len(buf) - i < _MAX_TOKEN_LEN:
            cut = i
        self._held = buf[cut:]
        return self._out.apply(buf[:cut])

    def finish(self) -> str:
        rest, self._held = self._held, ""
        return self._out.apply(rest)


# ── the layer ────────────────────────────────────────────────────────────────

AuditHook = Callable[[AuditEvent], None]


class PrivacyLayer:
    """Wraps the gateway entry points. `strict=True`: no policy, no call."""

    def __init__(
        self,
        *,
        strict: bool = True,
        on_audit: AuditHook | None = None,
        complete_with_usage: Callable[..., Awaitable[Any]] | None = None,
        chat: Callable[..., Awaitable[Any]] | None = None,
        stream_chat: Callable[..., AsyncIterator[Any]] | None = None,
    ) -> None:
        self.strict = strict
        self.on_audit = on_audit
        self._complete = complete_with_usage
        self._chat = chat
        self._stream = stream_chat

    # -- helpers

    def _resolve(self, privacy: Privacy | None, call: str) -> Privacy:
        if privacy is None:
            if self.strict:
                raise PrivacyPolicyMissingError(
                    f"{call}: no privacy policy given (strict mode). Pass privacy=Privacy(...) "
                    "or Privacy.passthrough(reason)."
                )
            return Privacy.passthrough("non-strict default", label=call)
        return privacy

    def _audit(self, call: str, p: Privacy, vault: Vault, out: _Output) -> None:
        if self.on_audit is not None:
            self.on_audit(
                AuditEvent(
                    call=p.label or call,
                    policy=p.output.value,
                    redacted=vault.counts(),
                    restored=out.restored,
                    stripped=out.stripped,
                    reason=p.reason,
                )
            )

    # -- complete_with_usage

    async def complete_with_usage(
        self, *, system: str, prompt: str, privacy: Privacy | None = None, **kwargs: Any
    ) -> Any:
        p = self._resolve(privacy, "complete_with_usage")
        vault = Vault(p.fmt)
        out = _Output(p, vault)
        try:
            fn = self._complete
            if fn is None:
                from .router import complete_with_usage as fn  # noqa: PLC0415
            completion = await fn(
                system=redact_text(system, p.system_redactor or p.redactor, vault),
                prompt=redact_text(prompt, p.redactor, vault),
                **kwargs,
            )
            result = self._finish_completion(completion, out, kwargs.get("output_schema"))
            self._audit("complete_with_usage", p, vault, out)
            return result
        finally:
            vault.clear()

    def _finish_completion(self, completion: Any, out: _Output, schema: Any) -> Any:
        if out.privacy.output is OutputPolicy.PASSTHROUGH:
            return completion
        text = out.apply_final(completion.text)
        parsed = getattr(completion, "parsed", None)
        new_parsed = parsed
        if parsed is not None:
            new_parsed = _rebuild_parsed(parsed, out)
        return replace(completion, text=text, parsed=new_parsed)

    # -- chat (non-streaming, tools)

    async def chat(
        self, *, messages: list[dict], privacy: Privacy | None = None, **kwargs: Any
    ) -> Any:
        p = self._resolve(privacy, "chat")
        vault = Vault(p.fmt)
        out = _Output(p, vault)
        try:
            fn = self._chat
            if fn is None:
                from .chat import chat as fn  # noqa: PLC0415
            result = await fn(
                messages=redact_messages(messages, p.redactor, vault, p.system_redactor), **kwargs
            )
            if p.output is not OutputPolicy.PASSTHROUGH:
                calls = [
                    replace(tc, arguments=out.apply_json(tc.arguments)) for tc in result.tool_calls
                ]
                result = replace(
                    result,
                    content=out.apply_final(result.content) if result.content else result.content,
                    tool_calls=calls,
                )
            self._audit("chat", p, vault, out)
            return result
        finally:
            vault.clear()

    # -- stream_chat

    async def stream_chat(
        self, *, messages: list[dict], privacy: Privacy | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        p = self._resolve(privacy, "stream_chat")
        vault = Vault(p.fmt)
        out = _Output(p, vault)
        text_filter = StreamFilter(out)
        arg_filters: dict[int, StreamFilter] = {}
        fn = self._stream
        if fn is None:
            from .streaming import stream_chat as fn  # noqa: PLC0415
        stream = fn(
            messages=redact_messages(messages, p.redactor, vault, p.system_redactor), **kwargs
        )
        try:
            async for delta in stream:
                yield self._filter_delta(delta, text_filter, arg_filters)
            tail = text_filter.finish()
            extra = [
                {"index": i, "function": {"arguments": f.finish()}}
                for i, f in sorted(arg_filters.items())
            ]
            extra = [e for e in extra if e["function"]["arguments"]]
            if tail or extra:
                yield _delta(content=tail or None, tool_call_deltas=extra or None)
            self._audit("stream_chat", p, vault, out)
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
            vault.clear()

    def _filter_delta(
        self, delta: Any, text_filter: StreamFilter, arg_filters: dict[int, StreamFilter]
    ) -> Any:
        changes: dict[str, Any] = {}
        if delta.content:
            changes["content"] = text_filter.feed(delta.content) or None
        if delta.tool_call_deltas:
            frags = []
            for frag in delta.tool_call_deltas:
                frag = copy.deepcopy(frag)
                fn_spec = frag.get("function") or {}
                args = fn_spec.get("arguments")
                if args:
                    idx = int(frag.get("index", 0))
                    f = arg_filters.setdefault(idx, StreamFilter(text_filter._out))
                    fn_spec["arguments"] = f.feed(args)
                    frag["function"] = fn_spec
                frags.append(frag)
            changes["tool_call_deltas"] = frags
        return replace(delta, **changes) if changes else delta


def _delta(**kw: Any) -> Any:
    from .providers.base import StreamDelta  # noqa: PLC0415

    return StreamDelta(**kw)


def _rebuild_parsed(parsed: Any, out: _Output) -> Any:
    """`Completion.parsed` after the output policy: a dict is mapped; a pydantic
    model is dumped, mapped and validated again. A model the policy emptied
    (a required string left blank) raises InvalidOutputError rather than
    leaking the token."""
    if isinstance(parsed, (dict, list)):
        return out.apply_json(parsed)
    dump = getattr(parsed, "model_dump", None)
    if dump is None:
        return parsed
    data = out.apply_json(dump(mode="json"))
    try:
        return type(parsed).model_validate(data)
    except Exception as exc:  # pydantic.ValidationError, kept generic to avoid a hard import
        raise InvalidOutputError("output no longer validates after the privacy policy") from exc
