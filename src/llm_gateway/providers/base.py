"""Shared provider-call contracts.

Every provider module exposes:
- `async def call(config, system, prompt, max_tokens, model) -> ProviderResult`
  — plain text completion, used by `router.complete()`.
- `async def chat(config, messages, tools, max_tokens, model) -> ChatResult`
  — tool-calling-capable, used by `chat.chat()`. `messages` and `tools` are
  OpenAI-wire-shaped dicts; each provider translates to/from its own native
  format internally (Anthropic's tool format is nothing like OpenAI's —
  see providers/anthropic.py).

Providers own their own client caching internally.
"""

import asyncio
import dataclasses
import json
import logging
from collections.abc import AsyncGenerator, Iterable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

# The same class as anthropic.Timeout — both SDKs re-export httpx2.Timeout.
from openai import Timeout
from openai.types.chat import ChatCompletionMessageParam

from ..config import GatewayConfig
from ..errors import EmptyCompletionError, InvalidOutputError, ToolCallTruncatedError
from ._wire import ToolCallDelta, ToolCallDeltaFunction

logger = logging.getLogger("llm_gateway")

C = TypeVar("C")


def sdk_messages(messages: list[dict[str, Any]]) -> list[ChatCompletionMessageParam]:
    """The gateway's OpenAI-format message dicts, typed for the OpenAI SDK.

    The shape is validated by `chat()` before any provider is called, so this
    is a typing-only cast (the SDK's TypedDict union cannot express a plain
    `dict`)."""
    return cast("list[ChatCompletionMessageParam]", messages)


# Kept at the SDKs' own default: a TCP connect that takes longer than this
# is not going to succeed, however long the request timeout is.
CONNECT_TIMEOUT_SECONDS = 5.0

# OpenAI-named sampling params accepted on requests. OpenAI/Groq accept these
# verbatim (same param names); Anthropic needs translation — see
# _anthropic_translate.to_anthropic_sampling for which of these it actually
# supports (seed, presence_penalty, frequency_penalty have no equivalent
# there and are silently dropped, not errored).
OPENAI_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
)


def openai_sampling_kwargs(sampling: dict[str, Any] | None) -> dict[str, Any]:
    """Filter to only the set (non-None) sampling keys — safe to **-expand
    directly into an OpenAI-compatible SDK call (OpenAI, Groq)."""
    if not sampling:
        return {}
    return {k: v for k, v in sampling.items() if k in OPENAI_SAMPLING_KEYS and v is not None}


def _sdk_timeout(seconds: float) -> Timeout | None:
    if seconds <= 0:
        return None
    return Timeout(seconds, connect=min(CONNECT_TIMEOUT_SECONDS, seconds))


def sdk_http_client(factory: Any, config: GatewayConfig) -> Any:
    """The SDK's own httpx2 client factory, with redirects off. The SDKs
    follow redirects by default, and httpx only strips `Authorization` across
    origins, so a hostile or compromised endpoint could bounce the request
    (with `x-api-key` / `api-key` headers) to another host. Honors ssl_verify."""
    kwargs: dict[str, Any] = {"follow_redirects": False}
    if not config.ssl_verify:
        kwargs["verify"] = False
    return factory(**kwargs)


def sdk_client_options(config: GatewayConfig) -> dict[str, Any]:
    """Constructor kwargs shared by every provider's SDK client: the
    per-attempt request timeout and the SDK's own retry count (see
    GatewayConfig.sdk_max_retries for why that is 0 by default). An
    omitted `timeout` keeps the SDK default — passing None would disable
    timeouts entirely."""
    options: dict[str, Any] = {"max_retries": max(config.sdk_max_retries, 0)}
    timeout = _sdk_timeout(config.request_timeout_seconds)
    if timeout is not None:
        options["timeout"] = timeout
    return options


# SDK clients are cached per (credential, settings) so every call reuses one
# connection pool. The cache is capped so credential rotation or per-tenant
# configs cannot grow it without bound; the least recently created client is
# evicted and closed after a grace period that lets its in-flight calls end.
MAX_CACHED_CLIENTS = 16
EVICTED_CLIENT_GRACE_SECONDS = 120.0
_closing: set[asyncio.Task[None]] = set()


async def _close_later(client: Any, grace: float) -> None:
    try:
        await asyncio.sleep(grace)
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001 — best effort, nothing to report to
            logger.debug("llm_gateway: closing an evicted client failed", exc_info=True)


def remember_client(cache: dict[tuple[Any, ...], C], key: tuple[Any, ...], client: C) -> C:
    """Store `client` under `key`, evicting the oldest entries beyond
    MAX_CACHED_CLIENTS. Returns `client`."""
    cache[key] = client
    while len(cache) > MAX_CACHED_CLIENTS:
        evicted = cache.pop(next(iter(cache)))
        try:
            task = asyncio.get_running_loop().create_task(
                _close_later(evicted, EVICTED_CLIENT_GRACE_SECONDS)
            )
        except RuntimeError:  # no running loop: nothing is in flight
            continue
        _closing.add(task)
        task.add_done_callback(_closing.discard)
    return client


async def close_clients(cache: dict[tuple[Any, ...], Any]) -> None:
    """Empty `cache` and close every client in it (and any evicted client
    still waiting out its grace period). One failing close does not stop the
    rest; cancellation still propagates."""
    clients = list(cache.values())
    cache.clear()
    pending = list(_closing)
    for task in pending:
        task.cancel()  # its `finally` closes the client now
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for client in clients:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            logger.warning("llm_gateway: closing an SDK client failed", exc_info=True)


def sdk_client_cache_key(api_key: str, config: GatewayConfig) -> tuple[Any, ...]:
    return (
        api_key,
        config.ssl_verify,
        config.request_timeout_seconds,
        config.sdk_max_retries,
    )


def stream_request_options(config: GatewayConfig) -> dict[str, Any]:
    """Per-request kwargs for a streaming SDK call: an httpx read timeout
    bounds the wait for the response to start and every gap between chunks,
    without capping how long a steadily-flowing stream may run."""
    timeout = _sdk_timeout(config.stream_idle_timeout_seconds)
    return {"timeout": timeout} if timeout is not None else {}


@dataclass(frozen=True)
class Usage:
    """Token usage for one served call, normalized across providers.

    `input_tokens` counts every prompt token the provider processed, cached
    or not, so it means the same thing for every provider: Anthropic reports
    uncached, cache-read and cache-write tokens as three disjoint numbers
    (they are summed here), while OpenAI/Groq's `prompt_tokens` already
    includes cached tokens. The two cache fields are subsets of
    `input_tokens`; use `uncached_input_tokens` for the part billed at the
    base input rate.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    # The subset of `output_tokens` the model spent reasoning (OpenAI/Groq
    # `completion_tokens_details.reasoning_tokens`, Anthropic
    # `output_tokens_details.thinking_tokens`). Already included in
    # `output_tokens` and billed as output; 0 when not reported.
    reasoning_tokens: int = 0

    @property
    def uncached_input_tokens(self) -> int:
        cached = self.cache_read_input_tokens + self.cache_creation_input_tokens
        return max(self.input_tokens - cached, 0)

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )

    @classmethod
    def total(cls, usages: "Iterable[Usage]") -> "Usage":
        """The sum of every step's usage, e.g. across a tool loop (each step
        resends the history, so input tokens add up; that is what is billed).
        Price it once with `cost_usd` when all steps used one model."""
        out = cls()
        for usage in usages:
            out = out + usage
        return out


_E = TypeVar("_E", "InvalidOutputError", "EmptyCompletionError")


def as_int(value: object) -> int:
    """An SDK usage field as an int: None (field absent) and anything
    non-integer count as 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def usage_from_openai_style(usage: Any) -> Usage:
    """Usage from an OpenAI/Groq `CompletionUsage` (or None)."""
    if usage is None:
        return Usage()
    details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return Usage(
        input_tokens=as_int(getattr(usage, "prompt_tokens", None)),
        output_tokens=as_int(getattr(usage, "completion_tokens", None)),
        cache_read_input_tokens=as_int(getattr(details, "cached_tokens", None)),
        # Only OpenRouter reports cache writes (`cache_write_tokens`, billed at
        # a premium for Anthropic models); OpenAI and Groq never set it.
        cache_creation_input_tokens=as_int(getattr(details, "cache_write_tokens", None)),
        reasoning_tokens=as_int(getattr(completion_details, "reasoning_tokens", None)),
    )


_LENGTH_REASONS = frozenset(
    {"length", "max_tokens", "model_length", "max_output_tokens", "model_context_window_exceeded"}
)
_FILTER_REASONS = frozenset(
    {
        "content_filter",
        "safety",
        "recitation",
        "blocklist",
        "prohibited_content",
        "spii",
        "refusal",
    }
)
_TOOL_REASONS = frozenset({"tool_calls", "function_call", "tool_use"})


def normalize_finish_reason(raw: str | None) -> str:
    """One finish-reason vocabulary for every provider, in chat and streams:
    "tool_calls", "length" (the token budget ran out), "content_filter" (a
    safety filter or refusal ended the reply) or "stop" (anything else:
    `stop`, `end_turn`, `stop_sequence`, ...)."""
    reason = (raw or "").lower()
    if reason in _TOOL_REASONS:
        return "tool_calls"
    if reason in _LENGTH_REASONS:
        return "length"
    if reason in _FILTER_REASONS:
        return "content_filter"
    return "stop"


# Reasoning models (OpenAI o-series / gpt-5, Azure reasoning deployments)
# reject these with a 400: only the default sampling is allowed.
REASONING_REJECTED_SAMPLING = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "stop",
)


def drop_keys(kwargs: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in keys}


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_think(text: str | None) -> str | None:
    """Remove a leading `<think>...</think>` block that some hosts (DeepSeek-R1
    and Qwen3 on vLLM/TGI/Groq, ...) put inline in `content`. An unclosed block
    (the budget ran out while thinking) leaves nothing."""
    if not text or not text.lstrip().startswith(_THINK_OPEN):
        return text
    end = text.find(_THINK_CLOSE)
    if end < 0:
        return ""
    return text[end + len(_THINK_CLOSE) :].lstrip()


class _ThinkStripper:
    """Streaming `strip_think()`: feed content pieces, get the visible part."""

    def __init__(self) -> None:
        self._state = "start"  # start | inside | trim | pass
        self._buf = ""

    def feed(self, text: str) -> str:
        if self._state == "pass":
            return text
        self._buf += text
        if self._state == "start":
            head = self._buf.lstrip()
            if not head:
                return ""
            if head.startswith(_THINK_OPEN):
                self._state, self._buf = "inside", head[len(_THINK_OPEN) :]
            elif _THINK_OPEN.startswith(head):
                return ""
            else:
                self._state, out, self._buf = "pass", self._buf, ""
                return out
        if self._state == "inside":
            end = self._buf.find(_THINK_CLOSE)
            if end < 0:
                self._buf = self._buf[-len(_THINK_CLOSE) :]
                return ""
            self._state, self._buf = "trim", self._buf[end + len(_THINK_CLOSE) :]
        if self._state == "trim":
            out, self._buf = self._buf.lstrip(), ""
            if out:
                self._state = "pass"
            return out
        return ""

    def flush(self) -> str:
        """Held-back text that turned out not to be a think block."""
        out = self._buf if self._state == "start" else ""
        self._buf = ""
        self._state = "pass"
        return out


def with_billed_usage(error: _E, usage: Usage | None) -> _E:
    """`error` with the usage the provider billed for the rejected reply."""
    error.usage = usage
    return error


def ensure_not_empty(
    text: str | None,
    *,
    has_tool_calls: bool = False,
    provider: str,
    model: str,
    finish_reason: str | None,
    reasoning_tokens: int | None,
    usage: Usage | None = None,
) -> None:
    """Raise `EmptyCompletionError` for a reply with no usable text (None,
    empty or whitespace) and no tool calls. Logs a warning with the provider,
    model, finish reason and reasoning tokens; never the content."""
    if has_tool_calls or (text and text.strip()):
        return
    logger.warning(
        "llm_gateway empty completion: provider=%s model=%s finish_reason=%s reasoning_tokens=%s",
        provider,
        model,
        finish_reason,
        reasoning_tokens,
    )
    error = EmptyCompletionError(
        provider=provider,
        model=model,
        finish_reason=finish_reason,
        reasoning_tokens=reasoning_tokens,
    )
    error.usage = usage
    raise error


@dataclass(frozen=True)
class ProviderResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    # Provider-native stop reason ("end_turn", "stop", "max_tokens", ...).
    # Excluded from equality: it is diagnostic, not part of the result.
    stop_reason: str | None = field(default=None, compare=False)

    @property
    def usage(self) -> Usage:
        return Usage(
            self.input_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
            self.reasoning_tokens,
        )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResult:
    content: str | None
    model: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    finish_reason: str = "stop"  # "stop" | "tool_calls"
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    # The call's correlation id, set by chat() (see tracing.resolve_request_id).
    # Excluded from equality: it identifies the call, not the answer.
    request_id: str | None = field(default=None, compare=False)

    @property
    def usage(self) -> Usage:
        return Usage(
            self.input_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
            self.reasoning_tokens,
        )


@dataclass(frozen=True)
class StreamDelta:
    """One normalized chunk of a streamed response. `tool_call_deltas` are
    OpenAI-shaped fragments (`[{"index":, "id":, "type":"function",
    "function":{"name":, "arguments":}}]`) — only the first fragment for a
    given tool call carries id/name, matching OpenAI's own streaming
    semantics exactly (so client-side aggregation code written against real
    OpenAI streaming, e.g. LangChain's, works unmodified)."""

    content: str | None = None
    tool_call_deltas: list[ToolCallDelta] | None = None
    finish_reason: str | None = None
    model: str | None = None
    usage: tuple[int, int] | None = None  # (input_tokens, output_tokens) — final chunk only
    # The same usage with the cache breakdown — final chunk only, when known.
    usage_details: Usage | None = None
    # Which provider served the stream. Set by `stream_chat()` on the chunk
    # carrying usage, so a caller can attribute cost after a failover.
    provider: str | None = None
    # The call's correlation id; set with `provider` on the usage chunk.
    request_id: str | None = None
    # Restart marker (`stream_chat(restart_on_failure=True)` only). A marker
    # carries no content and no tool-call deltas: it says the previous attempt
    # failed mid-stream and everything yielded so far, text and tool-call
    # fragments alike, is void. `provider`/`model` name the provider now
    # serving; `restart_attempt` counts restarts from 1.
    restart: bool = False
    restart_attempt: int | None = None
    failed_provider: str | None = None
    failed_error: str | None = None  # exception class name only


def _tool_arguments(raw: str | None) -> dict[str, Any]:
    """A tool call's arguments from the provider's JSON string. Truncated or
    malformed JSON (or anything but an object) is `InvalidOutputError`, so the
    call fails over, instead of an unclassified parse error (`UNKNOWN`) that
    counts against the provider's breaker."""
    try:
        arguments = json.loads(raw or "{}")
    except (ValueError, RecursionError):
        raise InvalidOutputError("Tool call arguments are not valid JSON.") from None
    if arguments is None:
        return {}
    if not isinstance(arguments, dict):
        raise InvalidOutputError("Tool call arguments are not a JSON object.")
    return cast("dict[str, Any]", arguments)


def _first_choice(resp: Any, model: str, provider: str) -> Any:
    """`resp.choices[0]`; a reply with no choices at all (Gemini blocking the
    prompt for safety, a filtered Azure reply) is `EmptyCompletionError` with
    finish reason "no_choices", not an IndexError that counts as UNKNOWN."""
    if not getattr(resp, "choices", None):
        usage = usage_from_openai_style(getattr(resp, "usage", None))
        ensure_not_empty(
            None,
            provider=provider,
            model=model,
            finish_reason="no_choices",
            reasoning_tokens=usage.reasoning_tokens,
            usage=usage,
        )
    return resp.choices[0]


def parse_openai_style_response(resp: Any, model: str, provider: str = "openai") -> ChatResult:
    """Shared response parser for any provider speaking the OpenAI chat-
    completions wire format natively (OpenAI itself, Groq).

    Null content is fine alongside tool calls. Without tool calls, a refusal
    is `InvalidOutputError` and empty content is `EmptyCompletionError`."""
    choice = _first_choice(resp, model, provider)
    msg = choice.message
    usage = usage_from_openai_style(resp.usage)
    if getattr(msg, "refusal", None) and not msg.tool_calls:
        raise with_billed_usage(
            InvalidOutputError("The model refused the request (message.refusal is set)."), usage
        )
    finish_reason = normalize_finish_reason(choice.finish_reason)
    raw_tool_calls = cast("list[Any]", msg.tool_calls or [])
    try:
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_tool_arguments(tc.function.arguments),
            )
            for tc in raw_tool_calls
        ]
    except InvalidOutputError:
        if finish_reason == "length":
            raise with_billed_usage(
                ToolCallTruncatedError(
                    "The reply was cut off by the token limit inside a tool call; "
                    "retry with a larger max_tokens.",
                    tuple(tc.function.name for tc in raw_tool_calls),
                ),
                usage,
            ) from None
        raise
    if tool_calls:
        finish_reason = "tool_calls"
    elif finish_reason == "tool_calls":
        finish_reason = "stop"
    content = strip_think(msg.content)
    ensure_not_empty(
        content,
        has_tool_calls=bool(tool_calls),
        provider=provider,
        model=model,
        finish_reason=choice.finish_reason,
        reasoning_tokens=usage.reasoning_tokens,
        usage=usage,
    )
    return ChatResult(
        # Mistral sends "" next to tool calls where OpenAI sends null; one shape.
        content=content or None,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )


def provider_result_from_openai_style(
    resp: Any, model: str, provider: str = "openai"
) -> ProviderResult:
    """Plain-completion parser shared by OpenAI and Groq.

    `message.content` is None when the model refuses (OpenAI sets
    `message.refusal`, which in practice only happens for structured-output
    requests) or answers with tool calls (never requested here). A refusal is
    an unusable answer (`InvalidOutputError`); empty or whitespace content is
    `EmptyCompletionError`. Both fail over instead of returning "".
    """
    choice = _first_choice(resp, model, provider)
    message = choice.message
    usage = usage_from_openai_style(resp.usage)
    if getattr(message, "refusal", None):
        raise with_billed_usage(
            InvalidOutputError("The model refused the request (message.refusal is set)."), usage
        )
    content = strip_think(message.content)
    ensure_not_empty(
        content,
        provider=provider,
        model=model,
        finish_reason=choice.finish_reason,
        reasoning_tokens=usage.reasoning_tokens,
        usage=usage,
    )
    return ProviderResult(
        text=(content or "").strip(),
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        stop_reason=choice.finish_reason,
    )


def parse_openai_style_chunk(chunk: Any, model: str) -> StreamDelta:
    """Shared streaming-chunk parser for OpenAI/Groq — their chunks are
    already shaped almost exactly like our normalized StreamDelta."""
    # OpenAI sends usage in a trailing chunk with no choices, but other hosts
    # (e.g. Mistral) put it on the last chunk that also carries the choice.
    details = usage_from_openai_style(chunk.usage) if chunk.usage else None
    usage = (details.input_tokens, details.output_tokens) if details else None
    if not chunk.choices:
        return StreamDelta(model=model, usage=usage, usage_details=details)

    choice = chunk.choices[0]
    delta = choice.delta
    tool_call_deltas: list[ToolCallDelta] | None = None
    if delta.tool_calls:
        tool_call_deltas = []
        for tc in cast("list[Any]", delta.tool_calls):
            fragment: ToolCallDelta = {"index": tc.index}
            if tc.id:
                fragment["id"] = tc.id
                fragment["type"] = "function"
            function: ToolCallDeltaFunction = {}
            if tc.function and tc.function.name:
                function["name"] = tc.function.name
            function["arguments"] = (tc.function.arguments or "") if tc.function else ""
            fragment["function"] = function
            tool_call_deltas.append(fragment)
    return StreamDelta(
        content=delta.content,
        tool_call_deltas=tool_call_deltas,
        finish_reason=(
            normalize_finish_reason(choice.finish_reason) if choice.finish_reason else None
        ),
        model=model,
        usage=usage,
        usage_details=details,
    )


class ToolIndexNormalizer:
    """Gives every streamed tool call its own `index`.

    OpenAI-shaped streams key a tool call's fragments by `index`, but some
    hosts do not keep to that: Gemini's OpenAI endpoint reports every call of
    a parallel batch at index 0 (told apart only by `id`), and a host may omit
    the index altogether. Aggregating such a stream by index glues unrelated
    calls into one, so a fragment that opens a new call (it carries an `id` not
    seen before) at an index another call already holds is moved to the next
    free index, and later fragments for that raw index follow it. Streams that
    already number their calls correctly pass through unchanged."""

    def __init__(self) -> None:
        self._by_id: dict[str, int] = {}
        self._by_raw: dict[int | None, int] = {}
        self._used: set[int] = set()
        self._last: int = 0

    def _free(self) -> int:
        index = 0
        while index in self._used:
            index += 1
        return index

    def _remap(self, fragment: ToolCallDelta) -> ToolCallDelta:
        raw = fragment.get("index")
        call_id = fragment.get("id")
        if call_id:
            index = self._by_id.get(call_id)
            if index is None:
                index = raw if isinstance(raw, int) and raw not in self._used else self._free()
                self._by_id[call_id] = index
                self._used.add(index)
            self._by_raw[raw] = index
        else:
            index = self._by_raw.get(raw, raw if isinstance(raw, int) else self._last)
        self._last = index
        return fragment if index == raw else {**fragment, "index": index}

    def apply(self, delta: StreamDelta) -> StreamDelta:
        if not delta.tool_call_deltas:
            return delta
        return dataclasses.replace(
            delta, tool_call_deltas=[self._remap(f) for f in delta.tool_call_deltas]
        )


async def _strip_think_stream(deltas: AsyncGenerator[StreamDelta]) -> AsyncGenerator[StreamDelta]:
    stripper = _ThinkStripper()
    async with aclosing(deltas):
        async for delta in deltas:
            if delta.content:
                visible = stripper.feed(delta.content)
                if not visible and not (
                    delta.tool_call_deltas or delta.finish_reason or delta.usage
                ):
                    continue
                delta = dataclasses.replace(delta, content=visible or None)
            if delta.finish_reason or delta.usage:
                held = stripper.flush()
                if held:
                    yield StreamDelta(content=held, model=delta.model)
            yield delta
        held = stripper.flush()
        if held:
            yield StreamDelta(content=held)


async def guard_empty_stream(
    deltas: AsyncGenerator[StreamDelta], *, provider: str, model: str, strip_think: bool = False
) -> AsyncGenerator[StreamDelta]:
    """Hold a stream's chunks back until one carries non-whitespace text or a
    tool call, then pass everything through in order.

    A stream that ends before that (a reasoning model streaming only
    reasoning, or nothing at all) raises `EmptyCompletionError` before its
    first chunk reaches the engine, so `stream_chat()` can still fail over.
    Once real content has been seen, chunks flow through unbuffered.

    `strip_think` removes a leading inline `<think>...</think>` block from the
    content first (see `strip_think()`), so reasoning-only streams count as
    empty."""
    stream: AsyncGenerator[StreamDelta] = _strip_think_stream(deltas) if strip_think else deltas
    pending: list[StreamDelta] = []
    started = False
    async with aclosing(stream):
        async for delta in stream:
            if started:
                yield delta
                continue
            pending.append(delta)
            if delta.tool_call_deltas or (delta.content and delta.content.strip()):
                started = True
                for item in pending:
                    yield item
                pending.clear()
    if not started:
        finish_reason = next((d.finish_reason for d in reversed(pending) if d.finish_reason), None)
        details = next((d.usage_details for d in reversed(pending) if d.usage_details), None)
        ensure_not_empty(
            None,
            provider=provider,
            model=model,
            finish_reason=finish_reason,
            reasoning_tokens=details.reasoning_tokens if details else None,
            usage=details,
        )
