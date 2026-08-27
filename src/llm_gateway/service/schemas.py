"""OpenAI-compatible request/response shapes for /v1/chat/completions.

Deliberately minimal — this is not a full OpenAI API reimplementation, just
enough of the wire format that any OpenAI-SDK-compatible client (including
LangChain's ChatOpenAI, which is what LangGraph agents use) can point at
this service by changing its base_url.
"""

import json
import time
import uuid

from pydantic import BaseModel, Field

from ..providers.base import ChatResult, StreamDelta


class ChatMessage(BaseModel):
    role: str
    # A plain string, or OpenAI's multi-part content list for multi-modal
    # messages (`[{"type": "text", "text": ...}, {"type": "image_url",
    # "image_url": {"url": ...}}]`) — passed through as-is for OpenAI/Groq,
    # translated for Anthropic (see providers/_anthropic_translate.py).
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None  # present on an assistant message requesting tool calls
    tool_call_id: str | None = None  # present on a role="tool" message (the result)

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d

    def text_length(self) -> int:
        """Length used for the max_prompt_chars guardrail — text only,
        images aren't charged against it (they have their own natural size
        limit via the HTTP request body)."""
        if isinstance(self.content, str):
            return len(self.content)
        if isinstance(self.content, list):
            return sum(len(p.get("text", "")) for p in self.content if p.get("type") == "text")
        return 0


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = 2000
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None
    stream: bool = False

    # OpenAI-named sampling params. Previously silently dropped (Pydantic's
    # default extra="ignore" swallowed any field not declared here) — a real
    # bug, since e.g. LangChain's ChatOpenAI(temperature=0) is extremely
    # common for deterministic agent behavior. Each has a provider-specific
    # translation (or is dropped where a provider has no equivalent) — see
    # providers/base.py:openai_sampling_kwargs and
    # providers/_anthropic_translate.py:to_anthropic_sampling.
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def as_message_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]

    def sampling_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }


def chat_completion_response_from_result(result: ChatResult) -> dict:
    """Build an OpenAI-shaped response from a chat()/tool-calling ChatResult."""
    message: dict = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in result.tool_calls
        ]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        },
    }


def stream_chunk_sse(delta: StreamDelta, *, chunk_id: str, created: int) -> list[str]:
    """Render one StreamDelta as 0-2 OpenAI-shaped `chat.completion.chunk`
    SSE events (`data: {...}\\n\\n`). A final delta commonly carries both
    finish_reason and usage together — OpenAI's own streaming sends those as
    two *separate* chunks (a finish_reason chunk, then a usage-only trailer
    with empty choices), so a delta with both is split the same way rather
    than collapsed into one non-standard chunk."""
    lines = []
    if (
        delta.content is not None
        or delta.tool_call_deltas is not None
        or delta.finish_reason is not None
    ):
        delta_obj: dict = {}
        if delta.content is not None:
            delta_obj["content"] = delta.content
        if delta.tool_call_deltas is not None:
            delta_obj["tool_calls"] = delta.tool_call_deltas
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": delta.model,
            "choices": [{"index": 0, "delta": delta_obj, "finish_reason": delta.finish_reason}],
        }
        lines.append(f"data: {json.dumps(payload)}\n\n")

    if delta.usage is not None:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": delta.model,
            "choices": [],
            "usage": {
                "prompt_tokens": delta.usage[0],
                "completion_tokens": delta.usage[1],
                "total_tokens": delta.usage[0] + delta.usage[1],
            },
        }
        lines.append(f"data: {json.dumps(payload)}\n\n")

    return lines


SSE_DONE = "data: [DONE]\n\n"


def sse_error_event(message: str) -> str:
    """A mid-or-pre-stream failure can't change the HTTP status code — the
    200 + SSE headers are already committed by the time a provider failure
    is known. This emits an OpenAI-error-shaped SSE event instead, followed
    by the caller sending SSE_DONE, so consumers see a clean end rather than
    a silently truncated connection."""
    payload = {"error": {"message": message, "type": "service_unavailable_error", "code": 503}}
    return f"data: {json.dumps(payload)}\n\n"
