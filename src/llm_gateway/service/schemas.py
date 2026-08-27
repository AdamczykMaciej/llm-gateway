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

from ..providers.base import ChatResult


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None  # present on an assistant message requesting tool calls
    tool_call_id: str | None = None  # present on a role="tool" message (the result)

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = 2000
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None

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
