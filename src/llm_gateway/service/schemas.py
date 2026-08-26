"""OpenAI-compatible request/response shapes for /v1/chat/completions.

Deliberately minimal — this is not a full OpenAI API reimplementation, just
enough of the wire format that any OpenAI-SDK-compatible client can point at
this service by changing its base_url.
"""

import time
import uuid

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = 2000

    def as_system_and_prompt(self) -> tuple[str, str]:
        system = "\n".join(m.content for m in self.messages if m.role == "system")
        user_messages = [m.content for m in self.messages if m.role == "user"]
        if not user_messages:
            raise ValueError("messages must include at least one role='user' entry")
        return system, user_messages[-1]


def chat_completion_response(*, text: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        # complete() intentionally returns text only, not token counts — usage
        # is captured in the OTel span (gen_ai.usage.*) instead, not surfaced
        # here. An empty dict is valid per the OpenAI response shape.
        "usage": {},
    }
