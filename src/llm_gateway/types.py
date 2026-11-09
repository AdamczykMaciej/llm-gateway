"""Precise request types for `chat()` / `stream_chat()`.

These are `TypedDict`s describing the OpenAI-wire shapes the gateway accepts.
They are typing-only: nothing validates or converts at runtime, and plain
`dict`s are still accepted everywhere (`Message | dict[str, Any]`), so adopting
them is optional and backward compatible.
"""

from typing import Any, Literal, NotRequired, Required, TypedDict


class TextPart(TypedDict):
    """A text content part: `{"type": "text", "text": "..."}`."""

    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    """The `image_url` object of an image part: a URL or a `data:` URI."""

    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ImagePart(TypedDict):
    """An image content part: `{"type": "image_url", "image_url": {"url": ...}}`."""

    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = TextPart | ImagePart
"""One element of a multi-part message `content` list."""


class FunctionCall(TypedDict):
    """The function of an assistant tool call; `arguments` is a JSON string."""

    name: str
    arguments: str


class ToolCallMessagePart(TypedDict):
    """A tool call inside an assistant message (`tool_calls[i]`)."""

    id: str
    type: Literal["function"]
    function: FunctionCall


class SystemMessage(TypedDict):
    """A `role: "system"` message."""

    role: Required[Literal["system"]]
    content: Required[str | list[TextPart]]
    name: NotRequired[str]


class UserMessage(TypedDict):
    """A `role: "user"` message; `content` is text or a list of parts."""

    role: Required[Literal["user"]]
    content: Required[str | list[ContentPart]]
    name: NotRequired[str]


class AssistantMessage(TypedDict):
    """A `role: "assistant"` message. `content` may be `None` when the turn is
    only `tool_calls`."""

    role: Required[Literal["assistant"]]
    content: NotRequired[str | list[TextPart] | None]
    tool_calls: NotRequired[list[ToolCallMessagePart]]
    name: NotRequired[str]


class ToolMessage(TypedDict):
    """A `role: "tool"` message answering the assistant tool call whose id is
    `tool_call_id`."""

    role: Required[Literal["tool"]]
    tool_call_id: Required[str]
    content: Required[str | list[TextPart]]


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage
"""Any chat message; accepted by `chat()` / `stream_chat()` next to plain dicts."""


class FunctionSpec(TypedDict):
    """A function tool's definition; `parameters` is a JSON Schema object."""

    name: Required[str]
    description: NotRequired[str]
    parameters: NotRequired[dict[str, Any]]
    strict: NotRequired[bool]


class ToolSpec(TypedDict):
    """One entry of `tools=`: `{"type": "function", "function": {...}}`."""

    type: Literal["function"]
    function: FunctionSpec


class NamedFunction(TypedDict):
    """The function a `NamedToolChoice` forces."""

    name: str


class NamedToolChoice(TypedDict):
    """Force one specific tool: `{"type": "function", "function": {"name": ...}}`."""

    type: Literal["function"]
    function: NamedFunction


ToolChoice = Literal["auto", "none", "required"] | NamedToolChoice
"""`tool_choice=`: a mode string or a forced tool."""

__all__ = [
    "AssistantMessage",
    "ContentPart",
    "FunctionCall",
    "FunctionSpec",
    "ImagePart",
    "ImageUrl",
    "Message",
    "NamedFunction",
    "NamedToolChoice",
    "SystemMessage",
    "TextPart",
    "ToolCallMessagePart",
    "ToolChoice",
    "ToolMessage",
    "ToolSpec",
    "UserMessage",
]
