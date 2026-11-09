"""TypedDicts for the wire-shaped dicts that cross provider boundaries.

They describe shapes only: at runtime every one of these is a plain `dict`,
so callers that build or read the dicts untyped are unaffected."""

from typing import Any, Literal, NotRequired, TypedDict, cast

# --- OpenAI wire format (what callers pass in and streams are normalized to) ---


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class OpenAIToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class OpenAIMessage(TypedDict):
    """One chat message. `content` is a string, null, or a list of parts
    (`{"type": "text" | "image_url", ...}`)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: NotRequired[str | list[dict[str, Any]] | None]
    tool_calls: NotRequired[list[OpenAIToolCall]]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


class OpenAIToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: NotRequired[dict[str, Any]]


class OpenAITool(TypedDict):
    type: Literal["function"]
    function: OpenAIToolFunction


class ToolCallDeltaFunction(TypedDict, total=False):
    name: str
    arguments: str


class ToolCallDelta(TypedDict, total=False):
    """A streamed tool-call fragment. Only the first fragment of a call
    carries `id`, `type` and `function.name`."""

    index: int
    id: str
    type: Literal["function"]
    function: ToolCallDeltaFunction


# --- Anthropic Messages API request blocks ---


class AnthropicTextBlock(TypedDict):
    type: Literal["text"]
    text: str
    cache_control: NotRequired[dict[str, str]]


class AnthropicImageSource(TypedDict, total=False):
    type: Literal["base64", "url"]
    media_type: str
    data: str
    url: str


class AnthropicImageBlock(TypedDict):
    type: Literal["image"]
    source: AnthropicImageSource


class AnthropicToolUseBlock(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class AnthropicToolResultBlock(TypedDict):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Any


AnthropicBlock = (
    AnthropicTextBlock | AnthropicImageBlock | AnthropicToolUseBlock | AnthropicToolResultBlock
)


class AnthropicMessage(TypedDict):
    role: str
    content: str | list[AnthropicBlock]


class AnthropicTool(TypedDict):
    name: str
    description: str
    input_schema: dict[str, Any]


def as_dict(value: object) -> dict[str, Any] | None:
    """`value` when it is a dict, else None. Caller-supplied message parts are
    untrusted whatever their type hints say, so readers narrow them with this."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None
