from . import anthropic, azure, groq, openai, vertex
from .base import ChatResult, ProviderResult, StreamDelta, ToolCall, Usage

CALLS = {
    "anthropic": anthropic.call,
    "azure": azure.call,
    "groq": groq.call,
    "openai": openai.call,
    "vertex": vertex.call,
}

CHAT_CALLS = {
    "anthropic": anthropic.chat,
    "azure": azure.chat,
    "groq": groq.chat,
    "openai": openai.chat,
    "vertex": vertex.chat,
}

STREAM_CALLS = {
    "anthropic": anthropic.stream_chat,
    "azure": azure.stream_chat,
    "groq": groq.stream_chat,
    "openai": openai.stream_chat,
    "vertex": vertex.stream_chat,
}

CONFIGURED = {
    "anthropic": anthropic.configured,
    "azure": azure.configured,
    "groq": groq.configured,
    "openai": openai.configured,
    "vertex": vertex.configured,
}

DEFAULT_MODEL = {
    "anthropic": anthropic.default_model,
    "azure": azure.default_model,
    "groq": groq.default_model,
    "openai": openai.default_model,
    "vertex": vertex.default_model,
}

__all__ = [
    "ProviderResult",
    "ChatResult",
    "StreamDelta",
    "ToolCall",
    "Usage",
    "CALLS",
    "CHAT_CALLS",
    "STREAM_CALLS",
    "CONFIGURED",
    "DEFAULT_MODEL",
]
