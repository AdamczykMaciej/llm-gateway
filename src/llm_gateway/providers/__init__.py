from . import anthropic, groq, openai
from .base import ChatResult, ProviderResult, ToolCall

CALLS = {
    "anthropic": anthropic.call,
    "groq": groq.call,
    "openai": openai.call,
}

CHAT_CALLS = {
    "anthropic": anthropic.chat,
    "groq": groq.chat,
    "openai": openai.chat,
}

CONFIGURED = {
    "anthropic": anthropic.configured,
    "groq": groq.configured,
    "openai": openai.configured,
}

DEFAULT_MODEL = {
    "anthropic": anthropic.default_model,
    "groq": groq.default_model,
    "openai": openai.default_model,
}

__all__ = [
    "ProviderResult",
    "ChatResult",
    "ToolCall",
    "CALLS",
    "CHAT_CALLS",
    "CONFIGURED",
    "DEFAULT_MODEL",
]
