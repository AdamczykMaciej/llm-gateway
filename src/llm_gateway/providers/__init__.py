from . import (
    anthropic,
    azure,
    gemini,
    groq,
    huggingface,
    mistral,
    openai,
    openai_compat,
    openrouter,
    vertex,
)
from .base import ChatResult, ProviderResult, StreamDelta, ToolCall, Usage

CALLS = {
    "anthropic": anthropic.call,
    "azure": azure.call,
    "gemini": gemini.call,
    "groq": groq.call,
    "huggingface": huggingface.call,
    "mistral": mistral.call,
    "openai": openai.call,
    "openai_compat": openai_compat.call,
    "openrouter": openrouter.call,
    "vertex": vertex.call,
}

CHAT_CALLS = {
    "anthropic": anthropic.chat,
    "azure": azure.chat,
    "gemini": gemini.chat,
    "groq": groq.chat,
    "huggingface": huggingface.chat,
    "mistral": mistral.chat,
    "openai": openai.chat,
    "openai_compat": openai_compat.chat,
    "openrouter": openrouter.chat,
    "vertex": vertex.chat,
}

STREAM_CALLS = {
    "anthropic": anthropic.stream_chat,
    "azure": azure.stream_chat,
    "gemini": gemini.stream_chat,
    "groq": groq.stream_chat,
    "huggingface": huggingface.stream_chat,
    "mistral": mistral.stream_chat,
    "openai": openai.stream_chat,
    "openai_compat": openai_compat.stream_chat,
    "openrouter": openrouter.stream_chat,
    "vertex": vertex.stream_chat,
}

CONFIGURED = {
    "anthropic": anthropic.configured,
    "azure": azure.configured,
    "gemini": gemini.configured,
    "groq": groq.configured,
    "huggingface": huggingface.configured,
    "mistral": mistral.configured,
    "openai": openai.configured,
    "openai_compat": openai_compat.configured,
    "openrouter": openrouter.configured,
    "vertex": vertex.configured,
}

DEFAULT_MODEL = {
    "anthropic": anthropic.default_model,
    "azure": azure.default_model,
    "gemini": gemini.default_model,
    "groq": groq.default_model,
    "huggingface": huggingface.default_model,
    "mistral": mistral.default_model,
    "openai": openai.default_model,
    "openai_compat": openai_compat.default_model,
    "openrouter": openrouter.default_model,
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
