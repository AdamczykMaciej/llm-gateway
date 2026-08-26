from . import anthropic, groq, openai
from .base import ProviderResult

CALLS = {
    "anthropic": anthropic.call,
    "groq": groq.call,
    "openai": openai.call,
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

__all__ = ["ProviderResult", "CALLS", "CONFIGURED", "DEFAULT_MODEL"]
