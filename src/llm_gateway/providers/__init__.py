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

__all__ = ["ProviderResult", "CALLS", "CONFIGURED"]
