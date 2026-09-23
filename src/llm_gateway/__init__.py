"""llm_gateway — a small, self-hosted multi-provider LLM gateway.

Use it in-process:

    from llm_gateway import complete, GatewayConfig
    text = await complete(system="...", prompt="...", config=GatewayConfig())

Token usage, structured output and prompt caching:

    from llm_gateway import complete_with_usage
    result = await complete_with_usage(
        system="...", prompt="...", output_schema=MyModel, cache_system=True
    )
    result.parsed, result.provider, result.usage.input_tokens

Tool-calling / ReAct-style agents:

    from llm_gateway import chat, GatewayConfig
    result = await chat(messages=[...], tools=[...], config=GatewayConfig())
    if result.tool_calls:
        ...

Or run it as an HTTP service (see `llm_gateway.service`) with an
OpenAI-compatible `/v1/chat/completions` API — including `tools`, so
LangGraph's prebuilt ReAct agent works against it with zero custom code,
just point `ChatOpenAI(base_url=...)` at it.
"""

from .breaker import reset as reset_circuit_breakers
from .chat import chat
from .config import GatewayConfig
from .errors import (
    AllProvidersExhaustedError,
    EmptyCompletionError,
    GatewayNotConfiguredError,
    InvalidOutputError,
    LLMDeadlineExceeded,
    PolicyViolationError,
    ProviderAuthError,
    UnsupportedCapabilityError,
)
from .pii import mask_pii
from .policy import BudgetCheck, ProviderMetadata, ProviderPolicy
from .pricing import ModelPrice
from .providers.base import ChatResult, StreamDelta, ToolCall, Usage
from .router import Completion, LLMError, complete, complete_with_usage
from .streaming import stream_chat

__all__ = [
    "ProviderPolicy",
    "ProviderMetadata",
    "ModelPrice",
    "BudgetCheck",
    "PolicyViolationError",
    "UnsupportedCapabilityError",
    "complete",
    "complete_with_usage",
    "Completion",
    "Usage",
    "chat",
    "stream_chat",
    "ChatResult",
    "StreamDelta",
    "ToolCall",
    "LLMError",
    "LLMDeadlineExceeded",
    "GatewayNotConfiguredError",
    "AllProvidersExhaustedError",
    "InvalidOutputError",
    "EmptyCompletionError",
    "ProviderAuthError",
    "GatewayConfig",
    "mask_pii",
    "reset_circuit_breakers",
]
