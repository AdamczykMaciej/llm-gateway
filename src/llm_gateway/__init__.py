"""llm_gateway — a small, self-hosted multi-provider LLM gateway.

Use it in-process:

    from llm_gateway import complete, GatewayConfig
    text = await complete(system="...", prompt="...", config=GatewayConfig())

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
from .pii import mask_pii
from .providers.base import ChatResult, ToolCall
from .router import LLMError, complete

__all__ = [
    "complete",
    "chat",
    "ChatResult",
    "ToolCall",
    "LLMError",
    "GatewayConfig",
    "mask_pii",
    "reset_circuit_breakers",
]
