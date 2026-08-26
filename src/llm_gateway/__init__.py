"""llm_gateway — a small, self-hosted multi-provider LLM gateway.

Use it in-process:

    from llm_gateway import complete, GatewayConfig
    text = await complete(system="...", prompt="...", config=GatewayConfig())

Or run it as an HTTP service (see `llm_gateway.service`) with an
OpenAI-compatible `/v1/chat/completions` API.
"""

from .breaker import reset as reset_circuit_breakers
from .config import GatewayConfig
from .pii import mask_pii
from .router import LLMError, complete

__all__ = [
    "complete",
    "LLMError",
    "GatewayConfig",
    "mask_pii",
    "reset_circuit_breakers",
]
