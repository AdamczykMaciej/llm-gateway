"""Which features a provider/model can serve, checked before a call.

Each entry is `True` (supported), `False` (known unsupported) or `None`
(unknown). routing.py always skips a provider whose model is known not to
support a feature the request uses, instead of spending a call on the 400
and failing over. With `require_parameters`, it also skips unknown support
and, for `output_schema`, models that only offer JSON mode instead of strict
schema enforcement.

Only facts from the vendors' docs are recorded (checked 2026-09-14); every
model not listed is unknown. `azure` models are deployment names chosen by
the operator: only a deployment named after gpt-oss is recognized, and every
other deployment is unknown.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Literal

from .providers.groq import STRICT_JSON_SCHEMA_MODELS

StructuredOutput = Literal["strict", "json_mode", "unsupported"] | None

FEATURES = ("structured_output", "tools", "images", "streaming")
_LABELS = {
    "structured_output": "structured output",
    "tools": "tool calling",
    "images": "image input",
    "streaming": "streaming",
}


@dataclass(frozen=True)
class ModelCapabilities:
    structured_output: StructuredOutput = None
    tools: bool | None = None
    images: bool | None = None
    streaming: bool | None = None


UNKNOWN = ModelCapabilities()

# Anthropic structured outputs, from the Claude API docs' supported-model
# list (https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
# Other Claude models are unknown, not unsupported.
_ANTHROPIC_STRICT_OUTPUT_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "claude-opus-4-5",
    "claude-opus-4-1",
)

# OpenAI strict json_schema (https://platform.openai.com/docs/guides/structured-outputs):
# gpt-4o-mini and gpt-4o from 2024-08-06, and the later gpt-4.1 / gpt-5
# families, which also take tools and image input.
_OPENAI_CURRENT_MODELS = ("gpt-4o", "gpt-4.1", "gpt-5")
_OPENAI_NON_CHAT_MARKERS = ("audio", "realtime", "search", "transcribe", "tts")
_OPENAI_PRE_STRICT_MODELS = ("gpt-4o-2024-05-13",)

# Groq text-only models (https://console.groq.com/docs/models lists no image
# input for them); the gateway's JSON-mode fallback covers `output_schema`
# on every Groq model without strict support.
_GROQ_TEXT_ONLY_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-")
_GROQ_TOOL_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-")


# Azure lists gpt-oss-120b with the Chat Completions API, streaming, function
# calling and structured outputs
# (https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure),
# and providers/azure.py always requests strict json_schema. gpt-oss takes no
# image input. Recognized only when the deployment is named after the model.
_AZURE_GPT_OSS_DEPLOYMENTS = ("gpt-oss-",)


def _azure(model: str) -> ModelCapabilities:
    if not model.startswith(_AZURE_GPT_OSS_DEPLOYMENTS):
        return UNKNOWN
    return ModelCapabilities(structured_output="strict", tools=True, images=False, streaming=True)


def _anthropic(model: str) -> ModelCapabilities:
    if not model.startswith("claude-"):
        return UNKNOWN
    return ModelCapabilities(
        structured_output="strict" if model.startswith(_ANTHROPIC_STRICT_OUTPUT_MODELS) else None,
        tools=True,
        images=True,
        streaming=True,
    )


def _vertex(model: str, *, structured_outputs: bool) -> ModelCapabilities:
    """Claude on Vertex AI: the same answers as `_anthropic` for the same model
    family. Google lists function calling, prompt caching and streaming for
    Claude Haiku 4.5, and structured outputs for every Claude 4.5 and later
    model
    (https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/structured-outputs),
    but organizations deny structured outputs until the policy
    `constraints/vertexai.allowedPartnerModelFeatures` allows them. So
    structured output is "unsupported" (known unsupported, always skipped)
    unless `GatewayConfig.vertex_structured_outputs` says the policy allows it.
    Vertex ids carry a `@version` suffix, which the prefix match covers. Image
    input is base64 only: Vertex doesn't accept URL image sources."""
    capabilities = _anthropic(model)
    if structured_outputs:
        return capabilities
    return replace(capabilities, structured_output="unsupported")


def _openai(model: str) -> ModelCapabilities:
    current = model.startswith(_OPENAI_CURRENT_MODELS) and not any(
        marker in model for marker in _OPENAI_NON_CHAT_MARKERS
    )
    if not current:
        return ModelCapabilities(streaming=True)
    strict = not model.startswith(_OPENAI_PRE_STRICT_MODELS)
    return ModelCapabilities(
        structured_output="strict" if strict else None,
        tools=True,
        images=True,
        streaming=True,
    )


def _groq(model: str) -> ModelCapabilities:
    return ModelCapabilities(
        structured_output="strict" if model.startswith(STRICT_JSON_SCHEMA_MODELS) else "json_mode",
        tools=True if model.startswith(_GROQ_TOOL_MODELS) else None,
        images=False if model.startswith(_GROQ_TEXT_ONLY_MODELS) else None,
        streaming=True,
    )


_TABLES: dict[str, Callable[[str], ModelCapabilities]] = {
    "anthropic": _anthropic,
    "azure": _azure,
    "openai": _openai,
    "groq": _groq,
}


def capabilities_for(
    provider: str, model: str, *, vertex_structured_outputs: bool = False
) -> ModelCapabilities:
    """`vertex_structured_outputs` is `GatewayConfig.vertex_structured_outputs`."""
    if provider == "vertex":
        return _vertex(model, structured_outputs=vertex_structured_outputs)
    table = _TABLES.get(provider)
    return table(model) if table else UNKNOWN


def missing_capabilities(
    provider: str,
    model: str,
    required: Iterable[str],
    *,
    require_parameters: bool,
    vertex_structured_outputs: bool = False,
) -> list[str]:
    """Why `provider`'s `model` can't serve the `required` features (empty
    when it can)."""
    capabilities = capabilities_for(
        provider, model, vertex_structured_outputs=vertex_structured_outputs
    )
    missing = []
    for feature in required:
        support = getattr(capabilities, feature)
        label = _LABELS[feature]
        if support is False or support == "unsupported":
            missing.append(f"model {model} does not support {label}")
        elif require_parameters and support is None:
            missing.append(f"{label} support is unknown for model {model} (require_parameters)")
        elif require_parameters and support == "json_mode":
            missing.append(
                f"model {model} has only JSON mode, not strict {label} (require_parameters)"
            )
    return missing
