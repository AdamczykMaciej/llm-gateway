"""The FastAPI flavor: an OpenAI-compatible /v1/chat/completions API in
front of the same chat()/stream_chat() engines the library exposes
in-process."""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .. import breaker
from ..chat import chat as chat_engine
from ..config import GatewayConfig
from ..errors import InvalidRequestError, PolicyViolationError, redact_secrets
from ..providers import CONFIGURED
from ..providers import azure as azure_provider
from ..providers import gemini as gemini_provider
from ..providers import vertex as vertex_provider
from ..router import LLMError
from ..streaming import stream_chat as stream_engine
from . import usage as usage_store
from .auth import require_api_key
from .body_limit import BodyLimitMiddleware
from .rate_limit import enforce_rate_limit
from .schemas import (
    SSE_DONE,
    ChatCompletionRequest,
    chat_completion_response_from_result,
    sse_error_event,
    stream_chunk_sse,
)

# HTTP status -> OpenAI error "type" string, matching what the openai-python
# SDK (and therefore LangChain's ChatOpenAI) knows how to parse for clean
# error messages and its own retry-on-rate-limit logic.
_OPENAI_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "invalid_request_error",
    429: "rate_limit_error",
    503: "service_unavailable_error",
}

logger = logging.getLogger("llm_gateway")


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    config = config or GatewayConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        # Entra and managed-identity credentials hold their own HTTP sessions.
        await azure_provider.aclose()
        await gemini_provider.aclose()
        await vertex_provider.aclose()

    # With keys configured, the interactive docs and the schema are not public.
    docs_kwargs: dict[str, Any] = (
        dict(docs_url=None, redoc_url=None, openapi_url=None)
        if config.gateway_api_keys_list
        else {}
    )
    app = FastAPI(title="llm-gateway", version="0.9.0", lifespan=lifespan, **docs_kwargs)
    if config.max_request_bytes > 0:
        app.add_middleware(BodyLimitMiddleware, max_bytes=config.max_request_bytes)
    rate_limit_dep = enforce_rate_limit(config)
    auth_dep = require_api_key(config)

    @app.exception_handler(HTTPException)
    async def openai_shaped_error(request: Request, exc: HTTPException) -> JSONResponse:
        error_type = _OPENAI_ERROR_TYPES.get(exc.status_code, "api_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": error_type, "code": exc.status_code}},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default body echoes each rejected `input` back, i.e. the
        # caller's prompt text; report only where and what, never the value.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', 'invalid')}"
            for err in exc.errors()[:20]
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": f"Invalid request: {problems}",
                    "type": "invalid_request_error",
                    "code": 422,
                }
            },
        )

    @app.get("/health")
    @app.get("/_internal/healthz")
    async def healthz() -> dict:
        # NOT /healthz: that exact path is reserved platform-wide on Cloud
        # Run (returns a Google edge 404 to external callers regardless of
        # this app's own routes or probe config — a known, documented
        # gotcha, not something scoped to this service). /health is the
        # public endpoint; the Cloud Run startup probe targets
        # /_internal/healthz (see infra/terraform/cloud_run.tf).
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(api_key: str = Depends(auth_dep)) -> dict:
        provider_models = {
            "anthropic": config.claude_model,
            "azure": config.azure_model,
            "groq": config.groq_model,
            "openai": config.openai_model,
            "vertex": config.vertex_model,
            "gemini": config.gemini_model,
            "huggingface": config.huggingface_model,
            "mistral": config.mistral_model,
            "openrouter": config.openrouter_model,
            "openai_compat": config.openai_compat_model,
        }
        data = [
            {
                "id": "auto",
                "object": "model",
                "available": any(
                    CONFIGURED[p](config) and not breaker.is_open(p) for p in provider_models
                ),
            }
        ]
        for provider, model_name in provider_models.items():
            configured = CONFIGURED[provider](config)
            data.append(
                {
                    "id": f"{provider}/{model_name}",
                    "object": "model",
                    "configured": configured,
                    "available": configured and not breaker.is_open(provider),
                }
            )
        return {"object": "list", "data": data}

    @app.get("/v1/usage")
    async def get_usage(api_key: str = Depends(auth_dep)) -> dict:
        """Cumulative usage for the calling key only — a key can't see any
        other key's usage. In-process and ephemeral (see usage.py); not a
        billing system, just cost visibility."""
        return usage_store.get(api_key)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest, api_key: str = Depends(rate_limit_dep)
    ) -> Any:  # dict, or a StreamingResponse when body.stream
        if not any(m.role != "system" for m in body.messages):
            raise HTTPException(
                status_code=400,
                detail="messages must include at least one non-system entry",
            )

        if config.max_tokens_ceiling > 0 and body.max_tokens > config.max_tokens_ceiling:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens {body.max_tokens} exceeds the configured ceiling "
                f"of {config.max_tokens_ceiling}",
            )
        total_chars = sum(m.text_length() for m in body.messages) + body.tools_length()
        if config.max_prompt_chars > 0 and total_chars > config.max_prompt_chars:
            raise HTTPException(
                status_code=400,
                detail=f"Total message content ({total_chars} chars) exceeds the "
                f"configured ceiling of {config.max_prompt_chars}",
            )
        total_image_bytes = sum(m.image_bytes() for m in body.messages)
        if config.max_image_bytes > 0 and total_image_bytes > config.max_image_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Total image content (~{total_image_bytes} bytes) exceeds the "
                f"configured ceiling of {config.max_image_bytes} bytes",
            )

        force_provider = None
        model = body.model
        if model and model != "auto" and "/" in model:
            provider_name, _, upstream_model = model.partition("/")
            # CONFIGURED (providers/__init__.py) is the live provider registry,
            # not a hand-maintained list here — so a new provider (vertex,
            # mistral, openrouter, openai_compat...) is force-selectable over
            # HTTP the moment it's registered, with no separate list to forget.
            if provider_name not in CONFIGURED:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown provider '{provider_name}' in model '{model}'",
                )
            force_provider = provider_name
            model = upstream_model

        # Merged with config.policy inside the engines, where it can only
        # narrow the server's global policy, never loosen it.
        try:
            policy = body.policy.to_policy() if body.policy is not None else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid policy: {exc}") from exc

        engine_kwargs: dict[str, Any] = dict(
            messages=body.as_message_dicts(),
            tools=body.tools,
            tool_choice=body.tool_choice,
            max_tokens=body.max_tokens,
            config=config,
            force_provider=force_provider,
            model=model if force_provider else None,
            sampling=body.sampling_dict(),
            response_format=body.response_format,
        )
        if policy is not None:
            engine_kwargs["policy"] = policy

        if body.stream:

            async def _sse():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                created = int(time.time())
                input_tokens = output_tokens = 0
                try:
                    # aclosing: if the client disconnects, the engine (and with
                    # it the upstream provider request) is closed right away,
                    # not whenever the garbage collector gets to it.
                    async with aclosing(stream_engine(**engine_kwargs)) as deltas:
                        async for delta in deltas:
                            if delta.usage:
                                input_tokens, output_tokens = delta.usage
                            for line in stream_chunk_sse(delta, chunk_id=chunk_id, created=created):
                                yield line
                except (PolicyViolationError, InvalidRequestError) as exc:
                    yield sse_error_event(str(exc), error_type="invalid_request_error", code=400)
                except LLMError as exc:
                    # Headers (200 + SSE) are already committed by the time a
                    # provider failure is known — can't change status code at
                    # this point, so emit an error event instead of crashing
                    # the connection. See schemas.sse_error_event.
                    yield sse_error_event(str(exc))
                except Exception as exc:  # noqa: BLE001 — raw provider/network error mid-stream
                    # Not an LLMError: a failure after the first chunk. End the
                    # stream cleanly with a generic event; the detail (which
                    # can carry hosts or request ids) stays in the log.
                    logger.warning(
                        "llm_gateway service: stream failed mid-response: %s: %s",
                        type(exc).__name__,
                        redact_secrets(str(exc), config),
                    )
                    yield sse_error_event("The upstream provider failed mid-response.")
                finally:
                    if input_tokens or output_tokens:
                        usage_store.record(api_key, input_tokens, output_tokens)
                yield SSE_DONE

            return StreamingResponse(_sse(), media_type="text/event-stream")

        try:
            result = await chat_engine(**engine_kwargs)
        except (PolicyViolationError, InvalidRequestError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        usage_store.record(api_key, result.input_tokens, result.output_tokens)
        return chat_completion_response_from_result(result)

    return app
