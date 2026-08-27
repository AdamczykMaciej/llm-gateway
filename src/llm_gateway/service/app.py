"""The FastAPI flavor: an OpenAI-compatible /v1/chat/completions API in
front of the same chat() engine the library exposes in-process."""

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import breaker
from ..chat import chat as chat_engine
from ..config import GatewayConfig
from ..providers import CONFIGURED
from ..router import LLMError
from .rate_limit import enforce_rate_limit
from .schemas import ChatCompletionRequest, chat_completion_response_from_result

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


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    config = config or GatewayConfig()
    app = FastAPI(title="llm-gateway", version="0.1.0")
    rate_limit_dep = enforce_rate_limit(config)

    @app.exception_handler(HTTPException)
    async def openai_shaped_error(request: Request, exc: HTTPException) -> JSONResponse:
        error_type = _OPENAI_ERROR_TYPES.get(exc.status_code, "api_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": error_type, "code": exc.status_code}},
            headers=exc.headers,
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
    async def models() -> dict:
        provider_models = {
            "anthropic": config.claude_model,
            "groq": config.groq_model,
            "openai": config.openai_model,
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

    @app.post("/v1/chat/completions", dependencies=[Depends(rate_limit_dep)])
    async def chat_completions(body: ChatCompletionRequest) -> dict:
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
        total_chars = sum(len(m.content or "") for m in body.messages)
        if config.max_prompt_chars > 0 and total_chars > config.max_prompt_chars:
            raise HTTPException(
                status_code=400,
                detail=f"Total message content ({total_chars} chars) exceeds the "
                f"configured ceiling of {config.max_prompt_chars}",
            )

        force_provider = None
        model = body.model
        if model and model != "auto" and "/" in model:
            provider_name, _, upstream_model = model.partition("/")
            if provider_name not in ("anthropic", "groq", "openai"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown provider '{provider_name}' in model '{model}'",
                )
            force_provider = provider_name
            model = upstream_model

        try:
            result = await chat_engine(
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
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return chat_completion_response_from_result(result)

    return app
