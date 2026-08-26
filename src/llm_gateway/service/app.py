"""The FastAPI flavor: an OpenAI-compatible /v1/chat/completions API in
front of the same router.complete() the library exposes in-process."""

from fastapi import Depends, FastAPI, HTTPException

from ..config import GatewayConfig
from ..router import LLMError, complete
from .auth import require_api_key
from .schemas import ChatCompletionRequest, chat_completion_response


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    config = config or GatewayConfig()
    app = FastAPI(title="llm-gateway", version="0.1.0")
    auth_dep = require_api_key(config)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [
                {"id": "auto", "object": "model"},
                {"id": f"anthropic/{config.claude_model}", "object": "model"},
                {"id": f"groq/{config.groq_model}", "object": "model"},
                {"id": f"openai/{config.openai_model}", "object": "model"},
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(auth_dep)])
    async def chat_completions(body: ChatCompletionRequest) -> dict:
        try:
            system, prompt = body.as_system_and_prompt()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
            text = await complete(
                system=system,
                prompt=prompt,
                max_tokens=body.max_tokens,
                config=config,
                force_provider=force_provider,
            )
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return chat_completion_response(text=text, model=model or "auto")

    return app
