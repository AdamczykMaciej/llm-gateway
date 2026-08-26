"""Deployable entrypoint: `uvicorn llm_gateway.service.main:app` or `llm-gateway`."""

import os

from ..config import GatewayConfig
from .app import create_app

app = create_app(GatewayConfig())


def main() -> None:
    import uvicorn

    uvicorn.run(
        "llm_gateway.service.main:app",
        host="0.0.0.0",  # noqa: S104 — intentional: this is the container's public bind address
        port=int(os.getenv("PORT", "8080")),
    )


if __name__ == "__main__":
    main()
