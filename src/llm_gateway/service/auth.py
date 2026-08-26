"""Bearer API-key auth for the HTTP service.

Deliberately minimal for v1: a static, deploy-configured allow-list of keys,
constant-time compared. No per-key identity, usage metering, or rate limiting
— documented as a known v1 limitation, not built here.
"""

import hmac

from fastapi import Header, HTTPException

from ..config import GatewayConfig


def require_api_key(config: GatewayConfig):
    """Return a FastAPI dependency bound to the given config's allowed keys."""
    allowed = config.gateway_api_keys_list

    async def _check(authorization: str | None = Header(default=None)) -> None:
        if not allowed:
            # No keys configured — the operator has deliberately left this open
            # (e.g. local dev). Documented, not a silent default in production
            # (README instructs setting GATEWAY_API_KEYS before deploying).
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer API key")
        supplied = authorization.removeprefix("Bearer ")
        if not any(hmac.compare_digest(k, supplied) for k in allowed):
            raise HTTPException(status_code=401, detail="Invalid API key")

    return _check
