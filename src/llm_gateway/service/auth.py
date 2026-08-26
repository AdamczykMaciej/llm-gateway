"""Bearer API-key auth for the HTTP service.

Deliberately minimal for v1: a static, deploy-configured allow-list of keys,
constant-time compared. No per-key quotas/identity beyond what's needed to
key the rate limiter (see rate_limit.py) off which key was used.
"""

import hmac

from fastapi import Header, HTTPException

from ..config import GatewayConfig

# Returned when no keys are configured — the operator has deliberately left
# the service open (e.g. local dev). Documented, not a silent default in
# production (README instructs setting GATEWAY_API_KEYS before deploying).
# Every unauthenticated caller shares this one rate-limit bucket.
ANONYMOUS = "anonymous"


def require_api_key(config: GatewayConfig):
    """Return a FastAPI dependency bound to the given config's allowed keys.

    Resolves to the caller's own key string (or ANONYMOUS) so it can be
    used downstream to key the per-caller rate limiter.
    """
    allowed = config.gateway_api_keys_list

    async def _check(authorization: str | None = Header(default=None)) -> str:
        if not allowed:
            return ANONYMOUS
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer API key")
        supplied = authorization.removeprefix("Bearer ")
        for key in allowed:
            if hmac.compare_digest(key, supplied):
                return supplied
        raise HTTPException(status_code=401, detail="Invalid API key")

    return _check
