"""
Firebase App Check verification middleware.
Spec: docs/superpowers/specs/2026-03-18-identity-security-hardening-design.md section 5

Rollout modes (controlled by APP_CHECK_ENFORCE_MODE env var):
  - "log": Log missing/invalid tokens, never reject (default)
  - "auth": Enforce on POST /auth/* only
  - "all": Enforce on all non-exempt endpoints
"""
import logging

import firebase_admin.app_check
import firebase_admin.exceptions
from fastapi import Request, HTTPException

from src.core.config import settings

logger = logging.getLogger(__name__)

# Endpoints exempt from App Check — verified against main.py router prefixes:
#   auth -> /api/auth, push -> /api, whatsapp-meta -> /api/whatsapp-meta
APP_CHECK_EXEMPT_PATHS = {
    "/health",
    "/api/whatsapp-meta",          # Meta webhook (HMAC-verified)
    "/api/auth/verify-email",      # Email link click (no app context)
    "/api/register-token",         # Push token registration (service worker calls)
}


def _is_exempt(path: str) -> bool:
    """Check if path is exempt from App Check. Uses startswith for prefix matching."""
    for exempt in APP_CHECK_EXEMPT_PATHS:
        if path == exempt or path.startswith(exempt + "/"):
            return True
    return False


async def verify_app_check(request: Request):
    """FastAPI dependency: verify Firebase App Check token."""
    path = request.url.path

    if _is_exempt(path):
        return

    token = request.headers.get("X-Firebase-AppCheck")
    mode = settings.APP_CHECK_ENFORCE_MODE

    # Log-only mode (default)
    if mode == "log":
        if not token:
            logger.info("App Check: missing token for %s", path)
        return

    # Auth-only mode: only enforce on /auth/* POST endpoints
    if mode == "auth":
        if not (path.startswith("/api/auth/") and request.method == "POST"):
            return

    # Enforce: token required
    if not token:
        raise HTTPException(status_code=401, detail="Missing App Check token")

    try:
        firebase_admin.app_check.verify_token(token)
    except firebase_admin.exceptions.FirebaseError:
        raise HTTPException(status_code=401, detail="Invalid App Check token")
    except Exception as e:
        # Service error -> fail OPEN (flood safety app — don't block during emergencies)
        logger.error("App Check service error (allowing request): %s", e)
