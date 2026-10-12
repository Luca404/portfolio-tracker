"""Public liveness endpoint: no authentication, storage or market-data work."""

import os

from fastapi import APIRouter, Response

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health(response: Response) -> dict[str, str]:
    response.headers["Cache-Control"] = "no-store"
    revision = os.environ.get("RENDER_GIT_COMMIT")
    if revision:
        response.headers["X-Backend-Revision"] = revision
    return {"status": "ok"}
