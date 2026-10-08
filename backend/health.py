"""Public liveness endpoint: no authentication, storage or market-data work."""

from fastapi import APIRouter, Response

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health(response: Response) -> dict[str, str]:
    response.headers["Cache-Control"] = "no-store"
    return {"status": "ok"}
