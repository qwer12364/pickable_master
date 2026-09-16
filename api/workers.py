from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_current_admin
from app.db.models import User

router = APIRouter(prefix="/workers", tags=["cluster"])


@router.get("")
async def list_workers(
    request: Request,
    _admin: User = Depends(get_current_admin),
) -> dict:
    redis = request.app.state.redis  # decode_responses=True，key 已是 str
    keys = await redis.keys("pickleball:worker:*")
    workers = []
    for key in keys:
        alive = await redis.get(key)
        if alive:
            workers.append(str(key).rsplit(":", 1)[-1])
    return {"workers": sorted(workers), "count": len(workers)}


@router.get("/queue")
async def queue_stats(
    request: Request,
    _admin: User = Depends(get_current_admin),
) -> dict:
    queue = request.app.state.queue
    return {
        "pending": await queue.pending_count(),
    }
