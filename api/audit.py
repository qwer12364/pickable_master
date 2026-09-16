from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin, get_db
from app.db.models import AuditRecord, User

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("")
async def list_audit(
    limit: int = Query(50, ge=1, le=200),
    decision: str | None = Query(None, pattern="^(approved|denied|error)$"),
    _admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    stmt = select(AuditRecord).order_by(AuditRecord.id.desc()).limit(limit)
    if decision:
        stmt = stmt.where(AuditRecord.decision == decision)
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "invocation_id": r.invocation_id,
            "user_id": r.user_id,
            "username": r.username,
            "tool": r.tool,
            "side_effect": r.side_effect,
            "decision": r.decision,
            "phase": r.phase,
            "args_summary": r.args_summary,
            "result_summary": r.result_summary,
            "duration_ms": r.duration_ms,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
