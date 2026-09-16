"""笔记本接口：查询 / 删除本人笔记（写入由 save_note 工具在五阶段管线内完成）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.db.models import User, UserNote

router = APIRouter(prefix="/notes", tags=["notes"])


class NoteOut(BaseModel):
    id: int
    category: str
    title: str
    content: str
    created_at: str


@router.get("", response_model=list[NoteOut])
async def list_notes(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[NoteOut]:
    rows = (await db.execute(
        select(UserNote)
        .where(UserNote.user_id == user.id)
        .order_by(UserNote.id.desc())
        .limit(100)
    )).scalars().all()
    return [NoteOut(id=n.id, category=n.category, title=n.title, content=n.content,
                    created_at=n.created_at.isoformat()) for n in rows]


@router.delete("/{note_id}")
async def delete_note(
    note_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    note = await db.get(UserNote, note_id)
    if note is None or note.user_id != user.id:  # 越权/不存在同 404（防探测）
        raise HTTPException(status.HTTP_404_NOT_FOUND, "笔记不存在")
    await db.delete(note)
    await db.commit()
    return {"ok": True}
