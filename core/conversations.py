from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.harness.audit import AuditStore, InvocationAudit
from app.core.harness.tools import SideEffectLevel
from app.db.models import Conversation, Message, User
from app.distributed.guard import conv_is_busy

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str = Field("新对话", max_length=128)


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    meta: dict
    created_at: str


@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ConversationOut]:
    rows = (await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
        .limit(100)
    )).scalars().all()
    return [ConversationOut(id=c.id, title=c.title,
                            created_at=c.created_at.isoformat()) for c in rows]


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationOut:
    conv = Conversation(user_id=user.id, title=body.title[:128])
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return ConversationOut(id=conv.id, title=conv.title,
                           created_at=conv.created_at.isoformat())


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[MessageOut]:
    conv = await db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在")
    rows = (await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id.asc())
    )).scalars().all()
    return [MessageOut(id=m.id, role=m.role, content=m.content,
                       meta=m.meta or {}, created_at=m.created_at.isoformat())
            for m in rows]


class RollbackRequest(BaseModel):
    message_id: int = Field(..., gt=0)


_audit = AuditStore()  # 模块级：环形缓冲可查 + best-effort 落库


@router.post("/{conversation_id}/rollback", response_model=list[MessageOut])
async def rollback_conversation(
    conversation_id: uuid.UUID,
    body: RollbackRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[MessageOut]:
    """回退会话到锚点消息：硬删除锚点之后的所有消息，锚点本身保留。

    不回自动重答——用户的下一轮消息从截断后的历史继续。
    """
    conv = await db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在")

    # 执行中守卫：会话正有任务在跑时禁止回退（Redis 不可用降级放行）
    if await conv_is_busy(request.app.state.redis, conversation_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "会话正在执行中，请稍后再试")

    anchor = (await db.execute(
        select(Message.id).where(Message.conversation_id == conversation_id,
                                 Message.id == body.message_id)
    )).scalar_one_or_none()
    if anchor is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "消息不存在")

    t0 = time.perf_counter()
    deleted = (await db.execute(
        delete(Message).where(Message.conversation_id == conversation_id,
                              Message.id > body.message_id)
    )).rowcount
    await db.commit()

    rows = (await db.execute(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.id.asc())
    )).scalars().all()

    await _audit.record(InvocationAudit(
        user_id=user.id,
        username=user.username,
        tool="conversation:rollback",
        side_effect=int(SideEffectLevel.WRITE_LOCAL),
        decision="approved",
        phase="execute",
        args_summary=f"conv={conversation_id} anchor={body.message_id}",
        result_summary=f"deleted={deleted} remaining={len(rows)}",
        duration_ms=int((time.perf_counter() - t0) * 1000),
    ))
    return [MessageOut(id=m.id, role=m.role, content=m.content,
                       meta=m.meta or {}, created_at=m.created_at.isoformat())
            for m in rows]
