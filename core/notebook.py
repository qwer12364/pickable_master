from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.harness.tools import SideEffectLevel, ToolContext, ToolError, ToolSpec
from app.db import get_sessionmaker
from app.db.models import UserNote

logger = logging.getLogger("pickleball.notebook")

NOTE_CATEGORIES = Literal["training", "match", "equipment", "other"]


class SaveNoteParams(BaseModel):
    category: NOTE_CATEGORIES = Field(..., description="分类：training=训练 match=比赛 equipment=装备 other=其他")
    title: str = Field(..., min_length=1, max_length=128, description="标题")
    content: str = Field(..., min_length=1, max_length=4000, description="正文（超 4000 字请精简）")


class ListNotesParams(BaseModel):
    category: NOTE_CATEGORIES | None = Field(None, description="按分类过滤；不填返回全部")


def make_notebook_tools() -> list[ToolSpec]:
    async def save_note(ctx: ToolContext, category: str, title: str, content: str) -> str:
        try:
            async with get_sessionmaker()() as db:
                note = UserNote(user_id=ctx.principal.user_id, category=category,
                                title=title, content=content)
                db.add(note)
                await db.commit()
        except Exception as exc:  # noqa: BLE001 降级：审计为 error，循环继续
            logger.warning("笔记保存失败 (user=%s): %s", ctx.principal.user_id, exc)
            raise ToolError(f"笔记保存失败: {exc}") from exc
        return (f"已保存到笔记本 #{note.id}（{category}）\n标题：{title}\n"
                f"摘要：{content[:80]}")

    async def list_notes(ctx: ToolContext, category: str | None = None) -> str:
        stmt = select(UserNote).where(UserNote.user_id == ctx.principal.user_id)
        if category:
            stmt = stmt.where(UserNote.category == category)
        stmt = stmt.order_by(UserNote.id.desc()).limit(20)
        async with get_sessionmaker()() as db:
            rows = (await db.execute(stmt)).scalars().all()
        if not rows:
            return "笔记本为空。"
        return "\n".join(
            f"{i + 1}. #{n.id} [{n.category}] {n.title}"
            f"（{n.created_at.strftime('%Y-%m-%d %H:%M')}）\n   {n.content[:60]}"
            for i, n in enumerate(rows)
        )

    return [
        ToolSpec(
            name="save_note",
            description=("把训练计划/比赛分析等产出保存到用户的个人笔记本（写数据库）。"
                         "仅应在用户明确要求或确认保存时调用；保存前先向用户征询同意。"),
            handler=save_note,
            params_model=SaveNoteParams,
            side_effect=SideEffectLevel.WRITE_LOCAL,
            permissions={"tools:write"},
        ),
        ToolSpec(
            name="list_notes",
            description="列出用户笔记本中最近的笔记（本人可见），供对话中回顾既往计划/分析。",
            handler=list_notes,
            params_model=ListNotesParams,
            side_effect=SideEffectLevel.READ_ONLY,
            permissions={"tools:read"},
            max_output_chars=4000,  # 20 条 × ~90 字符接近默认 2000 截断，放宽
        ),
    ]
