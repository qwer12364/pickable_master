from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np
from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.llm import ChatMessage
from app.core.retrieval.hashing import HashingEmbedder
from app.db import get_sessionmaker
from app.db.models import MemoryNote

logger = logging.getLogger("pickleball.memory.reflective")

# 相关性门槛：哈希向量余弦低于该值视为噪声（与 HybridRetriever.VEC_FLOOR 同口径）
SIM_FLOOR = 0.05
NOTE_MAX_CHARS = 80        # 单条笔记长度上限
WRITE_MAX_TOKENS = 400     # 提炼输出很小，收紧 max_tokens 控制成本与延迟
MAX_NOTES_PER_TURN = 3     # 每轮最多写 3 条

WRITE_PROMPT = (
    "你是对话记忆提炼器。根据用户问题与助手回答，提炼值得长期记住的用户信息，"
    "写入不超过 3 条反思笔记。\n"
    "只记跨对话仍有价值的事实：用户水平/偏好/装备/目标/身体条件/已确认的结论；"
    "不记本次问答本身的过程细节。\n"
    "每条不超过 60 字，用第三人称，如「用户是新手，使用中拍面碳纤维球拍」。\n"
    "没有值得记的内容时输出空数组。\n"
    "只输出一个 JSON 对象，不要输出任何其它内容：\n"
    '{"notes": ["…"]}'
)


def parse_notes(content: str | None) -> list[str]:
    """从 LLM 输出提取笔记数组；畸形输出返回空列表（本轮不写）。"""
    if not content:
        return []
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    notes = data.get("notes")
    if not isinstance(notes, list):
        return []
    out: list[str] = []
    for n in notes:
        if not isinstance(n, str):  # 非字符串元素视为畸形输出，跳过
            continue
        text = n.strip()
        if text and text not in out:
            out.append(text[:NOTE_MAX_CHARS])
        if len(out) >= MAX_NOTES_PER_TURN:
            break
    return out


class ReflectiveMemory:
    """ReMe 反思记忆：write（LLM 提炼落库）/ read（哈希向量排序取回）。
    """

    def __init__(self, llm: Any, *, settings: Settings | None = None) -> None:
        self._llm = llm
        self._settings = settings or get_settings()
        self._embedder = HashingEmbedder(dim=self._settings.embedding_dim)
    async def read(self, user_id: int, query: str, *,
                   top_k: int | None = None) -> list[str]:
        """取回与 query 最相关的 top_k 条笔记（按余弦相似度降序）。"""
        try:
            notes = await self._load_notes(user_id)
        except Exception as exc:  # noqa: BLE001 读失败降级为无记忆
            logger.warning("反思记忆读取失败 (user=%s): %s", user_id, exc)
            return []
        if not notes:
            return []
        k = top_k or self._settings.memory_reflective_top_k
        if k <= 0:
            return []
        q_vec = self._embedder.embed(query)
        mat = np.vstack([self._embedder.embed(n) for n in notes])
        sims = mat @ q_vec
        order = sorted(range(len(notes)), key=lambda i: float(sims[i]),
                       reverse=True)
        return [notes[i] for i in order if float(sims[i]) >= SIM_FLOOR][:k]

    async def _load_notes(self, user_id: int) -> list[str]:
        async with get_sessionmaker()() as db:
            rows = (await db.execute(
                select(MemoryNote.content)
                .where(MemoryNote.user_id == user_id)
                .order_by(MemoryNote.id.desc())
                .limit(self._settings.memory_reflective_recent)
            )).scalars().all()
        return list(rows)

    async def write(self, user_id: int, user_msg: str,
                    assistant_answer: str) -> int:
        prompt = (f"{WRITE_PROMPT}\n\n用户问题：{user_msg[:500]}\n"
                  f"助手回答：{assistant_answer[:1500]}")
        try:
            result = await self._llm.chat(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=WRITE_MAX_TOKENS,
            )
        except Exception as exc: 
            logger.warning("反思笔记提炼失败 (user=%s): %s", user_id, exc)
            return 0
        notes = parse_notes(result.content)
        if not notes:
            return 0
        try:
            async with get_sessionmaker()() as db:
                for text in notes:
                    db.add(MemoryNote(user_id=user_id, content=text))
                await db.commit()
        except Exception as exc:  #落库失败降级
            logger.warning("反思笔记落库失败 (user=%s): %s", user_id, exc)
            return 0
        return len(notes)
