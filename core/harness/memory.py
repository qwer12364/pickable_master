from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.llm import ChatMessage

# 过渡 re-export（deprecated）：新代码请用 `from app.core.retrieval import KnowledgeStore`
from app.core.retrieval.store import KnowledgeStore  # noqa: F401

# ---------------------------------------------------------------------------
# 会话记忆：滑动窗口 + 滚动摘要
# ---------------------------------------------------------------------------


class ConversationMemory:
    """长会话记忆：近 window_size 条保留原文，溢出部分压缩为滚动摘要。

    摘要由 LLM 生成；后续再溢出时基于旧摘要增量合并，保证上下文长度有界。
    """

    def __init__(self, llm: Any, *, window_size: int | None = None,
                 summary_trigger: int | None = None) -> None:
        settings = get_settings()
        self._llm = llm
        self.window_size = window_size or settings.memory_window_size
        self.summary_trigger = summary_trigger or settings.memory_summary_trigger
        self._messages: list[ChatMessage] = []
        self._summary: str | None = None

    def seed(self, history: list[ChatMessage]) -> None:
        self._messages = history

    def add(self, message: ChatMessage) -> None:
        self._messages.append(message)

    @property
    def size(self) -> int:
        return len(self._messages)

    async def build_messages(self) -> list[ChatMessage]:
        if len(self._messages) > self.summary_trigger:
            await self._summarize()
        out: list[ChatMessage] = []
        if self._summary:
            out.append(ChatMessage(
                role="system",
                content=f"[更早对话的滚动摘要]\n{self._summary}",
            ))
        out.extend(self._messages[-self.window_size:])
        return out

    async def _summarize(self) -> None:
        overflow = self._messages[: -self.window_size]
        if not overflow:
            return
        transcript = "\n".join(
            f"{m.role}: {m.content}" for m in overflow if m.content
        )
        prompt = (
            "请把下面的对话记录压缩成要点摘要（中文，不超过200字），"
            "保留：用户的问题与目标、已确认的关键事实、已给出的结论。"
            f"{('已有历史摘要：' + self._summary) if self._summary else ''}\n\n"
            f"待压缩对话：\n{transcript[:3000]}"
        )
        try:
            result = await self._llm.chat([ChatMessage(role="user", content=prompt)])
            self._summary = (result.content or "").strip()[:500]
        except Exception:  # noqa: BLE001 摘要失败不阻断对话
            self._summary = self._summary or "(历史摘要生成失败)"
        self._messages = self._messages[-self.window_size:]
