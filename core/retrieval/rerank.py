from __future__ import annotations

import json
import logging
import re

from app.core.llm import ChatMessage
from app.core.retrieval.base import RetrievedHit

logger = logging.getLogger("pickleball.retrieval.rerank")


RERANK_MAX_TOKENS = 150

RERANK_SYSTEM_PROMPT = (
    "你是检索结果重排器。根据用户问题，把候选知识片段按相关性从高到低排序。\n"
    "只输出一个 JSON 数组（候选片段序号，从 0 开始；必须包含全部序号、不得重复），"
    '例如：[2, 0, 3, 1]。不要输出任何其它内容。'
)


def _format_candidates(hits: list[RetrievedHit]) -> str:
    lines = [
        f"[{i}] 【{h.chunk.doc}/{h.chunk.section}】{h.chunk.text[:300]}"
        for i, h in enumerate(hits)
    ]
    return "\n".join(lines)


def parse_rank(content: str | None, n: int) -> list[int] | None:
    """
    匹配最内层方括号（`[^\[\]]*`）：容错 `[[2,0,1]]` 这类围栏嵌套产物。
    """
    if not content:
        return None
    match = re.search(r"\[[^\[\]]*\]", content)
    if not match:
        return None
    try:
        order = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(order, list) or len(order) != n:
        return None
    seen: set[int] = set()
    out: list[int] = []
    for x in order:
        if (not isinstance(x, int) or isinstance(x, bool)
                or x < 0 or x >= n or x in seen):
            return None
        seen.add(x)
        out.append(x)
    return out


async def rerank_with_llm(
    llm: object,
    query: str,
    hits: list[RetrievedHit],
    *,
    top_k: int | None = None,
    max_candidates: int = 8,
) -> list[RetrievedHit]:
    if len(hits) <= 1:
        return hits
    candidates = hits[:max_candidates]
    prompt = (f"用户问题：{query[:200]}\n\n候选片段：\n"
              f"{_format_candidates(candidates)}")
    try:
        result = await llm.chat(  # type: ignore[union-attr]
            [ChatMessage(role="system", content=RERANK_SYSTEM_PROMPT),
             ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=RERANK_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 重排失败降级原顺序，不阻断回答
        logger.warning("重排 LLM 调用失败，保持原排序: %s", exc)
        return hits
    order = parse_rank(result.content, len(candidates))
    if order is None:
        logger.warning("重排输出解析失败，保持原排序: %.200s", result.content)
        return hits
    reordered = [candidates[i] for i in order]
    return reordered[:top_k] if top_k is not None else reordered
