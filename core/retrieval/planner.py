from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.llm import ChatMessage
from app.core.retrieval.base import RetrievedHit
from app.core.retrieval.hybrid import merge_hits

logger = logging.getLogger("pickleball.retrieval.plan")

# 门控阈值：中文少于该字数不规划（「你好」、单词查询不值得一次 LLM 调用）
MIN_CJK_CHARS = 8
# 规划输出上限：JSON 很小，但含 HyDE 假设文档，收紧 max_tokens 控制成本与延迟
PLAN_MAX_TOKENS = 500

_CJK_RE = re.compile(r"[一-鿿]")

RetrieveFn = Callable[..., list[RetrievedHit]]


@dataclass
class QueryPlan:
    """一次规划调用的产物：rewritten 必有，step_back / sub_queries / hyde 可为空。"""

    rewritten: str
    step_back: str = ""
    sub_queries: list[str] = field(default_factory=list)
    hyde_doc: str = ""

    @property
    def variants(self) -> list[str]:
        """参与检索的变体序列（重写 → 退一步 → 子问题 → HyDE，去重去空，
        子问题最多 3 个）。"""
        out = [self.rewritten] if self.rewritten else []
        if self.step_back and self.step_back not in out:
            out.append(self.step_back)
        for q in self.sub_queries[:3]:
            if q and q not in out:
                out.append(q)
        if self.hyde_doc and self.hyde_doc not in out:
            out.append(self.hyde_doc)
        return out


PLAN_SYSTEM_PROMPT = (
    "你是匹克球领域知识库的检索规划器。对用户提问一次产出四项检索辅助信息：\n"
    "1. rewritten：把口语化、模糊的表达改写为精确的检索查询（术语规范化，"
    "如「那个厨房的规矩」→「匹克球非截击区规则」），含义与原问一致；\n"
    "2. step_back：把提问退一步抽象成概念背景问题"
    "（如「非截击区怎么截击」→「非截击区的设立目的」），防提问范围过窄；"
    "提问本身已足够宽泛时给空字符串；\n"
    "3. sub_queries：仅当提问是复杂复合句（含多个独立子问题）时，"
    "拆成不超过 3 个可独立检索的子问题；简单提问给空数组。\n"
    "4. hyde：写一段 60~150 字的假设性答案片段，内容为该问题答案应包含的"
    "知识要点，必须使用规范术语（如「匹克球非截击区规则规定……」），"
    "用于与知识库词汇桥接、提高召回率；所有提问都尽量给出。\n"
    "只输出一个 JSON 对象，不要输出任何其它内容：\n"
    '{"rewritten": "…", "step_back": "…", "sub_queries": ["…"], "hyde": "…"}'
)


def should_plan(query: str) -> bool:
    """门控：中文过少的短问句/纯英文不规划，省一次 LLM 调用。"""
    return len(_CJK_RE.findall(query)) >= MIN_CJK_CHARS


def parse_plan(content: str) -> QueryPlan | None:
    """从 LLM 输出中提取并校验 QueryPlan；任何畸形输出返回 None。"""
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    rewritten = str(data.get("rewritten") or "").strip()
    if not rewritten:
        return None
    step_back = str(data.get("step_back") or "").strip()
    subs = data.get("sub_queries")
    if not isinstance(subs, list):
        subs = []
    sub_queries = [str(q).strip() for q in subs if str(q).strip()][:3]
    hyde = str(data.get("hyde") or "").strip()
    return QueryPlan(rewritten=rewritten, step_back=step_back,
                     sub_queries=sub_queries, hyde_doc=hyde)


async def plan_query(llm: Any, query: str) -> QueryPlan | None:
    try:
        result = await llm.chat(
            [ChatMessage(role="system", content=PLAN_SYSTEM_PROMPT),
             ChatMessage(role="user", content=query)],
            temperature=0.0,
            max_tokens=PLAN_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 规划失败降级直查，不阻断回答
        logger.warning("查询规划 LLM 调用失败，降级为原查询单路检索: %s", exc)
        return None
    plan = parse_plan(result.content or "")
    if plan is None:
        logger.warning("查询规划 JSON 解析失败，降级为原查询单路检索: %.200s",
                       result.content)
    return plan


async def plan_and_retrieve(
    retrieve: RetrieveFn,
    llm: Any,
    query: str,
    kinds: list[str],
    *,
    per_kind: int,
    top_k: int,
    evidence_limit: int,
) -> tuple[list[RetrievedHit], QueryPlan | None, list[str]]:
    """规划 → 多路并行检索 → 汇总。

    返回 (合并命中, 规划结果, 实际参与检索的变体)：
    - 规划失败或 llm=None 时等价于经典单路直查；
    - 变体数为 1 时按原 top_k 截断；多路时放宽到 evidence_limit——
      不同变体命中不同块，需要更大注入预算覆盖各子主题；
    - 检索为本地 CPU 运算，放线程池并发，多路真正并行。
    """
    plan = await plan_query(llm, query) if llm is not None else None
    queries = plan.variants if plan else []
    if not queries:
        queries = [query]   
    merge_top = evidence_limit if len(queries) > 1 else top_k

    async def _fetch(q: str, kind: str) -> list[RetrievedHit]:
        return await asyncio.to_thread(retrieve, q, kind=kind, top_k=per_kind)

    groups = await asyncio.gather(*(_fetch(q, k) for q in queries for k in kinds))
    merged = merge_hits(list(groups), top_k=merge_top)
    return merged, plan, queries
