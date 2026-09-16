
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.core.config import Settings, get_settings
from app.core.harness.audit import AuditStore, InvocationAudit
from app.core.harness.loop import LoopConfig, LoopResult, ReActLoop
from app.core.harness.memory import KnowledgeStore
from app.core.harness.security import ExecutionPipeline
from app.core.harness.tools import ToolContext, ToolRegistry, ToolSpec
from app.core.harness.web_search import make_web_search_tool
from app.core.llm import LLMClient, TokenUsage
from app.core.retrieval import (
    format_hits,
    merge_hits,
    plan_and_retrieve,
    rerank_with_llm,
    should_plan,
)
from app.core.skills import SkillRegistry, make_load_skill_tool

logger = logging.getLogger("pickleball.agent")

Emitter = Callable[[dict], Awaitable[None]] | None
LLMFactory = Callable[..., LLMClient]


@dataclass(frozen=True)
class AgentInfo:
    key: str
    title: str
    description: str
    topics: tuple[str, ...] = ()


@dataclass
class AgentOutput:
    answer: str
    steps: int = 0
    tool_calls: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)


class BaseAgent:
    """领域专家基类。"""

    info: AgentInfo
    # 自动检索的知识领域：空元组 = 不检索
    retrieval_kinds: tuple[str, ...] = ()
    retrieval_query_chars: int = 200   # 检索 query 截断长度（哈希向量对长文本无增益）

    def __init__(
        self,
        llm_factory: LLMFactory,
        *,
        knowledge: KnowledgeStore,
        audit_store: AuditStore,
        settings: Settings | None = None,
        verify: bool = False,
        skills: SkillRegistry | None = None,
    ) -> None:
        self._llm_factory = llm_factory
        self._knowledge = knowledge
        self._audit_store = audit_store
        self._settings = settings or get_settings()
        self._registry = ToolRegistry()
        self._registry.register_many(self.tools())
        # 联网搜索
        self._web_search = make_web_search_tool(self._settings)
        if self._web_search is not None:
            self._registry.register(self._web_search)
        # 技能
        self._skills = skills
        self._load_skill = make_load_skill_tool(skills) if skills else None
        if self._load_skill is not None:
            self._registry.register(self._load_skill)
        self._pipeline = ExecutionPipeline(self._registry, self._audit_store)
        self._verify = verify

    def system_prompt(self) -> str:
        raise NotImplementedError

    def tools(self) -> list[ToolSpec]:
        return []

    # ---- 工具注册 ----
    def register_tools(self, tools: list[ToolSpec]) -> None:
        """追加注册共享工具（MCP 工具池等）；重名抛 ValueError 快速失败。"""
        self._registry.register_many(tools)

    # ---- RAG 证据注入：查询规划 + 多路并行检索 ----
    async def _retrieve_evidence(self, task: str, ctx: ToolContext,
                                 emit: Emitter,
                                 llm: LLMClient | None = None) -> str | None:
        """自动检索知识库证据。复杂查询先做查询规划（重写/退一步/拆分/HyDE，
        仿 ReMe 轻量版：单次 LLM 调用），多路并行检索后汇总，候选超过注入
        预算时再做 LLM 重排；规划关闭或失败时降级为单路直查，
        重排失败降级原顺序，检索失败降级为无证据继续跑。"""
        t0 = time.perf_counter()
        query = task[:self.retrieval_query_chars]
        kinds = list(self.retrieval_kinds)
        per_kind = max(1, -(-self._knowledge.top_k // len(kinds)))
        status = "done"
        planned = False
        reranked = False
        queries: list[str] = [query]
        sources: list[str] = []
        hits_total = 0

        async def _finish(text: str | None) -> str | None:
            # 审计：与工具调用同表（tool 列带 retrieval: 前缀），零命中也落审计
            args = query[:500] if not planned else f"{query[:300]} → 变体:{queries}"[:500]
            audit = InvocationAudit(
                user_id=ctx.principal.user_id,
                username=ctx.principal.username,
                tool=f"retrieval:{self.info.key}",
                side_effect=0,
                decision="error" if status == "error" else "approved",
                phase="execute",
                args_summary=args,
                result_summary=("0 hits" if not hits_total else
                                f"{hits_total} hits: {sources[:5]}")[:500],
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            await self._audit_store.record(audit)
            if emit is not None:
                await emit({"type": "retrieval", "agent": self.info.key,
                            "status": status, "query": query, "queries": queries,
                            "planned": planned, "reranked": reranked,
                            "kinds": kinds,
                            "hits": hits_total, "sources": sources[:10],
                            "duration_ms": audit.duration_ms})
            return text

        try:
            if (self._settings.retrieval_plan_enabled and llm is not None
                    and should_plan(query)):
                merged, plan, queries = await plan_and_retrieve(
                    self._knowledge.retrieve, llm, query, kinds,
                    per_kind=per_kind,
                    top_k=self._knowledge.top_k,
                    evidence_limit=max(self._knowledge.top_k,
                                       self._settings.retrieval_evidence_limit),
                )
                planned = plan is not None and len(queries) > 1
            else:
                groups = []
                for kind in kinds:
                    hits = self._knowledge.retrieve(query, kind=kind, top_k=per_kind)
                    groups.append(hits)
                merged = merge_hits(groups, top_k=self._knowledge.top_k)
            # 召回后重排：候选超过注入预算才有取舍空间（一次 LLM 调用）
            if (self._settings.retrieval_rerank_enabled and llm is not None
                    and len(merged) > self._knowledge.top_k):
                merged = await rerank_with_llm(
                    llm, query, merged, top_k=self._knowledge.top_k,
                    max_candidates=self._settings.retrieval_rerank_candidates)
                reranked = True
            hits_total = len(merged)
            if not merged:
                status = "empty"
                return await _finish(None)
            sources = [f"{h.chunk.doc}/{h.chunk.section}" for h in merged]
            return await _finish(format_hits(merged, query=query))
        except Exception as exc:  # noqa: BLE001 检索失败不阻断回答
            logger.warning("专家 %s 知识检索失败: %s", self.info.key, exc)
            status = "error"
            return await _finish(None)

    async def run(self, task: str, ctx: ToolContext, *,
                  emit: Emitter = None) -> AgentOutput:
        llm = self._llm_factory()
        config = LoopConfig.from_settings()
        config.verify_required = self._verify
        prompt = self.system_prompt()
        if self._web_search is not None:
            prompt += (
                "\n联网检索：遇到时效性问题"
                "或知识库检索不到答案时，可调用 web_search 工具联网搜索，"
                "引用外部信息时请注明来源。"
            )
        if self.retrieval_kinds:
            # llm 与 ReAct 循环共用实例：规划调用的 token 一并计入本轮成本观测
            evidence = await self._retrieve_evidence(task, ctx, emit, llm=llm)
            if evidence:
                prompt += ("\n\n【已检索知识库证据（系统自动注入，回答必须以此为准）】\n"
                           + evidence)
        if self._load_skill is not None and self._skills:
            prompt += ("\n\n【可用技能】需要时调用 load_skill 工具"
                       "加载完整说明并按其执行）\n" + self._skills.index_text())
        if ctx.attachments:
            prompt += (
                f"\n\n【本轮附件】用户上传了 {len(ctx.attachments)} 张图表图片。"
                "需要图表数据时先调用 extract_chart_data 工具（image_index 从 0 开始）"
                "把图表提取为表格，再基于提取结果作答；不要凭空猜测图片内容。"
            )
        loop = ReActLoop(
            llm,
            self._registry,
            self._pipeline,
            agent_key=self.info.key,
            system_prompt=prompt,
            config=config,
            verify_gate=None,
            emit=emit,
        )
        if emit:
            await emit({"type": "agent", "agent": self.info.key,
                        "title": self.info.title, "status": "started"})
        result: LoopResult = await loop.run(task, ctx)
        if emit:
            await emit({"type": "agent", "agent": self.info.key,
                        "title": self.info.title, "status": "finished",
                        "steps": result.steps, "tool_calls": result.tool_calls_made})
        return AgentOutput(answer=result.answer, steps=result.steps,
                           tool_calls=result.tool_calls_made, usage=result.usage)


