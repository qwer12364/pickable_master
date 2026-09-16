"""Agent 团队组装：4 个领域专家 + 全局工具注册表（含 admin 专属 EXEC 工具）。

build_agent_team() 是编排层（LangGraph）与 API 的唯一入口；
llm_factory(model=None) 每次调用返回一个全新 LLMClient，保证按调用独立计量 token。
"""
from __future__ import annotations

import asyncio
from typing import Callable

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.agents.equipment_advisor import EquipmentAdvisor
from app.agents.rule_expert import RuleExpert
from app.agents.strategy_analyst import StrategyAnalyst
from app.agents.technique_coach import TechniqueCoach
from app.core.config import Settings, get_settings
from app.core.harness.audit import AuditStore
from app.core.harness.tools import SideEffectLevel, ToolContext, ToolRegistry, ToolSpec
from app.core.llm import LLMClient
from app.core.retrieval import KnowledgeStore, get_knowledge_store
from app.core.skills import SkillRegistry


def default_llm_factory(settings: Settings | None = None):
    """默认工厂：每次调用返回全新客户端（独立 token 计量）。"""
    s = settings or get_settings()

    def factory(model: str | None = None) -> LLMClient:
        return LLMClient(s, model=model or s.llm_model)

    return factory


def build_agent_team(
    llm_factory: Callable | None = None,
    *,
    settings: Settings | None = None,
    knowledge: KnowledgeStore | None = None,
    audit_store: AuditStore | None = None,
    shared_tools: list[ToolSpec] | None = None,
    skills: SkillRegistry | None = None,
) -> dict[str, BaseAgent]:
    """构建并返回 {agent_key: agent} 团队。

    shared_tools: 注入到每个专家的共享工具（如 MCP 池工具），
    仍走五阶段管线按角色做权限控制。
    skills: 渐进式披露技能注册表；None 则不注册 load_skill 工具。
    """
    s = settings or get_settings()
    factory = llm_factory or default_llm_factory(s)
    knowledge = knowledge or get_knowledge_store()
    audit = audit_store or AuditStore()

    experts = [
        RuleExpert(factory, knowledge=knowledge, audit_store=audit, settings=s,
                   skills=skills),
        TechniqueCoach(factory, knowledge=knowledge, audit_store=audit, settings=s,
                       skills=skills),
        EquipmentAdvisor(factory, knowledge=knowledge, audit_store=audit, settings=s,
                         skills=skills),
        StrategyAnalyst(factory, knowledge=knowledge, audit_store=audit, settings=s,
                        skills=skills),
    ]
    if shared_tools:
        for agent in experts:
            agent.register_tools(shared_tools)
    return {agent.info.key: agent for agent in experts}


# ---------------------------------------------------------------------------
# 全局工具注册表：含 admin 专属的高副作用工具（安全边界演示/生产能力）
# ---------------------------------------------------------------------------

class ShellParams(BaseModel):
    command: str = Field(..., max_length=500, description="要执行的 shell 命令")


async def _run_shell(ctx: ToolContext, command: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    out = (stdout or b"").decode("utf-8", errors="replace")
    err = (stderr or b"").decode("utf-8", errors="replace")
    text = (out + (f"\n[stderr]\n{err}" if err else "")).strip()
    return f"exit={proc.returncode}\n{(text or '(无输出)')[:2000]}"


def build_global_registry() -> ToolRegistry:
    """全局注册表：领域工具 + admin-only 命令执行工具（演示五阶段权限管线）。"""
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="run_shell_command",
        description="执行 shell 命令（仅限 admin 角色，需 tools:exec 权限）",
        handler=_run_shell,
        params_model=ShellParams,
        side_effect=SideEffectLevel.EXEC,
        permissions={"tools:exec"},
        timeout=20.0,
    ))
    return registry
