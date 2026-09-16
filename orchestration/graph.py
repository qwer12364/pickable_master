from __future__ import annotations

import json
import logging
import operator
import re
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.core.config import Settings, get_settings
from app.core.harness.tools import Principal, ToolContext
from app.core.harness.verify import VerificationGate
from app.core.llm import ChatMessage, LLMClient
from app.orchestration.cost import usage_report

logger = logging.getLogger("pickleball.graph")


def _merge_dict(a: dict, b: dict) -> dict:
    out = dict(a)
    out.update(b)
    return out


class ChatState(TypedDict, total=False):
    task: str
    history: str
    memory: str
    session_id: str
    principal: dict
    emit: Any
    attachments: list[dict]
    routing: dict
    expert_outputs: Annotated[dict[str, str], _merge_dict]
    synthesis: str
    gate_verdict: dict
    repair_count: int
    usage_breakdown: Annotated[list[dict], operator.add]
    final: dict


ROUTE_PROMPT = """你是多专家调度器。根据用户问题，选择最相关的匹克球专家（可多选，最多2个）。
专家列表：
{experts}

对话历史（供上下文参考，可能为空）：
{history}

用户长期记忆（ReMe 反思笔记，供上下文参考，可能为空）：
{memory}
{attachment_note}
只输出一行 JSON：{{"agents": ["<key>", ...], "reason": "<一句话理由>"}}"""

SYNTHESIS_PROMPT = """你是匹克球首席专家。请整合以下专家意见，给用户一份完整、准确、结构清晰的最终回答（中文）。
要求：
1. 覆盖所有被咨询专家的核心观点，不要遗漏；
2. 保留专家引用知识库的来源标注（如【rules/发球规则】）；
3. 不引入专家意见之外的事实，不编造数据；
4. 面向业余爱好者，结论先行、具体可执行。

对话历史（供上下文参考，可能为空）：
{history}

用户长期记忆（ReMe 反思笔记，供上下文参考，可能为空）：
{memory}

用户问题：
{task}

专家意见：
{outputs}"""


def _principal_from_state(state: ChatState) -> Principal:
    p = state["principal"]
    return Principal(
        user_id=int(p["user_id"]),
        username=str(p.get("username", "")),
        role=str(p.get("role", "user")),
        permissions=frozenset(p.get("permissions") or ["tools:read"]),
    )


def _ctx_from_state(state: ChatState) -> ToolContext:
    return ToolContext(
        principal=_principal_from_state(state),
        session_id=state.get("session_id", ""),
        emit=state.get("emit"),
        attachments=state.get("attachments") or [],
    )


async def _safe_emit(state: ChatState, event: dict) -> None:
    emit = state.get("emit")
    if emit is not None:
        try:
            await emit(event)
        except Exception:  # noqa: BLE001
            logger.warning("事件推送失败: %s", event.get("type"))


def build_chat_graph(
    team: dict[str, BaseAgent],
    llm_factory: Callable[..., LLMClient],
    *,
    settings: Settings | None = None,
) -> Any:
    s = settings or get_settings()
    verify_gate = VerificationGate(llm_factory, s.llm_fast_model)
    expert_entries = "\n".join(
        f"- {agent.info.key}：{agent.info.description}" for agent in team.values()
    )
    async def route_node(state: ChatState) -> dict:
        llm = llm_factory()
        n_images = len(state.get("attachments") or [])
        attachment_note = (
            f"\n注意：用户本轮上传了 {n_images} 张图表图片，"
            "分析图表数据更适合 strategy_analyst 或 technique_coach。"
            if n_images else ""
        )
        prompt = ROUTE_PROMPT.format(experts=expert_entries,
                                     history=(state.get("history") or "（无）")[:1500],
                                     memory=(state.get("memory") or "（无）")[:800],
                                     attachment_note=attachment_note)
        result = await llm.chat([ChatMessage(role="user", content=prompt)],
                                temperature=0.0)
        raw = (result.content or "").strip()
        agents: list[str] = []
        reason = "自动调度"
        try:
            match = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(match.group(0) if match else raw)
            agents = [a for a in data.get("agents", []) if a in team]
            reason = str(data.get("reason", ""))[:200]
        except Exception:  # noqa: BLE001 调度解析失败 → 全部专家兜底
            logger.warning("调度结果解析失败，回退全专家: %r", raw[:200])
        if not agents:
            agents = list(team.keys())
        routing = {"agents": agents[:2], "reason": reason}
        await _safe_emit(state, {"type": "agent", "agent": "router", "title": "调度器",
                                 "status": "routed",
                                 "data": {"agents": routing["agents"], "reason": reason}})
        return {"routing": routing,
                "usage_breakdown": [{"agent": "router",
                                     "prompt_tokens": result.usage.prompt_tokens,
                                     "completion_tokens": result.usage.completion_tokens}]}

    def _expert_node(agent: BaseAgent):
        async def node(state: ChatState) -> dict:
            output = await agent.run(state["task"], _ctx_from_state(state),
                                     emit=state.get("emit"))
            return {
                "expert_outputs": {agent.info.key: output.answer},
                "usage_breakdown": [{
                    "agent": agent.info.key,
                    "prompt_tokens": output.usage.prompt_tokens,
                    "completion_tokens": output.usage.completion_tokens,
                }],
            }
        return node

    async def synthesize_node(state: ChatState) -> dict:
        llm = llm_factory()
        prompt = SYNTHESIS_PROMPT.format(
            history=(state.get("history") or "（无）")[:1500],
            memory=(state.get("memory") or "（无）")[:800],
            task=state["task"],
            outputs="\n\n---\n\n".join(
                f"【{key}】{text}" for key, text in state["expert_outputs"].items()
            ),
        )
        parts: list[str] = []
        usage: dict = {"agent": "synthesizer", "prompt_tokens": 0, "completion_tokens": 0}
        async for event in llm.stream_chat([ChatMessage(role="user", content=prompt)],
                                           temperature=0.4):
            if event.type == "text":
                parts.append(event.text)
                await _safe_emit(state, {"type": "delta", "text": event.text})
            elif event.type == "usage" and event.usage is not None:
                usage = {"agent": "synthesizer",
                         "prompt_tokens": event.usage.prompt_tokens,
                         "completion_tokens": event.usage.completion_tokens}
        return {"synthesis": "".join(parts), "usage_breakdown": [usage]}

    async def quality_gate_node(state: ChatState) -> dict:
        evidence = "\n\n".join(state["expert_outputs"].values())
        verdict = await verify_gate.verify(task=state["task"],
                                           answer=state["synthesis"],
                                           evidence=evidence)
        await _safe_emit(state, {"type": "verify", "agent": "gate",
                                 "passed": verdict.passed, "reason": verdict.reason,
                                 "repair_count": state.get("repair_count", 0)})
        usage = verdict.usage or {}
        return {
            "gate_verdict": {"passed": verdict.passed, "reason": verdict.reason},
            "usage_breakdown": [{
                "agent": "quality_gate",
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }],
        }



    async def repair_node(state: ChatState) -> dict:
        count = state.get("repair_count", 0) + 1
        reason = state["gate_verdict"]["reason"]
        await _safe_emit(state, {"type": "agent", "agent": "repair",
                                 "title": "修复管线", "status": "started",
                                 "data": {"reason": reason, "round": count}})
        task = (state["task"]
                + f"\n\n【质量门禁反馈】上一次综合回答未通过检查：{reason}\n"
                  "请针对反馈补充检索证据或修正你的回答。")
        outputs: dict[str, str] = {}
        breakdown: list[dict] = []
        for key in state["routing"]["agents"]:
            agent = team.get(key)
            if agent is None:
                continue
            output = await agent.run(task, _ctx_from_state(state), emit=state.get("emit"))
            outputs[key] = output.answer
            breakdown.append({"agent": f"repair:{key}",
                              "prompt_tokens": output.usage.prompt_tokens,
                              "completion_tokens": output.usage.completion_tokens})
        return {"expert_outputs": outputs, "repair_count": count,
                "usage_breakdown": breakdown}

    async def human_review_node(state: ChatState) -> dict:
        reason = state["gate_verdict"]["reason"]
        await _safe_emit(state, {"type": "human_review", "reason": reason})
        return {}

    async def finalize_node(state: ChatState) -> dict:
        report = usage_report(state.get("usage_breakdown", []))
        verdict = state["gate_verdict"]
        final: dict[str, Any] = {
            "answer": state["synthesis"],
            "agents_used": state["routing"]["agents"],
            "gate_passed": verdict["passed"],
            "gate_reason": verdict["reason"],
            "needs_human_review": not verdict["passed"],
            "repair_count": state.get("repair_count", 0),
            "usage": report,
        }
        return {"final": final}
    builder = StateGraph(ChatState)
    builder.add_node("route", route_node)
    for key, agent in team.items():
        builder.add_node(key, _expert_node(agent))
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("quality_gate", quality_gate_node)
    builder.add_node("repair", repair_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)

    def _expert_router(state: ChatState) -> list[str]:
        agents = state["routing"]["agents"]
        return agents or list(team.keys())

    builder.add_edge(START, "route")


    builder.add_conditional_edges("route", _expert_router)
    for key in team:
        builder.add_edge(key, "synthesize")
    builder.add_edge("synthesize", "quality_gate")

    def gate_router(state: ChatState) -> str:
        if state["gate_verdict"]["passed"]:
            return "finalize"
        if state.get("repair_count", 0) >= 1:
            return "human_review"
        return "repair"
    

    builder.add_conditional_edges(
        "quality_gate", gate_router,
        {"finalize": "finalize", "repair": "repair", "human_review": "human_review"},
    )
    builder.add_edge("repair", "synthesize")
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
