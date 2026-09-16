from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable

from sqlalchemy import select

from app.agents import build_agent_team, default_llm_factory
from app.core.config import get_settings
from app.core.harness.memory import ConversationMemory
from app.core.harness.reflective import ReflectiveMemory
from app.core.skills import get_skill_registry
from app.core.llm import ChatMessage
from app.db import get_sessionmaker
from app.db.models import Conversation, Message
from app.distributed.guard import (
    ConversationBusyError,
    acquire_conv_lock,
    get_redis,
    release_conv_lock,
)
from app.orchestration.graph import build_chat_graph

logger = logging.getLogger("pickleball.runner")

Emitter = Callable[[dict], Awaitable[None]] | None


_team_graph_future: asyncio.Future | None = None


def _attachments_from_job(job: dict) -> list[dict]:
    """把 job 里的 data URL 列表解析为 ToolContext.attachments（含 mime 与文件名）。"""
    out: list[dict] = []
    for i, data in enumerate(job.get("images") or []):
        header = str(data).partition(";")[0]
        mime = header.removeprefix("data:")
        ext = (mime.partition("/")[2].split("+")[0]) or "png"
        out.append({"name": f"image-{i + 1}.{ext}", "mime": mime, "data": data})
    return out


async def _discover_mcp_tools(settings) -> list:
    """MCP 共享工具发现（每个进程一次；池注册进模块级收尾表）。"""
    from app.core.mcp import McpToolPool, parse_mcp_servers

    servers = parse_mcp_servers(settings.mcp_servers)
    if not servers:
        return []
    pool = McpToolPool()  # 进程退出时由 aclose_all_pools() 统一关闭
    return await pool.discover(servers)


async def _default_team_and_graph() -> tuple[dict, Any]:
    """默认团队与状态图：进程内异步单例（缓存 Future，并发任务只构建一次）。"""
    global _team_graph_future
    if (_team_graph_future is not None
            and _team_graph_future.get_loop().is_closed()):
        _team_graph_future = None  # 旧事件循环残留（如测试环境）
    if _team_graph_future is None:
        _team_graph_future = asyncio.get_running_loop().create_future()
        try:
            settings = get_settings()
            factory = default_llm_factory(settings)
            shared_tools = await _discover_mcp_tools(settings)
            from app.core.notebook import make_notebook_tools

            shared_tools += make_notebook_tools()
            # 图表提取工具：未配置任何 API Key 时不注册（优雅降级）
            from app.core.chart import make_chart_tool

            chart_tool = make_chart_tool(settings)
            if chart_tool is not None:
                shared_tools.append(chart_tool)
            # worker 进程无 app.state：技能注册表靠进程内单例各自从磁盘加载一次
            skills = get_skill_registry()
            team = build_agent_team(factory, settings=settings,
                                    shared_tools=shared_tools,
                                    skills=skills)
            graph = build_chat_graph(team, factory, settings=settings)
            _team_graph_future.set_result((team, graph))
        except BaseException as exc:
            _team_graph_future.set_exception(exc)
            _team_graph_future = None  # 失败可重试
            raise
    return await _team_graph_future


async def _emit_safe(emitter: Emitter, event: dict) -> None:
    if emitter is None:
        return
    try:
        await emitter(event)
    except Exception:  # noqa: BLE001
        logger.warning("事件推送失败: %s", event.get("type"))


async def run_chat_job(
    job: dict,
    emitter: Emitter = None,
    *,
    team: dict | None = None,
    graph: Any = None,
) -> dict:
    """执行一个聊天任务。

    job 字段：{id, session_id, conversation_id?, message, images?, principal:{user_id,
    username, role, permissions[]}}
    """
    session_id = job.get("session_id") or uuid.uuid4().hex[:16]
    conversation_id: uuid.UUID | None = None
    acquired_lock = False

    async def emit(event: dict) -> None:
        await _emit_safe(emitter, event)

    try:
        user_id = int(job["principal"]["user_id"])
        message = str(job.get("message") or "").strip()
        if not message:
            raise ValueError("消息内容为空")

        # ---- 1. 会话加载 / 创建，保存用户消息 ----
        async with get_sessionmaker()() as db:
            conversation_id = job.get("conversation_id")
            conv: Conversation | None = None
            if conversation_id:
                conv = await db.get(Conversation, conversation_id)
                if conv is None or conv.user_id != user_id:
                    conversation_id = None
            if conv is None:
                conv = Conversation(user_id=user_id, title=message[:40] or "新对话")
                db.add(conv)
                await db.flush()
                conversation_id = conv.id
            # 会话级执行锁：同一会话并行任务/回退互斥（Redis 挂时降级放行）
            acquired_lock = await acquire_conv_lock(get_redis(), conversation_id,
                                                    session_id)
            if not acquired_lock:
                raise ConversationBusyError("该会话正在执行中，请稍后再试")
            # 全部历史消息（用于分层记忆）
            rows = (
                await db.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.id.asc())
                )
            ).scalars().all()
            history = [
                ChatMessage(role=m.role, content=m.content)
                for m in rows
                if m.role in ("user", "assistant")
            ]
            user_msg = Message(conversation_id=conversation_id, role="user",
                               content=message,
                               meta={"images": len(_attachments_from_job(job))})
            db.add(user_msg)
            await db.flush()
            user_message_id = user_msg.id
            await db.commit()

        # ---- 2. 分层记忆：滑窗 + 滚动摘要（超出阈值才触发摘要 LLM 调用）----
        settings = get_settings()
        memory_llm = default_llm_factory(settings)()
        memory = ConversationMemory(memory_llm)
        memory.seed(history)
        context_messages = await memory.build_messages()
        history_text = "\n".join(
            f"{m.role}: {m.content}" for m in context_messages if m.content
        )[:2000]

        # ---- 2.5 ReMe 反思记忆：读（本地向量排序，零 LLM 调用）----
        reflective = ReflectiveMemory(memory_llm)
        memory_text = ""
        if settings.memory_reflective_enabled:
            notes = await reflective.read(user_id, message)
            memory_text = "\n".join(f"- {n}" for n in notes)

        # ---- 3. 多 Agent 状态图执行 ----
        if team is None or graph is None:
            default_team, default_graph = await _default_team_and_graph()
            team = team or default_team
            graph = graph or default_graph

        state = {
            "task": message,
            "history": history_text,
            "memory": memory_text,
            "session_id": session_id,
            "principal": job["principal"],
            "attachments": _attachments_from_job(job),
            "emit": emit,
        }
        await emit({"type": "status", "status": "running", "session_id": session_id})
        result = await graph.ainvoke(state)
        final = dict(result.get("final") or {})
        final["conversation_id"] = str(conversation_id)
        final["session_id"] = session_id

        # ---- 4. 保存助手消息（附用量/成本/门禁元数据）----
        async with get_sessionmaker()() as db:
            asst_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=final.get("answer", ""),
                meta=final,
            )
            db.add(asst_msg)
            await db.flush()
            final["assistant_message_id"] = asst_msg.id
            final["user_message_id"] = user_message_id
            await db.commit()

        # ---- 5. ReMe 反思记忆：写（每轮 1 次 LLM 提炼；失败降级不阻断）----
        if settings.memory_reflective_enabled:
            try:
                await reflective.write(user_id, message,
                                       final.get("answer", ""))
            except Exception as exc:  # noqa: BLE001 兜底双保险
                logger.warning("反思记忆写入失败: %s", exc)

        await emit({"type": "done", **final})
        return final
    except ConversationBusyError:
        logger.info("会话 %s 忙，拒绝并行任务 %s", conversation_id, session_id)
        await emit({"type": "error", "message": "该会话正在执行中，请稍后再试"})
        return {"error": "该会话正在执行中，请稍后再试"}
    except Exception as exc:  # noqa: BLE001 执行器兜底：错误必须回到用户侧
        logger.exception("聊天任务失败: %s", exc)
        message_text = f"{type(exc).__name__}: {exc}"[:500]
        await emit({"type": "error", "message": message_text})
        return {"error": message_text}
    finally:
        if acquired_lock and conversation_id is not None:
            await release_conv_lock(get_redis(), conversation_id, session_id)
