"""前后端通信核心：
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.harness.tools import Principal
from app.db.models import User
from app.distributed.queue import RedisTaskQueue
from app.distributed.runner import run_chat_job

logger = logging.getLogger("pickleball.api.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_IMAGES = 4
MAX_IMAGE_CHARS = 3_000_000


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: uuid.UUID | None = None
    images: list[str] = Field(default_factory=list, max_length=MAX_IMAGES)

    @field_validator("images")
    @classmethod
    def _validate_images(cls, v: list[str]) -> list[str]:
        for img in v:
            if not img.startswith("data:image/"):
                raise ValueError("图片必须是 data:image/* 格式的 data URL")
            if len(img) > MAX_IMAGE_CHARS:
                raise ValueError("单张图片过大（请压缩后重试）")
        return v


def _make_job(body: ChatRequest, user: User, session_id: str) -> dict:
    principal = Principal.from_user(user)
    return {
        "id": uuid.uuid4().hex[:16],
        "session_id": session_id,
        "conversation_id": str(body.conversation_id) if body.conversation_id else None,
        "message": body.message,
        "images": body.images,
        "principal": {
            "user_id": principal.user_id,
            "username": principal.username,
            "role": principal.role,
            "permissions": sorted(principal.permissions),
        },
    }


@router.post("")
async def chat(
    body: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
) -> dict:
    """非流式对话：内联执行（不经队列），返回完整结果。"""
    job = _make_job(body, user, uuid.uuid4().hex[:16])
    events: list[dict] = []

    async def emitter(event: dict) -> None:
        events.append(event)

    result = await run_chat_job(
        job,
        emitter=emitter,
        team=request.app.state.team,
        graph=request.app.state.graph,
    )
    return {"result": result, "events": events}


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """SSE 流式对话：事件即推即达前端。"""
    settings = get_settings()
    session_id = uuid.uuid4().hex[:16]
    bus = request.app.state.bus
    queue: RedisTaskQueue = request.app.state.queue
    job = _make_job(body, user, session_id)
    raw = json.dumps(job, ensure_ascii=False)

    async def emitter(event: dict) -> None:
        await bus.publish(session_id, event)

    async def event_source():
        # 1. 先订阅，再派发（Redis PubSub 即发即弃，先订后发不丢事件）
        events = await bus.subscribe(session_id)  # async def 返回异步生成器，需先 await

        # 2. 派发：queue 模式入队等 worker；auto 模式 1.5s 无人认领则内联接管；inline 直接执行
        inline_task: asyncio.Task | None = None
        mode = settings.chat_execution_mode
        team = request.app.state.team
        graph = request.app.state.graph
        if mode == "inline":
            inline_task = asyncio.create_task(
                run_chat_job(job, emitter=emitter, team=team, graph=graph)
            )
        else:
            await queue.push(job)
            if mode == "auto":
                await asyncio.sleep(1.5)
                if await queue.try_claim_inline(job["id"], raw):
                    inline_task = asyncio.create_task(
                        run_chat_job(job, emitter=emitter, team=team, graph=graph)
                    )
                    logger.info("无 worker 认领，API 内联接管任务 %s", job["id"])

        yield f"event: ready\ndata: {json.dumps({'session_id': session_id})}\n\n"

        # 3. 转发事件 + 心跳 + 总超时
        # 事件由独立 drain 任务搬进队列：心跳超时只取消 q.get()，
        # 不会把 CancelledError 注入订阅生成器（否则生成器会退订并永久关闭）
        done = False
        q: asyncio.Queue[dict | None] = asyncio.Queue()

        async def _drain() -> None:
            async for ev in events:
                await q.put(ev)
            q.put_nowait(None)  # 订阅结束哨兵

        drain_task = asyncio.create_task(_drain())
        try:
            try:
                async with asyncio.timeout(settings.chat_max_total_seconds):
                    while True:
                        try:
                            event = await asyncio.wait_for(
                                q.get(), timeout=settings.stream_heartbeat_seconds
                            )
                        except TimeoutError:
                            yield ": ping\n\n"  # SSE 注释行心跳
                            continue
                        if event is None:
                            break
                        yield (f"event: {event['type']}\n"
                               f"data: {json.dumps(event, ensure_ascii=False)}\n\n")
                        if event.get("type") in ("done", "error"):
                            done = True
                            break
            except TimeoutError:
                yield ("event: error\n"
                       f"data: {json.dumps({'message': '对话总超时，请重试'}, ensure_ascii=False)}\n\n")
                done = True
        finally:
            drain_task.cancel()
            try:
                await drain_task  # 等待取消传播完成（订阅生成器随 drain 关闭）
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            await events.aclose()  # 幂等：已关闭则无操作，未关闭则退订释放连接

        # 4. 收尾：内联任务若异常，补发 error（run_chat_job 本身不抛，这里是双保险）
        if inline_task is not None:
            try:
                await inline_task
            except Exception as exc:  # noqa: BLE001
                if not done:
                    yield ("event: error\n"
                           f"data: {json.dumps({'message': str(exc)[:500]}, ensure_ascii=False)}\n\n")

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
