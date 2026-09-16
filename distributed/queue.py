from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("pickleball.queue")

PENDING = "pickleball:queue:pending"
PROCESSING = "pickleball:queue:processing"
STATUS_TTL = 1800  # 任务状态保留 30 分钟


def _status_key(job_id: str) -> str:
    return f"pickleball:job:{job_id}:status"


def _error_key(job_id: str) -> str:
    return f"pickleball:job:{job_id}:error"


class RedisTaskQueue:
    def __init__(self, redis: Any = None) -> None:
        self._r = redis

    @property
    def redis(self):
        if self._r is None:
            import redis.asyncio as aioredis

            self._r = aioredis.from_url(get_settings().redis_url, decode_responses=True)
        return self._r

    # ------------------------------------------------------------------
    # 生产者
    # ------------------------------------------------------------------
    async def push(self, job: dict) -> str:
        """入队。job 需含 id（缺失则生成），返回 job_id。"""
        job_id = job.get("id") or uuid.uuid4().hex[:16]
        job["id"] = job_id
        raw = json.dumps(job, ensure_ascii=False)
        await self.redis.rpush(PENDING, raw)
        await self.redis.set(_status_key(job_id), "queued", ex=STATUS_TTL)
        return job_id

    # ------------------------------------------------------------------
    # 消费者（worker）
    # ------------------------------------------------------------------
    async def worker_claim(self, timeout: float = 2.0) -> tuple[dict, str] | None:
        """阻塞认领：BRPOPLPUSH 到 processing 后，用 GETSET 原子裁决执行权。

        push 时状态已预置为 "queued"，因此裁决条件为旧值 ∈ {不存在, "queued"}；
        否则说明 API 已内联接管（或任务已结束），从 processing 摘除并放弃。
        """
        raw = await self.redis.brpoplpush(PENDING, PROCESSING, timeout=timeout)
        if not raw:
            return None
        job = json.loads(raw)
        prev = await self.redis.getset(_status_key(job["id"]), "running")
        if prev not in (None, "queued"):
            await self.redis.lrem(PROCESSING, 0, raw)
            return None
        await self.redis.expire(_status_key(job["id"]), STATUS_TTL)
        return job, raw

    async def ack(self, job_id: str, raw: str) -> None:
        await self.redis.lrem(PROCESSING, 0, raw)
        await self.redis.set(_status_key(job_id), "done", ex=STATUS_TTL)

    async def fail(self, job_id: str, raw: str, error: str) -> None:
        await self.redis.lrem(PROCESSING, 0, raw)
        await self.redis.set(_status_key(job_id), "failed", ex=STATUS_TTL)
        await self.redis.set(_error_key(job_id), error[:2000], ex=STATUS_TTL)

    async def recover_stale(self) -> int:
        """回收上次进程崩溃遗留的 processing 任务（worker 启动时调用）。"""
        count = 0
        while True:
            raw = await self.redis.rpoplpush(PROCESSING, PENDING)
            if not raw:
                break
            try:
                job = json.loads(raw)
                await self.redis.set(_status_key(job["id"]), "queued", ex=STATUS_TTL)
            except json.JSONDecodeError:
                pass
            count += 1
        if count:
            logger.info("回收 %d 个滞留任务", count)
        return count

    # ------------------------------------------------------------------
    # API 内联接管（auto 模式：无 worker 在线时 API 自己执行）
    # ------------------------------------------------------------------
    async def try_claim_inline(self, job_id: str, raw: str) -> bool:
        """原子认领执行权（GETSET 裁决）；成功则把任务从两个列表摘除（防 worker 重复执行）。"""
        prev = await self.redis.getset(_status_key(job_id), "running")
        if prev not in (None, "queued"):
            return False
        await self.redis.expire(_status_key(job_id), STATUS_TTL)
        await self.redis.lrem(PENDING, 0, raw)
        await self.redis.lrem(PROCESSING, 0, raw)
        return True

    async def status(self, job_id: str) -> str | None:
        return await self.redis.get(_status_key(job_id))

    async def pending_count(self) -> int:
        return int(await self.redis.llen(PENDING))
