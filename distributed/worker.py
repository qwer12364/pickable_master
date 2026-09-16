from __future__ import annotations

import argparse
import asyncio
import logging
import uuid

from app.core.config import get_settings
from app.db import init_db
from app.distributed.pubsub import EventBus
from app.distributed.queue import RedisTaskQueue
from app.distributed.runner import run_chat_job

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pickleball.worker")


async def _heartbeat(redis, worker_id: str, ttl: int) -> None:
    key = f"pickleball:worker:{worker_id}"
    while True:
        try:
            await redis.set(key, "alive", ex=ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("心跳失败: %s", exc)
        await asyncio.sleep(ttl / 2)


async def process_forever(
    queue: RedisTaskQueue,
    bus: EventBus,
    index: int,
) -> None:
    while True:
        try:
            claimed = await queue.worker_claim(timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%d] 认领失败: %s", index, exc)
            await asyncio.sleep(3)
            continue
        if claimed is None:
            continue
        job, raw = claimed
        session_id = job.get("session_id") or uuid.uuid4().hex[:16]
        logger.info("[%d] 领取任务 %s (会话 %s)", index, job.get("id"), session_id)

        async def emitter(event: dict) -> None:
            await bus.publish(session_id, event)

        try:
            await run_chat_job(job, emitter=emitter)
            await queue.ack(job["id"], raw)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%d] 任务执行异常", index)
            await bus.publish(session_id, {"type": "error",
                                           "message": f"worker 异常: {exc}"[:500]})
            await queue.fail(job["id"], raw, str(exc))


async def main(concurrency: int) -> None:
    settings = get_settings()
    await init_db()
    logger.info("worker 启动: 并发=%d, Redis=%s", concurrency, settings.redis_url)

    import redis.asyncio as aioredis

    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    queue = RedisTaskQueue(redis)
    recovered = await queue.recover_stale()
    if recovered:
        logger.info("已回收 %d 个滞留任务", recovered)

    bus = EventBus(redis_url=settings.redis_url)
    worker_id = uuid.uuid4().hex[:8]

    tasks = [asyncio.create_task(_heartbeat(redis, worker_id, settings.worker_heartbeat_ttl))]
    tasks += [asyncio.create_task(process_forever(queue, bus, i))
              for i in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    finally:
        from app.core.mcp import aclose_all_pools

        await aclose_all_pools()  # Windows 防 uvx 等 stdio 子进程残留
        await redis.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="匹克球助手分布式 worker")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="并发任务数（默认 2）")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency))
