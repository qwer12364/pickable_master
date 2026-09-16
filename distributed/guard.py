from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("pickleball.guard")

CONV_LOCK_TTL = 360  # 略大于 chat_max_total_seconds(300)，正常任务不可能超期


class ConversationBusyError(Exception):
    """同一会话已有任务在执行。"""


def conv_active_key(conversation_id: Any) -> str:
    return f"pickleball:conv:active:{conversation_id}"


_redis: Any = None


def get_redis():
    """懒加载客户端（每进程一份；首次 await 时绑定当前事件循环）。"""
    global _redis
    if _redis is None:
        import redis.asyncio as aioredis

        _redis = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis


_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


async def acquire_conv_lock(redis: Any, conversation_id: Any, token: str,
                            ttl: int = CONV_LOCK_TTL) -> bool:
    """SET NX 抢占；Redis 异常返回 True（降级为不加锁）。"""
    try:
        return bool(await redis.set(conv_active_key(conversation_id),
                                    token, nx=True, ex=ttl))
    except Exception as exc:  # noqa: BLE001
        logger.warning("会话锁获取失败（降级为不加锁）: %s", exc)
        return True


async def release_conv_lock(redis: Any, conversation_id: Any, token: str) -> None:
    """CAS 释放：只删自己持有的锁，避免误删 TTL 过期后新任务的锁。"""
    try:
        await redis.eval(_RELEASE_LUA, 1, conv_active_key(conversation_id), token)
    except Exception as exc:  # noqa: BLE001
        logger.warning("会话锁释放失败: %s", exc)


async def conv_is_busy(redis: Any, conversation_id: Any) -> bool:
    """回退接口守卫；Redis 异常返回 False（降级放行）。"""
    try:
        return bool(await redis.exists(conv_active_key(conversation_id)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("会话锁检查失败（降级放行）: %s", exc)
        return False
