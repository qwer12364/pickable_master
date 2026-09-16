from __future__ import annotations

import json
from typing import Any, AsyncIterator

from app.core.config import get_settings


class EventBus:
    def __init__(self, prefix: str = "pickleball", redis_url: str | None = None) -> None:
        self._prefix = prefix
        self._url = redis_url or get_settings().redis_url
        self._pub = None

    def channel(self, session_id: str) -> str:
        return f"{self._prefix}:events:{session_id}"

    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        if self._pub is None:
            import redis.asyncio as aioredis

            self._pub = aioredis.from_url(self._url, decode_responses=True)
        await self._pub.publish(
            self.channel(session_id), json.dumps(event, ensure_ascii=False)
        )

    async def subscribe(self, session_id: str) -> AsyncIterator[dict[str, Any]]:
        import redis.asyncio as aioredis

        client = aioredis.from_url(self._url, decode_responses=True)
        pubsub = client.pubsub()
        channel = self.channel(session_id)
        # 立即完成订阅（而非惰性到首次迭代）：保证"先订阅、后派发"，不丢早期事件
        await pubsub.subscribe(channel)

        async def gen() -> AsyncIterator[dict[str, Any]]:
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        yield json.loads(message["data"])
            finally:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.aclose()
                finally:
                    await client.aclose()
        return gen()


        