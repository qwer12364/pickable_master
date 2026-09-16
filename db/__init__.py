"""数据库引擎与会话管理（懒初始化，便于测试注入独立库）。

- ensure_engine()：首次调用时按配置创建 AsyncEngine（进程内单例）；
- init_db()：建表（开发期方案；生产建议引入 Alembic 迁移）；
- get_sessionmaker()：返回 async_sessionmaker 供依赖注入。
"""
import logging
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base

logger = logging.getLogger("pickleball.db")


@lru_cache
def ensure_engine() -> AsyncEngine:
    url = get_settings().database_url
    logger.info("初始化数据库引擎: %s", url.split("@")[-1])
    return create_async_engine(url, pool_pre_ping=True)


def get_sessionmaker() -> async_sessionmaker:
    return async_sessionmaker(ensure_engine(), expire_on_commit=False)


async def init_db() -> None:
    """建表（幂等）。"""
    engine = ensure_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据表就绪")
