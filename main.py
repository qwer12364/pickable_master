
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pickleball")


async def _check_redis(url: str) -> bool:
    import redis.asyncio as aioredis

    client = aioredis.from_url(url, decode_responses=True)
    try:
        return bool(await client.ping())
    except Exception as exc:  # noqa: BLE001 连通性检查需吞掉一切异常
        logger.warning("Redis 连通性检查失败: %s", exc)
        return False
    finally:
        await client.aclose()


async def _check_postgres(url: str) -> bool:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("PostgreSQL 连通性检查失败: %s", exc)
        return False
    finally:
        await engine.dispose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    redis_ok = await _check_redis(settings.redis_url)
    pg_ok = await _check_postgres(settings.database_url)
    logger.info("基础设施检查 => Redis: %s, PostgreSQL: %s", redis_ok, pg_ok)

    # 建表（幂等）
    from app.db import init_db

    await init_db()
    import redis.asyncio as aioredis

    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    from app.distributed.pubsub import EventBus
    from app.distributed.queue import RedisTaskQueue

    app.state.bus = EventBus(redis_url=settings.redis_url)
    app.state.queue = RedisTaskQueue(app.state.redis)
    from app.agents import build_agent_team, default_llm_factory
    from app.core.mcp import McpToolPool, aclose_all_pools, parse_mcp_servers
    from app.core.retrieval import get_knowledge_store
    from app.core.skills import get_skill_registry
    from app.orchestration.graph import build_chat_graph

    app.state.knowledge = get_knowledge_store()
    app.state.skills = get_skill_registry() 
    app.state.mcp_pool = McpToolPool()
    shared_tools = []
    servers = parse_mcp_servers(settings.mcp_servers)
    if servers:
        try:
            shared_tools = await app.state.mcp_pool.discover(servers)
        except Exception as exc:  # noqa: BLE001 优雅降级
            logger.warning("MCP 工具发现失败，已跳过: %s", exc)
    from app.core.notebook import make_notebook_tools

    shared_tools += make_notebook_tools()
    from app.core.chart import make_chart_tool

    chart_tool = make_chart_tool(settings)
    if chart_tool is not None:
        shared_tools.append(chart_tool)

    factory = default_llm_factory(settings)
    app.state.team = build_agent_team(factory, settings=settings,
                                      knowledge=app.state.knowledge,
                                      shared_tools=shared_tools,
                                      skills=app.state.skills)
    app.state.graph = build_chat_graph(app.state.team, factory, settings=settings)
    logger.info("Agent 团队就绪: %s", ", ".join(app.state.team.keys()))
    logger.info("技能注册表就绪: %d 个", len(app.state.skills))

    yield

    await aclose_all_pools() 
    await app.state.redis.aclose()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict:
        return {"status": "ok", "service": settings.app_name}

    @app.get("/api/v1/health", tags=["health"])
    async def health() -> dict:
        redis_ok = await _check_redis(settings.redis_url)
        pg_ok = await _check_postgres(settings.database_url)
        return {
            "status": "ok" if (redis_ok and pg_ok) else "degraded",
            "components": {"redis": redis_ok, "postgres": pg_ok},
        }

    return app


app = create_app()
