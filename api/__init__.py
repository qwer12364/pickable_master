"""API 路由汇总：认证 / 会话 / 对话（SSE）/ 审计 / 笔记本 / 集群观测。"""
from fastapi import APIRouter

from app.api import audit, auth, chat, notes, workers
from backend.app.core import conversations

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(conversations.router)
api_router.include_router(chat.router)
api_router.include_router(audit.router)
api_router.include_router(notes.router)
api_router.include_router(workers.router)
