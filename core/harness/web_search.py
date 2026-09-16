"""联网搜索工具：Tavily Search API 的薄封装。

设计要点：
- 与经典 RAG 注入的知识库证据互补：知识库查不到、或涉及时效性问题时由模型决定调用；
- TAVILY_API_KEY 未配置时 make_web_search_tool() 返回 None，Agent 侧自动跳过注册
  （优雅降级，模型看不到该工具就不会调用）；
- 走 REST（httpx），不引入官方 SDK；结果格式化为「标题 + 链接 + 摘要」，
  模型可直接引用来源链接；
- 错误统一抛 WebSearchError（ToolError 子类），由五阶段管线审计后回灌模型。
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from pydantic import Field, create_model

from app.core.config import Settings
from app.core.harness.tools import ToolContext, ToolError, ToolSpec

logger = logging.getLogger("pickleball.websearch")

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class WebSearchError(ToolError):
    """联网搜索失败（网络/额度/限流），可安全展示给模型促其改用知识库。"""


async def tavily_search(
    query: str,
    *,
    api_key: str,
    max_results: int = 5,
    search_depth: str = "basic",
    timeout: float = 20.0,
    client: httpx.AsyncClient | None = None,
) -> str:
    """调用 Tavily Search API 并格式化为文本（测试可注入 MockTransport 客户端）。"""
    if not api_key:
        raise WebSearchError("未配置 TAVILY_API_KEY，联网搜索不可用")
    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max(1, min(max_results, 10)),
        "include_answer": True,
    }
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        resp = await client.post(TAVILY_ENDPOINT, json=payload)
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = str(resp.json().get("detail", ""))[:200]
            except Exception:  # noqa: BLE001 非 JSON 错误体不解析
                pass
            raise WebSearchError(f"Tavily 返回 {resp.status_code}: {detail or resp.text[:200]}")
        data = resp.json()
    except WebSearchError:
        raise
    except Exception as exc:  # noqa: BLE001 网络层异常统一包装
        raise WebSearchError(f"联网搜索请求失败: {exc}") from exc
    finally:
        if owns_client:
            await client.aclose()
    return _format_response(data)


def _format_response(data: dict[str, Any]) -> str:
    answer = (data.get("answer") or "").strip()
    results = data.get("results") or []
    lines: list[str] = []
    if answer:
        lines.append(f"【综合回答】{answer}")
    for i, item in enumerate(results, 1):
        title = (item.get("title") or "").strip() or "(无标题)"
        url = (item.get("url") or "").strip()
        content = re.sub(r"\s+", " ", item.get("content") or "").strip()[:300]
        entry = f"{i}. {title}"
        if url:
            entry += f"\n   {url}"
        if content:
            entry += f"\n   {content}"
        lines.append(entry)
    return "\n\n".join(lines) if lines else "（无搜索结果）"


def make_web_search_tool(settings: Settings) -> ToolSpec | None:
    """构建 web_search 工具；未配置 API Key 时返回 None（调用方跳过注册）。"""
    if not settings.tavily_api_key:
        logger.info("TAVILY_API_KEY 未配置，web_search 工具不注册")
        return None

    params = create_model(
        "WebSearchParams",
        query=(str, Field(..., min_length=1, max_length=200,
                          description="搜索关键词（建议中英文结合，如：匹克球世锦赛 2025 pickleball world championship）")),
        max_results=(int, Field(settings.tavily_max_results, ge=1, le=10,
                                description="返回结果条数")),
    )

    async def handler(ctx: ToolContext, query: str, max_results: int) -> str:
        return await tavily_search(
            query,
            api_key=settings.tavily_api_key,
            max_results=max_results,
            search_depth=settings.tavily_search_depth,
        )

    return ToolSpec(
        name="web_search",
        description=(
            "联网搜索最新信息（赛事动态、规则更新、装备产品参数与行情等）。"
            "当内部知识库检索不到答案、或问题涉及时效性时使用；"
            "返回结果含标题、链接与摘要，回答时请注明来源链接。"
        ),
        handler=handler,
        params_model=params,
        timeout=20.0,
    )
