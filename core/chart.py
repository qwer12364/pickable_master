from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.harness.tools import SideEffectLevel, ToolContext, ToolError, ToolSpec

logger = logging.getLogger("pickleball.chart")

EXTRACT_PROMPT = (
    "你是比赛数据图表提取器。用户上传了一张图表（可能是饼图、柱状图、折线图"
    "或数据表截图）。请完成：\n"
    "1) 用 markdown 表格列出图中全部可见的数据项与数值；\n"
    "2) 用不超过 3 行列出关键指标。\n"
    "只输出提取结果本身，不要评论、不要分析、不要总结。"
    "图中无法辨认的数字一律用 ? 标注——绝不猜测或编造数字。"
)


def _effective(settings: Settings) -> tuple[str, str, str]:
    """VISION_* 缺省回落到 LLM_*（主模型支持图片则零配置可用）。"""
    return (
        settings.vision_model or settings.llm_model,
        settings.vision_base_url or settings.llm_base_url,
        settings.vision_api_key or settings.llm_api_key,
    )


async def extract_chart(
    settings: Settings,
    image: dict,
    *,
    focus: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    model, base_url, api_key = _effective(settings)
    if not api_key:
        raise ToolError("未配置任何 LLM API Key，图表提取不可用")
    prompt = EXTRACT_PROMPT
    if focus:
        prompt += f"\n用户关注的指标：{focus}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image["data"]}},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/chat/completions"
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=45.0)
    try:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = str(resp.json().get("error", {}).get("message", ""))[:200]
            except Exception:  # noqa: BLE001 非 JSON 错误体不解析
                pass
            raise ToolError(
                f"视觉模型返回 {resp.status_code}: {detail or resp.text[:200]}"
                "（主模型可能不支持图片输入，可配置 VISION_MODEL 指向视觉模型）"
            )
        data = resp.json()
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 网络层异常统一包装
        raise ToolError(f"图表提取请求失败: {exc}") from exc
    finally:
        if owns_client:
            await client.aclose()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    return (content or "").strip() or "（视觉模型未返回内容）"


class ChartExtractParams(BaseModel):
    image_index: int = Field(0, ge=0, description="要提取的图片序号（从 0 开始）")
    focus: str = Field("", max_length=200,
                       description="关注的指标，如：发球得分率、非受迫失误数；没有则留空")


def make_chart_tool(settings: Settings | None = None) -> ToolSpec | None:
    s = settings or get_settings()
    _model, _base, api_key = _effective(s)
    if not api_key:
        logger.info("未配置 LLM API Key，extract_chart_data 工具不注册")
        return None

    async def handler(ctx: ToolContext, image_index: int,
                      focus: str = "") -> str:
        attachments = ctx.attachments
        if not attachments:
            raise ToolError("本轮请求未附带图片，无法提取图表数据")
        if image_index >= len(attachments):
            raise ToolError(
                f"图片序号 {image_index} 超出范围（本轮共 {len(attachments)} 张）")
        image = attachments[image_index]
        text = await extract_chart(s, image, focus=focus)
        return f"【{image.get('name', '图片')}】\n{text}"

    return ToolSpec(
        name="extract_chart_data",
        description=(
            "提取用户本轮上传的图表图片（饼图/柱状图/折线图/数据表截图）中的数据，"
            "返回 markdown 表格与关键指标。仅在用户上传了图片、且需要图表数据时调用；"
            "拿到数据后再基于数据回答，绝不凭空解读图片。"
        ),
        handler=handler,
        params_model=ChartExtractParams,
        side_effect=SideEffectLevel.READ_ONLY,
        permissions={"tools:read"},
        timeout=45.0,
        max_output_chars=2500,  # 表格文本较长，放宽默认 2000 截断
    )
