
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.agents.base import AgentInfo, BaseAgent
from app.core.harness.tools import ToolContext, ToolSpec

INFO = AgentInfo(
    key="equipment_advisor",
    title="装备顾问",
    description="装备选购与对比：球拍（材质/核心/重量）、球（室内/室外）、球鞋、握把与配件",
    topics=("装备", "球拍", "球", "球鞋", "选购"),
)

# 内置装备目录（演示用确定性数据）
CATALOG: dict[str, list[dict]] = {
    "paddle": [
        {"name": "入门碳纤维拍 X1", "budget": "entry", "price": "¥200-300",
         "weight": "220-230g", "core": "聚丙烯蜂窝", "surface": "碳纤维",
         "fit": "新手入门，控球容错高"},
        {"name": "进阶热压成型拍 A2", "budget": "mid", "price": "¥500-800",
         "weight": "215-225g", "core": "热压成型蜂窝", "surface": "T700碳纤维",
         "fit": "进阶选手，旋转与力量均衡"},
        {"name": "专业款 J3K", "budget": "pro", "price": "¥1200-1800",
         "weight": "205-215g", "core": "热压成型+边缘加厚", "surface": "原丝碳纤维",
         "fit": "竞技选手，控制与爆发兼顾"},
        {"name": "训练玻璃纤维拍 T5", "budget": "entry", "price": "¥100-150",
         "weight": "230-240g", "core": "聚丙烯蜂窝", "surface": "玻璃纤维",
         "fit": "力量偏弱的新手，甜区大"},
    ],
    "ball": [
        {"name": "室外球 X-40（40洞）", "budget": "mid", "price": "¥60-90/3只",
         "weight": "约26g", "hardness": "偏硬", "fit": "室外场地标配，抗风耐打"},
        {"name": "室内球 Indoor（26洞）", "budget": "mid", "price": "¥60-90/3只",
         "weight": "约24g", "hardness": "偏软", "fit": "室内木地板/胶地，弹跳更可控"},
        {"name": "训练用耐用球 Dura Fast 40", "budget": "pro", "price": "¥100-150/3只",
         "weight": "约26g", "hardness": "硬", "fit": "赛事级，球速快"},
    ],
    "shoe": [
        {"name": "综合室内鞋 S1", "budget": "entry", "price": "¥300-500",
         "features": "non-marking橡胶底", "fit": "室内球场入门"},
        {"name": "专业匹克球鞋 P2", "budget": "mid", "price": "¥600-900",
         "features": "侧向支撑+耐磨大底", "fit": "频繁变向与急停"},
        {"name": "宽楦舒适款 W3", "budget": "mid", "price": "¥500-700",
         "features": "加宽鞋楦", "fit": "宽脚/足弓支撑需求者"},
    ],
    "grip": [
        {"name": "基础握把带 G1", "budget": "entry", "price": "¥20-40",
         "features": "PU材质", "fit": "吸汗防滑，入门够用"},
        {"name": "专业缠带 Pro Grip", "budget": "pro", "price": "¥50-80",
         "features": "聚氨酯+网眼", "fit": "手汗多的选手，耐用性好"},
    ],
}


class CatalogParams(BaseModel):
    category: Literal["paddle", "ball", "shoe", "grip"] = Field(
        ..., description="装备类别：paddle=球拍 ball=球 shoe=球鞋 grip=握把"
    )
    budget: Literal["entry", "mid", "pro"] | None = Field(
        None, description="预算档位：entry=入门 mid=进阶 pro=专业；不填则返回全部"
    )


def make_catalog_tool() -> ToolSpec:
    async def handler(ctx: ToolContext, category: str, budget: str | None) -> str:
        items = CATALOG.get(category, [])
        if budget:
            items = [i for i in items if i["budget"] == budget]
        if not items:
            return f"目录中没有符合条件（{category}/{budget}）的装备。"
        lines = [f"{i['name']}｜{i['price']}｜{i.get('weight') or i.get('hardness') or i.get('features')}｜适合：{i['fit']}"
                 for i in items]
        return "\n".join(lines)

    return ToolSpec(
        name="get_equipment_catalog",
        description="查询内置装备目录（球拍/球/球鞋/握把），返回型号、价格区间与适用人群",
        handler=handler,
        params_model=CatalogParams,
    )


class EquipmentAdvisor(BaseAgent):
    info = INFO
    retrieval_kinds = ("equipment",)

    def system_prompt(self) -> str:
        return (
            "你是匹克球装备顾问，熟悉球拍、球、球鞋与配件的选购。\n"
            "工作守则：\n"
            "1. 装备参数与选购原则以自动注入的知识库证据为准；"
            "需要具体型号推荐时调用 get_equipment_catalog。\n"
            "2. 推荐必须匹配用户水平与预算；目录里没有的型号不要编造。\n"
            "3. 回答结构：选购原则（简短）→ 具体推荐（含价格区间）→ 一句话理由。"
        )

    def tools(self) -> list[ToolSpec]:
        return [make_catalog_tool()]
