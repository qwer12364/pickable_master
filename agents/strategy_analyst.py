"""战术分析师"""
from app.agents.base import AgentInfo, BaseAgent

INFO = AgentInfo(
    key="strategy_analyst",
    title="战术分析师",
    description="比赛战术与策略：双打配合、抢占网前、第三拍战术、叠位(Stacking)、比分与心理策略",
    topics=("战术", "策略", "双打", "比赛", "叠位"),
)


class StrategyAnalyst(BaseAgent):
    info = INFO
    retrieval_kinds = ("strategy",)

    def system_prompt(self) -> str:
        return (
            "你是匹克球战术分析师，擅长比赛策略与双打配合。\n"
            "工作守则：\n"
            "1. 战术原则以自动注入的知识库证据为基础，结合实际场景给出打法建议。\n"
            "2. 回答结构：核心思路 → 具体执行（站位/线路/节奏）→ 常见反制与调整。\n"
            "3. 区分场景（单打/双打、水平差距、比分压力），不给出脱离场景的空泛建议。"
        )
