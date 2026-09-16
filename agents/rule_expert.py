"""规则专家"""
from app.agents.base import AgentInfo, BaseAgent

INFO = AgentInfo(
    key="rule_expert",
    title="规则专家",
    description="匹克球官方规则：场地尺寸、发球规则、双弹跳规则、计分、厨房区(NVZ)、犯规与判罚",
    topics=("规则", "计分", "犯规", "发球", "厨房区"),
)


class RuleExpert(BaseAgent):
    info = INFO
    retrieval_kinds = ("rules",)

    def system_prompt(self) -> str:
        return (
            "你是匹克球规则专家，精通 USAP/中国匹克球协会官方规则。\n"
            "工作守则：\n"
            "1. 涉及规则条文的回答，以自动注入的【已检索知识库证据】为准，"
            "禁止凭记忆编造数字（尺寸、分数、时间）；证据不足时可用 web_search 补充。\n"
            "2. 回答用中文，先给结论，再给依据；引用知识库来源（如【rules/发球规则】）。\n"
            "3. 规则有争议或版本差异时，说明适用版本（如 USAP 2024）。"
        )
