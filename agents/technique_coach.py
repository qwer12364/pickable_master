"""技术教练"""
from app.agents.base import AgentInfo, BaseAgent

INFO = AgentInfo(
    key="technique_coach",
    title="技术教练",
    description="技术动作指导：握拍、发球、第三拍吊球、丁克球、截击、步法与常见错误纠正",
    topics=("技术", "动作", "发球", "丁克", "吊球", "训练"),
)


class TechniqueCoach(BaseAgent):
    info = INFO
    retrieval_kinds = ("technique",)

    def system_prompt(self) -> str:
        return (
            "你是匹克球技术教练，擅长把复杂技术拆解成可执行的练习步骤。\n"
            "工作守则：\n"
            "1. 技术要领（握法、击球点、步法）以自动注入的知识库证据为基础，"
            "可补充训练方法。\n"
            "2. 回答结构：动作要点 → 常见错误 → 1-2 个针对性练习。\n"
            "3. 面向业余爱好者，用大白话，避免专业术语堆砌；不编造具体数字参数。"
        )
