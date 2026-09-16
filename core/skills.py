from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.harness.tools import SideEffectLevel, ToolContext, ToolSpec

logger = logging.getLogger("pickleball.skills")

# load_skill 回灌 ReAct 上下文的上限（与技能正文长度约定一致）
MAX_SKILL_BODY_CHARS = 8000


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str     # frontmatter 之后的 markdown 正文（原样保留）
    path: Path


def parse_skill_md(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("技能文件读取失败 %s: %s", path, exc)
        return None
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        logger.warning("技能 %s 缺少 YAML frontmatter，已跳过", path)
        return None
    try:
        end = next(i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---")
    except StopIteration:
        logger.warning("技能 %s frontmatter 未闭合，已跳过", path)
        return None
    try:
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        logger.warning("技能 %s frontmatter 解析失败: %s", path, exc)
        return None
    if not isinstance(meta, dict):
        logger.warning("技能 %s frontmatter 不是键值映射，已跳过", path)
        return None
    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    body = "\n".join(lines[end + 1:]).strip()
    if not name or not description or not body:
        logger.warning("技能 %s 加载失败，已跳过", path)
        return None
    return Skill(name=name, description=description, content=body, path=path)


class SkillRegistry:
    def __init__(self, dirs: list[Path] | None = None) -> None:
        self.dirs: list[Path] = dirs or [get_settings().skills_path]
        self._skills: dict[str, Skill] = {}

    def load(self) -> "SkillRegistry":
        for d in self.dirs:
            if not d.is_dir():
                logger.info("技能目录不存在，跳过: %s", d)
                continue
            for md in sorted(d.glob("*/SKILL.md")):
                skill = parse_skill_md(md)
                if skill is None:
                    continue
                #可以在这增加审查力度
                self._skills[skill.name] = skill
        return self

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)

    def index_text(self) -> str:
        """索引（名称+简介）"""
        return "\n".join(
            f"- {s.name}: {s.description}" for s in self._skills.values()
        )


class LoadSkillParams(BaseModel):
    name: str = Field(..., description="技能名称")

#
def make_load_skill_tool(registry: SkillRegistry) -> ToolSpec | None:
    """构造 load_skill 工具"""
    if not registry:
        return None

    async def handler(ctx: ToolContext, name: str) -> str:
        skill = registry.get(name)
        if skill is None:
            available = "、".join(registry.names()) or "（无）"
            return f"未知技能: {name}。可用技能: {available}"
        return f"【技能 {skill.name}】{skill.description}\n\n# 技能正文\n\n{skill.content}"

    return ToolSpec(
        name="load_skill",
        description=(
            "按需加载技能完整说明（渐进式披露）：系统提示词只含技能索引，"
            "调用本工具获取技能的详细操作步骤，随后严格按技能内容执行。"
        ),
        handler=handler,
        params_model=LoadSkillParams,
        side_effect=SideEffectLevel.READ_ONLY,
        permissions={"tools:read"},
        timeout=10.0,
        max_output_chars=MAX_SKILL_BODY_CHARS,
    )


@lru_cache(maxsize=1)
def get_skill_registry() -> SkillRegistry:
    return SkillRegistry().load()
