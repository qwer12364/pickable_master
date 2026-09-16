"""工具注册表：Harness 的第一块积木。

设计要点：
- 工具 = 处理器 + Pydantic 参数模型 + 副作用等级 + 所需权限，四要素显式声明；
- JSON Schema 自动从参数模型生成（$defs 内联），开发者无需手写 schema；
- 运行时二次校验：模型产出的 JSON 参数必须通过 Pydantic 校验才能执行（防参数注入）；
- 工具处理器一律异步，统一带 ToolContext（携带主体身份与会话上下文）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from app.core.harness.audit import InvocationAudit


class SideEffectLevel(IntEnum):
    """副作用等级：权限策略的判据。等级越高，越危险。"""

    READ_ONLY = 0     # 纯读取：检索、查询
    WRITE_LOCAL = 1   # 本地写：写文件、改数据库
    EXEC = 2          # 执行命令/外部调用


class ToolError(Exception):
    """工具执行失败（已审计，可安全展示给模型促其反思）。"""


@dataclass(frozen=True)
class Principal:
    """请求主体：来自 JWT / 系统内部。"""

    user_id: int
    username: str
    role: str
    #不可变什么时候？声明后？
    permissions: frozenset[str] = frozenset()

    @classmethod
    def system(cls) -> "Principal":
        return cls(user_id=0, username="system", role="admin",
                   permissions=frozenset({"tools:read", "tools:write", "tools:exec"}))

    @classmethod
    def from_user(cls, user: Any) -> "Principal":
        perms = {"tools:read", "tools:write"}
        if user.role == "admin":
            perms.add("tools:exec")
        return cls(user_id=user.id, username=user.username, role=user.role,
                   permissions=frozenset(perms))


@dataclass
class ToolContext:
    """工具执行上下文：主体身份 + 会话 + 可选共享资源。"""

    principal: Principal
    session_id: str = ""
    llm: Any = None                 # LLMClient，复合工具（如子 Agent 调用）可用
    knowledge: Any = None           # KnowledgeStore，检索类工具使用
    emit: Callable[[dict], Awaitable[None]] | None = None
    # 本轮请求附带的图片（extract_chart_data 等视觉工具使用）：
    # [{"name": str, "mime": str, "data": str(data URL)}]，仅请求内流转不落库
    attachments: list[dict] = field(default_factory=list)


@dataclass
class ToolSpec:
    """工具的完整声明。"""

    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    params_model: type[BaseModel] | None = None
    # 外部传入的 JSON Schema：
    #为 None时优先用它生成OpenAIschema 参数不Pydantic校验
    json_schema_override: dict[str, Any] | None = None
    side_effect: SideEffectLevel = SideEffectLevel.READ_ONLY
    permissions: set[str] = field(default_factory=lambda: {"tools:read"})
    timeout: float = 30.0
    max_output_chars: int = 2000   

    def openai_schema(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {"type": "object", "properties": {}}
        if self.json_schema_override is not None:
            raw = dict(self.json_schema_override)
            defs = raw.pop("$defs", {})
            parameters = _dereference(raw, defs)
            parameters.pop("title", None)
            parameters.pop("$schema", None)  # 元信息不能当属性发给模型
            parameters.setdefault("type", "object")
        elif self.params_model is not None:
            raw = self.params_model.model_json_schema()
            defs = raw.pop("$defs", {})
            parameters = _dereference(raw, defs)
            parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


def _dereference(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return _dereference(defs[name], defs)
        return {k: _dereference(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_dereference(x, defs) for x in node]
    return node


class UnknownToolError(ToolError):
    """工具名不在注册表中（模型幻觉出的工具）。"""


class ToolRegistry:
    """按名注册/查询工具"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}")
        self._tools[tool.name] = tool

    def register_many(self, tools: list[ToolSpec]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> ToolSpec:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"未知工具: {name}")
        return tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]


@dataclass
class ToolResult:

    ok: bool
    output: str
    error: str = ""
    duration_ms: int = 0
    audit: InvocationAudit | None = None


def parse_arguments(arguments: str) -> dict[str, Any]:
    if not arguments or not arguments.strip():
        return {}
    return json.loads(arguments)
