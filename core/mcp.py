from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncContextManager, Callable

from app.core.harness.tools import (
    SideEffectLevel,
    ToolContext,
    ToolError,
    ToolSpec,
)

import httpx2
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ImageContent, TextContent

MCP_AVAILABLE = True


_SIDE_EFFECT_LEVELS = {
    "read_only": SideEffectLevel.READ_ONLY,
    "write_local": SideEffectLevel.WRITE_LOCAL,
    "exec": SideEffectLevel.EXEC,
}


@dataclass
class McpServerConfig:

    name: str
    transport: str                      # "stdio" | "http"
    command: str = ""                   # stdio：可执行文件
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""                       # http：Streamable HTTP 端点
    headers: dict[str, str] = field(default_factory=dict)
    side_effect: str = "read_only"      # 该 server 全部工具的副作用等级
    timeout: float = 30.0               # 单次工具调用超时（秒）
    connect_timeout: float = 10.0       # 连接/握手/列举超时（秒）
    enabled: bool = True


def parse_mcp_servers(raw: str) -> list[McpServerConfig]:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("MCP_SERVERS 不是合法 JSON，MCP 已禁用: %s", exc)
        return []
    if not isinstance(data, list):
        logger.warning("MCP_SERVERS 必须是 JSON 数组，MCP 已禁用")
        return []
    servers: list[McpServerConfig] = []
    for i, item in enumerate(data):
        try:
            servers.append(_parse_one(item))
        except (TypeError, ValueError) as exc:
            logger.warning("MCP_SERVERS[%d] 配置无效，已跳过: %s", i, exc)
    return servers


def _parse_one(item: Any) -> McpServerConfig:
    if not isinstance(item, dict):
        raise ValueError("条目必须是对象")
    name = str(item.get("name", "")).strip()
    if not name:
        raise ValueError("缺少 name")
    transport = str(item.get("transport", "")).strip()
    if transport not in ("stdio", "http"):
        raise ValueError("transport 必须是 stdio 或 http")
    cfg = McpServerConfig(
        name=name,
        transport=transport,
        command=str(item.get("command", "")).strip(),
        args=[str(a) for a in item.get("args", [])],
        env={str(k): str(v) for k, v in (item.get("env") or {}).items()},
        url=str(item.get("url", "")).strip(),
        headers={str(k): str(v) for k, v in (item.get("headers") or {}).items()},
        side_effect=str(item.get("side_effect", "read_only")).strip() or "read_only",
        timeout=float(item.get("timeout", 30.0)),
        connect_timeout=float(item.get("connect_timeout", 10.0)),
        enabled=bool(item.get("enabled", True)),
    )
    if transport == "stdio" and not cfg.command:
        raise ValueError("stdio 传输需要 command")
    if transport == "http" and not cfg.url:
        raise ValueError("http 传输需要 url")
    if cfg.side_effect not in _SIDE_EFFECT_LEVELS:
        raise ValueError(f"side_effect 必须是 {'/'.join(_SIDE_EFFECT_LEVELS)} 之一")
    return cfg



_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_\-]")
_SCHEMA_WHITELIST = {
    "type", "properties", "items", "prefixItems", "enum", "const", "required",
    "anyOf", "oneOf", "allOf", "additionalProperties", "description", "default",
    "minimum", "maximum", "multipleOf", "minLength", "maxLength", "pattern",
    "format", "minItems", "maxItems", "uniqueItems", "$defs", "$ref",
}


def sanitize_server_name(name: str) -> str:
    clean = _SANITIZE_RE.sub("_", (name or "").strip())
    return clean[:32] or "mcp"


def mcp_tool_name(server: str, tool: str) -> str:
    t = _SANITIZE_RE.sub("_", (tool or "").strip())[:32] or "tool"
    return f"{sanitize_server_name(server)}__{t}"[:64]


# 这些键的值是「属性名 → 子 schema」映射：名字不是关键字，不做白名单过滤
_NAME_MAP_KEYS = {"properties", "patternProperties", "dependencies", "$defs"}


def adapt_schema(node: Any) -> Any:#

    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _NAME_MAP_KEYS and isinstance(v, dict):
                out[k] = {name: adapt_schema(schema) for name, schema in v.items()}
            elif k in _SCHEMA_WHITELIST:
                out[k] = adapt_schema(v)
        return out
    if isinstance(node, list):
        return [adapt_schema(x) for x in node]
    return node


def format_call_result(result: Any, tool_name: str) -> str:
    parts: list[str] = []
    for block in result.content or []:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ImageContent):
            parts.append(f"[图片: {block.mime_type or 'image'}, "
                         f"{len(block.data or '')} 字节，内容模型不可见]")
        else:
            parts.append(str(block)[:200])  # 兜底占位，截断防 base64 爆炸
    text = "\n".join(p for p in parts if p).strip()
    if result.structured_content is not None:
        extra = json.dumps(result.structured_content, ensure_ascii=False, default=str)
        text = f"{text}\n{extra}".strip()
    text = text or "(空结果)"
    if result.is_error:
        raise ToolError(f"MCP 工具 {tool_name} 返回错误: {text[:300]}")
    return text[:4000]


TransportFactory = Callable[[McpServerConfig], AsyncContextManager[tuple[Any, Any]]]


def _stdio_transport(cfg: McpServerConfig) -> AsyncContextManager[tuple[Any, Any]]:
    # StdioServerParameters.args 必须是 list（不能为 None），无参数时传空表
    params = StdioServerParameters(
        command=cfg.command, args=cfg.args or [], env=cfg.env or None)
    return stdio_client(params)


@asynccontextmanager
async def _http_transport(cfg: McpServerConfig):
    client = httpx2.AsyncClient(headers=cfg.headers or None,
                                timeout=cfg.connect_timeout)
    try:
        async with streamable_http_client(cfg.url, http_client=client,
                                          terminate_on_close=True) as streams:
            yield streams
    finally:
        await client.aclose()



class McpConnection:

    def __init__(self, cfg: McpServerConfig,
                 transport_factory: TransportFactory | None = None) -> None:
        self.cfg = cfg
        self._transport_factory = transport_factory
        self._owner: asyncio.Task | None = None
        self._queue: asyncio.Queue[tuple[str, Any, asyncio.Future]] = asyncio.Queue()

    def _submit(self, cmd: str, payload: Any) -> asyncio.Future:
        if self._owner is None or self._owner.done():
            self._owner = asyncio.create_task(
                self._owner_loop(),
                name=f"mcp-conn-{sanitize_server_name(self.cfg.name)}")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((cmd, payload, fut))
        return fut

    def _kill_owner(self, fut: asyncio.Future) -> None:
        fut.cancel()
        owner, self._owner = self._owner, None
        if owner is not None and not owner.done():
            owner.cancel()

    async def list_tools(self) -> list[Any]:
        fut = self._submit("list_tools", None)
        try:
            return await asyncio.wait_for(fut, timeout=self.cfg.connect_timeout)
        except asyncio.TimeoutError:
            self._kill_owner(fut)
            raise ToolError(
                f"MCP server「{self.cfg.name}」连接/列举超时"
                f"（{self.cfg.connect_timeout}s）") from None
        except Exception:
            self._kill_owner(fut) 
            raise

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        fut = self._submit("call", (name, arguments))
        try:
            result = await asyncio.wait_for(fut, timeout=self.cfg.timeout)
        except asyncio.TimeoutError:
            self._kill_owner(fut)
            raise ToolError(
                f"MCP 工具 {name} 超时（{self.cfg.timeout}s），连接已重置") from None
        except Exception as exc:  # noqa: BLE001 传输层故障：owner 已断开，可重连
            raise ToolError(
                f"MCP 调用失败: {type(exc).__name__}: {exc}") from exc
        # 工具级 isError（服务器正常返回错误结果）不伤连接
        return format_call_result(result, name)

    async def aclose(self) -> None:
        owner = self._owner
        self._owner = None
        if owner is None or owner.done():
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(("close", None, fut))
        try:
            await asyncio.wait_for(asyncio.shield(owner),
                                   timeout=self.cfg.connect_timeout)
        except BaseException: 
            owner.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(owner), timeout=2)
            except BaseException:
                pass

    async def _owner_loop(self) -> None:
        stack: AsyncExitStack | None = None
        session: ClientSession | None = None
        try:
            while True:
                cmd, payload, fut = await self._queue.get()
                if cmd == "close":
                    break
                try:
                    if cmd == "list_tools":
                        if session is None:
                            stack, session = await self._open()
                        assert session is not None
                        tools = await session.list_tools()
                        if not fut.done():
                            fut.set_result(list(tools.tools))
                    elif cmd == "call":
                        name, arguments = payload
                        if session is None:
                            stack, session = await self._open()
                        assert session is not None
                        result = await session.call_tool(name, arguments)
                        if not fut.done():
                            fut.set_result(result)
                except asyncio.CancelledError:
                    raise  # 交给 finally 收尾
                except BaseException as exc: 
                    if stack is not None:
                        await self._teardown(stack)
                        stack = session = None
                    if not fut.done():
                        fut.set_exception(exc)
        finally:
            if stack is not None:
                await self._teardown(stack)

    async def _open(self) -> tuple[AsyncExitStack, ClientSession]:
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(self._make_transport())
            session = ClientSession(read, write,
                                    read_timeout_seconds=self.cfg.connect_timeout)
            await stack.enter_async_context(session)
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        return stack, session

    @staticmethod
    async def _teardown(stack: AsyncExitStack) -> None:
        try:
            await stack.aclose()
        except Exception as exc:  # noqa: BLE001 收尾失败只告警
            logger.warning("MCP 连接收尾告警（可忽略）: %s", exc)

    def _make_transport(self) -> AsyncContextManager[tuple[Any, Any]]:
        if self._transport_factory is not None:
            return self._transport_factory(self.cfg)
        if self.cfg.transport == "http":
            return _http_transport(self.cfg)
        return _stdio_transport(self.cfg)


class McpToolPool:
    def __init__(self, transport_factory: TransportFactory | None = None) -> None:
        self._transport_factory = transport_factory
        self._connections: dict[str, McpConnection] = {}
        self._tools: dict[str, ToolSpec] = {}
        self._closed = False
        _active_pools.append(self)

    async def discover(self, servers: list[McpServerConfig]) -> list[ToolSpec]:
        if not MCP_AVAILABLE:
            logger.warning("mcp SDK 不可用，跳过 MCP 工具发现")
            return []
        for cfg in servers:
            if not cfg.enabled:
                continue
            conn = McpConnection(cfg, transport_factory=self._transport_factory)
            try:
                tools = await conn.list_tools()
            except Exception as exc:  # noqa: BLE001 优雅降级：跳过该 server
                logger.warning("MCP server「%s」连接失败，已跳过: %s", cfg.name, exc)
                await conn.aclose()
                continue
            added = 0
            for tool in tools:
                spec = make_mcp_tool_spec(cfg, conn, tool)
                if spec.name in self._tools:
                    logger.warning("MCP 工具名冲突，已跳过: %s", spec.name)
                    continue
                self._tools[spec.name] = spec
                added += 1
            self._connections[cfg.name] = conn
            logger.info("MCP server「%s」就绪: %d 个工具", cfg.name, added)
        return list(self._tools.values())

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._connections.values():
            await conn.aclose()
        self._connections.clear()
        self._tools.clear()
        if self in _active_pools:
            _active_pools.remove(self)


def make_mcp_tool_spec(cfg: McpServerConfig, conn: McpConnection,
                       tool: Any) -> ToolSpec:
    raw = tool.input_schema
    schema = adapt_schema(raw) if isinstance(raw, dict) else {
        "type": "object", "properties": {}}
    level = _SIDE_EFFECT_LEVELS.get(cfg.side_effect, SideEffectLevel.READ_ONLY)
    server_label = sanitize_server_name(cfg.name)

    async def handler(ctx: ToolContext, **kwargs: Any) -> str:
        return await conn.call_tool(tool.name, kwargs)

    return ToolSpec(
        name=mcp_tool_name(cfg.name, tool.name),
        description=f"[MCP:{server_label}] {tool.description or tool.name}".strip(),
        handler=handler,
        params_model=None,                 # 参数校验交给远端 server
        json_schema_override=schema,
        side_effect=level,
        permissions={"tools:read"} if level == SideEffectLevel.READ_ONLY
                    else {"tools:read", "tools:exec"},
        timeout=cfg.timeout,
    )



_active_pools: list[McpToolPool] = []


async def aclose_all_pools() -> None:
    for pool in list(_active_pools):
        await pool.aclose()
    _active_pools.clear()
