from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from app.core.harness.audit import AuditStore, InvocationAudit
from app.core.harness.tools import (
    Principal,
    SideEffectLevel,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    parse_arguments,
)

logger = logging.getLogger("pickleball.security")


class ToolSecurityError(ToolError):

    def __init__(self, phase: str, reason: str, tool: str, principal: Principal):
        self.phase = phase
        self.reason = reason
        self.tool = tool
        self.principal = principal
        super().__init__(f"[{phase}] {tool}: {reason}")


@dataclass
class TrustPolicy:
    role_max_side_effect: dict[str, SideEffectLevel] = field(default_factory=lambda: {
        "user": SideEffectLevel.WRITE_LOCAL,
        "admin": SideEffectLevel.EXEC,
    })
    permission_roles: dict[str, set[str]] = field(default_factory=lambda: {
        "tools:read": {"user", "admin"},
        "tools:write": {"user", "admin"},
        "tools:exec": {"admin"},
    })
    

    def check(self, tool, principal: Principal) -> None:
        # 信任策略
        limit = self.role_max_side_effect.get(principal.role, SideEffectLevel.READ_ONLY)
        if tool.side_effect > limit:
            raise ToolSecurityError(
                "trust",
                f"副作用等级 {tool.side_effect.name} 超出角色 {principal.role} 的上限 {limit.name}",
                tool.name, principal,
            )
        # 权限校验
        for perm in tool.permissions:
            if perm not in principal.permissions:
                raise ToolSecurityError(
                    "authorize",
                    f"缺少权限 {perm}",
                    tool.name, principal,
                )


class ExecutionPipeline:

    def __init__(
        self,
        registry: ToolRegistry,
        audit_store: AuditStore,
        policy: TrustPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.audit_store = audit_store
        self.policy = policy or TrustPolicy()

    async def execute(self, name: str, arguments: str, ctx: ToolContext) -> ToolResult:
        t0 = time.perf_counter()

        def _audit(decision: str, phase: str, result_summary: str = "") -> InvocationAudit:
            return InvocationAudit(
                user_id=ctx.principal.user_id,
                username=ctx.principal.username,
                tool=name,
                side_effect=0,
                decision=decision,
                phase=phase,
                args_summary=(arguments or "")[:500],
                result_summary=result_summary[:500],
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )

        try:
            tool = self.registry.get(name)
        except ToolError as exc:
            audit = _audit("denied", "discover", str(exc))
            await self.audit_store.record(audit)
            return ToolResult(ok=False, output=f"工具调用被拒绝: {exc}",
                              audit=audit)

        try:
            self.policy.check(tool, ctx.principal)
            try:
                raw = parse_arguments(arguments)#
            except Exception as exc:  # noqa: BLE001
                raise ToolError(f"参数不是合法 JSON: {exc}") from exc
            if tool.params_model is not None:
                params = tool.params_model.model_validate(raw)
                kwargs = params.model_dump()
            else:
                kwargs = raw
                
            async def _run() -> str:
                result = await tool.handler(ctx, **kwargs)
                return result if isinstance(result, str) else str(result)

            output = await asyncio.wait_for(_run(), timeout=tool.timeout)
            audit = _audit("approved", "execute", output)
            audit.side_effect = int(tool.side_effect)
            await self.audit_store.record(audit)
            return ToolResult(ok=True, output=output, duration_ms=audit.duration_ms,
                              audit=audit)

        except ToolSecurityError as exc:
            audit = _audit("denied", exc.phase, exc.reason)
            audit.side_effect = int(tool.side_effect)
            await self.audit_store.record(audit)
            logger.warning("工具调用被拒绝: %s", exc)
            return ToolResult(ok=False, output=f"工具调用被拒绝: {exc}", audit=audit)
        except (ToolError, Exception) as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"[:300]
            audit = _audit("error", "execute", msg)
            audit.side_effect = int(tool.side_effect)
            await self.audit_store.record(audit)
            return ToolResult(ok=False, output=f"工具执行失败: {msg}", audit=audit)
