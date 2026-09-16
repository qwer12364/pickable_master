from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("pickleball.audit")


@dataclass
class InvocationAudit:
    user_id: int
    username: str
    tool: str
    side_effect: int
    decision: str            # approved | denied | error
    phase: str               # discover | trust | authorize | execute
    args_summary: str = ""
    result_summary: str = ""
    duration_ms: int = 0
    invocation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "invocation_id": self.invocation_id,
            "user": self.username,
            "tool": self.tool,
            "side_effect": self.side_effect,
            "decision": self.decision,
            "phase": self.phase,
            "args": self.args_summary,
            "result": self.result_summary,
            "duration_ms": self.duration_ms,
            "ts": self.ts.isoformat(),
        }


class AuditStore:
    """工具记录存储。"""

    def __init__(self, ring_size: int = 200) -> None:
        self._ring: deque[InvocationAudit] = deque(maxlen=ring_size)

    async def record(self, audit: InvocationAudit) -> None:
        self._ring.append(audit)
        # 持久化（best-effort）
        try:
            from app.db import get_sessionmaker
            from app.db.models import AuditRecord

            async with get_sessionmaker()() as session:
                session.add(AuditRecord(
                    invocation_id=audit.invocation_id,
                    user_id=audit.user_id,
                    username=audit.username,
                    tool=audit.tool,
                    side_effect=audit.side_effect,
                    decision=audit.decision,
                    phase=audit.phase,
                    args_summary=audit.args_summary,
                    result_summary=audit.result_summary,
                    duration_ms=audit.duration_ms,
                ))
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("审计落库失败（不影响工具执行）: %s", exc)

    def recent(self, n: int = 50) -> list[dict]:
        return [a.to_dict() for a in list(self._ring)[-n:]]
