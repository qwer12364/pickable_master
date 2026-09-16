from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.core.config import get_settings
from app.core.harness.security import ExecutionPipeline
from app.core.harness.tools import ToolContext, ToolError, ToolRegistry
from app.core.harness.verify import Verdict, VerificationGate
from app.core.llm import ChatMessage, LLMClient, TokenUsage

logger = logging.getLogger("pickleball.loop")

Emitter = Callable[[dict], Awaitable[None]] | None

@dataclass
class LoopConfig:
    max_steps: int = 8
    timeout_seconds: float = 150.0
    verify_required: bool = True
    max_tool_calls: int = 4

    @classmethod
    def from_settings(cls) -> "LoopConfig":
        s = get_settings()
        return cls(
            max_steps=s.harness_max_steps,
            timeout_seconds=s.harness_timeout_seconds,
            verify_required=s.harness_verify_required,
            max_tool_calls=s.harness_max_tool_calls,
        )


@dataclass
class LoopResult:
    answer: str
    steps: int = 0
    tool_calls_made: int = 0
    verdict: Verdict | None = None
    timed_out: bool = False
    cancelled: bool = False
    usage: TokenUsage = field(default_factory=TokenUsage)
    events: list[dict] = field(default_factory=list)


class ReActLoop:
    """
    Agent的执行循环
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        pipeline: ExecutionPipeline,
        *,
        agent_key: str,
        system_prompt: str,
        config: LoopConfig | None = None,
        verify_gate: VerificationGate | None = None,
        emit: Emitter = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._pipeline = pipeline
        self._agent_key = agent_key
        self._system_prompt = system_prompt
        self._config = config or LoopConfig.from_settings()
        self._verify_gate = verify_gate
        self._emitter = emit
        self._cancel = asyncio.Event()

    def cancel(self) -> None:
        """外部取消"""
        self._cancel.set()

    async def _emit(self, event: dict) -> None:
        if self._emitter is not None:
            try:
                await self._emitter(event)
            except Exception:  #事件推送失败不影响主流程
                logger.warning("事件推送失败: %s", event.get("type"))

    def _max_output_chars(self, name: str) -> int:
        """
        工具输出回灌上下文的最大字符数（按工具规格，未知名回落默认值）。
        """
        try:
            return self._registry.get(name).max_output_chars
        except ToolError:
            return 2000

    async def run(self, task: str, ctx: ToolContext) -> LoopResult:
        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content=task),
        ]
        steps = tool_calls_made = 0
        last_answer = ""
        verdict: Verdict | None = None #业务实体
        events: list[dict] = []        #
        timed_out = cancelled = False
        agent_key = self._agent_key

        async def _ev(event: dict) -> None:
            events.append(event)
            await self._emit(event)

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                while steps < self._config.max_steps:
                    if self._cancel.is_set():
                        cancelled = True
                        await _ev({"type": "agent", "agent": agent_key,
                                   "status": "cancelled"})
                        break
                    steps += 1
                    await _ev({"type": "step", "agent": agent_key, "step": steps})
                    result = await self._llm.chat(messages, tools=self._registry.schemas())
                    await _ev({"type": "usage", "agent": agent_key,
                               "prompt_tokens": result.usage.prompt_tokens,
                               "completion_tokens": result.usage.completion_tokens})
                    assistant_msg = ChatMessage(
                        role="assistant",
                        content=result.content,
                        tool_calls=[tc.model_dump() for tc in result.tool_calls],
                    )
                    messages.append(assistant_msg)
                    if result.tool_calls:
                        for tc in result.tool_calls:
                            if self._cancel.is_set():
                                cancelled = True
                                break
                            if tool_calls_made >= self._config.max_tool_calls:
                                # 上限已满：为剩余 tool_call 补空结果，保证
                                # assistant(tool_calls) 后紧跟 tool 消息，消息序列合法
                                messages.append(ChatMessage(
                                    role="tool", tool_call_id=tc.id,
                                    name=tc.function.name,
                                    content="（系统提示）已达到工具调用上限，本工具未执行，"
                                            "请基于现有信息直接输出最终答案。",
                                ))
                                continue
                            tool_calls_made += 1
                            await _ev({"type": "tool", "tool": tc.function.name,
                                       "status": "called", "args": tc.function.arguments,
                                       "agent": agent_key})
                            tool_result = await self._pipeline.execute(
                                tc.function.name, tc.function.arguments, ctx
                            )
                            status = "success" if tool_result.ok else (
                                "denied" if tool_result.audit and tool_result.audit.decision == "denied"
                                else "error"
                            )
                            await _ev({"type": "tool", "tool": tc.function.name,
                                       "status": status,
                                       "result": tool_result.output[:600],
                                       "duration_ms": tool_result.duration_ms,
                                       "agent": agent_key})
                            messages.append(ChatMessage(
                                role="tool",
                                tool_call_id=tc.id,
                                name=tc.function.name,
                                content=tool_result.output[: self._max_output_chars(tc.function.name)],
                            ))
                        continue
                    last_answer = result.content or ""
                    if not self._config.verify_required or self._verify_gate is None:
                        verdict = Verdict(passed=True, reason="质量门禁未启用")
                    else:
                        verdict = await self._verify_gate.verify(task=task, answer=last_answer)
                        await _ev({"type": "verify", "agent": agent_key,
                                   "passed": verdict.passed, "reason": verdict.reason})
                    if verdict.passed:
                        return LoopResult(answer=last_answer, steps=steps,
                                          tool_calls_made=tool_calls_made,
                                          verdict=verdict, usage=self._llm.usage.snapshot,
                                          events=events)
                    await _ev({"type": "reflect", "agent": agent_key,
                               "reason": verdict.reason})
                    messages.append(ChatMessage(
                        role="user",
                        content=f"[质量门禁反馈] 你的答案未通过检查：{verdict.reason}\n"
                                "请修正后重新输出；若需要补充事实依据，可以继续调用工具。",
                    ))
        except TimeoutError:
            timed_out = True
            await _ev({"type": "agent", "agent": agent_key, "status": "timeout"})

        answer = last_answer or "（达到执行上限，未产出合格答案）"
        return LoopResult(answer=answer, steps=steps,
                          tool_calls_made=tool_calls_made, verdict=verdict,
                          timed_out=timed_out, cancelled=cancelled,
                          usage=self._llm.usage.snapshot, events=events)
