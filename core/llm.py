from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Literal

import openai
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings

logger = logging.getLogger("pickleball.llm")

_RETRYABLE_EXC = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@dataclass
class TokenUsage:
    """单次 LLM 调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class ChatMessage:
    """对话消息：字段与 OpenAI 协议一一对应，to_dict() 后可直接交给 SDK。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  
    tool_call_id: str | None = None                 
    name: str | None = None                       

    def to_dict(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls is not None:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        return msg


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[dict[str, Any]]
    usage: TokenUsage
    finish_reason: str | None = None


@dataclass
class StreamEvent:

    type: Literal["text", "usage"]
    text: str = ""
    usage: TokenUsage | None = None


class UsageAccumulator:

    def __init__(self) -> None:
        self._usage = TokenUsage()

    def add(self, usage: TokenUsage | None) -> None:
        if usage is not None:
            self._usage = self._usage + usage

    @property
    def snapshot(self) -> TokenUsage:
        return self._usage


def _extract_usage(raw_usage: Any) -> TokenUsage:
    if raw_usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
    )


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.llm_model
        self.usage = UsageAccumulator()
        self._client = AsyncOpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            timeout=60.0,
            max_retries=0,
        )
#
    @staticmethod
    def _normalize(messages: list[Any]) -> list[ChatMessage]:
        return [m if isinstance(m, ChatMessage) else ChatMessage(**m) for m in messages]

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> ChatResult:
        async def _call() -> ChatResult:
            resp = await self._client.chat.completions.create(
                model=model or self.model,
                messages=[m.to_dict() for m in self._normalize(messages)],
                tools=tools,
                tool_choice=tool_choice,
                temperature=(
                    temperature
                    if temperature is not None
                    else self.settings.llm_temperature
                ),
                max_tokens=max_tokens or self.settings.llm_max_tokens,
            )
            choice = resp.choices[0]
            return ChatResult(
                content=choice.message.content,
                tool_calls=choice.message.tool_calls or [],
                usage=_extract_usage(resp.usage),
                finish_reason=choice.finish_reason,
            )

        result = await self._with_retry(_call)
        self.usage.add(result.usage)
        return result

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        async def _call() -> AsyncIterator[StreamEvent]:
            stream = await self._client.chat.completions.create(
                model=model or self.model,
                messages=[m.to_dict() for m in self._normalize(messages)],
                tools=tools,
                temperature=(
                    temperature
                    if temperature is not None
                    else self.settings.llm_temperature
                ),
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                stream=True,
                # 流式默认不带用量，需显式要求；SDK 会在流尾补一个 usage chunk
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.usage is not None:
                    yield StreamEvent(type="usage", usage=_extract_usage(chunk.usage))
                    break
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield StreamEvent(type="text", text=delta.content)
            # 个别兼容端点不返回 usage chunk：补一个空用量事件，保证流总有终止事件
            else:
                yield StreamEvent(type="usage", usage=None)

        async for event in self._with_retry_stream(_call):
            if event.type == "usage":
                self.usage.add(event.usage)
            yield event
    # 重试：指数退避 + 抖动，仅可恢复错误
    async def _with_retry(
        self,
        fn: Callable,
        *,
        retries: int = 3,
        base_delay: float = 1.0,
    ):
        for attempt in range(retries + 1):
            try:
                return await fn()
            except _RETRYABLE_EXC as exc:
                if attempt >= retries:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 0.3)
                logger.warning(
                    "LLM 调用失败(可恢复) 第 %d/%d 次，%.2fs 后重试: %s",
                    attempt + 1, retries, delay, exc,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _with_retry_stream(
        self,
        gen_fn: Callable,
        *,
        retries: int = 2,
        base_delay: float = 1.0,
    ):
        """流式重试：请求发起阶段可重试；一旦开始产出 token 就不再重试（避免重复输出）。"""
        for attempt in range(retries + 1):
            started = False
            try:
                async for event in gen_fn():
                    started = True
                    yield event
                return
            except _RETRYABLE_EXC as exc:
                if started or attempt >= retries:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 0.3)
                logger.warning(
                    "LLM 流式调用失败 第 %d/%d 次，%.2fs 后重试: %s",
                    attempt + 1, retries, delay, exc,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover
