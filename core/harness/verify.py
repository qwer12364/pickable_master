from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

VERIFY_PROMPT = """你是严格的质量门禁审查员。请判断「最终答案」是否合格。

【任务】
{task}

【最终答案】
{answer}

【参考证据】（各专家检索到的材料，可能为空）
{evidence}

审查标准：
1. 答案是否完整回应了任务，结论明确；
2. 答案中的事实性内容是否与证据矛盾，或存在明显编造；
3. 若证据为空且答案给出具体规则条文/产品参数，视为不可靠。

只输出一行 JSON：{{"passed": true 或 false, "reason": "一句话中文说明"}}"""


@dataclass
class Verdict:
    passed: bool
    reason: str
    raw: str = ""
    usage: Any = None  # 门禁审查调用自身的 TokenUsage（供成本观测）


class VerificationGate:

    def __init__(self, llm_factory: Callable[..., Any], fast_model: str) -> None:
        self._factory = llm_factory
        self._fast_model = fast_model

    async def verify(self, task: str, answer: str, *,
                     evidence: str = "") -> Verdict:
        llm = self._factory(self._fast_model)
        prompt = VERIFY_PROMPT.format(
            task=task[:2000], answer=answer[:4000], evidence=(evidence or "（无）")[:3000],
        )
        result = await llm.chat(
            [{"role": "user", "content": prompt}],  # 兼容 dict 与 ChatMessage
            temperature=0.0,
        )
        raw = (result.content or "").strip()
        verdict = _parse_verdict(raw)
        return Verdict(passed=verdict[0], reason=verdict[1], raw=raw,
                       usage=result.usage)


def _parse_verdict(raw: str) -> tuple[bool, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        return bool(data.get("passed")), str(data.get("reason", ""))[:300]
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r'"passed"\s*:\s*(true|false)', raw, re.IGNORECASE)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]{0,300})"', raw)
    if match:
        return match.group(1).lower() == "true", (reason_match.group(1) if reason_match else "")
    return "通过" not in raw[:50] or "未通过" in raw, raw[:300]
