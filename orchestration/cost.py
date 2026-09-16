from __future__ import annotations

from typing import Any

from app.core.config import get_settings


def aggregate_usage(breakdown: list[dict[str, Any]]) -> dict[str, int]:
    """把逐调用用量明细聚合为总量。"""
    prompt = sum(int(x.get("prompt_tokens") or 0) for x in breakdown)
    completion = sum(int(x.get("completion_tokens") or 0) for x in breakdown)
    return {
        "calls": len(breakdown),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def estimate_cost(breakdown: list[dict[str, Any]]) -> dict[str, float]:
    """按配置的单价估算成本（美元 + 人民币，汇率按 7.2 估算）。"""
    settings = get_settings()
    totals = aggregate_usage(breakdown)
    usd = (
        totals["prompt_tokens"] / 1_000_000 * settings.cost_input_per_mtok
        + totals["completion_tokens"] / 1_000_000 * settings.cost_output_per_mtok
    )
    return {
        "cost_usd": round(usd, 6),
        "cost_cny": round(usd * 7.2, 4),
    }


def usage_report(breakdown: list[dict[str, Any]]) -> dict[str, Any]:
    """完整用量报告：明细聚合 + 成本估算。"""
    return {**aggregate_usage(breakdown), **estimate_cost(breakdown)}
