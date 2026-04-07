"""日志格式化工具。"""

from __future__ import annotations

from typing import Mapping


def format_metrics(metrics: Mapping[str, float], keys: list[str]) -> str:
    """将指标字典按固定键顺序拼接成日志字符串。"""
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


