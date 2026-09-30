from __future__ import annotations

from typing import Mapping


def format_metrics(metrics: Mapping[str, float], keys: list[str]) -> str:
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


