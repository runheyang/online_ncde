"""权重保存与加载。"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """保存 checkpoint（含 model、optimizer 状态及 epoch）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """加载 checkpoint，可选同时恢复 optimizer 状态。返回原始 payload。"""
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
