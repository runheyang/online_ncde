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
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def load_checkpoint_for_eval(
    path: str,
    model: torch.nn.Module,
    strict: bool = False,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    ema_blob = payload.get("ema")
    if isinstance(ema_blob, dict) and "module" in ema_blob:
        model.load_state_dict(ema_blob["module"], strict=strict)
        print(f"[ckpt] loaded EMA weights from {path}")
    else:
        state = payload.get("model", payload)
        model.load_state_dict(state, strict=strict)
        print(f"[ckpt] no EMA found, loaded raw weights from {path}")
    return payload
