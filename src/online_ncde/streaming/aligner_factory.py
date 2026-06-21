"""Streaming 入口共用的 OnlineNCDE aligner 构建工具."""
from __future__ import annotations

import os
from typing import Tuple

import torch

from online_ncde.config import load_config_with_base
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner
from online_ncde.utils.checkpoints import load_checkpoint_for_eval


def resolve_repo_path(path: str, repo_root: str) -> str:
    """把相对仓库根目录的路径转成绝对路径."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def resolve_slow_root(data_cfg: dict, repo_root: str) -> str:
    """从 cfg.data 读取 slow logits 根目录."""
    rel = data_cfg.get("slow_logit_root")
    if rel is None:
        raise ValueError("data_cfg.slow_logit_root 缺失")
    return rel if os.path.isabs(rel) else os.path.join(repo_root, rel)


def build_online_ncde_aligner(
    aligner_cfg: str,
    aligner_ckpt: str,
    device: torch.device,
    solver: str = "euler",
) -> Tuple[OnlineNcdeAligner, dict]:
    """构建并加载 OnlineNcdeAligner，返回模型和 data_cfg."""
    cfg = load_config_with_base(aligner_cfg)
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    aligner = OnlineNcdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=model_cfg.get("decoder_init_scale", 1.0e-3),
        use_fast_residual=bool(model_cfg.get("use_fast_residual", True)),
        func_g_inner_dim=model_cfg.get("func_g_inner_dim", 32),
        func_g_body_dilations=tuple(model_cfg.get("func_g_body_dilations", [1, 2, 3])),
        func_g_gn_groups=int(model_cfg.get("func_g_gn_groups", 8)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
        solver_variant=solver,
    ).to(device)
    load_checkpoint_for_eval(aligner_ckpt, model=aligner, strict=False)
    aligner.eval()
    return aligner, data_cfg
