"""Streaming baseline 的统一配置加载与模型构建。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from evoocc.baselines import (
    LearnedDirectAttentionAligner,
    LearnedDirectFusionAligner,
    NeuralOdeDt100Aligner,
)
from evoocc.config import load_config, load_config_with_base, merge_dict
from evoocc.utils.checkpoints import load_checkpoint_for_eval


SUPPORTED_BASELINES = (
    "learned_direct_attention",
    "learned_direct_fusion",
    "neural_ode_dt_100",
)

_BASELINES_ROOT = Path(__file__).resolve().parents[1] / "baselines"


def load_config_with_baseline_overlay(
    config_path: str,
    baseline_name: str,
) -> Dict[str, Any]:
    """加载主配置并叠加 baseline 专属配置。"""
    if baseline_name not in SUPPORTED_BASELINES:
        raise ValueError(
            f"未知 baseline: {baseline_name!r}，可选 {SUPPORTED_BASELINES}"
        )
    overlay_path = _BASELINES_ROOT / baseline_name / "occ3d_config.yaml"
    return merge_dict(
        load_config_with_base(config_path),
        load_config(str(overlay_path)),
    )


def _common_model_kwargs(
    cfg: Dict[str, Any],
    baseline_name: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    data_cfg = cfg["data"]
    baseline_cfg = cfg["model"][baseline_name]
    input_grid_size = tuple(
        baseline_cfg.get("input_grid_size", [200, 200, 16])
    )
    if tuple(data_cfg["grid_size"]) != input_grid_size:
        raise ValueError(
            f"数据 grid_size 与 {baseline_name} 输入网格不一致："
            f"{tuple(data_cfg['grid_size'])} vs {input_grid_size}"
        )
    kwargs = {
        "num_classes": int(data_cfg["num_classes"]),
        "encoder_in_channels": int(
            cfg["model"].get("encoder_in_channels", data_cfg["num_classes"])
        ),
        "free_index": int(data_cfg["free_index"]),
        "pc_range": tuple(data_cfg["pc_range"]),
        "voxel_size": tuple(data_cfg["voxel_size"]),
        "latent_dim": int(baseline_cfg.get("latent_dim", 288)),
        "decoder_channels": int(baseline_cfg.get("decoder_channels", 32)),
        "decoder_init_scale": baseline_cfg.get(
            "decoder_init_scale", 1.0e-6
        ),
        "use_fast_residual": bool(
            baseline_cfg.get("use_fast_residual", True)
        ),
        "input_grid_size": input_grid_size,
        "latent_grid_size": tuple(
            baseline_cfg.get("latent_grid_size", [50, 50, 16])
        ),
        "timestamp_scale": float(data_cfg.get("timestamp_scale", 1.0e-6)),
    }
    return kwargs, baseline_cfg


def build_baseline_model_from_config(
    cfg: Dict[str, Any],
    baseline_name: str,
) -> torch.nn.Module:
    """从已合并配置构建 baseline，不加载 checkpoint。"""
    common, baseline_cfg = _common_model_kwargs(cfg, baseline_name)

    if baseline_name == "learned_direct_attention":
        return LearnedDirectAttentionAligner(
            **common,
            attention_inner_dim=int(
                baseline_cfg.get("attention_inner_dim", 96)
            ),
            attention_num_heads=int(
                baseline_cfg.get("attention_num_heads", 8)
            ),
            attention_window_size=tuple(
                baseline_cfg.get("attention_window_size", [5, 5, 4])
            ),
            attention_local_dilations=tuple(
                baseline_cfg.get("attention_local_dilations", [1, 2])
            ),
            attention_gn_groups=int(
                baseline_cfg.get("attention_gn_groups", 8)
            ),
            attention_mlp_ratio=float(
                baseline_cfg.get("attention_mlp_ratio", 2.0)
            ),
        )

    if baseline_name == "learned_direct_fusion":
        return LearnedDirectFusionAligner(
            **common,
            fusion_inner_dim=int(
                baseline_cfg.get("fusion_inner_dim", 104)
            ),
            fusion_body_dilations=tuple(
                baseline_cfg.get("fusion_body_dilations", [1, 3, 5])
            ),
            fusion_gn_groups=int(
                baseline_cfg.get("fusion_gn_groups", 8)
            ),
        )

    if baseline_name == "neural_ode_dt_100":
        return NeuralOdeDt100Aligner(
            **common,
            func_g_inner_dim=int(
                baseline_cfg.get("func_g_inner_dim", 104)
            ),
            func_g_body_dilations=tuple(
                baseline_cfg.get("func_g_body_dilations", [1, 3, 5])
            ),
            func_g_gn_groups=int(
                baseline_cfg.get("func_g_gn_groups", 8)
            ),
            solver_variant=str(
                baseline_cfg.get("solver_variant", "euler")
            ).lower(),
        )

    raise ValueError(
        f"未知 baseline: {baseline_name!r}，可选 {SUPPORTED_BASELINES}"
    )


def build_streaming_baseline(
    config_path: str,
    checkpoint_path: str,
    baseline_name: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict]:
    """构建 baseline、加载评估权重并返回 data 配置。"""
    cfg = load_config_with_baseline_overlay(config_path, baseline_name)
    model = build_baseline_model_from_config(cfg, baseline_name).to(device)
    load_checkpoint_for_eval(checkpoint_path, model=model, strict=False)
    model.eval()
    return model, cfg["data"]
