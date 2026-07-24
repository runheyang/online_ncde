#!/usr/bin/env python3
"""50×50×16 learned direct fusion baseline 训练入口。

复用 train_rwfa.py 的数据、DDP、EMA、loss、Trainer 和 checkpoint 流程，
只替换模型构造并叠加 baseline 专属配置。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts" / "baselines"))

import train_rwfa as upstream  # noqa: E402
from evoocc.baselines import LearnedDirectFusionAligner  # noqa: E402
from evoocc.config import load_config, merge_dict  # noqa: E402


BASELINE_CONFIG_PATH = (
    ROOT
    / "src"
    / "evoocc"
    / "baselines"
    / "learned_direct_fusion"
    / "occ3d_config.yaml"
)


def _load_overlay() -> dict:
    return load_config(str(BASELINE_CONFIG_PATH))


def _build_model(
    model_kind: str,
    model_cfg: dict,
    data_cfg: dict,
    device: torch.device,
    use_fast_residual: bool,
) -> LearnedDirectFusionAligner:
    del model_kind
    baseline_cfg = model_cfg["learned_direct_fusion"]
    configured_residual = bool(baseline_cfg.get("use_fast_residual", True))
    if bool(use_fast_residual) != configured_residual:
        raise ValueError(
            "learned direct fusion 的 use_fast_residual 必须与专属配置一致，"
            f"当前入口={use_fast_residual}, 配置={configured_residual}"
        )
    input_grid_size = tuple(baseline_cfg.get("input_grid_size", [200, 200, 16]))
    if tuple(data_cfg["grid_size"]) != input_grid_size:
        raise ValueError(
            "数据 grid_size 与 direct fusion 输入网格不一致："
            f"{tuple(data_cfg['grid_size'])} vs {input_grid_size}"
        )
    return LearnedDirectFusionAligner(
        num_classes=int(data_cfg["num_classes"]),
        encoder_in_channels=int(model_cfg["encoder_in_channels"]),
        free_index=int(data_cfg["free_index"]),
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        latent_dim=int(baseline_cfg.get("latent_dim", 288)),
        fusion_inner_dim=int(baseline_cfg.get("fusion_inner_dim", 104)),
        fusion_body_dilations=tuple(
            baseline_cfg.get("fusion_body_dilations", [1, 3, 5])
        ),
        fusion_gn_groups=int(baseline_cfg.get("fusion_gn_groups", 8)),
        decoder_channels=int(baseline_cfg.get("decoder_channels", 32)),
        decoder_init_scale=baseline_cfg.get("decoder_init_scale", 1.0e-6),
        use_fast_residual=configured_residual,
        input_grid_size=input_grid_size,
        latent_grid_size=tuple(
            baseline_cfg.get("latent_grid_size", [50, 50, 16])
        ),
        timestamp_scale=float(data_cfg.get("timestamp_scale", 1.0e-6)),
    ).to(device)


def main() -> None:
    overlay = _load_overlay()
    original_parse_args = upstream.parse_args
    original_load_config = upstream.load_config_with_base

    def parse_args_fixed():
        args = original_parse_args()
        args.model_kind = "learned-direct-fusion"
        # Rebuttal 对照固定训练协议，不接受 CLI 改写。
        args.epochs = 10
        args.lambda_fast_kl = 0.0
        args.use_fast_residual = True
        return args

    def load_config_with_overlay(path: str) -> dict:
        return merge_dict(original_load_config(path), overlay)

    upstream.parse_args = parse_args_fixed
    upstream.load_config_with_base = load_config_with_overlay
    upstream._build_model = _build_model
    print(
        "[learned-direct-fusion] "
        "channels=288/104 latent=50x50x16 "
        "epochs=10 gradient_accumulation_steps=2"
    )
    upstream.main()


if __name__ == "__main__":
    main()
