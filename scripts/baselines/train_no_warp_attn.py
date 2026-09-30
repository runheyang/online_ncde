#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts" / "baselines"))

import train_rwfa as upstream  # noqa: E402
from evoocc.baselines import NoWarpMotionBiasAttnAligner  # noqa: E402


def _build_model(
    model_kind: str,
    model_cfg: dict,
    data_cfg: dict,
    device: torch.device,
    use_fast_residual: bool,
) -> NoWarpMotionBiasAttnAligner:
    inner_dim = int(model_cfg.get("no_warp_inner_dim", model_cfg.get("func_g_inner_dim", 24)))
    num_heads = int(model_cfg.get("no_warp_attn_num_heads", 3))
    if inner_dim % num_heads != 0:
        raise ValueError(
            f"no-warp attention inner_dim={inner_dim} 必须能被 num_heads={num_heads} 整除"
        )
    if use_fast_residual:
        decoder_init_scale = model_cfg.get("decoder_init_scale", 1.0e-3)
    else:
        decoder_init_scale = None

    return NoWarpMotionBiasAttnAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=decoder_init_scale,
        use_fast_residual=use_fast_residual,
        fusion_inner_dim=inner_dim,
        fusion_attn_num_heads=num_heads,
        fusion_attn_window_size=tuple(model_cfg.get("no_warp_attn_window_size", [8, 8, 4])),
        fusion_attn_head_dilations=tuple(model_cfg.get("no_warp_attn_head_dilations", [1, 2])),
        fusion_gn_groups=int(model_cfg.get("no_warp_gn_groups", model_cfg.get("func_g_gn_groups", 8))),
        fusion_attn_mlp_ratio=float(model_cfg.get("no_warp_attn_mlp_ratio", 2.0)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
    ).to(device)


def main() -> None:
    original_parse_args = upstream.parse_args

    def parse_args_with_kind():
        args = original_parse_args()
        args.model_kind = "no-warp-attn"
        if args.epochs == 0:
            args.epochs = 10
        # No fast KL here; override locally instead of editing the shared config.
        args.lambda_fast_kl = 0.0
        return args

    upstream.parse_args = parse_args_with_kind
    upstream._build_model = _build_model
    upstream.main()


if __name__ == "__main__":
    main()
