#!/usr/bin/env python3
"""50×50×16 learned direct window-attention baseline 评估入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

import eval_evoocc as upstream  # noqa: E402
from evoocc.baselines import LearnedDirectAttentionAligner  # noqa: E402
from evoocc.config import load_config, merge_dict  # noqa: E402


BASELINE_CONFIG_PATH = (
    ROOT
    / "src"
    / "evoocc"
    / "baselines"
    / "learned_direct_attention"
    / "occ3d_config.yaml"
)


class _LearnedDirectAttentionAsAligner:
    """适配eval_evoocc.py的EvoOccAligner构造签名。"""

    def __init__(self, baseline_cfg: dict, data_cfg: dict) -> None:
        self.baseline_cfg = baseline_cfg
        self.data_cfg = data_cfg

    def __call__(
        self,
        num_classes,
        feat_dim,
        hidden_dim,
        encoder_in_channels,
        free_index,
        pc_range,
        voxel_size,
        decoder_init_scale=1.0e-6,
        use_fast_residual=True,
        func_g_inner_dim=32,
        func_g_body_dilations=(1, 2, 3),
        func_g_gn_groups=8,
        timestamp_scale=1.0e-6,
        solver_variant=None,
    ) -> LearnedDirectAttentionAligner:
        del (
            feat_dim,
            hidden_dim,
            decoder_init_scale,
            use_fast_residual,
            func_g_inner_dim,
            func_g_body_dilations,
            func_g_gn_groups,
            solver_variant,
        )
        cfg = self.baseline_cfg
        input_grid_size = tuple(cfg.get("input_grid_size", [200, 200, 16]))
        if tuple(self.data_cfg["grid_size"]) != input_grid_size:
            raise ValueError(
                "数据 grid_size 与 direct attention 输入网格不一致："
                f"{tuple(self.data_cfg['grid_size'])} vs {input_grid_size}"
            )
        return LearnedDirectAttentionAligner(
            num_classes=int(num_classes),
            encoder_in_channels=int(encoder_in_channels),
            free_index=int(free_index),
            pc_range=tuple(pc_range),
            voxel_size=tuple(voxel_size),
            latent_dim=int(cfg.get("latent_dim", 288)),
            attention_inner_dim=int(cfg.get("attention_inner_dim", 96)),
            attention_num_heads=int(cfg.get("attention_num_heads", 8)),
            attention_window_size=tuple(
                cfg.get("attention_window_size", [5, 5, 4])
            ),
            attention_local_dilations=tuple(
                cfg.get("attention_local_dilations", [1, 2])
            ),
            attention_gn_groups=int(cfg.get("attention_gn_groups", 8)),
            attention_mlp_ratio=float(
                cfg.get("attention_mlp_ratio", 2.0)
            ),
            decoder_channels=int(cfg.get("decoder_channels", 32)),
            decoder_init_scale=cfg.get("decoder_init_scale", 1.0e-6),
            use_fast_residual=bool(cfg.get("use_fast_residual", True)),
            input_grid_size=input_grid_size,
            latent_grid_size=tuple(
                cfg.get("latent_grid_size", [50, 50, 16])
            ),
            timestamp_scale=float(timestamp_scale),
        )


def _peek_config_path() -> Optional[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    known, _ = parser.parse_known_args()
    return known.config


def main() -> None:
    config_path = _peek_config_path()
    if config_path is None:
        upstream.main()
        return

    overlay = load_config(str(BASELINE_CONFIG_PATH))
    original_load_config = upstream.load_config_with_base

    def load_config_with_overlay(path: str) -> dict:
        return merge_dict(original_load_config(path), overlay)

    merged_cfg = load_config_with_overlay(config_path)
    baseline_cfg = merged_cfg["model"]["learned_direct_attention"]
    upstream.load_config_with_base = load_config_with_overlay
    upstream.EvoOccAligner = _LearnedDirectAttentionAsAligner(
        baseline_cfg=baseline_cfg,
        data_cfg=merged_cfg["data"],
    )
    print(
        "[learned-direct-attention] "
        "channels=288/96 latent=50x50x16 window=5x5x4 heads=8"
    )
    upstream.main()


if __name__ == "__main__":
    main()
