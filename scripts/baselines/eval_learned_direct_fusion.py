#!/usr/bin/env python3
"""100×100×16 learned direct fusion baseline 评估入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

import eval_evoocc as upstream  # noqa: E402
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


class _LearnedDirectFusionAsAligner:
    """适配 eval_evoocc.py 的 EvoOccAligner 构造签名。"""

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
    ) -> LearnedDirectFusionAligner:
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
                "数据 grid_size 与 direct fusion 输入网格不一致："
                f"{tuple(self.data_cfg['grid_size'])} vs {input_grid_size}"
            )
        return LearnedDirectFusionAligner(
            num_classes=int(num_classes),
            encoder_in_channels=int(encoder_in_channels),
            free_index=int(free_index),
            pc_range=tuple(pc_range),
            voxel_size=tuple(voxel_size),
            latent_dim=int(cfg.get("latent_dim", 128)),
            fusion_inner_dim=int(cfg.get("fusion_inner_dim", 48)),
            fusion_body_dilations=tuple(
                cfg.get("fusion_body_dilations", [1, 3, 5])
            ),
            fusion_gn_groups=int(cfg.get("fusion_gn_groups", 8)),
            decoder_channels=int(cfg.get("decoder_channels", 32)),
            decoder_init_scale=cfg.get("decoder_init_scale", 1.0e-6),
            use_fast_residual=bool(cfg.get("use_fast_residual", True)),
            input_grid_size=input_grid_size,
            latent_grid_size=tuple(
                cfg.get("latent_grid_size", [100, 100, 16])
            ),
            timestamp_scale=float(timestamp_scale),
        )


def _peek_config_path() -> str | None:
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
    baseline_cfg = merged_cfg["model"]["learned_direct_fusion"]
    upstream.load_config_with_base = load_config_with_overlay
    upstream.EvoOccAligner = _LearnedDirectFusionAsAligner(
        baseline_cfg=baseline_cfg,
        data_cfg=merged_cfg["data"],
    )
    print("[learned-direct-fusion] latent=100x100x16 direct anchor-to-target warp")
    upstream.main()


if __name__ == "__main__":
    main()
