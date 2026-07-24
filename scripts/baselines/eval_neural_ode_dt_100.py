#!/usr/bin/env python3
"""100×100×16 Neural ODE Δt baseline 评估入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

import eval_evoocc as upstream  # noqa: E402
from evoocc.baselines import NeuralOdeDt100Aligner  # noqa: E402
from evoocc.config import load_config, merge_dict  # noqa: E402


BASELINE_CONFIG_PATH = (
    ROOT
    / "src"
    / "evoocc"
    / "baselines"
    / "neural_ode_dt_100"
    / "occ3d_config.yaml"
)


class _NeuralOdeDt100AsAligner:
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
        solver_variant="euler",
    ) -> NeuralOdeDt100Aligner:
        del (
            feat_dim,
            hidden_dim,
            decoder_init_scale,
            use_fast_residual,
            func_g_inner_dim,
            func_g_body_dilations,
            func_g_gn_groups,
        )
        cfg = self.baseline_cfg
        configured_solver = str(cfg.get("solver_variant", "euler")).lower()
        if str(solver_variant).lower() != configured_solver:
            raise ValueError(
                "评估 solver 必须与训练配置一致："
                f"CLI={solver_variant}, 配置={configured_solver}"
            )
        input_grid_size = tuple(cfg.get("input_grid_size", [200, 200, 16]))
        if tuple(self.data_cfg["grid_size"]) != input_grid_size:
            raise ValueError(
                "数据 grid_size 与 Neural ODE Δt 输入网格不一致："
                f"{tuple(self.data_cfg['grid_size'])} vs {input_grid_size}"
            )
        return NeuralOdeDt100Aligner(
            num_classes=int(num_classes),
            encoder_in_channels=int(encoder_in_channels),
            free_index=int(free_index),
            pc_range=tuple(pc_range),
            voxel_size=tuple(voxel_size),
            latent_dim=int(cfg.get("latent_dim", 120)),
            func_g_inner_dim=int(cfg.get("func_g_inner_dim", 48)),
            func_g_body_dilations=tuple(
                cfg.get("func_g_body_dilations", [1, 3, 5])
            ),
            func_g_gn_groups=int(cfg.get("func_g_gn_groups", 8)),
            decoder_channels=int(cfg.get("decoder_channels", 32)),
            decoder_init_scale=cfg.get("decoder_init_scale", 1.0e-6),
            use_fast_residual=bool(cfg.get("use_fast_residual", True)),
            input_grid_size=input_grid_size,
            latent_grid_size=tuple(
                cfg.get("latent_grid_size", [100, 100, 16])
            ),
            timestamp_scale=float(timestamp_scale),
            solver_variant=configured_solver,
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
    baseline_cfg = merged_cfg["model"]["neural_ode_dt_100"]
    upstream.load_config_with_base = load_config_with_overlay
    upstream.EvoOccAligner = _NeuralOdeDt100AsAligner(
        baseline_cfg=baseline_cfg,
        data_cfg=merged_cfg["data"],
    )
    print(
        "[neural-ode-dt-100] "
        "channels=120 latent_grid=100x100x16 solver=euler"
    )
    upstream.main()


if __name__ == "__main__":
    main()
