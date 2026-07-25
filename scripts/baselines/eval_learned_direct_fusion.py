#!/usr/bin/env python3
"""50×50×16 learned direct fusion baseline 评估入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

import eval_evoocc as upstream  # noqa: E402
from evoocc.streaming.baseline_factory import (  # noqa: E402
    build_baseline_model_from_config,
    load_config_with_baseline_overlay,
)


class _LearnedDirectFusionAsAligner:
    """适配 eval_evoocc.py 的 EvoOccAligner 构造签名。"""

    def __init__(self, merged_cfg: dict) -> None:
        self.merged_cfg = merged_cfg

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
    ):
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
        return build_baseline_model_from_config(
            self.merged_cfg,
            "learned_direct_fusion",
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

    def load_config_with_overlay(path: str) -> dict:
        return load_config_with_baseline_overlay(
            path,
            "learned_direct_fusion",
        )

    merged_cfg = load_config_with_overlay(config_path)
    upstream.load_config_with_base = load_config_with_overlay
    upstream.EvoOccAligner = _LearnedDirectFusionAsAligner(merged_cfg)
    print(
        "[learned-direct-fusion] "
        "channels=288/104 latent=50x50x16 direct anchor-to-target warp"
    )
    upstream.main()


if __name__ == "__main__":
    main()
