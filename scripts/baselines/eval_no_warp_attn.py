#!/usr/bin/env python3
"""No-warp motion-conditioned attention baseline 评估入口。

复用 scripts/eval_evoocc.py 的 dataset / loader / Trainer.evaluate /
mIoU + RayIoU 流程，只把 EvoOccAligner 替换为 NoWarpMotionBiasAttnAligner。

示例：
    python scripts/baselines/eval_no_warp_attn.py \
        --config configs/xxx.yaml --checkpoint ckpt.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

import eval_evoocc as upstream  # noqa: E402
from evoocc.baselines import NoWarpMotionBiasAttnAligner  # noqa: E402


class _NoWarpAsAlignerCallable:
    """让 NoWarpMotionBiasAttnAligner 与 EvoOccAligner 构造签名兼容。"""

    def __init__(self, model_cfg: dict, use_fast_residual: bool) -> None:
        self._model_cfg = model_cfg
        self._use_fast_residual = bool(use_fast_residual)

    def _resolve_attn_channels(self) -> Tuple[int, int]:
        """与 train_no_warp_attn.py 保持一致：默认走 EvoOcc 主干 24 维。"""
        cfg = self._model_cfg
        inner_dim = int(cfg.get("no_warp_inner_dim", cfg.get("func_g_inner_dim", 24)))
        num_heads = int(cfg.get("no_warp_attn_num_heads", 3))
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"no-warp attention inner_dim={inner_dim} 必须能被 num_heads={num_heads} 整除"
            )
        return inner_dim, num_heads

    def __call__(
        self,
        num_classes,
        feat_dim,
        hidden_dim,
        encoder_in_channels,
        free_index,
        pc_range,
        voxel_size,
        decoder_init_scale=1.0e-3,
        use_fast_residual=True,
        func_g_inner_dim=32,           # EvoOcc-only，忽略
        func_g_body_dilations=(1, 2, 3),  # EvoOcc-only，忽略
        func_g_gn_groups=8,            # EvoOcc-only，忽略
        timestamp_scale=1.0e-6,
        solver_variant=None,           # EvoOcc-only，忽略
    ):
        cfg = self._model_cfg
        resolved_use_fast_residual = self._use_fast_residual
        resolved_decoder_init_scale = decoder_init_scale if resolved_use_fast_residual else None
        inner_dim, num_heads = self._resolve_attn_channels()

        return NoWarpMotionBiasAttnAligner(
            num_classes=num_classes,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            encoder_in_channels=encoder_in_channels,
            free_index=free_index,
            pc_range=pc_range,
            voxel_size=voxel_size,
            decoder_init_scale=resolved_decoder_init_scale,
            use_fast_residual=resolved_use_fast_residual,
            fusion_inner_dim=inner_dim,
            fusion_attn_num_heads=num_heads,
            fusion_attn_window_size=tuple(cfg.get("no_warp_attn_window_size", [8, 8, 4])),
            fusion_attn_head_dilations=tuple(cfg.get("no_warp_attn_head_dilations", [1, 2])),
            fusion_gn_groups=int(cfg.get("no_warp_gn_groups", cfg.get("func_g_gn_groups", 8))),
            fusion_attn_mlp_ratio=float(cfg.get("no_warp_attn_mlp_ratio", 2.0)),
            timestamp_scale=timestamp_scale,
        )


def _peek_baseline_args() -> bool:
    """提取 no-warp 专属参数，剩余参数交给上游 parser。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--use-fast-residual",
        dest="use_fast_residual",
        action="store_true",
        default=False,
        help="覆盖 use_fast_residual；默认关闭，与 train_no_warp_attn.py 对齐",
    )
    parser.add_argument(
        "--no-use-fast-residual",
        dest="use_fast_residual",
        action="store_false",
        help="覆盖 use_fast_residual；默认关闭，与 train_no_warp_attn.py 对齐",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known.use_fast_residual


def _peek_config_path() -> Optional[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    known, _ = parser.parse_known_args()
    return known.config


def _print_extra_help_if_needed() -> None:
    if not any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return
    print(
        "No-warp attention baseline 额外参数:\n"
        "  --use-fast-residual / --no-use-fast-residual\n"
        "                                      默认关闭，与 train_no_warp_attn.py 对齐\n"
    )


def main() -> None:
    _print_extra_help_if_needed()
    use_fast_residual_override = _peek_baseline_args()
    config_path = _peek_config_path()

    if config_path is None:
        upstream.main()
        return

    from evoocc.config import load_config_with_base

    cfg = load_config_with_base(config_path)
    model_cfg = cfg.get("model", {})

    upstream.EvoOccAligner = _NoWarpAsAlignerCallable(
        model_cfg=model_cfg,
        use_fast_residual=use_fast_residual_override,
    )
    print(f"[no-warp-attn] use_fast_residual={use_fast_residual_override}")
    upstream.main()


if __name__ == "__main__":
    main()
