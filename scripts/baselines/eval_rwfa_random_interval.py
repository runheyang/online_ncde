#!/usr/bin/env python3
"""RWFA baseline 随机/显式帧间隔评估（末帧 mIoU + RayIoU）。

复用 tests/online_ncde/eval_online_ncde_random_interval.py 的抽帧 dataset、
mIoU/RayIoU 与 CLI，只把 OnlineNcdeAligner 替换为 RecurrentWarpFusionAligner。

示例：
  python scripts/baselines/eval_rwfa_random_interval.py \
      --config configs/xxx.yaml --checkpoint ckpt.pt --model-kind rwfa-attn \
      --random --gap-choices "1,2,3" --target-last-step 12 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "tests" / "online_ncde"))

import eval_online_ncde_random_interval as upstream  # noqa: E402
from online_ncde.baselines import RecurrentWarpFusionAligner  # noqa: E402


class _RwfaAsAlignerCallable:
    """让 RWFA 看起来和 OnlineNcdeAligner 同构造签名。"""

    def __init__(
        self,
        model_kind: str,
        model_cfg: dict,
        use_fast_residual: bool,
    ) -> None:
        self._fusion_kind = "conv" if model_kind == "rwfa-conv" else "attn"
        self._model_cfg = model_cfg
        self._use_fast_residual = bool(use_fast_residual)

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
        func_g_inner_dim=32,
        func_g_body_dilations=(1, 2, 3),
        func_g_gn_groups=8,
        timestamp_scale=1.0e-6,
        solver_variant=None,  # NCDE-only，忽略
    ):
        cfg = self._model_cfg
        resolved_use_fast_residual = self._use_fast_residual
        # 训练时若关闭 residual，DenseDecoder 使用默认初始化；结构不变，但保持构造语义一致。
        resolved_decoder_init_scale = decoder_init_scale if resolved_use_fast_residual else None

        return RecurrentWarpFusionAligner(
            num_classes=num_classes,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            encoder_in_channels=encoder_in_channels,
            free_index=free_index,
            pc_range=pc_range,
            voxel_size=voxel_size,
            decoder_init_scale=resolved_decoder_init_scale,
            use_fast_residual=resolved_use_fast_residual,
            fusion_kind=self._fusion_kind,
            # 与 train_rwfa.py 对齐：RWFA 默认值独立于 NCDE 的 func_g_* 配置。
            fusion_inner_dim=int(cfg.get("fusion_inner_dim", 32)),
            fusion_body_dilations=tuple(cfg.get("fusion_body_dilations", [1, 2, 3])),
            fusion_gn_groups=int(cfg.get("fusion_gn_groups", 8)),
            fusion_attn_num_heads=int(cfg.get("fusion_attn_num_heads", 4)),
            fusion_attn_window_size=tuple(cfg.get("fusion_attn_window_size", [8, 8, 4])),
            fusion_attn_head_dilations=tuple(cfg.get("fusion_attn_head_dilations", [1, 2])),
            fusion_attn_mlp_ratio=float(cfg.get("fusion_attn_mlp_ratio", 2.0)),
            timestamp_scale=timestamp_scale,
        )


def _peek_baseline_args() -> tuple[str, bool]:
    """提取 RWFA 专属参数，剩余参数交给上游 random-interval parser。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-kind", choices=["rwfa-conv", "rwfa-attn"], default="rwfa-attn")
    parser.add_argument(
        "--use-fast-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="覆盖 RWFA use_fast_residual；默认关闭，与 train_rwfa.py 对齐",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known.model_kind, known.use_fast_residual


def _peek_config_path() -> str | None:
    """提前读取 --config，以便构造 RWFA factory；未提供时让上游处理 --help/报错。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    known, _ = parser.parse_known_args()
    return known.config


def _print_extra_help_if_needed() -> None:
    if not any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return
    print(
        "RWFA baseline 额外参数:\n"
        "  --model-kind {rwfa-conv,rwfa-attn}    默认 rwfa-attn\n"
        "  --use-fast-residual / --no-use-fast-residual\n"
        "                                      默认关闭，与 train_rwfa.py 对齐\n"
    )


def main() -> None:
    _print_extra_help_if_needed()
    model_kind, use_fast_residual_override = _peek_baseline_args()
    config_path = _peek_config_path()

    if config_path is None:
        upstream.main()
        return

    from online_ncde.config import load_config_with_base

    cfg = load_config_with_base(config_path)
    model_cfg = cfg.get("model", {})

    upstream.OnlineNcdeAligner = _RwfaAsAlignerCallable(
        model_kind=model_kind,
        model_cfg=model_cfg,
        use_fast_residual=use_fast_residual_override,
    )
    print(
        f"[rwfa-random-interval] model_kind={model_kind} "
        f"use_fast_residual={use_fast_residual_override}"
    )
    upstream.main()


if __name__ == "__main__":
    main()
