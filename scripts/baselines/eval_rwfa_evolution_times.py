#!/usr/bin/env python3
"""RWFA baseline 的演化时长评估（0.5/1.0/1.5/2.0s 桶 mIoU + RayIoU）。

复用 scripts/eval_online_ncde_evolution_times.main() 的全套桶分配、fallback、
RayIoU 收集逻辑，仅通过 monkey-patch 把 OnlineNcdeAligner 替换成签名兼容的
RWFA factory（NCDE-only 的 func_g_* / solver_variant 字段被吸收/忽略）。

CLI 与上游一致：--config / --checkpoint / --evolution-times / --batch-size / ...
新增参数：
  - --model-kind {rwfa-conv, rwfa-attn}（默认 rwfa-attn）
  - --use-fast-residual / --no-use-fast-residual（默认关闭，与 train_rwfa.py 对齐）
其余参数（--solver 等）保留以兼容上游签名，但对 RWFA 无影响。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

import eval_online_ncde_evolution_times as upstream  # noqa: E402
from online_ncde.baselines import RecurrentWarpFusionAligner  # noqa: E402


class _RwfaAsAlignerCallable:
    """让 RWFA 看起来和 OnlineNcdeAligner 同构造签名。

    NCDE-only 参数（func_g_*, solver_variant）被映射到 RWFA 对应字段或忽略。
    fusion_attn_* 走 RWFA 默认值（论文实验里固定）；如需覆盖请直接在配置中
    新增对应字段并在此处读取。
    """

    def __init__(self, model_kind: str, model_cfg: dict, use_fast_residual: bool) -> None:
        self._model_kind = model_kind
        self._fusion_kind = "conv" if model_kind == "rwfa-conv" else "attn"
        self._model_cfg = model_cfg
        self._use_fast_residual = bool(use_fast_residual)

    def _resolve_fusion_channels(self) -> tuple[int, int]:
        """与 train_rwfa.py 保持一致：attn 默认用 NCDE 的 24 维计算主干。"""
        cfg = self._model_cfg
        if self._model_kind == "rwfa-attn":
            inner_dim = int(cfg.get("fusion_inner_dim", cfg.get("func_g_inner_dim", 24)))
            num_heads = int(cfg.get("fusion_attn_num_heads", 3))
        else:
            inner_dim = int(cfg.get("fusion_inner_dim", 32))
            num_heads = int(cfg.get("fusion_attn_num_heads", 4))
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"fusion_inner_dim={inner_dim} 必须能被 fusion_attn_num_heads={num_heads} 整除"
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
        use_fast_residual=False,
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
        fusion_inner_dim, fusion_attn_num_heads = self._resolve_fusion_channels()
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
            fusion_inner_dim=fusion_inner_dim,
            fusion_body_dilations=tuple(cfg.get("fusion_body_dilations", [1, 2, 3])),
            fusion_gn_groups=int(cfg.get("fusion_gn_groups", 8)),
            fusion_attn_num_heads=fusion_attn_num_heads,
            fusion_attn_window_size=tuple(cfg.get("fusion_attn_window_size", [8, 8, 4])),
            fusion_attn_head_dilations=tuple(cfg.get("fusion_attn_head_dilations", [1, 2])),
            fusion_attn_mlp_ratio=float(cfg.get("fusion_attn_mlp_ratio", 2.0)),
            timestamp_scale=timestamp_scale,
        )


def _peek_baseline_args() -> tuple[str, bool]:
    """提取 RWFA 专属参数，剩余参数留给上游 parse_args。"""
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
    """偷一份 --config 路径用来加载 model_cfg；若未提供让上游正常处理（--help 等情形）。"""
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
    model_kind, use_fast_residual = _peek_baseline_args()
    config_path = _peek_config_path()

    if config_path is None:
        # 没拿到 --config（或用户 --help），直接走上游让它接管参数解析/帮助打印
        upstream.main()
        return

    from online_ncde.config import load_config_with_base
    cfg = load_config_with_base(config_path)
    model_cfg = cfg.get("model", {})

    # monkey-patch：让上游脚本里 model = OnlineNcdeAligner(...) 实际构造 RWFA
    upstream.OnlineNcdeAligner = _RwfaAsAlignerCallable(
        model_kind=model_kind,
        model_cfg=model_cfg,
        use_fast_residual=use_fast_residual,
    )
    print(
        f"[rwfa-eval] model_kind={model_kind} "
        f"use_fast_residual={use_fast_residual} "
        "(monkey-patched OnlineNcdeAligner)"
    )
    upstream.main()


if __name__ == "__main__":
    main()
