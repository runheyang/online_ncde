#!/usr/bin/env python3
"""RWFA baseline 的演化时长评估（0.5/1.0/1.5/2.0s 桶 mIoU + RayIoU）。

复用 scripts/eval_online_ncde_evolution_times.main() 的全套桶分配、fallback、
RayIoU 收集逻辑，仅通过 monkey-patch 把 OnlineNcdeAligner 替换成签名兼容的
RWFA factory（NCDE-only 的 func_g_* / solver_variant 字段被吸收/忽略）。

CLI 与上游一致：--config / --checkpoint / --evolution-times / --batch-size / ...
唯一新增：--model-kind {rwfa-conv, rwfa-attn}（默认 rwfa-conv）。
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

    def __init__(self, model_kind: str, model_cfg: dict) -> None:
        self._fusion_kind = "conv" if model_kind == "rwfa-conv" else "attn"
        self._model_cfg = model_cfg

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
        return RecurrentWarpFusionAligner(
            num_classes=num_classes,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            encoder_in_channels=encoder_in_channels,
            free_index=free_index,
            pc_range=pc_range,
            voxel_size=voxel_size,
            decoder_init_scale=decoder_init_scale,
            use_fast_residual=use_fast_residual,
            fusion_kind=self._fusion_kind,
            # 优先读 fusion_* 字段，兜底沿用 func_g_*（兼容直接复用 NCDE config）
            fusion_inner_dim=int(cfg.get("fusion_inner_dim", func_g_inner_dim)),
            fusion_body_dilations=tuple(cfg.get("fusion_body_dilations", func_g_body_dilations)),
            fusion_gn_groups=int(cfg.get("fusion_gn_groups", func_g_gn_groups)),
            fusion_attn_num_heads=int(cfg.get("fusion_attn_num_heads", 4)),
            fusion_attn_window_size=tuple(cfg.get("fusion_attn_window_size", [8, 8, 4])),
            fusion_attn_head_dilations=tuple(cfg.get("fusion_attn_head_dilations", [1, 2])),
            fusion_attn_mlp_ratio=float(cfg.get("fusion_attn_mlp_ratio", 2.0)),
            timestamp_scale=timestamp_scale,
        )


def _peek_model_kind() -> str:
    """从 sys.argv 里偷偷把 --model-kind 提取出来，剩余参数留给上游 parse_args。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-kind", choices=["rwfa-conv", "rwfa-attn"], default="rwfa-conv")
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known.model_kind


def _peek_config_path() -> str | None:
    """偷一份 --config 路径用来加载 model_cfg；若未提供让上游正常处理（--help 等情形）。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    known, _ = parser.parse_known_args()
    return known.config


def main() -> None:
    model_kind = _peek_model_kind()
    config_path = _peek_config_path()

    if config_path is None:
        # 没拿到 --config（或用户 --help），直接走上游让它接管参数解析/帮助打印
        upstream.main()
        return

    from online_ncde.config import load_config_with_base
    cfg = load_config_with_base(config_path)
    model_cfg = cfg.get("model", {})

    # monkey-patch：让上游脚本里 model = OnlineNcdeAligner(...) 实际构造 RWFA
    upstream.OnlineNcdeAligner = _RwfaAsAlignerCallable(model_kind, model_cfg)
    print(f"[rwfa-eval] model_kind={model_kind} (monkey-patched OnlineNcdeAligner)")
    upstream.main()


if __name__ == "__main__":
    main()
