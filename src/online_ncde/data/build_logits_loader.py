"""根据配置构造 LogitsLoader 实例的工厂函数。"""

from __future__ import annotations

from typing import Any, Dict

from online_ncde.data.logits_loader import (
    AloccDenseTopkLoader,
    CompositeLogitsLoader,
    LogitsLoader,
    OpusSparseFullLoader,
    OpusSparseTopkLoader,
)


def build_logits_loader(
    data_cfg: Dict[str, Any],
    root_path: str,
) -> LogitsLoader:
    """根据 data.logits_format 构造 LogitsLoader。"""
    logits_format = str(data_cfg.get("logits_format", "")).strip().lower()
    if not logits_format:
        raise KeyError("data.logits_format 为必填项。")

    if logits_format == "alocc_dense_topk":
        fast_logits_root = data_cfg.get("fast_logits_root", "")
        slow_logit_root = data_cfg.get("slow_logit_root", "")
        if not fast_logits_root or not slow_logit_root:
            raise KeyError(
                "logits_format=alocc_dense_topk 时，"
                "data.fast_logits_root 和 data.slow_logit_root 为必填项。"
            )
        return AloccDenseTopkLoader(
            root_path=root_path,
            fast_logits_root=fast_logits_root,
            slow_logit_root=slow_logit_root,
            num_classes=int(data_cfg["num_classes"]),
            grid_size=tuple(data_cfg["grid_size"]),
            fill_value=float(data_cfg.get("alocc_fill_value", -12.0)),
            clamp_min=float(data_cfg.get("alocc_clamp_min", -12.0)),
            topk_k=int(data_cfg.get("alocc_topk_k", 3)),
            max_centering=bool(data_cfg.get("alocc_max_centering", True)),
        )

    if logits_format == "composite":
        # 要求 data.fast_logits_format / data.slow_logit_format 指定具体子格式。
        fast_sub = str(data_cfg.get("fast_logits_format", "")).strip().lower()
        slow_sub = str(data_cfg.get("slow_logit_format", "")).strip().lower()
        for name, sub in (("fast_logits_format", fast_sub), ("slow_logit_format", slow_sub)):
            if sub in ("", "composite", "opus"):
                raise ValueError(
                    f"composite.{name}={sub!r} 非法（需为具体非 composite 子格式）。"
                )
        fast_loader = build_logits_loader({**data_cfg, "logits_format": fast_sub}, root_path)
        slow_loader = build_logits_loader({**data_cfg, "logits_format": slow_sub}, root_path)
        return CompositeLogitsLoader(fast_loader=fast_loader, slow_loader=slow_loader)

    if logits_format == "opus_sparse_full":
        fast_logits_root = data_cfg.get("fast_logits_root", "")
        slow_logit_root = data_cfg.get("slow_logit_root", "")
        if not fast_logits_root or not slow_logit_root:
            raise KeyError(
                "logits_format=opus_sparse_full 时，"
                "data.fast_logits_root 和 data.slow_logit_root 为必填项。"
            )
        return OpusSparseFullLoader(
            root_path=root_path,
            fast_logits_root=fast_logits_root,
            slow_logit_root=slow_logit_root,
            num_classes=int(data_cfg["num_classes"]),
            free_index=int(data_cfg["free_index"]),
            grid_size=tuple(data_cfg["grid_size"]),
            topk_k=int(data_cfg.get("opus_full_topk_k", 3)),
            other_fill_value=float(data_cfg.get("opus_other_fill_value", -5.0)),
            free_fill_value=float(data_cfg.get("opus_free_fill_value", 5.0)),
        )

    if logits_format == "opus_sparse_topk":
        fast_logits_root = data_cfg.get("fast_logits_root", "")
        slow_logit_root = data_cfg.get("slow_logit_root", "")
        if not fast_logits_root or not slow_logit_root:
            raise KeyError(
                "logits_format=opus_sparse_topk 时，"
                "data.fast_logits_root 和 data.slow_logit_root 为必填项。"
            )
        return OpusSparseTopkLoader(
            root_path=root_path,
            fast_logits_root=fast_logits_root,
            slow_logit_root=slow_logit_root,
            num_classes=int(data_cfg["num_classes"]),
            free_index=int(data_cfg["free_index"]),
            grid_size=tuple(data_cfg["grid_size"]),
            other_fill_value=float(data_cfg.get("opus_other_fill_value", -5.0)),
            free_fill_value=float(data_cfg.get("opus_free_fill_value", 5.0)),
        )

    raise ValueError(f"不支持的 logits_format: {logits_format!r}")
