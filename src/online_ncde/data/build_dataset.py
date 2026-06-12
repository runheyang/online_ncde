"""构造 online_ncde Occ3D 数据集。"""

from __future__ import annotations

from typing import Any, Dict

from online_ncde.data.logits_loader import LogitsLoader
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset


def build_online_ncde_dataset(
    data_cfg: Dict[str, Any],
    *,
    info_path: str,
    root_path: str,
    logits_loader: LogitsLoader,
    ray_sidecar_dir: str | None = None,
    ray_sidecar_split: str | None = None,
    fast_frame_stride: int | None = None,
    min_history_completeness: int | None = None,
    eval_only_mode: bool = False,
) -> Occ3DOnlineNcdeDataset:
    """根据 data_cfg 构造 Occ3D 数据集。"""
    variant = str(data_cfg.get("dataset_variant", "occ3d")).strip().lower()
    if variant != "occ3d":
        raise ValueError(f"未知的 data.dataset_variant: {variant!r}（仅支持 'occ3d'）")
    if fast_frame_stride is None:
        fast_frame_stride = int(data_cfg.get("fast_frame_stride", 1))

    common_kwargs = dict(
        info_path=info_path,
        root_path=root_path,
        gt_root=data_cfg["gt_root"],
        num_classes=int(data_cfg["num_classes"]),
        free_index=int(data_cfg["free_index"]),
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg.get("gt_mask_key", "mask_camera"),
        logits_loader=logits_loader,
        ray_sidecar_dir=ray_sidecar_dir,
        ray_sidecar_split=ray_sidecar_split,
        fast_frame_stride=fast_frame_stride,
        min_history_completeness=min_history_completeness,
        eval_only_mode=eval_only_mode,
    )

    return Occ3DOnlineNcdeDataset(**common_kwargs)
