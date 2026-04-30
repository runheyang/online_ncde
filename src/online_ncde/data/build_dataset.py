"""按 data.dataset_variant 分发到具体 online_ncde 数据集实现。"""

from __future__ import annotations

from typing import Any, Dict

from online_ncde.data.logits_loader import LogitsLoader
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset
from online_ncde.data.openoccupancy_online_ncde_dataset import (
    OpenOccupancyOnlineNcdeDataset,
)


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
    """根据 data_cfg.dataset_variant 构造 Occ3D / OpenOccupancy 数据集。

    - dataset_variant 缺省为 'occ3d'（向后兼容）
    - 'openoccupancy' 走 OO 子类，需要 oo_token_map_path（默认 configs/online_ncde/oo_token_map.pkl）
    """
    variant = str(data_cfg.get("dataset_variant", "occ3d")).strip().lower()
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

    if variant == "occ3d":
        return Occ3DOnlineNcdeDataset(**common_kwargs)
    if variant == "openoccupancy":
        return OpenOccupancyOnlineNcdeDataset(
            **common_kwargs,
            oo_token_map_path=data_cfg.get(
                "oo_token_map_path", "configs/online_ncde/oo_token_map.pkl"
            ),
        )
    raise ValueError(
        f"未知的 data.dataset_variant: {variant!r}（可选: 'occ3d', 'openoccupancy'）"
    )
