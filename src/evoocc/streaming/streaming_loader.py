"""DataLoader 包装: 后台 worker 异步预取 OccStudio sample, 主线程顺序消费.

OccStudio NuScenesDataset 的 __getitem__ 涉及 6 路相机 imread + resize + normalize +
LiDAR load, 单线程串行约 185ms/keyframe. 用 DataLoader(num_workers=N, prefetch_factor=2)
后台预取, 把这部分跟 GPU 上的 fast forward + align 重叠.

关键:
  - shuffle=False 必须保留: alocc 内部 do_history 依赖 dataset 顺序连续
  - collate_fn 在 worker 内 (CPU) 完成 mmcv 包装, 不触发 CUDA
  - scatter_to_device 在 main thread 做 (worker 不能 access CUDA)
"""
from __future__ import annotations
from typing import Iterable

from torch.utils.data import DataLoader, Subset
from mmcv.parallel import collate as _mmcv_collate
from mmcv.parallel import scatter as _mmcv_scatter


def _streaming_collate(batch):
    """module-level 函数, 满足 multi-worker pickle 要求."""
    return _mmcv_collate(batch, samples_per_gpu=1)


def make_streaming_loader(
    occ_dataset,
    flat_indices: Iterable[int],
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> DataLoader:
    """构建顺序消费的 DataLoader.

    Args:
        occ_dataset: OccStudio NuScenesDataset 实例 (已 build).
        flat_indices: 按消费顺序排好的 dataset index 列表 (跨 scene flatten).
        num_workers: 后台 worker 数. 0 表示同步 (退化到原行为).
        prefetch_factor: 每个 worker 预取 batch 数. num_workers=0 时忽略.
    """
    subset = Subset(occ_dataset, list(flat_indices))
    kwargs = dict(
        batch_size=1, shuffle=False,
        num_workers=num_workers,
        collate_fn=_streaming_collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(subset, **kwargs)


def scatter_to_device(batch, device_idx: int = 0):
    """mmcv collate 后的 batch → GPU (main thread)."""
    return _mmcv_scatter(batch, [device_idx])[0]
