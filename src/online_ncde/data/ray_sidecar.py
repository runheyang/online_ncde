"""Ray first-hit sidecar 加载器（供 RayLoss 训练使用）。

数据布局（由 scripts/gen_online_ncde_ray_sidecar.py 产出，经合并后放在 data 下）：

    <dir>/
      <split>_dist.npy      (N, 4, R) float16   NaN = 无效 ray
      <split>_origin.npy    (N, 4, 3) float32   lidar origin per sup
      <split>_sup_mask.npy  (N, 4)    uint8     1=该 sup 有效
      <split>_meta.pkl      {"token_to_idx", "supervision_labels", ...}

RaySidecar 以 mmap 方式打开 npy 文件，按 token 查表，对每个样本返回该样本
的 4 个监督时刻的 (dist, origin, sup_mask)。worker 进程各自构造即可。
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import numpy as np


class RaySidecar:
    """按 token 查 ray sidecar 的轻量包装。"""

    def __init__(self, sidecar_dir: str, split: str) -> None:
        prefix = f"{split}_"
        dist_path = os.path.join(sidecar_dir, prefix + "dist.npy")
        origin_path = os.path.join(sidecar_dir, prefix + "origin.npy")
        mask_path = os.path.join(sidecar_dir, prefix + "sup_mask.npy")
        meta_path = os.path.join(sidecar_dir, prefix + "meta.pkl")
        for p in (dist_path, origin_path, mask_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"ray sidecar 缺文件: {p}")

        # mmap 打开，worker 进程间共享 page cache
        self.dist = np.load(dist_path, mmap_mode="r")          # (N, 4, R) fp16
        self.origin = np.load(origin_path, mmap_mode="r")      # (N, 4, 3) fp32
        self.sup_mask = np.load(mask_path, mmap_mode="r")      # (N, 4) uint8
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.token_to_idx: Dict[str, int] = dict(meta["token_to_idx"])
        self.supervision_labels = list(meta.get("supervision_labels", []))
        self.num_rays = int(meta.get("num_rays", self.dist.shape[-1]))

        if self.dist.shape[0] != self.origin.shape[0] or self.dist.shape[0] != self.sup_mask.shape[0]:
            raise ValueError(
                f"sidecar N 不一致: dist={self.dist.shape}, "
                f"origin={self.origin.shape}, sup_mask={self.sup_mask.shape}"
            )

    def __len__(self) -> int:
        return int(self.dist.shape[0])

    def has(self, token: str) -> bool:
        return token in self.token_to_idx

    def query(self, token: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """返回 (dist (4,R) fp32, origin (4,3) fp32, sup_mask (4,) uint8)。

        token 不存在时返回 None，调用方负责 fallback。
        dist 保留 NaN 表示无效 ray；fp16 → fp32 转换在这里做。
        """
        idx = self.token_to_idx.get(token)
        if idx is None:
            return None
        # mmap 切片可能是非 writable view，显式 copy 避免 torch.from_numpy 的 UB 警告
        dist = np.array(self.dist[idx], dtype=np.float32, copy=True)
        origin = np.array(self.origin[idx], dtype=np.float32, copy=True)
        sup_mask = np.array(self.sup_mask[idx], dtype=np.uint8, copy=True)
        return dist, origin, sup_mask
