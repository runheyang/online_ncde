"""Ray first-hit sidecar 加载器（供 RayLoss 训练使用）。

数据布局（由 scripts/gen_online_ncde_ray_sidecar.py 产出）：

    <dir>/
      <split>_dist.npy        (N, 4, K, R) float16   NaN = 无效 ray
      <split>_origin.npy      (N, 4, K, 3) float32
      <split>_origin_mask.npy (N, 4, K)    uint8     1 = 该 origin 有效
      <split>_sup_mask.npy    (N, 4)       uint8     1 = 该 sup 有效
      <split>_meta.pkl        {"schema_version", "num_origins", "token_to_idx", ...}

RaySidecar 以 mmap 方式打开 npy，按 token 查表，返回 (num_sup, K, *) 形状。
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import numpy as np


_SCHEMA_V2 = "online_ncde_ray_sidecar_v2"


class RaySidecar:
    """按 token 查 ray sidecar 的轻量包装。"""

    def __init__(self, sidecar_dir: str, split: str) -> None:
        prefix = f"{split}_"
        dist_path = os.path.join(sidecar_dir, prefix + "dist.npy")
        origin_path = os.path.join(sidecar_dir, prefix + "origin.npy")
        sup_mask_path = os.path.join(sidecar_dir, prefix + "sup_mask.npy")
        origin_mask_path = os.path.join(sidecar_dir, prefix + "origin_mask.npy")
        meta_path = os.path.join(sidecar_dir, prefix + "meta.pkl")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        schema_version = str(meta.get("schema_version", ""))
        if schema_version != _SCHEMA_V2:
            raise ValueError(
                f"仅支持 schema={_SCHEMA_V2}，实际 {schema_version!r}；"
                "请用 scripts/gen_online_ncde_ray_sidecar.py 重新生成 sidecar。"
            )
        self.schema_version = schema_version
        self.token_to_idx: Dict[str, int] = dict(meta["token_to_idx"])
        self.supervision_labels = list(meta.get("supervision_labels", []))

        # mmap 打开，worker 进程 / DDP rank 间共享 page cache
        self.dist = np.load(dist_path, mmap_mode="r")
        self.origin = np.load(origin_path, mmap_mode="r")
        self.sup_mask = np.load(sup_mask_path, mmap_mode="r")
        self.origin_mask = np.load(origin_mask_path, mmap_mode="r")

        if self.dist.ndim != 4:
            raise ValueError(f"dist 应是 4D (N,sup,K,R)，实际 {self.dist.shape}")
        self.num_origins = int(self.dist.shape[2])
        self.num_rays = int(meta.get("num_rays", self.dist.shape[-1]))

        if (
            self.dist.shape[0] != self.origin.shape[0]
            or self.dist.shape[0] != self.sup_mask.shape[0]
            or self.dist.shape[0] != self.origin_mask.shape[0]
        ):
            raise ValueError(
                f"sidecar N 不一致: dist={self.dist.shape}, "
                f"origin={self.origin.shape}, sup_mask={self.sup_mask.shape}, "
                f"origin_mask={self.origin_mask.shape}"
            )

    def __len__(self) -> int:
        return int(self.dist.shape[0])

    def has(self, token: str) -> bool:
        return token in self.token_to_idx

    def query(
        self, token: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """按 token 查一条样本。

        返回 tuple:
            dist:        (num_sup, K, R) float32   NaN = 无效 ray
            origin:      (num_sup, K, 3) float32
            sup_mask:    (num_sup,)      uint8     1 = 该 sup 有效
            origin_mask: (num_sup, K)    uint8     1 = 该 origin 有效

        token 不存在时返回 None。
        """
        idx = self.token_to_idx.get(token)
        if idx is None:
            return None
        # mmap 切片 copy 成 writable
        dist = np.array(self.dist[idx], dtype=np.float32, copy=True)
        origin = np.array(self.origin[idx], dtype=np.float32, copy=True)
        sup_mask = np.array(self.sup_mask[idx], dtype=np.uint8, copy=True)
        origin_mask = np.array(self.origin_mask[idx], dtype=np.uint8, copy=True)
        return dist, origin, sup_mask, origin_mask
