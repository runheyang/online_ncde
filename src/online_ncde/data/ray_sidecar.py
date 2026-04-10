"""Ray first-hit sidecar 加载器（供 RayLoss 训练使用）。

数据布局（由 scripts/gen_online_ncde_ray_sidecar.py 产出）：

v2（多原点，与 RayIoU 评估对齐）：

    <dir>/
      <split>_dist.npy        (N, 4, K, R) float16   NaN = 无效 ray
      <split>_origin.npy      (N, 4, K, 3) float32
      <split>_origin_mask.npy (N, 4, K)    uint8     1 = 该 origin 有效
      <split>_sup_mask.npy    (N, 4)       uint8     1 = 该 sup 有效
      <split>_meta.pkl        {"schema_version", "num_origins", "token_to_idx", ...}

v1（单原点，仅用于读老 sidecar）：dist/origin 少一维 K，无 origin_mask.npy。

RaySidecar 以 mmap 方式打开 npy，按 token 查表。query() 统一把两个 schema 升到
v2 的 (num_sup, K, R) / (num_sup, K, 3) / (num_sup, K) 形状，caller 无需区分。
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import numpy as np


_SCHEMA_V1 = "online_ncde_ray_sidecar_v1"
_SCHEMA_V2 = "online_ncde_ray_sidecar_v2"


class RaySidecar:
    """按 token 查 ray sidecar 的轻量包装，统一 v1/v2 schema 输出接口。"""

    def __init__(self, sidecar_dir: str, split: str) -> None:
        prefix = f"{split}_"
        dist_path = os.path.join(sidecar_dir, prefix + "dist.npy")
        origin_path = os.path.join(sidecar_dir, prefix + "origin.npy")
        sup_mask_path = os.path.join(sidecar_dir, prefix + "sup_mask.npy")
        meta_path = os.path.join(sidecar_dir, prefix + "meta.pkl")
        origin_mask_path = os.path.join(sidecar_dir, prefix + "origin_mask.npy")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.schema_version: str = str(meta.get("schema_version", _SCHEMA_V1))
        self.token_to_idx: Dict[str, int] = dict(meta["token_to_idx"])
        self.supervision_labels = list(meta.get("supervision_labels", []))

        # mmap 打开，worker 进程 / DDP rank 间共享 page cache
        self.dist = np.load(dist_path, mmap_mode="r")
        self.origin = np.load(origin_path, mmap_mode="r")
        self.sup_mask = np.load(sup_mask_path, mmap_mode="r")

        if self.schema_version == _SCHEMA_V2:
            self.origin_mask = np.load(origin_mask_path, mmap_mode="r")
            if self.dist.ndim != 4:
                raise ValueError(
                    f"v2 dist 应是 4D (N,sup,K,R)，实际 {self.dist.shape}"
                )
            self.num_origins = int(self.dist.shape[2])
            self.num_rays = int(meta.get("num_rays", self.dist.shape[-1]))
        elif self.schema_version == _SCHEMA_V1:
            # v1: (N, sup, R) / (N, sup, 3)，没有独立 origin_mask
            if self.dist.ndim != 3:
                raise ValueError(
                    f"v1 dist 应是 3D (N,sup,R)，实际 {self.dist.shape}"
                )
            self.origin_mask = None
            self.num_origins = 1
            self.num_rays = int(meta.get("num_rays", self.dist.shape[-1]))
        else:
            raise ValueError(f"未知 ray sidecar schema_version: {self.schema_version!r}")

        # 形状一致性
        if (
            self.dist.shape[0] != self.origin.shape[0]
            or self.dist.shape[0] != self.sup_mask.shape[0]
        ):
            raise ValueError(
                f"sidecar N 不一致: dist={self.dist.shape}, "
                f"origin={self.origin.shape}, sup_mask={self.sup_mask.shape}"
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

        v1 schema 时 K=1，origin_mask 按 sup_mask 广播填充（有效 sup 的唯一
        origin 记为有效，其余置 0），让上层统一当 v2 处理。
        token 不存在时返回 None。
        """
        idx = self.token_to_idx.get(token)
        if idx is None:
            return None

        if self.schema_version == _SCHEMA_V2:
            # mmap 切片 copy 成 writable
            dist = np.array(self.dist[idx], dtype=np.float32, copy=True)          # (sup,K,R)
            origin = np.array(self.origin[idx], dtype=np.float32, copy=True)      # (sup,K,3)
            sup_mask = np.array(self.sup_mask[idx], dtype=np.uint8, copy=True)    # (sup,)
            origin_mask = np.array(self.origin_mask[idx], dtype=np.uint8, copy=True)  # (sup,K)
        else:
            # v1: 升维到 (sup, 1, R) / (sup, 1, 3) / (sup, 1)
            dist_v1 = np.array(self.dist[idx], dtype=np.float32, copy=True)       # (sup,R)
            origin_v1 = np.array(self.origin[idx], dtype=np.float32, copy=True)   # (sup,3)
            sup_mask = np.array(self.sup_mask[idx], dtype=np.uint8, copy=True)    # (sup,)
            dist = dist_v1[:, None, :]                                            # (sup,1,R)
            origin = origin_v1[:, None, :]                                        # (sup,1,3)
            origin_mask = sup_mask[:, None]                                       # (sup,1)

        return dist, origin, sup_mask, origin_mask
