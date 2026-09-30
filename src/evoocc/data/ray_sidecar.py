from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import numpy as np


_SCHEMA_V3 = "evoocc_ray_sidecar_v3"
_LEGACY_SCHEMA_V3 = "online_ncde_ray_sidecar_v3"
_SUPPORTED_SCHEMAS = {_SCHEMA_V3, _LEGACY_SCHEMA_V3}


class RaySidecar:
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
        if schema_version not in _SUPPORTED_SCHEMAS:
            raise ValueError(
                f"仅支持 schema={sorted(_SUPPORTED_SCHEMAS)}，实际 {schema_version!r}；"
                "请用 scripts/gen_evoocc_ray_sidecar.py 重新生成 sidecar。"
            )
        self.schema_version = schema_version
        self.token_to_idx: Dict[str, int] = dict(meta["token_to_idx"])
        self.supervision_labels = list(meta.get("supervision_labels", []))
        self.dist_semantics = str(
            meta.get("dist_semantics", "finite=hit_dist_m, inf=no_hit, nan=ignore")
        )
        self.ray_horizon_m = float(meta.get("ray_horizon_m", 0.0))

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

    # dist (num_sup, K, R): finite = hit distance in meters, inf = no-hit within view, NaN = ignore
    def query(
        self, token: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        idx = self.token_to_idx.get(token)
        if idx is None:
            return None
        dist = np.array(self.dist[idx], dtype=np.float32, copy=True)
        origin = np.array(self.origin[idx], dtype=np.float32, copy=True)
        sup_mask = np.array(self.sup_mask[idx], dtype=np.uint8, copy=True)
        origin_mask = np.array(self.origin_mask[idx], dtype=np.uint8, copy=True)
        return dist, origin, sup_mask, origin_mask
