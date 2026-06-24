"""SurroundOcc-nuScenes online_ncde 数据集。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch

from online_ncde.config import resolve_path
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset


class SurroundOccOnlineNcdeDataset(Occ3DOnlineNcdeDataset):
    """读取 SurroundOcc sparse GT，并映射到内部 17 类空间。

    外部 SurroundOcc/OccStudio label space 为 1..17，其中 17 是 free。
    本数据集内部压缩为 0..16，其中 16 是 free。
    """

    def __init__(
        self,
        *args,
        nuscenes_root: str = "data/nuscenes",
        version: str = "v1.0-trainval",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.nuscenes_root = resolve_path(self.root_path, nuscenes_root)
        self.nuscenes_version = str(version)
        self._lidar_filename_by_sample = self._build_lidar_filename_index()

    def _build_lidar_filename_index(self) -> dict[str, str]:
        """建立 sample_token -> LIDAR_TOP 文件名索引。"""
        table_dir = os.path.join(self.nuscenes_root, self.nuscenes_version)
        sample_path = os.path.join(table_dir, "sample.json")
        sample_data_path = os.path.join(table_dir, "sample_data.json")
        calibrated_sensor_path = os.path.join(table_dir, "calibrated_sensor.json")
        sensor_path = os.path.join(table_dir, "sensor.json")
        for table_path in (sample_path, sample_data_path, calibrated_sensor_path, sensor_path):
            if not os.path.exists(table_path):
                raise FileNotFoundError(f"缺少 nuScenes table: {table_path}")

        with open(sample_path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        with open(sample_data_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        with open(calibrated_sensor_path, "r", encoding="utf-8") as f:
            calibrated_sensor = json.load(f)
        with open(sensor_path, "r", encoding="utf-8") as f:
            sensor = json.load(f)

        sensor_by_token = {str(rec["token"]): rec for rec in sensor}
        calibrated_by_token = {str(rec["token"]): rec for rec in calibrated_sensor}
        sample_ts_by_token = {str(rec["token"]): int(rec["timestamp"]) for rec in sample}

        candidates: dict[str, list[tuple[int, bool, str]]] = {}
        for rec in sample_data:
            calibrated = calibrated_by_token.get(str(rec.get("calibrated_sensor_token", "")), {})
            sensor_rec = sensor_by_token.get(str(calibrated.get("sensor_token", "")), {})
            if sensor_rec.get("channel") != "LIDAR_TOP":
                continue
            sample_token = str(rec.get("sample_token", ""))
            filename = str(rec.get("filename", ""))
            if sample_token and filename:
                target_ts = sample_ts_by_token.get(sample_token, int(rec.get("timestamp", 0)))
                dt = abs(int(rec.get("timestamp", 0)) - int(target_ts))
                candidates.setdefault(sample_token, []).append(
                    (dt, bool(rec.get("is_key_frame", False)), os.path.basename(filename))
                )

        index: dict[str, str] = {}
        for sample_token, items in candidates.items():
            # 优先 keyframe；若表里没有标记，则退回到 timestamp 最近。
            items_sorted = sorted(items, key=lambda item: (not item[1], item[0]))
            index[sample_token] = items_sorted[0][2]
        if not index:
            raise RuntimeError(f"未能从 {sample_data_path} 建立 LIDAR_TOP 索引")
        return index

    def _gt_path_from_sample_token(self, sample_token: str) -> str:
        """由 sample token 定位 SurroundOcc sparse GT 文件。"""
        filename = self._lidar_filename_by_sample.get(str(sample_token), "")
        if not filename:
            raise KeyError(f"sample_token={sample_token} 缺少 LIDAR_TOP 文件索引")
        return os.path.join(self.gt_root, "samples", f"{filename}.npy")

    def _load_surroundocc_gt(self, sample_token: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取并 dense 化 SurroundOcc GT，返回内部 0..16 标签和全 1 mask。"""
        gt_path = self._gt_path_from_sample_token(sample_token)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"缺少 SurroundOcc GT: {gt_path}")

        sparse = np.load(gt_path)
        if sparse.ndim != 2 or sparse.shape[1] < 4:
            raise ValueError(f"SurroundOcc GT 形状异常: {gt_path}, shape={sparse.shape}")

        coords = sparse[:, :3].astype(np.int64, copy=False)
        labels = sparse[:, 3].astype(np.uint8, copy=False)
        occ = np.zeros(self.grid_size, dtype=np.uint8)
        occ[coords[:, 0], coords[:, 1], coords[:, 2]] = labels

        # 对齐 OccStudio: dense 中 label 0 视为 free，再压缩 1..17 -> 0..16。
        occ[occ == 0] = 17
        occ = occ - 1
        mask = np.ones(self.grid_size, dtype=np.float32)
        return torch.from_numpy(occ.astype(np.int64)), torch.from_numpy(mask)

    def _load_curr_gt(
        self,
        info: Dict[str, Any],
        load_npz_cached,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取当前帧 SurroundOcc GT。"""
        return self._load_surroundocc_gt(str(info.get("token", "")))

    def _load_sup_gt(
        self,
        info: Dict[str, Any],
        sup_index: int,
        gt_rel: str,
        load_npz_cached,
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        """读取指定 supervision 帧 SurroundOcc GT。"""
        tokens = info.get("supervision_gt_tokens", [])
        if sup_index >= len(tokens):
            return None
        token = str(tokens[sup_index])
        if not token:
            return None
        return self._load_surroundocc_gt(token)
