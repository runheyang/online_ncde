"""从 source sweep pkl 计算每个样本的多视角 lidar origin（用于 RayIoU 评估）。

逻辑等价于 OPUS 的 EgoPoseDataset：对同一 scene 内所有 keyframe，
把各帧 lidar 原点变换到当前帧 ego 坐标系下，最多选 8 个。
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from pyquaternion import Quaternion


def _trans_matrix(translation: Any, rotation: Any) -> np.ndarray:
    """构建 4×4 齐次变换矩阵。"""
    tm = np.eye(4, dtype=np.float64)
    tm[:3, :3] = Quaternion(rotation).rotation_matrix
    tm[:3, 3] = np.asarray(translation, dtype=np.float64)
    return tm


def _global_from_lidar(info: Dict[str, Any]) -> np.ndarray:
    """global_from_ego @ ego_from_lidar → global_from_lidar。"""
    global_from_ego = _trans_matrix(
        info["ego2global_translation"], info["ego2global_rotation"]
    )
    ego_from_lidar = _trans_matrix(
        info["lidar2ego_translation"], info["lidar2ego_rotation"]
    )
    return global_from_ego @ ego_from_lidar


def _ego_from_lidar(info: Dict[str, Any]) -> np.ndarray:
    return _trans_matrix(info["lidar2ego_translation"], info["lidar2ego_rotation"])


def compute_lidar_origins(
    source_infos: List[Dict[str, Any]],
    max_origins: int = 8,
    range_limit: float = 39.0,
) -> Dict[str, torch.Tensor]:
    """计算每个样本的多视角 lidar origin。

    Args:
        source_infos: nuscenes_infos_*_sweep.pkl 中的 infos 列表。
        max_origins: 每个样本最多选多少个 origin。
        range_limit: origin 在 ego 坐标系下 x/y 方向的最大范围。

    Returns:
        token → [1, T, 3] float32 tensor 的字典。
    """
    # 按 scene 分组
    scene_frames: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    for info in source_infos:
        scene_key = info.get("scene_token", info.get("scene_name", ""))
        if scene_key not in scene_frames:
            scene_frames[scene_key] = []
        scene_frames[scene_key].append(info)

    origins_by_token: Dict[str, torch.Tensor] = {}

    for _scene_key, frames in scene_frames.items():
        for ref_idx, ref_info in enumerate(frames):
            ref_lidar_from_global = np.linalg.inv(_global_from_lidar(ref_info))
            ref_ego_from_lidar = _ego_from_lidar(ref_info)

            origin_list: List[np.ndarray] = []
            for curr_idx, curr_info in enumerate(frames):
                if curr_idx == ref_idx:
                    origin_lidar = np.zeros(3, dtype=np.float64)
                else:
                    global_from_curr = _global_from_lidar(curr_info)
                    ref_lidar_from_curr = ref_lidar_from_global @ global_from_curr
                    origin_lidar = ref_lidar_from_curr[:3, 3]

                # lidar 坐标 → ego 坐标
                origin_pad = np.ones(4, dtype=np.float64)
                origin_pad[:3] = origin_lidar
                origin_ego = (ref_ego_from_lidar[:3] @ origin_pad).astype(np.float32)

                if abs(origin_ego[0]) < range_limit and abs(origin_ego[1]) < range_limit:
                    origin_list.append(origin_ego)

            # 降采样到 max_origins
            if len(origin_list) > max_origins:
                sel = np.round(
                    np.linspace(0, len(origin_list) - 1, max_origins)
                ).astype(np.int64)
                origin_list = [origin_list[i] for i in sel]

            token = str(ref_info["token"])
            # [1, T, 3]
            origins_by_token[token] = torch.from_numpy(
                np.stack(origin_list)
            ).unsqueeze(0)

    return origins_by_token


def load_origins_from_sweep_pkl(
    sweep_pkl_path: str | Path,
    max_origins: int = 8,
) -> Dict[str, torch.Tensor]:
    """从 nuscenes_infos_*_sweep.pkl 加载并计算 lidar origins。"""
    path = Path(sweep_pkl_path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "infos" in payload:
        infos = payload["infos"]
    else:
        infos = payload
    return compute_lidar_origins(infos, max_origins=max_origins)
