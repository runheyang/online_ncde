"""共享的 14040 条虚拟 lidar ray 方向生成函数。

同时被 RayIoU 评估（`ray_metrics.py`）和 RayLoss 训练（`ray_loss.py`）使用，
单点定义防止两侧 pitch/azimuth 不一致导致 ray 集漂移。

单独成文件是为了避免 import `ray_metrics` 时连带触发顶层的 DVR CUDA 扩展编译——
训练主链路不需要 DVR，只需要这套纯 numpy 的方向向量。
"""

from __future__ import annotations

import math

import numpy as np


def generate_lidar_rays() -> np.ndarray:
    """返回 `(R, 3)` 的 float32 ndarray，R=14040。"""
    pitch_angles: list[float] = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    rays: list[tuple[float, float, float]] = []
    for pitch_angle in pitch_angles:
        for azimuth_deg in np.arange(0, 360, 1):
            azimuth = np.deg2rad(azimuth_deg)
            x = float(np.cos(pitch_angle) * np.cos(azimuth))
            y = float(np.cos(pitch_angle) * np.sin(azimuth))
            z = float(np.sin(pitch_angle))
            rays.append((x, y, z))

    return np.asarray(rays, dtype=np.float32)
