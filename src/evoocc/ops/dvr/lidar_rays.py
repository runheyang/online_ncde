from __future__ import annotations

import math

import numpy as np


def generate_lidar_rays() -> np.ndarray:
    """Virtual lidar ray directions (R, 3) float32 with R=14040, shared by RayIoU eval and RayLoss."""
    pitch_angles: list[float] = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuScenes lidar pitch fov is [-0.544, 0.211] rad
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
