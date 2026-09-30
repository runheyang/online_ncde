from __future__ import annotations

from typing import Sequence

import numpy as np

OCC3D_CLASS_NAMES: list[str] = [
    "others", "barrier", "bicycle", "bus", "car",
    "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone",
    "trailer", "truck", "driveable_surface", "other_flat", "sidewalk",
    "terrain", "manmade", "vegetation", "free",
]

OCC3D_COLORS: np.ndarray = np.array([
    [  0,   0,   0],  # 0 others
    [255, 120,  50],  # 1 barrier
    [255, 192, 203],  # 2 bicycle
    [255, 255,   0],  # 3 bus
    [  0, 150, 245],  # 4 car
    [  0, 255, 255],  # 5 construction
    [255, 127,   0],  # 6 motorcycle
    [255,   0,   0],  # 7 pedestrian
    [255, 240, 150],  # 8 traffic_cone
    [135,  60,   0],  # 9 trailer
    [160,  32, 240],  # 10 truck
    [255,   0, 255],  # 11 driveable
    [139, 137, 137],  # 12 other_flat
    [ 75,   0,  75],  # 13 sidewalk
    [150, 240,  80],  # 14 terrain
    [230, 230, 250],  # 15 manmade
    [  0, 175,   0],  # 16 vegetation
    [255, 255, 255],  # 17 free (not rendered)
], dtype=np.float32) / 255.0


def find_visible(voxel: np.ndarray, free_index: int) -> np.ndarray:
    occ = voxel != free_index
    interior = np.zeros_like(occ)
    interior[1:-1, 1:-1, 1:-1] = (
        occ[2:, 1:-1, 1:-1] & occ[:-2, 1:-1, 1:-1]
        & occ[1:-1, 2:, 1:-1] & occ[1:-1, :-2, 1:-1]
        & occ[1:-1, 1:-1, 2:] & occ[1:-1, 1:-1, :-2]
    )
    return occ & ~interior


def clear_figure(figure) -> None:
    from mayavi import mlab
    mlab.clf(figure=figure)


def render_voxel_into_figure(
    figure,
    voxel: np.ndarray,
    *,
    voxel_size: float = 0.4,
    pc_range: Sequence[float] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
    free_index: int = 17,
    colors: np.ndarray | None = None,
    show_interior: bool = False,
    apply_mask: np.ndarray | None = None,
):
    from mayavi import mlab

    if colors is None:
        colors = OCC3D_COLORS

    voxel = np.asarray(voxel)
    if apply_mask is not None:
        voxel = voxel.copy()
        voxel[np.asarray(apply_mask) == 0] = free_index

    visible = (voxel != free_index) if show_interior else find_visible(voxel, free_index)
    xs, ys, zs = np.where(visible)
    if xs.size == 0:
        return mlab.points3d(
            [0.0], [0.0], [0.0], [0.0],
            mode="cube", scale_factor=voxel_size, scale_mode="none",
            opacity=0.0, figure=figure,
        )
    labels = voxel[xs, ys, zs].astype(float)

    x_min, y_min, z_min = pc_range[:3]
    px = xs.astype(np.float32) * voxel_size + x_min + voxel_size * 0.5
    py = ys.astype(np.float32) * voxel_size + y_min + voxel_size * 0.5
    pz = zs.astype(np.float32) * voxel_size + z_min + voxel_size * 0.5

    pts = mlab.points3d(
        px, py, pz, labels,
        mode="cube",
        scale_factor=voxel_size,
        scale_mode="none",
        opacity=1.0,
        vmin=0,
        vmax=len(colors) - 1,
        figure=figure,
    )

    lut = (colors * 255).astype(np.uint8)
    lut = np.concatenate(
        [lut, 255 * np.ones((len(lut), 1), dtype=np.uint8)], axis=1
    )
    lut_mgr = pts.module_manager.scalar_lut_manager
    lut_mgr.lut.number_of_colors = len(lut)
    lut_mgr.lut.table = lut
    return pts
