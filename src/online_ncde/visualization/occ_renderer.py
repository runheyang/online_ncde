"""Occ3D-nuScenes 体素渲染：底层函数库，GUI 与离屏渲染共用。

关键概念：
- 输入是 (X, Y, Z) 形状的 int 体素，类别 0..num_classes-1，free 类用 free_index 标记
- 渲染只画"暴露在外"的体素（默认开），相比全画可加速 5-10 倍
- 配色与 SurroundOcc / OccFormer 论文一致
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Occ3D-nuScenes 18 类官方语义 + 论文风配色（RGB 0-255）
OCC3D_CLASS_NAMES: list[str] = [
    "others", "barrier", "bicycle", "bus", "car",
    "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone",
    "trailer", "truck", "driveable_surface", "other_flat", "sidewalk",
    "terrain", "manmade", "vegetation", "free",
]

OCC3D_COLORS: np.ndarray = np.array([
    [  0,   0,   0],   # 0  others
    [255, 120,  50],   # 1  barrier        橙
    [255, 192, 203],   # 2  bicycle        浅粉
    [255, 255,   0],   # 3  bus            黄
    [  0, 150, 245],   # 4  car            蓝
    [  0, 255, 255],   # 5  construction   青
    [255, 127,   0],   # 6  motorcycle     深橙
    [255,   0,   0],   # 7  pedestrian     红
    [255, 240, 150],   # 8  traffic_cone   浅黄
    [135,  60,   0],   # 9  trailer        棕
    [160,  32, 240],   # 10 truck          紫
    [255,   0, 255],   # 11 driveable      粉（论文里那个标志性粉路面）
    [139, 137, 137],   # 12 other_flat     深灰
    [ 75,   0,  75],   # 13 sidewalk       暗紫
    [150, 240,  80],   # 14 terrain        浅绿
    [230, 230, 250],   # 15 manmade        浅灰白
    [  0, 175,   0],   # 16 vegetation     绿
    [255, 255, 255],   # 17 free（不渲染）
], dtype=np.float32) / 255.0


def find_visible(voxel: np.ndarray, free_index: int) -> np.ndarray:
    """只保留被空气暴露的体素（6 邻居有任一为空），加速渲染。

    返回 bool 数组，形状与 voxel 相同。边界体素永远不会被判为内部，
    所以即使数据撑到边缘也不会丢面。
    """
    occ = voxel != free_index
    interior = np.zeros_like(occ)
    interior[1:-1, 1:-1, 1:-1] = (
        occ[2:, 1:-1, 1:-1] & occ[:-2, 1:-1, 1:-1]
        & occ[1:-1, 2:, 1:-1] & occ[1:-1, :-2, 1:-1]
        & occ[1:-1, 1:-1, 2:] & occ[1:-1, 1:-1, :-2]
    )
    return occ & ~interior


def clear_figure(figure) -> None:
    """清空 Mayavi figure（删除全部已绘制对象）。"""
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
    """把 voxel 绘制到指定 Mayavi figure 上。

    参数:
        figure: Mayavi figure（GUI 内嵌时传 ``MlabSceneModel.mayavi_scene``，
            离屏渲染时传 ``mlab.figure(...)`` 返回值）
        voxel: (X, Y, Z) int 数组，类别 id
        apply_mask: 可选 (X, Y, Z) 二值掩码，为 0 的位置当作 free（用于
            画 GT 时套上 mask_camera）。

    返回:
        mlab.points3d 对象（用于后续修改 LUT 或删除）。
    """
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
        # 全空，画一个隐形点占位避免报错
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

    # 自定义离散 LUT（每类一种颜色）
    lut = (colors * 255).astype(np.uint8)
    lut = np.concatenate(
        [lut, 255 * np.ones((len(lut), 1), dtype=np.uint8)], axis=1
    )
    lut_mgr = pts.module_manager.scalar_lut_manager
    lut_mgr.lut.number_of_colors = len(lut)
    lut_mgr.lut.table = lut
    return pts
