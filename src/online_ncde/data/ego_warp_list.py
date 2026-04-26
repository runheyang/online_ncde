"""体素网格 ego-warp 算子：稠密 backward trilinear warp。

对 t 帧的每个体素中心 r，计算其在 t-1 帧中的对应位置 r' = T_{t→t-1}(r)，
再从 t-1 帧的特征图做 trilinear 插值，得到 r 处的特征值。
全程 GPU 计算：基于 F.grid_sample，无 CPU↔GPU 来回搬运。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_transform_prev_to_curr(
    pose_prev_ego2global: torch.Tensor, pose_curr_ego2global: torch.Tensor
) -> torch.Tensor:
    """计算从 prev ego 到 curr ego 的 4×4 刚体变换矩阵 T_{t-1→t}。"""
    return torch.linalg.inv(pose_curr_ego2global) @ pose_prev_ego2global


def _xyz_to_metric(
    xyz: torch.Tensor,
    pc_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
) -> torch.Tensor:
    """体素索引 → 体素中心度量坐标。公式：metric = (xyz + 0.5) * vsize + pc_min。"""
    pc_min = torch.tensor(pc_range[:3], device=xyz.device, dtype=torch.float32)
    vsize = torch.tensor(voxel_size, device=xyz.device, dtype=torch.float32)
    return (xyz.float() + 0.5) * vsize + pc_min


def _metric_to_xyz_round(
    metric_xyz: torch.Tensor,
    pc_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
) -> torch.Tensor:
    """度量坐标 → 最近邻体素索引。公式：xyz = round((metric - pc_min) / vsize - 0.5)。"""
    pc_min = torch.tensor(pc_range[:3], device=metric_xyz.device, dtype=torch.float32)
    vsize = torch.tensor(voxel_size, device=metric_xyz.device, dtype=torch.float32)
    xyz = torch.round((metric_xyz - pc_min) / vsize - 0.5).to(torch.long)
    return xyz


def _build_sampling_grid(
    transform_prev_to_curr: torch.Tensor,
    spatial_shape_xyz: Tuple[int, int, int],
    pc_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
) -> torch.Tensor:
    """构建 F.grid_sample 所需的采样网格，形状 (1, Z, Y, X, 3)。

    对 t 帧每个体素中心 r，计算其在 t-1 帧中的归一化体素坐标 r'：
        r' = T_{t→t-1}(r) = inv(T_{t-1→t})(r)

    坐标归一化约定（align_corners=True）：
        norm = 2 * idx / (size - 1) - 1
    其中 idx 为分数体素索引（0 ~ size-1），-1/1 分别对应第 0/N-1 个体素中心。

    F.grid_sample 5D 输入约定：
        输入张量：(N, C, D, H, W) = (N, C, Z, Y, X)
        采样网格：(N, D_out, H_out, W_out, 3)，其中
            grid[..., 0] 对应 W(X) 维，grid[..., 1] 对应 H(Y) 维，grid[..., 2] 对应 D(Z) 维。
    全程在 GPU 上完成，无 CPU 坐标计算。
    """
    x_size, y_size, z_size = spatial_shape_xyz
    device = transform_prev_to_curr.device

    # 在目标帧（t 帧）构建完整体素中心度量坐标，保持全程 GPU
    pc_min = transform_prev_to_curr.new_tensor(pc_range[:3])   # (3,)，在同设备
    vsize = transform_prev_to_curr.new_tensor(voxel_size)      # (3,)

    xs = (torch.arange(x_size, device=device, dtype=torch.float32) + 0.5) * vsize[0] + pc_min[0]
    ys = (torch.arange(y_size, device=device, dtype=torch.float32) + 0.5) * vsize[1] + pc_min[1]
    zs = (torch.arange(z_size, device=device, dtype=torch.float32) + 0.5) * vsize[2] + pc_min[2]

    # meshgrid 后展平为 (X*Y*Z, 3) 的度量坐标矩阵
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")    # 各 (X, Y, Z)
    tgt_metric = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (X*Y*Z, 3)

    # 应用 T_{t→t-1} 将目标帧坐标变换到源帧（t-1 帧）
    T_curr_to_prev = torch.linalg.inv(transform_prev_to_curr.float())
    rot = T_curr_to_prev[:3, :3]
    trans = T_curr_to_prev[:3, 3]
    src_metric = tgt_metric @ rot.T + trans   # (X*Y*Z, 3)

    # 度量坐标 → 分数体素索引，范围约 [0, size-1]
    # idx = (metric - pc_min) / vsize - 0.5
    src_idx = (src_metric - pc_min) / vsize - 0.5   # (X*Y*Z, 3)

    # align_corners=True 归一化：norm = 2*idx/(size-1) - 1
    def _norm(coord: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(coord)
        return 2.0 * coord / float(size - 1) - 1.0

    # grid[..., 0]=X(W), [..1]=Y(H), [..2]=Z(D)，与 F.grid_sample 5D 约定一致
    x_norm = _norm(src_idx[:, 0], x_size)
    y_norm = _norm(src_idx[:, 1], y_size)
    z_norm = _norm(src_idx[:, 2], z_size)
    src_norm = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # (X*Y*Z, 3)

    # reshape 到 (1, Z_out, Y_out, X_out, 3)
    grid = src_norm.reshape(x_size, y_size, z_size, 3)          # (X, Y, Z, 3)
    grid = grid.permute(2, 1, 0, 3).unsqueeze(0).contiguous()   # (1, Z, Y, X, 3)
    return grid


def backward_warp_dense_trilinear(
    dense_prev_feat: torch.Tensor,
    transform_prev_to_curr: torch.Tensor,
    spatial_shape_xyz: Tuple[int, int, int],
    pc_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
    padding_mode: str = "border",
    prebuilt_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """对稠密张量做全图 backward trilinear warp，将 t-1 帧特征对齐到 t 帧坐标系。

    Args:
        dense_prev_feat       : (C, X, Y, Z)，t-1 帧稠密特征/隐藏状态
        transform_prev_to_curr: T_{t-1→t}，4×4 刚体变换矩阵（GPU 张量）
        padding_mode          : 默认 'border'，越界时取最近边界值
        prebuilt_grid         : 预构建的采样网格 (1, Z, Y, X, 3)，若提供则跳过 grid 构建

    Returns:
        warped : (C, X, Y, Z)，对齐到 t 帧坐标系的稠密特征
    """
    # 转换轴顺序：(C, X, Y, Z) → (1, C, Z, Y, X) 以匹配 F.grid_sample 5D 输入
    feat_5d = dense_prev_feat.permute(0, 3, 2, 1).unsqueeze(0).contiguous()

    if prebuilt_grid is not None:
        grid = prebuilt_grid
    else:
        grid = _build_sampling_grid(
            transform_prev_to_curr, spatial_shape_xyz, pc_range, voxel_size
        )  # (1, Z, Y, X, 3)，全程 GPU

    warped_5d = F.grid_sample(
        feat_5d,
        grid,
        mode="bilinear",         # 3D 张量上 bilinear = trilinear
        padding_mode=padding_mode,
        align_corners=True,
    )  # (1, C, Z, Y, X)

    # 恢复轴顺序：(1, C, Z, Y, X) → (C, X, Y, Z)
    return warped_5d.squeeze(0).permute(0, 3, 2, 1).contiguous()


def build_sampling_grid(
    transform_prev_to_curr: torch.Tensor,
    spatial_shape_xyz: Tuple[int, int, int],
    pc_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
) -> torch.Tensor:
    """公开的 grid 构建接口，供外部预计算复用。返回 (1, Z, Y, X, 3)。"""
    return _build_sampling_grid(transform_prev_to_curr, spatial_shape_xyz, pc_range, voxel_size)

