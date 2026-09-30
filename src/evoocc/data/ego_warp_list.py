from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_transform_prev_to_curr(
    pose_prev_ego2global: torch.Tensor, pose_curr_ego2global: torch.Tensor
) -> torch.Tensor:
    return torch.linalg.inv(pose_curr_ego2global) @ pose_prev_ego2global


def _xyz_to_metric(
    xyz: torch.Tensor,
    pc_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
) -> torch.Tensor:
    pc_min = torch.tensor(pc_range[:3], device=xyz.device, dtype=torch.float32)
    vsize = torch.tensor(voxel_size, device=xyz.device, dtype=torch.float32)
    return (xyz.float() + 0.5) * vsize + pc_min


def _metric_to_xyz_round(
    metric_xyz: torch.Tensor,
    pc_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
) -> torch.Tensor:
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
    x_size, y_size, z_size = spatial_shape_xyz
    device = transform_prev_to_curr.device

    pc_min = transform_prev_to_curr.new_tensor(pc_range[:3])
    vsize = transform_prev_to_curr.new_tensor(voxel_size)

    xs = (torch.arange(x_size, device=device, dtype=torch.float32) + 0.5) * vsize[0] + pc_min[0]
    ys = (torch.arange(y_size, device=device, dtype=torch.float32) + 0.5) * vsize[1] + pc_min[1]
    zs = (torch.arange(z_size, device=device, dtype=torch.float32) + 0.5) * vsize[2] + pc_min[2]

    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    tgt_metric = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    T_curr_to_prev = torch.linalg.inv(transform_prev_to_curr.float())
    rot = T_curr_to_prev[:3, :3]
    trans = T_curr_to_prev[:3, 3]
    # Apply T_{t->t-1} to target-frame voxel centers to get source-frame coordinates.
    src_metric = tgt_metric @ rot.T + trans

    # Fractional voxel index in the source frame: (metric - pc_min) / vsize - 0.5.
    src_idx = (src_metric - pc_min) / vsize - 0.5

    # align_corners=True normalization: 2*idx/(size-1) - 1.
    def _norm(coord: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(coord)
        return 2.0 * coord / float(size - 1) - 1.0

    x_norm = _norm(src_idx[:, 0], x_size)
    y_norm = _norm(src_idx[:, 1], y_size)
    z_norm = _norm(src_idx[:, 2], z_size)
    src_norm = torch.stack([x_norm, y_norm, z_norm], dim=-1)

    grid = src_norm.reshape(x_size, y_size, z_size, 3)
    # Output layout (1, Z, Y, X, 3); last dim ordered (x, y, z) as F.grid_sample expects.
    grid = grid.permute(2, 1, 0, 3).unsqueeze(0).contiguous()
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
    feat_5d = dense_prev_feat.permute(0, 3, 2, 1).unsqueeze(0).contiguous()

    if prebuilt_grid is not None:
        grid = prebuilt_grid
    else:
        grid = _build_sampling_grid(
            transform_prev_to_curr, spatial_shape_xyz, pc_range, voxel_size
        )

    warped_5d = F.grid_sample(
        feat_5d,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )

    return warped_5d.squeeze(0).permute(0, 3, 2, 1).contiguous()


def build_sampling_grid(
    transform_prev_to_curr: torch.Tensor,
    spatial_shape_xyz: Tuple[int, int, int],
    pc_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
) -> torch.Tensor:
    return _build_sampling_grid(transform_prev_to_curr, spatial_shape_xyz, pc_range, voxel_size)

