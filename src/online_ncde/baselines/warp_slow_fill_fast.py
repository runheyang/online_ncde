"""Baseline：slow → current 一次性 ego-warp，越界区用 current fast 填充。

无参数、纯几何。流程：
  1) 取 slow_logits（锚在 frame[rollout_start_step] 的 ego 坐标系）。
  2) 一次性计算 T_{slow → curr} = inv(pose_curr) @ pose_slow，不逐帧滚动。
  3) 用 F.grid_sample 做 zeros padding 的 trilinear warp。
  4) 同一个 grid 再 warp 一份全 1 张量，阈值 0.5 得到 valid_mask
     （区分"确实从源体积采到"与"越界"）。
  5) 输出 argmax：valid_mask 区域用 warp 后 slow，其余用 fast_logits[-1]。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from online_ncde.data.ego_warp_list import (
    build_sampling_grid,
    compute_transform_prev_to_curr,
)


def _grid_sample_with_padding(
    dense_feat: torch.Tensor,   # (C, X, Y, Z)
    grid: torch.Tensor,         # (1, Z, Y, X, 3)
    padding_mode: str,
) -> torch.Tensor:
    """与 backward_warp_dense_trilinear 同口径，显式允许 zeros padding。"""
    with torch.amp.autocast("cuda", enabled=False):
        feat_5d = dense_feat.float().permute(0, 3, 2, 1).unsqueeze(0).contiguous()
        warped_5d = F.grid_sample(
            feat_5d,
            grid,
            mode="bilinear",      # 3D 张量上 bilinear = trilinear
            padding_mode=padding_mode,
            align_corners=True,
        )
    warped = warped_5d.squeeze(0).permute(0, 3, 2, 1).contiguous()
    return warped.to(dense_feat.dtype)


class WarpSlowFillFastBaseline:
    """纯几何 baseline：warp slow 到 current，越界填 fast。"""

    name = "warp_slow_fill_fast"

    def __init__(
        self,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        free_index: int,
        mask_thresh: float = 0.5,
    ) -> None:
        self.pc_range = tuple(pc_range)
        self.voxel_size = tuple(voxel_size)
        self.free_index = int(free_index)
        self.mask_thresh = float(mask_thresh)

    @torch.inference_mode()
    def predict_sample(
        self,
        fast_logits: torch.Tensor,        # (T, C, X, Y, Z)
        slow_logits: torch.Tensor,        # (C, X, Y, Z)
        frame_ego2global: torch.Tensor,   # (T, 4, 4)
        rollout_start_step: int = 0,
    ) -> dict:
        """返回 dict：pred (X,Y,Z) long、valid_mask (X,Y,Z) bool。"""
        num_frames = int(fast_logits.shape[0])
        spatial_shape_xyz = (
            int(fast_logits.shape[2]),
            int(fast_logits.shape[3]),
            int(fast_logits.shape[4]),
        )

        fast_pred = fast_logits[-1].argmax(dim=0).long()

        # scene 首帧退化：slow 本身即当前帧，无需 warp
        if rollout_start_step >= num_frames - 1:
            slow_pred = slow_logits.argmax(dim=0).long()
            valid_mask = torch.ones(spatial_shape_xyz, dtype=torch.bool, device=slow_logits.device)
            return {"pred": slow_pred, "valid_mask": valid_mask, "fast_pred": fast_pred}

        transform = compute_transform_prev_to_curr(
            pose_prev_ego2global=frame_ego2global[rollout_start_step],
            pose_curr_ego2global=frame_ego2global[num_frames - 1],
        )
        grid = build_sampling_grid(
            transform, spatial_shape_xyz, self.pc_range, self.voxel_size
        )

        slow_warped = _grid_sample_with_padding(slow_logits, grid, padding_mode="zeros")

        ones = torch.ones(
            (1, *spatial_shape_xyz),
            device=slow_logits.device,
            dtype=slow_logits.dtype,
        )
        mask_warped = _grid_sample_with_padding(ones, grid, padding_mode="zeros")
        valid_mask = mask_warped[0] > self.mask_thresh  # (X, Y, Z) bool

        slow_pred = slow_warped.argmax(dim=0).long()
        pred = torch.where(valid_mask, slow_pred, fast_pred)

        return {"pred": pred, "valid_mask": valid_mask, "fast_pred": fast_pred}
