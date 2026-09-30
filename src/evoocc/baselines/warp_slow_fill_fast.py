from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from evoocc.data.ego_warp_list import (
    build_sampling_grid,
    compute_transform_prev_to_curr,
)


def _grid_sample_with_padding(
    dense_feat: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str,
) -> torch.Tensor:
    feat_5d = dense_feat.permute(0, 3, 2, 1).unsqueeze(0).contiguous()
    warped_5d = F.grid_sample(
        feat_5d,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    return warped_5d.squeeze(0).permute(0, 3, 2, 1).contiguous()


class WarpSlowFillFastBaseline:
    name = "warp_slow_fill_fast"

    def __init__(
        self,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        free_index: int,
    ) -> None:
        self.pc_range = tuple(pc_range)
        self.voxel_size = tuple(voxel_size)
        self.free_index = int(free_index)

    @torch.inference_mode()
    def predict_sample(
        self,
        fast_logits: torch.Tensor,  # (T, C, X, Y, Z); only [-1] is used
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        rollout_start_step: int = 0,
    ) -> dict:
        num_frames = int(frame_ego2global.shape[0])
        fast_last = fast_logits[-1]
        spatial_shape_xyz = (
            int(fast_last.shape[1]),
            int(fast_last.shape[2]),
            int(fast_last.shape[3]),
        )

        # First frame of a scene: slow is already current, coverage is all ones
        if rollout_start_step >= num_frames - 1:
            slow_pred = slow_logits.argmax(dim=0).long()
            coverage = torch.ones(
                spatial_shape_xyz, dtype=torch.float32, device=slow_logits.device
            )
            return {"pred": slow_pred, "coverage": coverage}

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
        coverage = mask_warped[0].clamp(0.0, 1.0).float()

        # Soft blend keeps continuity at the warp boundary
        w = coverage.unsqueeze(0)
        merged = w * slow_warped + (1.0 - w) * fast_last
        pred = merged.argmax(dim=0).long()

        return {"pred": pred, "coverage": coverage}
