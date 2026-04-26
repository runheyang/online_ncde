"""Baseline：slow → current 一次性 ego-warp，按覆盖率与 current fast 做 logits 级融合。

无参数、纯几何。流程：
  1) 取 slow_logits（锚在 frame[rollout_start_step] 的 ego 坐标系）。
  2) 一次性计算 T_{slow → curr} = inv(pose_curr) @ pose_slow，不逐帧滚动。
  3) 用 F.grid_sample 做 zeros padding 的 trilinear warp，得到
     slow_warped (C, X, Y, Z)。
  4) 同一 grid 再 warp 一份全 1 张量，得到连续 coverage ∈ [0,1]：
     边界处 0<coverage<1，完全越界处 coverage=0。
  5) logits 级融合：merged = coverage * slow_warped + (1-coverage) * fast[-1]，
     再 argmax。边界连续、保留置信度，和 NCDE aligner 的 logits 空间一致。
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
    feat_5d = dense_feat.permute(0, 3, 2, 1).unsqueeze(0).contiguous()
    warped_5d = F.grid_sample(
        feat_5d,
        grid,
        mode="bilinear",      # 3D 张量上 bilinear = trilinear
        padding_mode=padding_mode,
        align_corners=True,
    )
    return warped_5d.squeeze(0).permute(0, 3, 2, 1).contiguous()


class WarpSlowFillFastBaseline:
    """纯几何 baseline：warp slow 到 current，按连续覆盖率与 fast 做 logits 级融合。"""

    name = "warp_slow_fill_fast"

    def __init__(
        self,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        free_index: int,
    ) -> None:
        self.pc_range = tuple(pc_range)
        self.voxel_size = tuple(voxel_size)
        # 保留 free_index 供未来融合策略扩展（例如"越界区强制置 free"），当前不使用
        self.free_index = int(free_index)

    @torch.inference_mode()
    def predict_sample(
        self,
        fast_logits: torch.Tensor,        # (T, C, X, Y, Z)；实际仅用 [-1]
        slow_logits: torch.Tensor,        # (C, X, Y, Z)
        frame_ego2global: torch.Tensor,   # (T, 4, 4)
        rollout_start_step: int = 0,
    ) -> dict:
        """返回 dict：pred (X,Y,Z) long、coverage (X,Y,Z) float ∈ [0,1]。

        num_frames 从 frame_ego2global 读取，允许 fast_logits 只含末帧
        （省 I/O 与显存，上游可预先 slice 到 T=1 再传入）。
        """
        num_frames = int(frame_ego2global.shape[0])
        fast_last = fast_logits[-1]  # (C, X, Y, Z)
        spatial_shape_xyz = (
            int(fast_last.shape[1]),
            int(fast_last.shape[2]),
            int(fast_last.shape[3]),
        )

        # scene 首帧退化：slow 本身即当前帧，coverage 全 1
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
        coverage = mask_warped[0].clamp(0.0, 1.0).float()  # (X, Y, Z)

        # logits 级融合：边界连续、保留置信度
        w = coverage.unsqueeze(0)  # (1, X, Y, Z)，广播到 C 维
        merged = w * slow_warped + (1.0 - w) * fast_last
        pred = merged.argmax(dim=0).long()

        return {"pred": pred, "coverage": coverage}
