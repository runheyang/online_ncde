"""新增 baseline 的逐 keyframe streaming 适配器。"""
from __future__ import annotations

from typing import Optional

import torch

from evoocc.data.ego_warp_list import (
    backward_warp_dense_trilinear,
    build_sampling_grid,
    compute_transform_prev_to_curr,
)
from evoocc.streaming.stream_aligner import StreamAligner


class LearnedDirectStreamAligner:
    """最近 slow anchor 到当前帧的非递归 direct 对齐。"""

    def __init__(self, model) -> None:
        self.m = model
        self.m.eval()
        self.hidden: Optional[torch.Tensor] = None
        self.anchor_ego: Optional[torch.Tensor] = None

    def reset_scene(self) -> None:
        self.hidden = None
        self.anchor_ego = None

    @torch.no_grad()
    def reset_with_slow(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        del fast_logits, t_us
        self.hidden = self.m._encode_slow_anchor(slow_logits)
        self.anchor_ego = ego2global
        return slow_logits.float()

    @torch.no_grad()
    def evolve(
        self,
        fast_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        del t_us
        if self.hidden is None or self.anchor_ego is None:
            raise RuntimeError(
                "evolve called before reset_with_slow; "
                "first keyframe must be a slow injection"
            )
        target_fast_logits = fast_logits.unsqueeze(0)
        target_fast_features = self.m._encode_target_fast(
            target_fast_logits
        )
        warped_slow = self.m._warp_anchor_to_target(
            slow_anchor=self.hidden,
            pose_anchor=self.anchor_ego,
            pose_target=ego2global,
        ).unsqueeze(0)
        fused = self.m.fusion(warped_slow, target_fast_features)
        return self.m._decode_targets(
            fused,
            target_fast_logits,
        )[0].float()


class NeuralOdeDtStreamAligner(StreamAligner):
    """复用 EvoOcc 流式状态管理，仅改为标量 Δt 驱动 solver。"""

    @torch.no_grad()
    def evolve(
        self,
        fast_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        if (
            self.hidden is None
            or self.prev_ego is None
            or self.prev_t_us is None
            or self.prev_fast_feat is None
        ):
            raise RuntimeError(
                "evolve called before reset_with_slow; "
                "first keyframe must be a slow injection"
            )

        fast_feat = self._encode_fast(fast_logits)
        if self._spatial_shape is None:
            self._spatial_shape = tuple(
                int(value) for value in fast_feat.shape[1:]
            )

        transform = compute_transform_prev_to_curr(
            pose_prev_ego2global=self.prev_ego,
            pose_curr_ego2global=ego2global,
        )
        grid = build_sampling_grid(
            transform,
            self._spatial_shape,
            self._pc_range,
            self._voxel_size,
        )
        hidden_warped = backward_warp_dense_trilinear(
            dense_prev_feat=self.hidden,
            transform_prev_to_curr=None,
            spatial_shape_xyz=self._spatial_shape,
            pc_range=self._pc_range,
            voxel_size=self._voxel_size,
            padding_mode="border",
            prebuilt_grid=grid,
        )
        fast_prev_warped = backward_warp_dense_trilinear(
            dense_prev_feat=self.prev_fast_feat,
            transform_prev_to_curr=None,
            spatial_shape_xyz=self._spatial_shape,
            pc_range=self._pc_range,
            voxel_size=self._voxel_size,
            padding_mode="border",
            prebuilt_grid=grid,
        )

        dt = fast_feat.new_tensor(
            float(t_us - self.prev_t_us) * self._ts_scale
        )
        hidden_next, _delta_scene = self.m.solver.step(
            h_adv=hidden_warped,
            f_prev_adv=fast_prev_warped,
            f_t=fast_feat,
            dt=dt,
        )
        logits_delta = self.m._decode_dense_state(hidden_next)
        aligned = (
            logits_delta + fast_logits
            if self.m.use_fast_residual
            else logits_delta
        )

        self.hidden = hidden_next
        self.prev_ego = ego2global
        self.prev_t_us = int(t_us)
        self.prev_fast_feat = fast_feat
        self.prev_tau += float(dt.item())
        return aligned.float()


def build_baseline_stream_aligner(baseline_name: str, model):
    """按 baseline 类型构造统一 streaming 接口。"""
    if baseline_name in (
        "learned_direct_attention",
        "learned_direct_fusion",
    ):
        return LearnedDirectStreamAligner(model)
    if baseline_name == "neural_ode_dt_100":
        return NeuralOdeDtStreamAligner(model)
    raise ValueError(f"未知 baseline: {baseline_name!r}")
