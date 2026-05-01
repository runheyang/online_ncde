"""StreamAligner: 把 OnlineNcdeAligner 的逐 step 演化拆成可单步调用的两个 API.

reset_with_slow(fast_logits, slow_logits, ego, t_us)
    在 slow 注入时刻调用. hidden = slow_feat - fast_feat (use_fast_residual=True),
    aligned = slow_logits 直接输出.

evolve(fast_logits, ego, t_us)
    在仅有 fast 的 keyframe 调用. 走 ego_warp + euler/heun solver + decode + 残差融合.

复用现有 OnlineNcdeAligner ckpt, 不改模型, 只重构控制流以适配流式场景.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from online_ncde.data.ego_warp_list import (
    backward_warp_dense_trilinear,
    build_sampling_grid,
    compute_transform_prev_to_curr,
)
from online_ncde.data.scene_delta import build_scene_delta_ctrl


class StreamAligner:
    """逐 keyframe 调用的对齐器封装."""

    def __init__(self, base_aligner: Any) -> None:
        """base_aligner: OnlineNcdeAligner 已 build + load_state_dict + .eval() 后的实例."""
        self.m = base_aligner
        self.m.eval()
        self.hidden: Optional[torch.Tensor] = None      # (D, X, Y, Z)
        self.prev_ego: Optional[torch.Tensor] = None    # (4, 4)
        self.prev_t_us: Optional[int] = None
        self.prev_fast_feat: Optional[torch.Tensor] = None  # (D, X, Y, Z)
        self.prev_tau: float = 0.0                       # 累计 tau (秒)
        self._spatial_shape: Optional[tuple] = None      # (X, Y, Z)
        self._pc_range = self.m.pc_range
        self._voxel_size = self.m.voxel_size
        self._ts_scale = float(self.m.timestamp_scale)   # 一般 = 1e-6 (us → s)

    def reset_scene(self) -> None:
        """跨 scene 边界, 清空所有 stream 状态."""
        self.hidden = None
        self.prev_ego = None
        self.prev_t_us = None
        self.prev_fast_feat = None
        self.prev_tau = 0.0

    @torch.no_grad()
    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        """fast_logits (C, X, Y, Z) → (D, X, Y, Z)."""
        return self.m._encode_fast(fast_logits.unsqueeze(0))[0]

    @torch.no_grad()
    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        """slow_logits (C, X, Y, Z) → (D, X, Y, Z)."""
        return self.m._encode_slow(slow_logits)

    @torch.no_grad()
    def reset_with_slow(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        """slow keyframe: 重置隐藏状态, aligned 输出 = slow_logits."""
        fast_feat = self._encode_fast(fast_logits)
        slow_feat = self._encode_slow(slow_logits)
        if self.m.use_fast_residual:
            self.hidden = slow_feat - fast_feat
        else:
            self.hidden = slow_feat
        self.prev_ego = ego2global
        self.prev_t_us = int(t_us)
        self.prev_fast_feat = fast_feat
        self.prev_tau = 0.0
        if self._spatial_shape is None:
            self._spatial_shape = (
                int(fast_feat.shape[1]),
                int(fast_feat.shape[2]),
                int(fast_feat.shape[3]),
            )
        return slow_logits

    @torch.no_grad()
    def evolve(
        self,
        fast_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        """非 slow keyframe: ego_warp + solver + decode + 残差融合."""
        if self.hidden is None or self.prev_ego is None or self.prev_t_us is None:
            raise RuntimeError("evolve called before reset_with_slow; first keyframe must be a slow injection")

        fast_feat = self._encode_fast(fast_logits)
        if self._spatial_shape is None:
            self._spatial_shape = (
                int(fast_feat.shape[1]),
                int(fast_feat.shape[2]),
                int(fast_feat.shape[3]),
            )

        # --- warp 段 ---
        transform = compute_transform_prev_to_curr(
            pose_prev_ego2global=self.prev_ego,
            pose_curr_ego2global=ego2global,
        )
        grid = build_sampling_grid(transform, self._spatial_shape, self._pc_range, self._voxel_size)
        Z_warp = backward_warp_dense_trilinear(
            dense_prev_feat=self.hidden,
            transform_prev_to_curr=None,
            spatial_shape_xyz=self._spatial_shape,
            pc_range=self._pc_range,
            voxel_size=self._voxel_size,
            padding_mode="border",
            prebuilt_grid=grid,
        )
        f_prev_adv = backward_warp_dense_trilinear(
            dense_prev_feat=self.prev_fast_feat,
            transform_prev_to_curr=None,
            spatial_shape_xyz=self._spatial_shape,
            pc_range=self._pc_range,
            voxel_size=self._voxel_size,
            padding_mode="border",
            prebuilt_grid=grid,
        )

        # --- solver 段 ---
        dt = float(t_us - self.prev_t_us) * self._ts_scale
        tau_curr = self.prev_tau + dt
        tau_prev = self.prev_tau
        device = fast_feat.device
        delta_info = build_scene_delta_ctrl(
            fast_curr=fast_feat,
            fast_prev_adv=f_prev_adv,
            tau_curr=torch.tensor(tau_curr, device=device, dtype=fast_feat.dtype),
            tau_prev=torch.tensor(tau_prev, device=device, dtype=fast_feat.dtype),
        )
        Z_dense, _delta_scene = self.m.solver.step(
            h_adv=Z_warp,
            f_prev_adv=f_prev_adv,
            f_t=fast_feat,
            delta_ctrl=delta_info["delta_ctrl"],
        )

        # --- decode + 残差融合 ---
        logits_delta = self.m._decode_dense_state(Z_dense)
        aligned = logits_delta + fast_logits if self.m.use_fast_residual else logits_delta

        # 更新 stream 状态
        self.hidden = Z_dense
        self.prev_ego = ego2global
        self.prev_t_us = int(t_us)
        self.prev_fast_feat = fast_feat
        self.prev_tau = tau_curr
        return aligned
