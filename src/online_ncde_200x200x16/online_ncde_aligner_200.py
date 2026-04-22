"""Online NCDE 200x200x16 全分辨率对齐模型。

与原始 OnlineNcdeAligner 的区别：
  - encoder 不做空间下采样（stride=1），隐藏状态保持 200x200x16
  - decoder 不做上采样（无 interpolate），直接 3x3x3 + 1x1x1 输出
  - warp 使用原始 voxel_size，不需要 voxel_size_ds
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用原始 online_ncde 的共享组件
from online_ncde.data.ego_warp_list import (
    backward_warp_dense_trilinear,
    build_sampling_grid,
    compute_transform_prev_to_curr,
)
from online_ncde.data.scene_delta import build_scene_delta_ctrl
from online_ncde.data.time_series import compute_segment_dt, cumulative_tau
from online_ncde.models.func_g import FuncG
from online_ncde.models.heads import CtrlProjector
from online_ncde.models.solver_euler import EulerNextFastSolver
from online_ncde.models.solver_heun import HeunSolver

# 200x200x16 专用编解码器
from online_ncde_200x200x16.decoder import DenseDecoder200
from online_ncde_200x200x16.encoder import DenseEncoder200


def _compute_m_occ(fast_logits: torch.Tensor, free_index: int) -> torch.Tensor:
    """m_occ = max_{c != free}(logit[c]) - logit[free]，Fast-KL 的 conf 权重来源。"""
    masked = fast_logits.clone()
    masked.narrow(-4, free_index, 1).fill_(float("-inf"))
    max_non_free = masked.amax(dim=-4, keepdim=True)
    free_logit = fast_logits.narrow(-4, free_index, 1)
    return max_non_free - free_logit


class OnlineNcdeAligner200(nn.Module):
    """全分辨率 200x200x16 编码 + Dense Warp + Dense Dynamics + 解码。

    与原版 OnlineNcdeAligner 的关键区别：
      - 隐藏状态在 200x200x16 全分辨率空间演化，无空间下采样
      - voxel_size 直接使用原始值，无需 voxel_size_ds
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        hidden_dim: int,
        encoder_in_channels: int,
        free_index: int,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        decoder_init_scale: float = 0.0,
        use_fast_residual: bool = True,
        func_g_inner_dim: int = 32,
        func_g_body_dilations: tuple[int, ...] = (1, 2, 3),
        func_g_gn_groups: int = 8,
        timestamp_scale: float = 1.0e-6,
        amp_fp16: bool = False,
        solver_variant: str = "heun",
    ) -> None:
        super().__init__()
        self.amp_fp16 = bool(amp_fp16)
        self.use_fast_residual = bool(use_fast_residual)
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder_in_channels = int(encoder_in_channels)
        if self.hidden_dim != self.feat_dim:
            raise ValueError(
                f"hidden_dim 必须与 feat_dim 相同，当前 hidden_dim={self.hidden_dim}, "
                f"feat_dim={self.feat_dim}。请在配置中只设置一个值并保持一致。"
            )
        self.free_index = int(free_index)

        pc_range_tuple = tuple(pc_range)
        if len(pc_range_tuple) != 6:
            raise ValueError(f"pc_range 必须长度为 6，当前: {pc_range_tuple}")
        self.pc_range = cast(Tuple[float, float, float, float, float, float], pc_range_tuple)
        self.voxel_size = tuple(voxel_size)
        self.timestamp_scale = float(timestamp_scale)

        # 全分辨率：不需要 voxel_size_ds，直接使用原始 voxel_size

        self.fast_encoder = DenseEncoder200(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        self.slow_encoder = DenseEncoder200(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        self.ctrl_proj = CtrlProjector(feat_dim + 1, hidden_dim)
        self.func_g = FuncG(
            in_channels=hidden_dim + feat_dim,
            hidden_channels=hidden_dim,
            inner_dim=func_g_inner_dim,
            body_dilations=func_g_body_dilations,
            gn_groups=func_g_gn_groups,
        )
        solver_variant_lower = str(solver_variant).lower()
        if solver_variant_lower == "heun":
            self.solver = HeunSolver(func_g=self.func_g, ctrl_proj=self.ctrl_proj)
        elif solver_variant_lower == "euler":
            # Euler + next-fast：func_g 仅喂 f_t，单次求值
            self.solver = EulerNextFastSolver(func_g=self.func_g, ctrl_proj=self.ctrl_proj)
        else:
            raise ValueError(
                f"未知的 solver_variant: {solver_variant!r}，可选: 'heun', 'euler'"
            )
        self.solver_variant = solver_variant_lower
        self.decoder = DenseDecoder200(
            in_channels=hidden_dim,
            out_channels=num_classes,
            init_scale=decoder_init_scale,
        )

        # Trainer 根据 lambda_fast_kl 注入；False 时 forward 跳过 KL 计算。
        self._fast_kl_active: bool = False

    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        """编码快系统序列，输出 dense (T, C_f, X, Y, Z)。"""
        return self.fast_encoder(fast_logits)

    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        """编码慢系统输入，输出 dense (C_f, X, Y, Z)。"""
        return self.slow_encoder(slow_logits.unsqueeze(0))[0]

    def _decode_dense_state(self, z_dense: torch.Tensor) -> torch.Tensor:
        """将稠密隐藏状态解码为残差 logits，输出 (C, X, Y, Z)。"""
        z_tensor = z_dense.unsqueeze(0)  # (1, C, X, Y, Z)
        z_tensor = z_tensor.permute(0, 1, 4, 3, 2).contiguous()  # (1, C, Z, Y, X)
        out_dense = self.decoder(z_tensor)
        return out_dense.permute(0, 1, 4, 3, 2).contiguous()[0]

    def _compute_fast_kl_step(
        self,
        fast_logits_t: torch.Tensor,    # (C, X, Y, Z) raw fast logits at step t (detached by caller)
        aligned_logits: torch.Tensor,   # (C, X, Y, Z)
    ) -> torch.Tensor:
        """Conf-weighted KL(aligned || fast)：fast 自信判占用处把 aligned 拉回 fast。

        用 m_occ.clamp(min=0) 做权重 —— 只惩罚"fast 自信判占用"处偏离，保留
        "fast 自信判空"处 aligner 自由填补（aligner 真推演的主要价值场景）。
        """
        # 显式 .float() 避免 AMP autocast 下 fp16 输入导致 kl_div 混算
        aligned_f = aligned_logits.float()
        fast_f = fast_logits_t.float()
        w = _compute_m_occ(fast_f, self.free_index).clamp(min=0.0)
        log_p_fast = F.log_softmax(fast_f, dim=0)
        log_p_aligned = F.log_softmax(aligned_f, dim=0)
        # F.kl_div(log_q, log_p, log_target=True) = Σ exp(log_p) (log_p - log_q) = KL(P || Q)
        # P=aligned, Q=fast：input=log_p_fast, target=log_p_aligned
        kl_per_voxel = F.kl_div(
            log_p_fast, log_p_aligned, log_target=True, reduction="none"
        ).sum(dim=0, keepdim=True)
        return (w * kl_per_voxel).mean()

    def _forward_single(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """处理单样本（不含 batch 维），全分辨率 Dense 流程。"""
        num_frames = fast_logits.shape[0]

        fast_feat = self._encode_fast(fast_logits)
        slow_feat = self._encode_slow(slow_logits)

        spatial_shape_xyz = (
            int(fast_feat.shape[2]),
            int(fast_feat.shape[3]),
            int(fast_feat.shape[4]),
        )
        pc_range_6 = cast(Tuple[float, float, float, float, float, float], self.pc_range)
        # 全分辨率：直接使用原始 voxel_size
        voxel_size_3 = cast(Tuple[float, float, float], self.voxel_size)

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)
        tau = cumulative_tau(dt).to(device=fast_logits.device)

        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[0]
        else:
            Z_dense = slow_feat

        delta_mag_values: list[float] = []

        for k in range(num_frames - 1):
            transform = compute_transform_prev_to_curr(
                pose_prev_ego2global=frame_ego2global[k],
                pose_curr_ego2global=frame_ego2global[k + 1],
            )
            grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)

            Z_warp_dense = backward_warp_dense_trilinear(
                dense_prev_feat=Z_dense,
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_prev_adv = backward_warp_dense_trilinear(
                dense_prev_feat=fast_feat[k],
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_t = fast_feat[k + 1]

            delta_info = build_scene_delta_ctrl(
                fast_curr=f_t,
                fast_prev_adv=f_prev_adv,
                tau_curr=tau[k + 1],
                tau_prev=tau[k],
            )
            Z_dense, delta_scene = self.solver.step(
                h_adv=Z_warp_dense,
                f_prev_adv=f_prev_adv,
                f_t=f_t,
                delta_ctrl=delta_info["delta_ctrl"],
            )

            delta_mag_values.append(delta_scene.abs().mean().item())

        logits_delta = self._decode_dense_state(Z_dense)
        if self.use_fast_residual:
            logits = logits_delta + fast_logits[-1]
        else:
            logits = logits_delta

        avg_delta = sum(delta_mag_values) / max(len(delta_mag_values), 1)
        diagnostics = {
            "delta_scene_abs_mean": torch.tensor(avg_delta, device=fast_logits.device),
        }
        return {"aligned": logits.float(), "diagnostics": {k: v.float() for k, v in diagnostics.items()}}

    def _forward_single_stepwise_train(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        max_step_index: int | None = None,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """逐步训练单样本：每步更新后解码并返回全时刻 logits。"""
        num_frames = fast_logits.shape[0]
        rollout_steps = num_frames - 1
        if max_step_index is not None:
            rollout_steps = min(rollout_steps, max(int(max_step_index), 0))

        fast_feat = self._encode_fast(fast_logits)
        slow_feat = self._encode_slow(slow_logits)
        spatial_shape_xyz = (
            int(fast_feat.shape[2]),
            int(fast_feat.shape[3]),
            int(fast_feat.shape[4]),
        )
        pc_range_6 = cast(Tuple[float, float, float, float, float, float], self.pc_range)
        voxel_size_3 = cast(Tuple[float, float, float], self.voxel_size)

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)
        tau = cumulative_tau(dt).to(device=fast_logits.device)

        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[0]
        else:
            Z_dense = slow_feat

        delta_mag_values: list[float] = []
        fast_kl_accum: torch.Tensor | None = None
        fast_kl_step_count = 0
        step_logits_list: list[torch.Tensor] = []
        compute_fast_kl = self._fast_kl_active and self.use_fast_residual

        for k in range(rollout_steps):
            transform = compute_transform_prev_to_curr(
                pose_prev_ego2global=frame_ego2global[k],
                pose_curr_ego2global=frame_ego2global[k + 1],
            )
            grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)

            Z_warp_dense = backward_warp_dense_trilinear(
                dense_prev_feat=Z_dense,
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_prev_adv = backward_warp_dense_trilinear(
                dense_prev_feat=fast_feat[k],
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_t = fast_feat[k + 1]

            delta_info = build_scene_delta_ctrl(
                fast_curr=f_t,
                fast_prev_adv=f_prev_adv,
                tau_curr=tau[k + 1],
                tau_prev=tau[k],
            )
            Z_dense, delta_scene = self.solver.step(
                h_adv=Z_warp_dense,
                f_prev_adv=f_prev_adv,
                f_t=f_t,
                delta_ctrl=delta_info["delta_ctrl"],
            )

            logits_delta = self._decode_dense_state(Z_dense)
            if self.use_fast_residual:
                logits_now = logits_delta + fast_logits[k + 1]
            else:
                logits_now = logits_delta
            step_logits_list.append(logits_now.float())

            if compute_fast_kl:
                kl_step = self._compute_fast_kl_step(
                    fast_logits_t=fast_logits[k + 1].detach(),
                    aligned_logits=logits_now,
                )
                fast_kl_accum = kl_step if fast_kl_accum is None else fast_kl_accum + kl_step
                fast_kl_step_count += 1

            delta_mag_values.append(delta_scene.abs().mean().item())

        if step_logits_list:
            step_logits = torch.stack(step_logits_list, dim=0)
        else:
            step_logits = fast_logits.new_zeros(
                (0, self.num_classes, fast_logits.shape[2], fast_logits.shape[3], fast_logits.shape[4])
            )
        step_indices = torch.arange(1, rollout_steps + 1, device=fast_logits.device, dtype=torch.long)
        avg_delta = sum(delta_mag_values) / max(len(delta_mag_values), 1)
        diagnostics = {
            "delta_scene_abs_mean": torch.tensor(avg_delta, device=fast_logits.device),
        }
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "step_logits": step_logits,
            "step_indices": step_indices,
            "diagnostics": {k: v.float() for k, v in diagnostics.items()},
        }
        if fast_kl_accum is not None and fast_kl_step_count > 0:
            out["fast_kl"] = fast_kl_accum / fast_kl_step_count
        return out

    def _forward_single_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """逐步评估单样本：每步更新后都做解码并和当前 fast logits 残差融合。"""
        num_frames = fast_logits.shape[0]

        fast_feat = self._encode_fast(fast_logits)
        slow_feat = self._encode_slow(slow_logits)
        spatial_shape_xyz = (
            int(fast_feat.shape[2]),
            int(fast_feat.shape[3]),
            int(fast_feat.shape[4]),
        )
        pc_range_6 = cast(Tuple[float, float, float, float, float, float], self.pc_range)
        voxel_size_3 = cast(Tuple[float, float, float], self.voxel_size)

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)
        tau = cumulative_tau(dt).to(device=fast_logits.device)

        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[0]
        else:
            Z_dense = slow_feat

        delta_mag_values: list[float] = []

        step_logits_list: list[torch.Tensor] = []
        # 分三段统计：warp(ego-warp/grid_sample) / solver(build_delta + self.solver.step) / decode(解码+残差)
        step_warp_ms_values: list[float] = []
        step_solver_ms_values: list[float] = []
        step_decode_ms_values: list[float] = []
        step_time_events: list[tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = []
        use_cuda_timing = fast_logits.is_cuda

        for k in range(num_frames - 1):
            if use_cuda_timing:
                ev_t0 = torch.cuda.Event(enable_timing=True)
                ev_t1 = torch.cuda.Event(enable_timing=True)
                ev_t2 = torch.cuda.Event(enable_timing=True)
                ev_t3 = torch.cuda.Event(enable_timing=True)
                ev_t0.record()
            else:
                tp0 = time.perf_counter()

            # ---- warp 段 ----
            transform = compute_transform_prev_to_curr(
                pose_prev_ego2global=frame_ego2global[k],
                pose_curr_ego2global=frame_ego2global[k + 1],
            )
            grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)

            Z_warp_dense = backward_warp_dense_trilinear(
                dense_prev_feat=Z_dense,
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_prev_adv = backward_warp_dense_trilinear(
                dense_prev_feat=fast_feat[k],
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )

            if use_cuda_timing:
                ev_t1.record()
            else:
                tp1 = time.perf_counter()

            # ---- solver 段 ----
            f_t = fast_feat[k + 1]
            delta_info = build_scene_delta_ctrl(
                fast_curr=f_t,
                fast_prev_adv=f_prev_adv,
                tau_curr=tau[k + 1],
                tau_prev=tau[k],
            )
            Z_dense, delta_scene = self.solver.step(
                h_adv=Z_warp_dense,
                f_prev_adv=f_prev_adv,
                f_t=f_t,
                delta_ctrl=delta_info["delta_ctrl"],
            )

            if use_cuda_timing:
                ev_t2.record()
            else:
                tp2 = time.perf_counter()

            # ---- decode 段 ----
            logits_delta = self._decode_dense_state(Z_dense)
            if self.use_fast_residual:
                logits_now = logits_delta + fast_logits[k + 1]
            else:
                logits_now = logits_delta
            step_logits_list.append(logits_now.float())

            delta_mag_values.append(delta_scene.abs().mean().item())

            if use_cuda_timing:
                ev_t3.record()
                step_time_events.append((ev_t0, ev_t1, ev_t2, ev_t3))
            else:
                tp3 = time.perf_counter()
                step_warp_ms_values.append((tp1 - tp0) * 1000.0)
                step_solver_ms_values.append((tp2 - tp1) * 1000.0)
                step_decode_ms_values.append((tp3 - tp2) * 1000.0)

        if use_cuda_timing:
            torch.cuda.synchronize(device=fast_logits.device)
            step_warp_ms_values = [t0.elapsed_time(t1) for t0, t1, _, _ in step_time_events]
            step_solver_ms_values = [t1.elapsed_time(t2) for _, t1, t2, _ in step_time_events]
            step_decode_ms_values = [t2.elapsed_time(t3) for _, _, t2, t3 in step_time_events]
        step_time_ms_values = [
            w + s + d
            for w, s, d in zip(step_warp_ms_values, step_solver_ms_values, step_decode_ms_values)
        ]

        if step_logits_list:
            step_logits = torch.stack(step_logits_list, dim=0)
        else:
            step_logits = fast_logits.new_zeros(
                (0, self.num_classes, fast_logits.shape[2], fast_logits.shape[3], fast_logits.shape[4])
            )

        step_indices = torch.arange(1, num_frames, device=fast_logits.device, dtype=torch.long)
        step_time_ms = torch.tensor(step_time_ms_values, device=fast_logits.device, dtype=torch.float32)
        step_warp_ms = torch.tensor(step_warp_ms_values, device=fast_logits.device, dtype=torch.float32)
        step_solver_ms = torch.tensor(step_solver_ms_values, device=fast_logits.device, dtype=torch.float32)
        step_decode_ms = torch.tensor(step_decode_ms_values, device=fast_logits.device, dtype=torch.float32)
        avg_delta = sum(delta_mag_values) / max(len(delta_mag_values), 1)
        diagnostics = {
            "delta_scene_abs_mean": torch.tensor(avg_delta, device=fast_logits.device),
        }
        return {
            "step_logits": step_logits,
            "step_time_ms": step_time_ms,
            "step_warp_ms": step_warp_ms,
            "step_solver_ms": step_solver_ms,
            "step_decode_ms": step_decode_ms,
            "step_indices": step_indices,
            "diagnostics": {k: v.float() for k, v in diagnostics.items()},
        }

    def forward_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """评估接口：返回每个循环步的 logits 与耗时（批处理包装）。"""
        if fast_logits.dim() == 5:
            fast_logits = fast_logits.unsqueeze(0)
            slow_logits = slow_logits.unsqueeze(0)
            frame_ego2global = frame_ego2global.unsqueeze(0)
            if frame_timestamps is not None:
                frame_timestamps = frame_timestamps.unsqueeze(0)
            if frame_dt is not None:
                frame_dt = frame_dt.unsqueeze(0)

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if self.amp_fp16 else nullcontext()
        step_logits_list: list[torch.Tensor] = []
        step_time_list: list[torch.Tensor] = []
        step_warp_list: list[torch.Tensor] = []
        step_solver_list: list[torch.Tensor] = []
        step_decode_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        step_indices: torch.Tensor | None = None
        with amp_ctx:
            for b in range(fast_logits.shape[0]):
                out = self._forward_single_stepwise_eval(
                    fast_logits=fast_logits[b],
                    slow_logits=slow_logits[b],
                    frame_ego2global=frame_ego2global[b],
                    frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                    frame_dt=frame_dt[b] if frame_dt is not None else None,
                )
                sample_step_logits = cast(torch.Tensor, out["step_logits"])
                sample_step_time = cast(torch.Tensor, out["step_time_ms"])
                sample_step_warp = cast(torch.Tensor, out["step_warp_ms"])
                sample_step_solver = cast(torch.Tensor, out["step_solver_ms"])
                sample_step_decode = cast(torch.Tensor, out["step_decode_ms"])
                sample_step_indices = cast(torch.Tensor, out["step_indices"])
                if step_indices is None:
                    step_indices = sample_step_indices
                elif sample_step_indices.shape != step_indices.shape:
                    raise ValueError(
                        f"batch 内 step 数不一致: {sample_step_indices.shape} vs {step_indices.shape}"
                    )

                step_logits_list.append(sample_step_logits)
                step_time_list.append(sample_step_time)
                step_warp_list.append(sample_step_warp)
                step_solver_list.append(sample_step_solver)
                step_decode_list.append(sample_step_decode)
                diag_list.append(cast(dict[str, torch.Tensor], out["diagnostics"]))

        if step_indices is None:
            step_indices = torch.zeros((0,), dtype=torch.long, device=fast_logits.device)
        step_logits = torch.stack(step_logits_list, dim=0)
        step_time_ms = torch.stack(step_time_list, dim=0)
        step_warp_ms = torch.stack(step_warp_list, dim=0)
        step_solver_ms = torch.stack(step_solver_list, dim=0)
        step_decode_ms = torch.stack(step_decode_list, dim=0)
        return {
            "step_logits": step_logits,
            "step_time_ms": step_time_ms,
            "step_warp_ms": step_warp_ms,
            "step_solver_ms": step_solver_ms,
            "step_decode_ms": step_decode_ms,
            "step_indices": step_indices,
            "diagnostics": diag_list,
        }

    def _unsqueeze_inputs(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ):
        """若输入无 batch 维则自动补齐。"""
        if fast_logits.dim() == 5:
            fast_logits = fast_logits.unsqueeze(0)
            slow_logits = slow_logits.unsqueeze(0)
            frame_ego2global = frame_ego2global.unsqueeze(0)
            if frame_timestamps is not None:
                frame_timestamps = frame_timestamps.unsqueeze(0)
            if frame_dt is not None:
                frame_dt = frame_dt.unsqueeze(0)
        return fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt

    def forward(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        mode: str = "default",
        max_step_index: int | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """统一前向入口，通过 mode 分发到不同路径（兼容 DDP）。

        mode:
          - "default": 仅返回最终 aligned logits
          - "stepwise_train": 逐步训练，返回每步 logits（完整 BPTT）
          - "stepwise_eval": 逐步评估，返回每步 logits 与耗时
        """
        fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt = (
            self._unsqueeze_inputs(fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt)
        )

        if mode == "stepwise_train":
            return self._forward_batched_stepwise_train(
                fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt,
                max_step_index=max_step_index,
            )
        elif mode == "stepwise_eval":
            return self._forward_batched_stepwise_eval(
                fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt,
            )
        elif mode == "default":
            return self._forward_batched_default(
                fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt,
            )
        else:
            raise ValueError(f"未知的 forward mode: {mode!r}，可选: 'default', 'stepwise_train', 'stepwise_eval'")

    def _forward_batched_default(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if self.amp_fp16 else nullcontext()
        aligned_list = []
        diag_list = []
        with amp_ctx:
            for b in range(fast_logits.shape[0]):
                out = self._forward_single(
                    fast_logits=fast_logits[b],
                    slow_logits=slow_logits[b],
                    frame_ego2global=frame_ego2global[b],
                    frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                    frame_dt=frame_dt[b] if frame_dt is not None else None,
                )
                aligned_list.append(out["aligned"])
                diag_list.append(out["diagnostics"])
        aligned = torch.stack(aligned_list, dim=0)
        return {"aligned": aligned, "diagnostics": diag_list}

    def _forward_batched_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """评估：返回每个循环步的 logits 与耗时。"""
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if self.amp_fp16 else nullcontext()
        step_logits_list: list[torch.Tensor] = []
        step_time_list: list[torch.Tensor] = []
        step_warp_list: list[torch.Tensor] = []
        step_solver_list: list[torch.Tensor] = []
        step_decode_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        step_indices: torch.Tensor | None = None
        with amp_ctx:
            for b in range(fast_logits.shape[0]):
                out = self._forward_single_stepwise_eval(
                    fast_logits=fast_logits[b],
                    slow_logits=slow_logits[b],
                    frame_ego2global=frame_ego2global[b],
                    frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                    frame_dt=frame_dt[b] if frame_dt is not None else None,
                )
                sample_step_logits = cast(torch.Tensor, out["step_logits"])
                sample_step_time = cast(torch.Tensor, out["step_time_ms"])
                sample_step_warp = cast(torch.Tensor, out["step_warp_ms"])
                sample_step_solver = cast(torch.Tensor, out["step_solver_ms"])
                sample_step_decode = cast(torch.Tensor, out["step_decode_ms"])
                sample_step_indices = cast(torch.Tensor, out["step_indices"])
                if step_indices is None:
                    step_indices = sample_step_indices
                elif sample_step_indices.shape != step_indices.shape:
                    raise ValueError(
                        f"batch 内 step 数不一致: {sample_step_indices.shape} vs {step_indices.shape}"
                    )

                step_logits_list.append(sample_step_logits)
                step_time_list.append(sample_step_time)
                step_warp_list.append(sample_step_warp)
                step_solver_list.append(sample_step_solver)
                step_decode_list.append(sample_step_decode)
                diag_list.append(cast(dict[str, torch.Tensor], out["diagnostics"]))

        if step_indices is None:
            step_indices = torch.zeros((0,), dtype=torch.long, device=fast_logits.device)
        step_logits = torch.stack(step_logits_list, dim=0)
        step_time_ms = torch.stack(step_time_list, dim=0)
        step_warp_ms = torch.stack(step_warp_list, dim=0)
        step_solver_ms = torch.stack(step_solver_list, dim=0)
        step_decode_ms = torch.stack(step_decode_list, dim=0)
        return {
            "step_logits": step_logits,
            "step_time_ms": step_time_ms,
            "step_warp_ms": step_warp_ms,
            "step_solver_ms": step_solver_ms,
            "step_decode_ms": step_decode_ms,
            "step_indices": step_indices,
            "diagnostics": diag_list,
        }

    def _forward_batched_stepwise_train(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        max_step_index: int | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """训练：返回逐步 logits（完整 BPTT），不做计时。"""
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if self.amp_fp16 else nullcontext()
        step_logits_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        fast_kl_list: list[torch.Tensor] = []
        step_indices: torch.Tensor | None = None
        with amp_ctx:
            for b in range(fast_logits.shape[0]):
                out = self._forward_single_stepwise_train(
                    fast_logits=fast_logits[b],
                    slow_logits=slow_logits[b],
                    frame_ego2global=frame_ego2global[b],
                    frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                    frame_dt=frame_dt[b] if frame_dt is not None else None,
                    max_step_index=max_step_index,
                )
                sample_step_logits = cast(torch.Tensor, out["step_logits"])
                sample_step_indices = cast(torch.Tensor, out["step_indices"])
                if step_indices is None:
                    step_indices = sample_step_indices
                elif sample_step_indices.shape != step_indices.shape:
                    raise ValueError(
                        f"batch 内 step 数不一致: {sample_step_indices.shape} vs {step_indices.shape}"
                    )

                step_logits_list.append(sample_step_logits)
                diag_list.append(cast(dict[str, torch.Tensor], out["diagnostics"]))
                if "fast_kl" in out:
                    fast_kl_list.append(cast(torch.Tensor, out["fast_kl"]))

        if step_indices is None:
            step_indices = torch.zeros((0,), dtype=torch.long, device=fast_logits.device)
        step_logits = torch.stack(step_logits_list, dim=0)
        result: Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "step_logits": step_logits,
            "step_indices": step_indices,
            "diagnostics": diag_list,
        }
        if fast_kl_list:
            result["fast_kl"] = torch.stack(fast_kl_list).mean()
        return result
