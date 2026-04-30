"""Neural ODE 离散化 baseline。

与 OnlineNcdeAligner 的唯一差异：
  - NCDE：每步控制增量 = (Fast_t - Fast_{t-1→t}) ‖ τ 拼接 → 1x1x1 conv → hidden_dim，
    再与 func_g 输出做 odot；
  - 本 baseline：每步控制增量 = 标量 Δt，广播为 (hidden_dim, X, Y, Z) 后与 func_g
    输出做 odot。形式上即 h' = g(h, f) * Δt 的经典 Neural ODE 离散化。

其余结构（双独立 encoder、全分辨率、ego-warp、FuncG、fast 残差、stepwise 接口、
fast-KL 协议）全部对齐 OnlineNcdeAligner，方便消融归因 NCDE 控制增量构造。
"""

from __future__ import annotations

import time
from typing import Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from online_ncde.data.ego_warp_list import (
    backward_warp_dense_trilinear,
    build_sampling_grid,
    compute_transform_prev_to_curr,
)
from online_ncde.data.time_series import compute_segment_dt
from online_ncde.models.decoder import DenseDecoder
from online_ncde.models.encoder import DenseEncoder
from online_ncde.models.func_g import FuncG


def _compute_m_occ(fast_logits: torch.Tensor, free_index: int) -> torch.Tensor:
    """与 NCDE 同口径：m_occ = max_{c != free}(logit[c]) - logit[free]。"""
    masked = fast_logits.clone()
    masked.narrow(-4, free_index, 1).fill_(float("-inf"))
    max_non_free = masked.amax(dim=-4, keepdim=True)
    free_logit = fast_logits.narrow(-4, free_index, 1)
    return max_non_free - free_logit


class NeuralOdeDtSolver(nn.Module):
    """Δt 驱动的更新器，签名与 HeunSolver / EulerNextFastSolver 风格保持平行。

    - heun  : h_hat = h + Δt * g(h, f_prev_adv); h_next = h + 0.5 * Δt * (g(h, f_prev) + g(h_hat, f_t))
    - euler : h_next = h + Δt * g(h, f_t)
    """

    def __init__(self, func_g: FuncG, variant: str = "heun") -> None:
        super().__init__()
        variant_lower = str(variant).lower()
        if variant_lower not in ("heun", "euler"):
            raise ValueError(f"未知的 solver variant: {variant!r}，可选: 'heun', 'euler'")
        self.func_g = func_g
        self.variant = variant_lower

    def step(
        self,
        h_adv: torch.Tensor,
        f_prev_adv: torch.Tensor,
        f_t: torch.Tensor,
        dt: torch.Tensor,  # 0-dim 标量张量，与 h_adv 同 device/dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 标量 Δt 直接与 (C_h, X, Y, Z) 广播相乘 —— 等价于把 Δt 广播成同形向量
        if self.variant == "heun":
            s1 = self.func_g(h_adv, f_prev_adv)
            k1 = s1 * dt
            h_hat = h_adv + k1
            s2 = self.func_g(h_hat, f_t)
            k2 = s2 * dt
            h_next = h_adv + 0.5 * (k1 + k2)
            # 诊断量：与 NCDE 的 delta_scene 对齐 —— 整体平均更新幅度
            delta_scene = 0.5 * (k1 + k2)
        else:
            slope = self.func_g(h_adv, f_t)
            delta_scene = slope * dt
            h_next = h_adv + delta_scene
        return h_next, delta_scene


class NeuralOdeDtAligner(nn.Module):
    """Neural ODE 离散化对齐器（Δt 控制增量 baseline）。

    forward / forward_stepwise_eval 接口与 OnlineNcdeAligner 一致，可直接喂给
    Trainer / 各类 evaluation 脚本，无需任何分支改动。
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
        solver_variant: str = "heun",
    ) -> None:
        super().__init__()
        self.use_fast_residual = bool(use_fast_residual)
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder_in_channels = int(encoder_in_channels)
        if self.hidden_dim != self.feat_dim:
            raise ValueError(
                f"hidden_dim 必须与 feat_dim 相同，当前 hidden_dim={self.hidden_dim}, "
                f"feat_dim={self.feat_dim}。"
            )
        self.free_index = int(free_index)

        pc_range_tuple = tuple(pc_range)
        if len(pc_range_tuple) != 6:
            raise ValueError(f"pc_range 必须长度为 6，当前: {pc_range_tuple}")
        self.pc_range = cast(Tuple[float, float, float, float, float, float], pc_range_tuple)
        self.voxel_size = tuple(voxel_size)
        self.timestamp_scale = float(timestamp_scale)

        # 双独立 encoder + FuncG + Decoder：与 NCDE 一致；唯一差别是没有 CtrlProjector
        self.fast_encoder = DenseEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        self.slow_encoder = DenseEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        self.func_g = FuncG(
            in_channels=hidden_dim + feat_dim,
            hidden_channels=hidden_dim,
            inner_dim=func_g_inner_dim,
            body_dilations=func_g_body_dilations,
            gn_groups=func_g_gn_groups,
        )
        self.solver = NeuralOdeDtSolver(func_g=self.func_g, variant=solver_variant)
        self.solver_variant = self.solver.variant
        self.decoder = DenseDecoder(
            in_channels=hidden_dim,
            out_channels=num_classes,
            init_scale=decoder_init_scale,
        )

        # Trainer 根据 lambda_fast_kl 注入；False 时跳过 KL（与 NCDE 同协议）
        self._fast_kl_active: bool = False

    # ----------------- 编码 / 解码 -----------------

    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        return self.fast_encoder(fast_logits)

    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        return self.slow_encoder(slow_logits.unsqueeze(0))[0]

    def _decode_dense_state(self, z_dense: torch.Tensor) -> torch.Tensor:
        z_tensor = z_dense.unsqueeze(0)
        z_tensor = z_tensor.permute(0, 1, 4, 3, 2).contiguous()  # (1, C, Z, Y, X)
        out_dense = self.decoder(z_tensor)
        return out_dense.permute(0, 1, 4, 3, 2).contiguous()[0]

    def _compute_fast_kl_step(
        self,
        fast_logits_t: torch.Tensor,
        aligned_logits: torch.Tensor,
    ) -> torch.Tensor:
        aligned_f = aligned_logits.float()
        fast_f = fast_logits_t.float()
        w = _compute_m_occ(fast_f, self.free_index).clamp(min=0.0)
        log_p_fast = F.log_softmax(fast_f, dim=0)
        log_p_aligned = F.log_softmax(aligned_f, dim=0)
        kl_per_voxel = F.kl_div(
            log_p_fast, log_p_aligned, log_target=True, reduction="none"
        ).sum(dim=0, keepdim=True)
        return (w * kl_per_voxel).mean()

    # ----------------- 共用：单帧 warp + solver 一步 -----------------

    def _rollout_step(
        self,
        Z_dense: torch.Tensor,
        fast_feat: torch.Tensor,
        frame_ego2global: torch.Tensor,
        spatial_shape_xyz: Tuple[int, int, int],
        pc_range_6: Tuple[float, float, float, float, float, float],
        voxel_size_3: Tuple[float, float, float],
        dt_k: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """k -> k+1 的 warp + solver 单步；返回 (Z_next, delta_scene)。"""
        transform = compute_transform_prev_to_curr(
            pose_prev_ego2global=frame_ego2global[k],
            pose_curr_ego2global=frame_ego2global[k + 1],
        )
        grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)
        Z_warp = backward_warp_dense_trilinear(
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
        Z_next, delta_scene = self.solver.step(
            h_adv=Z_warp, f_prev_adv=f_prev_adv, f_t=f_t, dt=dt_k
        )
        return Z_next, delta_scene

    # ----------------- 单样本：默认（仅末帧） -----------------

    def _forward_single(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: int = 0,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        num_frames = fast_logits.shape[0]

        # h=0 退化：scene 第一帧 keyframe
        if rollout_start_step >= num_frames - 1:
            return {
                "aligned": slow_logits.float(),
                "diagnostics": {
                    "delta_scene_abs_mean": torch.tensor(0.0, device=fast_logits.device),
                },
            }

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

        # 与 NCDE 同步：use_fast_residual 时初始锚到最老真实 keyframe
        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[rollout_start_step]
        else:
            Z_dense = slow_feat

        delta_mag_values: list[float] = []

        for k in range(rollout_start_step, num_frames - 1):
            Z_dense, delta_scene = self._rollout_step(
                Z_dense=Z_dense,
                fast_feat=fast_feat,
                frame_ego2global=frame_ego2global,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range_6=pc_range_6,
                voxel_size_3=voxel_size_3,
                dt_k=dt[k],
                k=k,
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

    # ----------------- 单样本：stepwise train -----------------

    def _forward_single_stepwise_train(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        max_step_index: int | None = None,
        rollout_start_step: int = 0,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        num_frames = fast_logits.shape[0]

        if rollout_start_step >= num_frames - 1:
            step_logits = slow_logits.unsqueeze(0).float()
            step_indices = torch.tensor(
                [num_frames - 1], device=fast_logits.device, dtype=torch.long
            )
            return {
                "step_logits": step_logits,
                "step_indices": step_indices,
                "diagnostics": {
                    "delta_scene_abs_mean": torch.tensor(0.0, device=fast_logits.device),
                },
            }

        rollout_steps = (num_frames - 1) - rollout_start_step
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

        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[rollout_start_step]
        else:
            Z_dense = slow_feat

        delta_mag_values: list[float] = []
        fast_kl_accum: torch.Tensor | None = None
        fast_kl_step_count = 0
        step_logits_list: list[torch.Tensor] = []
        compute_fast_kl = self._fast_kl_active and self.use_fast_residual

        for k_off in range(rollout_steps):
            k = rollout_start_step + k_off
            Z_dense, delta_scene = self._rollout_step(
                Z_dense=Z_dense,
                fast_feat=fast_feat,
                frame_ego2global=frame_ego2global,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range_6=pc_range_6,
                voxel_size_3=voxel_size_3,
                dt_k=dt[k],
                k=k,
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
        step_indices = torch.arange(
            rollout_start_step + 1,
            rollout_start_step + 1 + rollout_steps,
            device=fast_logits.device,
            dtype=torch.long,
        )
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

    # ----------------- 单样本：stepwise eval（含分段计时） -----------------

    def _forward_single_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: int = 0,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        num_frames = fast_logits.shape[0]

        if rollout_start_step >= num_frames - 1:
            step_logits = slow_logits.unsqueeze(0).float()
            step_indices = torch.tensor(
                [num_frames - 1], device=fast_logits.device, dtype=torch.long
            )
            zero_step = torch.zeros((1,), device=fast_logits.device, dtype=torch.float32)
            return {
                "step_logits": step_logits,
                "step_time_ms": zero_step,
                "step_warp_ms": zero_step,
                "step_solver_ms": zero_step,
                "step_decode_ms": zero_step,
                "step_indices": step_indices,
                "diagnostics": {
                    "delta_scene_abs_mean": torch.tensor(0.0, device=fast_logits.device),
                },
            }

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

        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[rollout_start_step]
        else:
            Z_dense = slow_feat

        delta_mag_values: list[float] = []
        step_logits_list: list[torch.Tensor] = []
        step_warp_ms_values: list[float] = []
        step_solver_ms_values: list[float] = []
        step_decode_ms_values: list[float] = []
        step_time_events: list[tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = []
        use_cuda_timing = fast_logits.is_cuda

        for k in range(rollout_start_step, num_frames - 1):
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
            Z_warp = backward_warp_dense_trilinear(
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
            Z_dense, delta_scene = self.solver.step(
                h_adv=Z_warp, f_prev_adv=f_prev_adv, f_t=f_t, dt=dt[k]
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
        step_indices = torch.arange(
            rollout_start_step + 1, num_frames, device=fast_logits.device, dtype=torch.long
        )
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

    # ----------------- batch 包装 / 公共入口 -----------------

    def _unsqueeze_inputs(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ):
        if fast_logits.dim() == 5:
            fast_logits = fast_logits.unsqueeze(0)
            slow_logits = slow_logits.unsqueeze(0)
            frame_ego2global = frame_ego2global.unsqueeze(0)
            if frame_timestamps is not None:
                frame_timestamps = frame_timestamps.unsqueeze(0)
            if frame_dt is not None:
                frame_dt = frame_dt.unsqueeze(0)
        return fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt

    def forward_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt = (
            self._unsqueeze_inputs(fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt)
        )
        return self._forward_batched_stepwise_eval(
            fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt,
            rollout_start_step=rollout_start_step,
        )

    def forward(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        mode: str = "default",
        max_step_index: int | None = None,
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt = (
            self._unsqueeze_inputs(fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt)
        )
        if mode == "stepwise_train":
            return self._forward_batched_stepwise_train(
                fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt,
                max_step_index=max_step_index,
                rollout_start_step=rollout_start_step,
            )
        elif mode == "stepwise_eval":
            return self._forward_batched_stepwise_eval(
                fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt,
                rollout_start_step=rollout_start_step,
            )
        elif mode == "default":
            return self._forward_batched_default(
                fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt,
                rollout_start_step=rollout_start_step,
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
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        aligned_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        for b in range(fast_logits.shape[0]):
            rss_b = int(rollout_start_step[b].item()) if rollout_start_step is not None else 0
            out = self._forward_single(
                fast_logits=fast_logits[b],
                slow_logits=slow_logits[b],
                frame_ego2global=frame_ego2global[b],
                frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                frame_dt=frame_dt[b] if frame_dt is not None else None,
                rollout_start_step=rss_b,
            )
            aligned_list.append(cast(torch.Tensor, out["aligned"]))
            diag_list.append(cast(dict[str, torch.Tensor], out["diagnostics"]))
        aligned = torch.stack(aligned_list, dim=0)
        return {"aligned": aligned, "diagnostics": diag_list}

    def _forward_batched_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        step_logits_list: list[torch.Tensor] = []
        step_time_list: list[torch.Tensor] = []
        step_warp_list: list[torch.Tensor] = []
        step_solver_list: list[torch.Tensor] = []
        step_decode_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        step_indices: torch.Tensor | None = None
        for b in range(fast_logits.shape[0]):
            rss_b = int(rollout_start_step[b].item()) if rollout_start_step is not None else 0
            out = self._forward_single_stepwise_eval(
                fast_logits=fast_logits[b],
                slow_logits=slow_logits[b],
                frame_ego2global=frame_ego2global[b],
                frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                frame_dt=frame_dt[b] if frame_dt is not None else None,
                rollout_start_step=rss_b,
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
            step_time_list.append(cast(torch.Tensor, out["step_time_ms"]))
            step_warp_list.append(cast(torch.Tensor, out["step_warp_ms"]))
            step_solver_list.append(cast(torch.Tensor, out["step_solver_ms"]))
            step_decode_list.append(cast(torch.Tensor, out["step_decode_ms"]))
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
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        step_logits_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        fast_kl_list: list[torch.Tensor] = []
        step_indices: torch.Tensor | None = None
        for b in range(fast_logits.shape[0]):
            rss_b = int(rollout_start_step[b].item()) if rollout_start_step is not None else 0
            out = self._forward_single_stepwise_train(
                fast_logits=fast_logits[b],
                slow_logits=slow_logits[b],
                frame_ego2global=frame_ego2global[b],
                frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                frame_dt=frame_dt[b] if frame_dt is not None else None,
                max_step_index=max_step_index,
                rollout_start_step=rss_b,
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
