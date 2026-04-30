"""下采样 Online NCDE Aligner（OpenOccupancy 等大网格分支专用）。

输入 logits 在 (X, Y, Z) 全分辨率（如 512×512×40）；编码器一次 stride=(2,2,2)
下采样到隐藏分辨率（256×256×20），主干（FuncG / Solver / warp）全部在隐藏分辨率
上演化；解码器先在隐藏分辨率做 3×3×3 卷积，再 trilinear 上采样回全分辨率，
最后 depthwise(3×3×3) + pointwise 映射到类别 logits。残差范式下输出 = decoder
输出 + 当前帧 fast logits（同样在全分辨率）。
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from online_ncde.utils.nn import resolve_group_norm_groups


class DownsampleEncoder(nn.Module):
    """单层 stride=(2,2,2) 下采样编码器：(B, C_in, X, Y, Z) → (B, C_out, X//2, Y//2, Z//2)。"""

    def __init__(self, in_channels: int, out_channels: int, gn_groups: int = 8) -> None:
        super().__init__()
        groups = resolve_group_norm_groups(num_channels=out_channels, preferred_groups=gn_groups)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            padding=1,
            bias=False,
        )
        self.gn = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv3d 约定空间维 (D, H, W)，把 Z 放到 D。
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        x = self.relu(self.gn(self.conv(x)))
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        return x


class UpsampleDecoder(nn.Module):
    """隐藏分辨率 3×3×3 卷积 → trilinear 上采样 ×2 → DW(3×3×3) + PW(1×1×1) → logits。

    init_scale 与 DenseDecoder 一致：None 走默认初始化；<=0 输出权重/bias 全 0；
    >0 用 std=init_scale 的正态采样（残差范式下首步输出≈0）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_scale: Tuple[int, int, int] = (2, 2, 2),
        init_scale: Optional[float] = 1.0e-6,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        groups = resolve_group_norm_groups(num_channels=in_channels, preferred_groups=gn_groups)

        # 隐藏分辨率：3×3×3 语义增强
        self.conv_hidden = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.gn_hidden = nn.GroupNorm(groups, in_channels)
        self.relu_hidden = nn.ReLU(inplace=True)

        # 上采样到全分辨率（不可学习 trilinear）
        self.upsample_scale = tuple(int(s) for s in upsample_scale)

        # 全分辨率：depthwise 3×3×3
        self.refine_dw = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.refine_gn = nn.GroupNorm(groups, in_channels)
        self.refine_relu = nn.ReLU(inplace=True)

        # 全分辨率：pointwise 1×1×1 映射到 num_classes
        self.out_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self._init_output(init_scale)

    def _init_output(self, init_scale: Optional[float]) -> None:
        """init_scale=None 走默认；<=0 全零；>0 小方差正态。"""
        if init_scale is None:
            return
        if init_scale <= 0.0:
            nn.init.constant_(self.out_conv.weight, 0.0)
            if self.out_conv.bias is not None:
                nn.init.constant_(self.out_conv.bias, 0.0)
        else:
            nn.init.normal_(self.out_conv.weight, mean=0.0, std=init_scale)
            if self.out_conv.bias is not None:
                nn.init.constant_(self.out_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, Z_h, Y_h, X_h) 隐藏分辨率
        Returns:
            (B, C_out, Z_full, Y_full, X_full) 全分辨率 logits
        """
        x = self.relu_hidden(self.gn_hidden(self.conv_hidden(x)))

        # F.interpolate 在 (B, C, D, H, W) 上工作，scale_factor 顺序对应 (D, H, W)=(Z, Y, X)
        sz, sy, sx = self.upsample_scale[2], self.upsample_scale[1], self.upsample_scale[0]
        x = F.interpolate(x, scale_factor=(sz, sy, sx), mode="trilinear", align_corners=False)

        x = self.refine_relu(self.refine_gn(self.refine_dw(x)))
        x = self.out_conv(x)
        return x


def _compute_m_occ(fast_logits: torch.Tensor, free_index: int) -> torch.Tensor:
    """m_occ = max_{c != free}(logit[c]) - logit[free]，与全分辨率 aligner 共用语义。"""
    masked = fast_logits.clone()
    masked.narrow(-4, free_index, 1).fill_(float("-inf"))
    max_non_free = masked.amax(dim=-4, keepdim=True)
    free_logit = fast_logits.narrow(-4, free_index, 1)
    return max_non_free - free_logit


class OnlineNcdeAlignerDS(nn.Module):
    """下采样 Online NCDE Aligner：encoder ↓2 → 隐藏分辨率主干演化 → decoder ↑2 → 残差。

    输入约定：fast_logits (T, C, X, Y, Z)；slow_logits (C, X, Y, Z)。X/Y/Z 必须能被
    encoder_downsample_stride 整除。warp / FuncG / solver 在 (X//s, Y//s, Z//s) 上算，
    voxel_size 自动按 stride 放大。
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
        encoder_downsample_stride: Tuple[int, int, int] = (2, 2, 2),
        decoder_init_scale: float = 1.0e-6,
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

        voxel_size_3 = tuple(float(v) for v in voxel_size)
        if len(voxel_size_3) != 3:
            raise ValueError(f"voxel_size 必须长度为 3，当前: {voxel_size_3}")
        self.voxel_size = cast(Tuple[float, float, float], voxel_size_3)

        stride_3 = tuple(int(s) for s in encoder_downsample_stride)
        if len(stride_3) != 3 or any(s <= 0 for s in stride_3):
            raise ValueError(f"encoder_downsample_stride 必须为 3 个正整数，当前: {stride_3}")
        self.downsample_stride = cast(Tuple[int, int, int], stride_3)

        # 隐藏分辨率下的 voxel_size：覆盖范围不变，单 voxel 物理尺寸 ×stride
        self.hidden_voxel_size = cast(
            Tuple[float, float, float],
            tuple(float(v) * float(s) for v, s in zip(self.voxel_size, self.downsample_stride)),
        )
        self.timestamp_scale = float(timestamp_scale)

        self.fast_encoder = DownsampleEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        self.slow_encoder = DownsampleEncoder(
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
            self.solver = EulerNextFastSolver(func_g=self.func_g, ctrl_proj=self.ctrl_proj)
        else:
            raise ValueError(
                f"未知的 solver_variant: {solver_variant!r}，可选: 'heun', 'euler'"
            )
        self.solver_variant = solver_variant_lower

        self.decoder = UpsampleDecoder(
            in_channels=hidden_dim,
            out_channels=num_classes,
            upsample_scale=self.downsample_stride,
            init_scale=decoder_init_scale,
        )

        # OO 分支默认关闭 fast_kl；trainer 看 lambda_fast_kl>0 时才置 True。
        self._fast_kl_active: bool = False

    # ---------- 编解码 ---------- #
    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        """(T, C_in, X, Y, Z) → (T, C_f, X_h, Y_h, Z_h)。"""
        return self.fast_encoder(fast_logits)

    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        """(C_in, X, Y, Z) → (C_f, X_h, Y_h, Z_h)。"""
        return self.slow_encoder(slow_logits.unsqueeze(0))[0]

    def _decode_dense_state(self, z_dense: torch.Tensor) -> torch.Tensor:
        """隐藏态 (C, X_h, Y_h, Z_h) → 全分辨率残差 logits (C_out, X, Y, Z)。"""
        z_tensor = z_dense.unsqueeze(0)  # (1, C, X_h, Y_h, Z_h)
        z_tensor = z_tensor.permute(0, 1, 4, 3, 2).contiguous()  # (1, C, Z, Y, X)
        out = self.decoder(z_tensor)
        return out.permute(0, 1, 4, 3, 2).contiguous()[0]

    def _compute_fast_kl_step(
        self,
        fast_logits_t: torch.Tensor,
        aligned_logits: torch.Tensor,
    ) -> torch.Tensor:
        """与 OnlineNcdeAligner 对齐的 conf-加权 KL；OO 分支默认不调用。"""
        aligned_f = aligned_logits.float()
        fast_f = fast_logits_t.float()
        w = _compute_m_occ(fast_f, self.free_index).clamp(min=0.0)
        log_p_fast = F.log_softmax(fast_f, dim=0)
        log_p_aligned = F.log_softmax(aligned_f, dim=0)
        kl_per_voxel = F.kl_div(
            log_p_fast, log_p_aligned, log_target=True, reduction="none"
        ).sum(dim=0, keepdim=True)
        return (w * kl_per_voxel).mean()

    # ---------- 单样本 forward ---------- #
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
        pc_range_6 = self.pc_range
        voxel_size_3 = self.hidden_voxel_size

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)
        tau = cumulative_tau(dt).to(device=fast_logits.device)

        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[rollout_start_step]
        else:
            Z_dense = slow_feat

        delta_mag_values: list[float] = []

        for k in range(rollout_start_step, num_frames - 1):
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
        pc_range_6 = self.pc_range
        voxel_size_3 = self.hidden_voxel_size

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)
        tau = cumulative_tau(dt).to(device=fast_logits.device)

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
        pc_range_6 = self.pc_range
        voxel_size_3 = self.hidden_voxel_size

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)
        tau = cumulative_tau(dt).to(device=fast_logits.device)

        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[rollout_start_step]
        else:
            Z_dense = slow_feat

        delta_mag_values: list[float] = []
        step_logits_list: list[torch.Tensor] = []
        step_warp_ms_values: list[float] = []
        step_solver_ms_values: list[float] = []
        step_decode_ms_values: list[float] = []
        step_time_events: list[
            tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]
        ] = []
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

    # ---------- 批处理包装（与 OnlineNcdeAligner 接口对齐） ---------- #
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
            raise ValueError(
                f"未知的 forward mode: {mode!r}，可选: 'default', 'stepwise_train', 'stepwise_eval'"
            )

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

    def _forward_batched_default(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        aligned_list = []
        diag_list = []
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
