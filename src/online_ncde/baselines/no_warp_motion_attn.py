"""No-warp motion-conditioned attention baseline.

该 baseline 用于验证显式几何 warp / 两阶段演化是否必要：每步不对隐藏状态或
fast 特征做 grid_sample，只把相邻帧 ego motion 转成 dense motion field 作为
attention 条件，让窗口 cross-attention 隐式学习跨帧对应关系。
"""

from __future__ import annotations

import time
from typing import Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from online_ncde.baselines.recurrent_warp_fusion import (
    RecurrentWarpFusionAligner,
    _ResidualDilatedBlock,
)
from online_ncde.data.ego_warp_list import compute_transform_prev_to_curr
from online_ncde.data.time_series import compute_segment_dt
from online_ncde.utils.nn import resolve_group_norm_groups


class _WindowCrossAttention3D(nn.Module):
    """3D window cross-attention：query 来自当前 fast，key/value 来自上一帧 hidden。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} 必须能被 num_heads={num_heads} 整除")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.window_size = tuple(int(v) for v in window_size)
        self.shift_size = tuple(int(v) for v in shift_size)
        self.q = nn.Conv3d(dim, dim, kernel_size=1, bias=True)
        self.k = nn.Conv3d(dim, dim, kernel_size=1, bias=True)
        self.v = nn.Conv3d(dim, dim, kernel_size=1, bias=True)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1, bias=True)

    def _to_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int, int]]:
        B, C, X, Y, Z = x.shape
        Wx, Wy, Wz = self.window_size
        Sx, Sy, Sz = self.shift_size
        if (X % Wx) or (Y % Wy) or (Z % Wz):
            raise ValueError(
                f"空间形状 ({X},{Y},{Z}) 必须能被 window ({Wx},{Wy},{Wz}) 整除"
            )
        if Sx or Sy or Sz:
            x = torch.roll(x, shifts=(-Sx, -Sy, -Sz), dims=(2, 3, 4))
        nWx, nWy, nWz = X // Wx, Y // Wy, Z // Wz
        x = x.view(B, C, nWx, Wx, nWy, Wy, nWz, Wz)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.view(B * nWx * nWy * nWz, Wx * Wy * Wz, C)
        return x, (B, C, X, Y, Z, nWx * nWy * nWz)

    def _from_windows(self, x: torch.Tensor, meta: tuple[int, int, int, int, int, int]) -> torch.Tensor:
        B, C, X, Y, Z, _ = meta
        Wx, Wy, Wz = self.window_size
        Sx, Sy, Sz = self.shift_size
        nWx, nWy, nWz = X // Wx, Y // Wy, Z // Wz
        x = x.view(B, nWx, nWy, nWz, Wx, Wy, Wz, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, X, Y, Z)
        if Sx or Sy or Sz:
            x = torch.roll(x, shifts=(Sx, Sy, Sz), dims=(2, 3, 4))
        return x

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # 输入/输出: (B, C, X, Y, Z)
        q = self.q(query)
        k = self.k(key_value)
        v = self.v(key_value)
        q_w, meta = self._to_windows(q)
        k_w, _ = self._to_windows(k)
        v_w, _ = self._to_windows(v)

        N = q_w.shape[1]
        q_w = q_w.view(-1, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_w = k_w.view(-1, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_w = v_w.view(-1, N, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q_w, k_w, v_w)
        out = out.transpose(1, 2).reshape(-1, N, self.dim)
        out = self._from_windows(out, meta)
        return self.proj(out)


class _CrossAttentionBlock(nn.Module):
    """Pre-GN cross-attention + 1x1 FFN。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int],
        gn_groups: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        groups = resolve_group_norm_groups(num_channels=dim, preferred_groups=gn_groups)
        self.norm_q = nn.GroupNorm(groups, dim)
        self.norm_kv = nn.GroupNorm(groups, dim)
        self.attn = _WindowCrossAttention3D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
        )
        hidden = max(int(round(dim * float(mlp_ratio))), dim)
        self.norm2 = nn.GroupNorm(groups, dim)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, dim, kernel_size=1, bias=True),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        x = query + self.attn(self.norm_q(query), self.norm_kv(key_value))
        x = x + self.ffn(self.norm2(x))
        return x


class NoWarpMotionBiasAttnFusion(nn.Module):
    """无显式 warp 的 motion-conditioned cross-attention 主干。

    主干计算维度默认 24，与当前 NCDE 的 func_g_inner_dim 对齐；hidden state 仍为 32。
    motion_field=(dx,dy,dz,in_bounds) 只作为条件通道输入，不用于采样特征。
    """

    def __init__(
        self,
        hidden_dim: int,
        inner_dim: int = 24,
        num_heads: int = 3,
        window_size: Tuple[int, int, int] = (8, 8, 4),
        head_dilations: Tuple[int, int] = (1, 2),
        gn_groups: int = 8,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if len(head_dilations) != 2:
            raise ValueError(f"head_dilations 需恰好 2 个值，当前: {head_dilations}")
        self.window_size = tuple(int(v) for v in window_size)
        groups = resolve_group_norm_groups(num_channels=inner_dim, preferred_groups=gn_groups)
        cond_channels = 1 + 4  # dt + motion(dx,dy,dz,in_bounds)
        self.q_stem = nn.Sequential(
            nn.Conv3d(hidden_dim + cond_channels, inner_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, inner_dim),
            nn.SiLU(inplace=True),
        )
        self.kv_stem = nn.Sequential(
            nn.Conv3d(hidden_dim + cond_channels, inner_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, inner_dim),
            nn.SiLU(inplace=True),
        )
        shift = (window_size[0] // 2, window_size[1] // 2, 0)
        self.block0 = _CrossAttentionBlock(
            dim=inner_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=(0, 0, 0),
            gn_groups=gn_groups,
            mlp_ratio=mlp_ratio,
        )
        self.local0 = _ResidualDilatedBlock(inner_dim, int(head_dilations[0]), groups)
        self.block1 = _CrossAttentionBlock(
            dim=inner_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift,
            gn_groups=gn_groups,
            mlp_ratio=mlp_ratio,
        )
        self.local1 = _ResidualDilatedBlock(inner_dim, int(head_dilations[1]), groups)
        self.head = nn.Conv3d(inner_dim, hidden_dim, kernel_size=1, bias=True)

    def forward(
        self,
        h_prev: torch.Tensor,
        fast_curr: torch.Tensor,
        dt_channel: torch.Tensor,
        motion_field: torch.Tensor,
    ) -> torch.Tensor:
        cond = torch.cat([dt_channel, motion_field], dim=0)
        # 与 RWFA/FusionAttnNet 对齐：Conv3d 直接吃 (B,C,X,Y,Z)，不做轴置换。
        q = torch.cat([fast_curr, cond], dim=0).unsqueeze(0).contiguous()
        kv = torch.cat([h_prev, cond], dim=0).unsqueeze(0).contiguous()
        q = self.q_stem(q)
        kv = self.kv_stem(kv)
        x = self.block0(q, kv)
        x = self.local0(x)
        x = self.block1(x, kv)
        x = self.local1(x)
        return self.head(x).squeeze(0).contiguous()


class NoWarpMotionBiasAttnAligner(RecurrentWarpFusionAligner):
    """No-warp attention baseline，与 RWFA/NCDE 共用 forward 接口。"""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        hidden_dim: int,
        encoder_in_channels: int,
        free_index: int,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        decoder_init_scale: float | None = 0.0,
        use_fast_residual: bool = True,
        fusion_inner_dim: int = 24,
        fusion_attn_num_heads: int = 3,
        fusion_attn_window_size: Tuple[int, int, int] = (8, 8, 4),
        fusion_attn_head_dilations: Tuple[int, int] = (1, 2),
        fusion_gn_groups: int = 8,
        fusion_attn_mlp_ratio: float = 2.0,
        timestamp_scale: float = 1.0e-6,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            encoder_in_channels=encoder_in_channels,
            free_index=free_index,
            pc_range=pc_range,
            voxel_size=voxel_size,
            decoder_init_scale=decoder_init_scale,
            use_fast_residual=use_fast_residual,
            fusion_kind="attn",
            fusion_inner_dim=fusion_inner_dim,
            fusion_gn_groups=fusion_gn_groups,
            fusion_attn_num_heads=fusion_attn_num_heads,
            fusion_attn_window_size=fusion_attn_window_size,
            fusion_attn_head_dilations=fusion_attn_head_dilations,
            fusion_attn_mlp_ratio=fusion_attn_mlp_ratio,
            timestamp_scale=timestamp_scale,
        )
        self.fusion = NoWarpMotionBiasAttnFusion(
            hidden_dim=hidden_dim,
            inner_dim=fusion_inner_dim,
            num_heads=fusion_attn_num_heads,
            window_size=fusion_attn_window_size,
            head_dilations=fusion_attn_head_dilations,
            gn_groups=fusion_gn_groups,
            mlp_ratio=fusion_attn_mlp_ratio,
        )

    def _make_motion_field(
        self,
        transform_prev_to_curr: torch.Tensor,
        spatial_shape_xyz: Tuple[int, int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """生成 motion 条件通道，不做任何特征采样。"""
        x_size, y_size, z_size = spatial_shape_xyz
        device = transform_prev_to_curr.device
        pc_min = transform_prev_to_curr.new_tensor(self.pc_range[:3])
        vsize = transform_prev_to_curr.new_tensor(self.voxel_size)

        xs = torch.arange(x_size, device=device, dtype=torch.float32)
        ys = torch.arange(y_size, device=device, dtype=torch.float32)
        zs = torch.arange(z_size, device=device, dtype=torch.float32)
        ix, iy, iz = torch.meshgrid(xs, ys, zs, indexing="ij")
        tgt_idx = torch.stack([ix, iy, iz], dim=-1).reshape(-1, 3)
        tgt_metric = (tgt_idx + 0.5) * vsize + pc_min

        curr_to_prev = torch.linalg.inv(transform_prev_to_curr.float())
        src_metric = tgt_metric @ curr_to_prev[:3, :3].T + curr_to_prev[:3, 3]
        src_idx = (src_metric - pc_min) / vsize - 0.5
        offset = src_idx - tgt_idx

        win = transform_prev_to_curr.new_tensor(self.fusion.window_size).clamp(min=1.0)
        offset_norm = (offset / win).reshape(x_size, y_size, z_size, 3)
        in_bounds = (
            (src_idx[:, 0] >= 0) & (src_idx[:, 0] <= x_size - 1)
            & (src_idx[:, 1] >= 0) & (src_idx[:, 1] <= y_size - 1)
            & (src_idx[:, 2] >= 0) & (src_idx[:, 2] <= z_size - 1)
        ).to(dtype=torch.float32).reshape(x_size, y_size, z_size, 1)
        motion = torch.cat([offset_norm, in_bounds], dim=-1).permute(3, 0, 1, 2)
        return motion.to(device=device, dtype=dtype).contiguous()

    def _advance_no_warp(
        self,
        h_dense: torch.Tensor,
        fast_curr: torch.Tensor,
        dt_value: torch.Tensor,
        transform_prev_to_curr: torch.Tensor,
        spatial_shape_xyz: Tuple[int, int, int],
    ) -> torch.Tensor:
        dt_ch = self._make_dt_channel(dt_value, spatial_shape_xyz, h_dense.device, h_dense.dtype)
        motion = self._make_motion_field(transform_prev_to_curr, spatial_shape_xyz, h_dense.dtype)
        return self.fusion(
            h_prev=h_dense,
            fast_curr=fast_curr,
            dt_channel=dt_ch,
            motion_field=motion,
        )

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
                "diagnostics": {"delta_scene_abs_mean": torch.tensor(0.0, device=fast_logits.device)},
            }

        fast_feat = self._encode_fast(fast_logits)
        h_dense = self._encode_slow(slow_logits)
        spatial_shape_xyz = (int(fast_feat.shape[2]), int(fast_feat.shape[3]), int(fast_feat.shape[4]))
        dt = compute_segment_dt(frame_timestamps, frame_dt, num_frames, self.timestamp_scale).to(fast_logits.device)

        delta_mag_values: list[float] = []
        for k in range(rollout_start_step, num_frames - 1):
            transform = compute_transform_prev_to_curr(frame_ego2global[k], frame_ego2global[k + 1])
            h_new = self._advance_no_warp(h_dense, fast_feat[k + 1], dt[k], transform, spatial_shape_xyz)
            delta_mag_values.append((h_new - h_dense).abs().mean().item())
            h_dense = h_new

        logits_delta = self._decode_dense_state(h_dense)
        logits = logits_delta + fast_logits[-1] if self.use_fast_residual else logits_delta
        avg_delta = sum(delta_mag_values) / max(len(delta_mag_values), 1)
        return {
            "aligned": logits.float(),
            "diagnostics": {"delta_scene_abs_mean": torch.tensor(avg_delta, device=fast_logits.device)},
        }

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
            return {
                "step_logits": slow_logits.unsqueeze(0).float(),
                "step_indices": torch.tensor([num_frames - 1], device=fast_logits.device, dtype=torch.long),
                "diagnostics": {"delta_scene_abs_mean": torch.tensor(0.0, device=fast_logits.device)},
            }

        rollout_steps = (num_frames - 1) - rollout_start_step
        if max_step_index is not None:
            rollout_steps = min(rollout_steps, max(int(max_step_index), 0))

        fast_feat = self._encode_fast(fast_logits)
        h_dense = self._encode_slow(slow_logits)
        spatial_shape_xyz = (int(fast_feat.shape[2]), int(fast_feat.shape[3]), int(fast_feat.shape[4]))
        dt = compute_segment_dt(frame_timestamps, frame_dt, num_frames, self.timestamp_scale).to(fast_logits.device)

        step_logits_list: list[torch.Tensor] = []
        delta_mag_values: list[float] = []
        fast_kl_accum: torch.Tensor | None = None
        fast_kl_step_count = 0
        compute_fast_kl = self._fast_kl_active and self.use_fast_residual
        for k_off in range(rollout_steps):
            k = rollout_start_step + k_off
            transform = compute_transform_prev_to_curr(frame_ego2global[k], frame_ego2global[k + 1])
            h_new = self._advance_no_warp(h_dense, fast_feat[k + 1], dt[k], transform, spatial_shape_xyz)
            delta_mag_values.append((h_new - h_dense).abs().mean().item())
            h_dense = h_new

            logits_delta = self._decode_dense_state(h_dense)
            logits_now = logits_delta + fast_logits[k + 1] if self.use_fast_residual else logits_delta
            step_logits_list.append(logits_now.float())
            if compute_fast_kl:
                kl_step = self._compute_fast_kl_step(fast_logits[k + 1].detach(), logits_now)
                fast_kl_accum = kl_step if fast_kl_accum is None else fast_kl_accum + kl_step
                fast_kl_step_count += 1

        step_logits = torch.stack(step_logits_list, dim=0) if step_logits_list else fast_logits.new_zeros(
            (0, self.num_classes, fast_logits.shape[2], fast_logits.shape[3], fast_logits.shape[4])
        )
        step_indices = torch.arange(
            rollout_start_step + 1,
            rollout_start_step + 1 + rollout_steps,
            device=fast_logits.device,
            dtype=torch.long,
        )
        avg_delta = sum(delta_mag_values) / max(len(delta_mag_values), 1)
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "step_logits": step_logits,
            "step_indices": step_indices,
            "diagnostics": {"delta_scene_abs_mean": torch.tensor(avg_delta, device=fast_logits.device)},
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
            zero = torch.zeros((1,), device=fast_logits.device, dtype=torch.float32)
            return {
                "step_logits": step_logits,
                "step_time_ms": zero,
                "step_warp_ms": zero,
                "step_solver_ms": zero,
                "step_decode_ms": zero,
                "step_indices": torch.tensor([num_frames - 1], device=fast_logits.device, dtype=torch.long),
                "diagnostics": {"delta_scene_abs_mean": torch.tensor(0.0, device=fast_logits.device)},
            }

        fast_feat = self._encode_fast(fast_logits)
        h_dense = self._encode_slow(slow_logits)
        spatial_shape_xyz = (int(fast_feat.shape[2]), int(fast_feat.shape[3]), int(fast_feat.shape[4]))
        dt = compute_segment_dt(frame_timestamps, frame_dt, num_frames, self.timestamp_scale).to(fast_logits.device)

        use_cuda_timing = fast_logits.is_cuda
        step_logits_list: list[torch.Tensor] = []
        motion_ms_values: list[float] = []
        attn_ms_values: list[float] = []
        decode_ms_values: list[float] = []
        events: list[tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = []
        delta_mag_values: list[float] = []

        for k in range(rollout_start_step, num_frames - 1):
            if use_cuda_timing:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev2 = torch.cuda.Event(enable_timing=True)
                ev3 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            else:
                t0 = time.perf_counter()

            transform = compute_transform_prev_to_curr(frame_ego2global[k], frame_ego2global[k + 1])
            dt_ch = self._make_dt_channel(dt[k], spatial_shape_xyz, h_dense.device, h_dense.dtype)
            motion = self._make_motion_field(transform, spatial_shape_xyz, h_dense.dtype)
            if use_cuda_timing:
                ev1.record()
            else:
                t1 = time.perf_counter()

            h_new = self.fusion(h_dense, fast_feat[k + 1], dt_ch, motion)
            delta_mag_values.append((h_new - h_dense).abs().mean().item())
            h_dense = h_new
            if use_cuda_timing:
                ev2.record()
            else:
                t2 = time.perf_counter()

            logits_delta = self._decode_dense_state(h_dense)
            logits_now = logits_delta + fast_logits[k + 1] if self.use_fast_residual else logits_delta
            step_logits_list.append(logits_now.float())
            if use_cuda_timing:
                ev3.record()
                events.append((ev0, ev1, ev2, ev3))
            else:
                t3 = time.perf_counter()
                motion_ms_values.append((t1 - t0) * 1000.0)
                attn_ms_values.append((t2 - t1) * 1000.0)
                decode_ms_values.append((t3 - t2) * 1000.0)

        if use_cuda_timing:
            torch.cuda.synchronize(device=fast_logits.device)
            motion_ms_values = [a.elapsed_time(b) for a, b, _, _ in events]
            attn_ms_values = [b.elapsed_time(c) for _, b, c, _ in events]
            decode_ms_values = [c.elapsed_time(d) for _, _, c, d in events]
        step_time_values = [a + b + c for a, b, c in zip(motion_ms_values, attn_ms_values, decode_ms_values)]

        step_logits = torch.stack(step_logits_list, dim=0) if step_logits_list else fast_logits.new_zeros(
            (0, self.num_classes, fast_logits.shape[2], fast_logits.shape[3], fast_logits.shape[4])
        )
        avg_delta = sum(delta_mag_values) / max(len(delta_mag_values), 1)
        return {
            "step_logits": step_logits,
            "step_time_ms": torch.tensor(step_time_values, device=fast_logits.device, dtype=torch.float32),
            # 复用上游字段名；这里表示 motion-field 构造耗时，不包含显式 warp。
            "step_warp_ms": torch.tensor(motion_ms_values, device=fast_logits.device, dtype=torch.float32),
            "step_solver_ms": torch.tensor(attn_ms_values, device=fast_logits.device, dtype=torch.float32),
            "step_decode_ms": torch.tensor(decode_ms_values, device=fast_logits.device, dtype=torch.float32),
            "step_indices": torch.arange(rollout_start_step + 1, num_frames, device=fast_logits.device, dtype=torch.long),
            "diagnostics": {"delta_scene_abs_mean": torch.tensor(avg_delta, device=fast_logits.device)},
        }
