from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from evoocc.data.ego_warp_list import (
    backward_warp_dense_trilinear,
    build_sampling_grid,
    compute_transform_prev_to_curr,
)
from evoocc.data.time_series import compute_segment_dt
from evoocc.models.decoder import DenseDecoder
from evoocc.models.encoder import DenseEncoder
from evoocc.utils.nn import resolve_group_norm_groups


class _ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, gn_groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.gn(self.conv(x)))


class _WindowAttention3D(nn.Module):
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
        self.scale = self.head_dim ** -0.5
        self.window_size = tuple(int(v) for v in window_size)
        self.shift_size = tuple(int(v) for v in shift_size)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # (B, C, X, Y, Z) -> (B, nWx, Wx, nWy, Wy, nWz, Wz, C) -> (B*nW, N, C)
        h = x.view(B, C, nWx, Wx, nWy, Wy, nWz, Wz)
        h = h.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        N_tok = Wx * Wy * Wz
        h = h.view(B * nWx * nWy * nWz, N_tok, C)

        qkv = self.qkv(h).view(-1, N_tok, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(-1, N_tok, C)
        out = self.proj(out)

        out = out.view(B, nWx, nWy, nWz, Wx, Wy, Wz, C)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        out = out.view(B, C, X, Y, Z)

        if Sx or Sy or Sz:
            out = torch.roll(out, shifts=(Sx, Sy, Sz), dims=(2, 3, 4))
        return out


class _WindowAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int],
        gn_groups: int,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=gn_groups, num_channels=dim)
        self.attn = _WindowAttention3D(
            dim=dim, num_heads=num_heads,
            window_size=window_size, shift_size=shift_size,
        )
        hidden = max(int(round(dim * float(mlp_ratio))), dim)
        self.norm2 = nn.GroupNorm(num_groups=gn_groups, num_channels=dim)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FusionAttnNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inner_dim: int = 32,
        num_heads: int = 4,
        window_size: Tuple[int, int, int] = (8, 8, 4),
        head_dilations: Tuple[int, ...] = (1, 2),
        gn_groups: int = 8,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if len(head_dilations) != 2:
            raise ValueError(
                f"head_dilations 需恰好 2 个值（首尾各一），当前: {head_dilations}"
            )
        groups = resolve_group_norm_groups(num_channels=inner_dim, preferred_groups=gn_groups)

        self.stem_conv = nn.Conv3d(in_channels, inner_dim, kernel_size=1, bias=False)
        self.stem_gn = nn.GroupNorm(num_groups=groups, num_channels=inner_dim)
        self.stem_act = nn.SiLU(inplace=True)

        # No shift along Z: with 16 layers it would wrap top and bottom layers together.
        shift = (window_size[0] // 2, window_size[1] // 2, 0)
        self.body = nn.ModuleList(
            [
                _ResidualDilatedBlock(
                    channels=inner_dim, dilation=int(head_dilations[0]), gn_groups=groups
                ),
                _WindowAttentionBlock(
                    dim=inner_dim, num_heads=num_heads,
                    window_size=window_size, shift_size=(0, 0, 0),
                    gn_groups=groups, mlp_ratio=mlp_ratio,
                ),
                _WindowAttentionBlock(
                    dim=inner_dim, num_heads=num_heads,
                    window_size=window_size, shift_size=shift,
                    gn_groups=groups, mlp_ratio=mlp_ratio,
                ),
                _ResidualDilatedBlock(
                    channels=inner_dim, dilation=int(head_dilations[1]), gn_groups=groups
                ),
            ]
        )
        self.head_conv = nn.Conv3d(inner_dim, out_channels, kernel_size=1, bias=True)

    def forward(
        self,
        h_warp: torch.Tensor,
        fast_prev_adv: torch.Tensor,
        fast_curr: torch.Tensor,
        dt_channel: torch.Tensor,
    ) -> torch.Tensor:
        # Input channels: h_warp(C_h) + fast_prev_adv(C_f) + fast_curr(C_f) + dt(1).
        x = torch.cat([h_warp, fast_prev_adv, fast_curr, dt_channel], dim=0).unsqueeze(0)
        x = self.stem_act(self.stem_gn(self.stem_conv(x)))
        for block in self.body:
            x = block(x)
        x = self.head_conv(x)
        return x.squeeze(0)


def _compute_m_occ(fast_logits: torch.Tensor, free_index: int) -> torch.Tensor:
    """m_occ = max_{c != free} logit[c] - logit[free]."""
    masked = fast_logits.clone()
    masked.narrow(-4, free_index, 1).fill_(float("-inf"))
    max_non_free = masked.amax(dim=-4, keepdim=True)
    free_logit = fast_logits.narrow(-4, free_index, 1)
    return max_non_free - free_logit


class FusionNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inner_dim: int = 32,
        body_dilations: Tuple[int, ...] = (1, 2, 3),
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        if not body_dilations:
            raise ValueError("body_dilations 不能为空，至少需要一个 dilation。")
        if any(int(d) <= 0 for d in body_dilations):
            raise ValueError(f"body_dilations 中每个 dilation 必须 > 0，当前: {body_dilations}")

        groups = resolve_group_norm_groups(num_channels=inner_dim, preferred_groups=gn_groups)
        self.stem_conv = nn.Conv3d(in_channels, inner_dim, kernel_size=1, bias=False)
        self.stem_gn = nn.GroupNorm(num_groups=groups, num_channels=inner_dim)
        self.stem_act = nn.SiLU(inplace=True)
        self.body = nn.ModuleList(
            [
                _ResidualDilatedBlock(channels=inner_dim, dilation=int(d), gn_groups=groups)
                for d in body_dilations
            ]
        )
        self.head_conv = nn.Conv3d(inner_dim, out_channels, kernel_size=1, bias=True)

    def forward(
        self,
        h_warp: torch.Tensor,
        fast_prev_adv: torch.Tensor,
        fast_curr: torch.Tensor,
        dt_channel: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([h_warp, fast_prev_adv, fast_curr, dt_channel], dim=0).unsqueeze(0)
        x = self.stem_act(self.stem_gn(self.stem_conv(x)))
        for block in self.body:
            x = block(x)
        x = self.head_conv(x)
        return x.squeeze(0)


class RecurrentWarpFusionAligner(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        hidden_dim: int,
        encoder_in_channels: int,
        free_index: int,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        decoder_init_scale: Optional[float] = 0.0,
        use_fast_residual: bool = True,
        fusion_kind: str = "conv",
        fusion_inner_dim: int = 32,
        fusion_body_dilations: Tuple[int, ...] = (1, 2, 3),
        fusion_gn_groups: int = 8,
        fusion_attn_num_heads: int = 4,
        fusion_attn_window_size: Tuple[int, int, int] = (8, 8, 4),
        fusion_attn_head_dilations: Tuple[int, ...] = (1, 2),
        fusion_attn_mlp_ratio: float = 2.0,
        timestamp_scale: float = 1.0e-6,
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

        self.fast_encoder = DenseEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        self.slow_encoder = DenseEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        fusion_in_channels = hidden_dim + 2 * feat_dim + 1
        fusion_kind_lower = str(fusion_kind).lower()
        self.fusion_kind = fusion_kind_lower
        if fusion_kind_lower == "conv":
            self.fusion: nn.Module = FusionNet(
                in_channels=fusion_in_channels,
                out_channels=hidden_dim,
                inner_dim=fusion_inner_dim,
                body_dilations=tuple(fusion_body_dilations),
                gn_groups=fusion_gn_groups,
            )
        elif fusion_kind_lower == "attn":
            self.fusion = FusionAttnNet(
                in_channels=fusion_in_channels,
                out_channels=hidden_dim,
                inner_dim=fusion_inner_dim,
                num_heads=int(fusion_attn_num_heads),
                window_size=tuple(fusion_attn_window_size),
                head_dilations=tuple(fusion_attn_head_dilations),
                gn_groups=fusion_gn_groups,
                mlp_ratio=float(fusion_attn_mlp_ratio),
            )
        else:
            raise ValueError(
                f"未知的 fusion_kind: {fusion_kind!r}，可选: 'conv', 'attn'"
            )
        self.decoder = DenseDecoder(
            in_channels=hidden_dim,
            out_channels=num_classes,
            init_scale=decoder_init_scale,
        )

        self._fast_kl_active: bool = False

    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        return self.fast_encoder(fast_logits)

    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        return self.slow_encoder(slow_logits.unsqueeze(0))[0]

    def _decode_dense_state(self, h_dense: torch.Tensor) -> torch.Tensor:
        h_tensor = h_dense.unsqueeze(0)
        h_tensor = h_tensor.permute(0, 1, 4, 3, 2).contiguous()
        out_dense = self.decoder(h_tensor)
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

    @staticmethod
    def _make_dt_channel(dt_value: torch.Tensor, spatial_shape_xyz: Tuple[int, int, int],
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # expand (not fill_) avoids a GPU->CPU sync per rollout step.
        x_size, y_size, z_size = spatial_shape_xyz
        return (
            dt_value.to(device=device, dtype=dtype)
            .reshape(1, 1, 1, 1)
            .expand(1, x_size, y_size, z_size)
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

        h_dense = slow_feat
        delta_mag_values: list[float] = []

        for k in range(rollout_start_step, num_frames - 1):
            transform = compute_transform_prev_to_curr(
                pose_prev_ego2global=frame_ego2global[k],
                pose_curr_ego2global=frame_ego2global[k + 1],
            )
            grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)

            h_warp = backward_warp_dense_trilinear(
                dense_prev_feat=h_dense,
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
            dt_ch = self._make_dt_channel(
                dt[k], spatial_shape_xyz, fast_logits.device, h_warp.dtype
            )

            h_new = self.fusion(
                h_warp=h_warp, fast_prev_adv=f_prev_adv, fast_curr=f_t, dt_channel=dt_ch
            )
            delta_mag_values.append((h_new - h_warp).abs().mean().item())
            h_dense = h_new

        logits_delta = self._decode_dense_state(h_dense)
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
        pc_range_6 = cast(Tuple[float, float, float, float, float, float], self.pc_range)
        voxel_size_3 = cast(Tuple[float, float, float], self.voxel_size)

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)

        h_dense = slow_feat
        delta_mag_values: list[float] = []
        step_logits_list: list[torch.Tensor] = []
        fast_kl_accum: torch.Tensor | None = None
        fast_kl_step_count = 0
        compute_fast_kl = self._fast_kl_active and self.use_fast_residual

        for k_off in range(rollout_steps):
            k = rollout_start_step + k_off
            transform = compute_transform_prev_to_curr(
                pose_prev_ego2global=frame_ego2global[k],
                pose_curr_ego2global=frame_ego2global[k + 1],
            )
            grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)

            h_warp = backward_warp_dense_trilinear(
                dense_prev_feat=h_dense,
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
            dt_ch = self._make_dt_channel(
                dt[k], spatial_shape_xyz, fast_logits.device, h_warp.dtype
            )

            h_new = self.fusion(
                h_warp=h_warp, fast_prev_adv=f_prev_adv, fast_curr=f_t, dt_channel=dt_ch
            )
            delta_mag_values.append((h_new - h_warp).abs().mean().item())
            h_dense = h_new

            logits_delta = self._decode_dense_state(h_dense)
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
        pc_range_6 = cast(Tuple[float, float, float, float, float, float], self.pc_range)
        voxel_size_3 = cast(Tuple[float, float, float], self.voxel_size)

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)

        h_dense = slow_feat
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

            transform = compute_transform_prev_to_curr(
                pose_prev_ego2global=frame_ego2global[k],
                pose_curr_ego2global=frame_ego2global[k + 1],
            )
            grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)
            h_warp = backward_warp_dense_trilinear(
                dense_prev_feat=h_dense,
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
            dt_ch = self._make_dt_channel(
                dt[k], spatial_shape_xyz, fast_logits.device, h_warp.dtype
            )
            h_new = self.fusion(
                h_warp=h_warp, fast_prev_adv=f_prev_adv, fast_curr=f_t, dt_channel=dt_ch
            )
            delta_mag_values.append((h_new - h_warp).abs().mean().item())
            h_dense = h_new

            if use_cuda_timing:
                ev_t2.record()
            else:
                tp2 = time.perf_counter()

            logits_delta = self._decode_dense_state(h_dense)
            if self.use_fast_residual:
                logits_now = logits_delta + fast_logits[k + 1]
            else:
                logits_now = logits_delta
            step_logits_list.append(logits_now.float())

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
