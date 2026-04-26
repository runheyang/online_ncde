"""Recurrent Warp-Fusion Aligner (RWFA) baseline。

可学习的对比方法：每一步对运行隐状态做 ego-warp，然后用一个 FusionNet
融合当前帧 fast 特征与 Δt 标量通道，得到 t+1 时刻的隐状态；解码后与
当前 fast logits 残差相加。

设计目标：除把 NCDE 的连续时间动力学换成普通递归 CNN 融合外，其余结构
（双独立 encoder、全分辨率、warp 算子、Δt 条件、fast 残差、多步监督接口）
全部对齐 OnlineNcdeAligner —— 算力/参数量在 ±少量误差内可比，便于消融
归因 NCDE formulation 本身的贡献。

forward / forward_stepwise_eval 接口签名与 OnlineNcdeAligner 一致，
方便复用 trainer 与 evaluation 脚本。
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
from online_ncde.utils.nn import resolve_group_norm_groups


class _ResidualDilatedBlock(nn.Module):
    """Conv3d(dilation=d) + GN + SiLU + Residual。结构与 FuncG 内部块一致。"""

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
    """3D (shifted) window multi-head self-attention。

    把 (X, Y, Z) 切成不重叠的 (Wx, Wy, Wz) 窗口，窗口内做 MHSA；shift_size > 0 时
    先做 cyclic shift，attn 后再 shift 回去 —— 与 Swin 一致。窗口边界 attn mask
    暂未实现（shift 引入的"环绕"语义对体素任务影响微弱，且 mask 会显著拖慢前向；
    若实证表明影响明显，可后续补 mask）。

    输入/输出形状均为 (B, C, X, Y, Z)。要求 X/Wx, Y/Wy, Z/Wz 整除。
    """

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
        # x: (B, C, X, Y, Z)
        B, C, X, Y, Z = x.shape
        Wx, Wy, Wz = self.window_size
        Sx, Sy, Sz = self.shift_size

        if (X % Wx) or (Y % Wy) or (Z % Wz):
            raise ValueError(
                f"空间形状 ({X},{Y},{Z}) 必须能被 window ({Wx},{Wy},{Wz}) 整除"
            )

        # cyclic shift（提供跨窗口交互）
        if Sx or Sy or Sz:
            x = torch.roll(x, shifts=(-Sx, -Sy, -Sz), dims=(2, 3, 4))

        nWx, nWy, nWz = X // Wx, Y // Wy, Z // Wz
        # (B, C, X, Y, Z) -> (B, nWx, Wx, nWy, Wy, nWz, Wz, C) -> (B*nW, N, C)
        h = x.view(B, C, nWx, Wx, nWy, Wy, nWz, Wz)
        h = h.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        N_tok = Wx * Wy * Wz
        h = h.view(B * nWx * nWy * nWz, N_tok, C)

        # MHSA：用 F.scaled_dot_product_attention 走 SDPA backend（FlashAttention/MEA），
        # fp16 下显著加速且数值稳定，比展开式 matmul+softmax 更高效。
        qkv = self.qkv(h).view(-1, N_tok, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, Bw, heads, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)  # (Bw, heads, N, hd)
        out = out.transpose(1, 2).reshape(-1, N_tok, C)
        out = self.proj(out)

        # 还原窗口
        out = out.view(B, nWx, nWy, nWz, Wx, Wy, Wz, C)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        out = out.view(B, C, X, Y, Z)

        # reverse cyclic shift
        if Sx or Sy or Sz:
            out = torch.roll(out, shifts=(Sx, Sy, Sz), dims=(2, 3, 4))
        return out


class _WindowAttentionBlock(nn.Module):
    """Swin 风格 block：Pre-GN + W-MSA + Residual，再 Pre-GN + 1x1 FFN + Residual。"""

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
    """RWFA-Attn 主干：dilated conv → W-MSA → SW-MSA → dilated conv。

    设计取舍：头尾保留 dilation conv 提供局部归纳偏置 + 跨窗口扩散，中间一对
    Swin 风格 block（unshifted + shifted）做窗口内长程交互。窗口默认 (8, 8, 4)，
    在 200×200×16 上整除（25×25×4=2500 windows，256 tokens/window）；sanity
    测试用的 40×40×8 也整除（5×5×2 windows）。

    参数量与 FusionNet 相近：head/stem 与 FusionNet 一致；body 中 attn pair 的
    Q/K/V/proj+FFN 参数量比 dilated conv 略低（attn ≈ 3·d² + d²，conv ≈ 27·d²），
    整体在 ±10% 内可比。
    """

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

        # stem 与 FusionNet 一致
        self.stem_conv = nn.Conv3d(in_channels, inner_dim, kernel_size=1, bias=False)
        self.stem_gn = nn.GroupNorm(num_groups=groups, num_channels=inner_dim)
        self.stem_act = nn.SiLU(inplace=True)

        # Z 轴不做 cyclic shift：16 层高度下 shift_z=2 会让顶/底层直接互相 attend，
        # 语义错误明显；XY 上 shift 的"环绕"对体素任务影响微弱可接受。
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
        x = torch.cat([h_warp, fast_prev_adv, fast_curr, dt_channel], dim=0).unsqueeze(0)
        x = self.stem_act(self.stem_gn(self.stem_conv(x)))
        for block in self.body:
            x = block(x)
        x = self.head_conv(x)
        return x.squeeze(0)


def _compute_m_occ(fast_logits: torch.Tensor, free_index: int) -> torch.Tensor:
    """conf 权重：m_occ = max_{c != free}(logit[c]) - logit[free]。与 NCDE 同口径。"""
    masked = fast_logits.clone()
    masked.narrow(-4, free_index, 1).fill_(float("-inf"))
    max_non_free = masked.amax(dim=-4, keepdim=True)
    free_logit = fast_logits.narrow(-4, free_index, 1)
    return max_non_free - free_logit


class FusionNet(nn.Module):
    """与 FuncG 同构的融合主干：1x1 stem + N 层膨胀残差 3x3x3 + 1x1 head。

    输入由 (h_warp, fast_feat, dt_channel) 沿 channel 维拼接而成。不带 tanh 头部，
    因为输出是状态本体而非 dh/dt（避免对运行特征范围的隐式截断）。
    """

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
        h_warp: torch.Tensor,         # (C_h, X, Y, Z)
        fast_prev_adv: torch.Tensor,  # (C_f, X, Y, Z)
        fast_curr: torch.Tensor,      # (C_f, X, Y, Z)
        dt_channel: torch.Tensor,     # (1, X, Y, Z)
    ) -> torch.Tensor:
        x = torch.cat([h_warp, fast_prev_adv, fast_curr, dt_channel], dim=0).unsqueeze(0)
        x = self.stem_act(self.stem_gn(self.stem_conv(x)))
        for block in self.body:
            x = block(x)
        x = self.head_conv(x)
        return x.squeeze(0)


class RecurrentWarpFusionAligner(nn.Module):
    """RWFA baseline：与 OnlineNcdeAligner 同接口的可学习对比方法。"""

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
        fusion_kind: str = "conv",
        fusion_inner_dim: int = 32,
        fusion_body_dilations: Tuple[int, ...] = (1, 2, 3),
        fusion_gn_groups: int = 8,
        # attn 分支专用配置（fusion_kind="attn" 时生效）
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

        # 双独立 encoder：与 NCDE 一致
        self.fast_encoder = DenseEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        self.slow_encoder = DenseEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        # FusionNet：输入 = h_warp(C_h) + fast_prev_adv(C_f) + fast_curr(C_f) + dt(1)
        # 输入信号与 NCDE 对齐：每步多 warp 一次 fast_feat[k] 提供 ego-aligned 的上一帧
        # fast 副本，让 fusion 自己学是否需要做差分 —— 把"用增量做动力学"留作 NCDE 独有特性。
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

        # Trainer 根据 lambda_fast_kl 注入；False 时 forward 跳过 KL 计算（与 NCDE 同协议）。
        self._fast_kl_active: bool = False

    # ----------------- 编码 / 解码 -----------------

    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        return self.fast_encoder(fast_logits)

    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        return self.slow_encoder(slow_logits.unsqueeze(0))[0]

    def _decode_dense_state(self, h_dense: torch.Tensor) -> torch.Tensor:
        h_tensor = h_dense.unsqueeze(0)
        h_tensor = h_tensor.permute(0, 1, 4, 3, 2).contiguous()  # (1, C, Z, Y, X)
        out_dense = self.decoder(h_tensor)
        return out_dense.permute(0, 1, 4, 3, 2).contiguous()[0]

    def _compute_fast_kl_step(
        self,
        fast_logits_t: torch.Tensor,
        aligned_logits: torch.Tensor,
    ) -> torch.Tensor:
        """conf-weighted KL(aligned || fast)：与 NCDE 实现一致。

        权重 m_occ.clamp(min=0) —— 仅在 fast 自信判占用处把 aligned 拉回 fast，
        保留 fast 自信判空处 aligner 自由填补。"""
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
        """把 0-dim dt 张量广播成 (1, X, Y, Z) 通道。

        关键：用 .expand 而非 .fill_(float(...))，避免把 GPU 标量同步回 CPU
        触发 host sync —— 否则 rollout 循环里每步都会 stall 一次。
        """
        x_size, y_size, z_size = spatial_shape_xyz
        return (
            dt_value.to(device=device, dtype=dtype)
            .reshape(1, 1, 1, 1)
            .expand(1, x_size, y_size, z_size)
        )

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

            # ---- warp 段（h + fast_prev）----
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

            # ---- fusion 段（对应 NCDE 的 solver 段）----
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

            # ---- decode 段 ----
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
