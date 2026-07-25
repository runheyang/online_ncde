"""Learned direct window-attention baseline 的融合模块。"""

from __future__ import annotations

from itertools import product
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from evoocc.baselines.learned_direct_fusion.modules import (
    _ResidualDilatedBlock,
)
from evoocc.utils.nn import resolve_group_norm_groups


class _WindowCrossAttention3D(nn.Module):
    """带边界 mask 的3D窗口交叉注意力。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} 必须能被 num_heads={num_heads} 整除")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.window_size = tuple(int(value) for value in window_size)
        self.shift_size = tuple(int(value) for value in shift_size)
        if any(value <= 0 for value in self.window_size):
            raise ValueError(f"window_size 必须为正整数，当前 {self.window_size}")
        if any(
            shift < 0 or shift >= window
            for shift, window in zip(self.shift_size, self.window_size)
        ):
            raise ValueError(
                "shift_size 必须位于 [0, window_size)，"
                f"当前 shift={self.shift_size}, window={self.window_size}"
            )

        self.q = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)
        self.k = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)
        self.v = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)
        self.proj = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)

    def _to_windows(
        self,
        tensor: torch.Tensor,
        *,
        apply_shift: bool,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
        batch, channels, x_size, y_size, z_size = tensor.shape
        wx, wy, wz = self.window_size
        if x_size % wx or y_size % wy or z_size % wz:
            raise ValueError(
                f"空间形状 {(x_size, y_size, z_size)} 必须能被窗口 "
                f"{self.window_size} 整除"
            )
        if apply_shift and any(self.shift_size):
            tensor = torch.roll(
                tensor,
                shifts=tuple(-value for value in self.shift_size),
                dims=(2, 3, 4),
            )

        nx, ny, nz = x_size // wx, y_size // wy, z_size // wz
        tensor = tensor.view(
            batch,
            channels,
            nx,
            wx,
            ny,
            wy,
            nz,
            wz,
        )
        tensor = tensor.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        windows = tensor.view(batch * nx * ny * nz, wx * wy * wz, channels)
        return windows, (batch, x_size, y_size, z_size, channels)

    def _from_windows(
        self,
        windows: torch.Tensor,
        meta: tuple[int, int, int, int, int],
    ) -> torch.Tensor:
        batch, x_size, y_size, z_size, channels = meta
        wx, wy, wz = self.window_size
        nx, ny, nz = x_size // wx, y_size // wy, z_size // wz
        tensor = windows.view(
            batch,
            nx,
            ny,
            nz,
            wx,
            wy,
            wz,
            channels,
        )
        tensor = tensor.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        tensor = tensor.view(batch, channels, x_size, y_size, z_size)
        if any(self.shift_size):
            tensor = torch.roll(
                tensor,
                shifts=self.shift_size,
                dims=(2, 3, 4),
            )
        return tensor

    @staticmethod
    def _axis_slices(window: int, shift: int) -> tuple[slice, ...]:
        if shift == 0:
            return (slice(None),)
        return (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )

    def _build_shift_mask(
        self,
        spatial_shape: tuple[int, int, int],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not any(self.shift_size):
            return None

        x_size, y_size, z_size = spatial_shape
        region = torch.zeros(
            (1, 1, x_size, y_size, z_size),
            device=device,
            dtype=torch.int64,
        )
        slices = [
            self._axis_slices(window, shift)
            for window, shift in zip(self.window_size, self.shift_size)
        ]
        region_id = 0
        for x_slice, y_slice, z_slice in product(*slices):
            region[:, :, x_slice, y_slice, z_slice] = region_id
            region_id += 1

        region_windows, _ = self._to_windows(region, apply_shift=False)
        region_windows = region_windows.squeeze(-1)
        different_region = (
            region_windows.unsqueeze(1) != region_windows.unsqueeze(2)
        )
        mask = torch.zeros(
            different_region.shape,
            device=device,
            dtype=dtype,
        ).masked_fill(different_region, float("-inf"))
        return mask.unsqueeze(1).repeat(batch, 1, 1, 1)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape != key_value.shape:
            raise ValueError(
                "query/key_value 形状必须一致，"
                f"当前 {tuple(query.shape)} vs {tuple(key_value.shape)}"
            )

        q_windows, meta = self._to_windows(
            self.q(query),
            apply_shift=True,
        )
        k_windows, _ = self._to_windows(
            self.k(key_value),
            apply_shift=True,
        )
        v_windows, _ = self._to_windows(
            self.v(key_value),
            apply_shift=True,
        )
        batch, x_size, y_size, z_size, _ = meta
        tokens = q_windows.shape[1]

        q_windows = q_windows.view(
            -1,
            tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        k_windows = k_windows.view(
            -1,
            tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v_windows = v_windows.view(
            -1,
            tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        attention_mask = self._build_shift_mask(
            spatial_shape=(x_size, y_size, z_size),
            batch=batch,
            device=query.device,
            dtype=q_windows.dtype,
        )
        output = F.scaled_dot_product_attention(
            q_windows,
            k_windows,
            v_windows,
            attn_mask=attention_mask,
        )
        output = output.transpose(1, 2).reshape(-1, tokens, self.dim)
        output = self._from_windows(output, meta)
        return self.proj(output)


class _CrossAttentionBlock(nn.Module):
    """Pre-GN窗口交叉注意力与1×1 FFN。"""

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
        groups = resolve_group_norm_groups(dim, gn_groups)
        hidden_dim = max(int(round(dim * float(mlp_ratio))), dim)
        self.norm_q = nn.GroupNorm(groups, dim)
        self.norm_kv = nn.GroupNorm(groups, dim)
        self.attention = _WindowCrossAttention3D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
        )
        self.norm_ffn = nn.GroupNorm(groups, dim)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_dim, dim, kernel_size=1, bias=True),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        hidden = query + self.attention(
            self.norm_q(query),
            self.norm_kv(key_value),
        )
        return hidden + self.ffn(self.norm_ffn(hidden))


class DirectWindowAttentionNet(nn.Module):
    """以current fast为query、warped slow为key/value的融合主干。"""

    def __init__(
        self,
        feature_dim: int = 288,
        inner_dim: int = 96,
        num_heads: int = 8,
        window_size: Tuple[int, int, int] = (5, 5, 4),
        local_dilations: Tuple[int, int] = (1, 2),
        gn_groups: int = 8,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if len(local_dilations) != 2 or any(
            int(value) <= 0 for value in local_dilations
        ):
            raise ValueError(
                "local_dilations 必须包含两个正整数，"
                f"当前 {local_dilations}"
            )
        groups = resolve_group_norm_groups(inner_dim, gn_groups)
        self.window_size = tuple(int(value) for value in window_size)
        self.q_stem = nn.Sequential(
            nn.Conv3d(feature_dim, inner_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, inner_dim),
            nn.SiLU(inplace=True),
        )
        self.kv_stem = nn.Sequential(
            nn.Conv3d(feature_dim, inner_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, inner_dim),
            nn.SiLU(inplace=True),
        )
        shift = (
            self.window_size[0] // 2,
            self.window_size[1] // 2,
            0,
        )
        self.block0 = _CrossAttentionBlock(
            dim=inner_dim,
            num_heads=num_heads,
            window_size=self.window_size,
            shift_size=(0, 0, 0),
            gn_groups=gn_groups,
            mlp_ratio=mlp_ratio,
        )
        self.local0 = _ResidualDilatedBlock(
            channels=inner_dim,
            dilation=int(local_dilations[0]),
            gn_groups=gn_groups,
        )
        self.block1 = _CrossAttentionBlock(
            dim=inner_dim,
            num_heads=num_heads,
            window_size=self.window_size,
            shift_size=shift,
            gn_groups=gn_groups,
            mlp_ratio=mlp_ratio,
        )
        self.local1 = _ResidualDilatedBlock(
            channels=inner_dim,
            dilation=int(local_dilations[1]),
            gn_groups=gn_groups,
        )
        self.head = nn.Conv3d(inner_dim, feature_dim, kernel_size=1, bias=True)

    def forward(
        self,
        warped_slow: torch.Tensor,
        current_fast: torch.Tensor,
    ) -> torch.Tensor:
        if warped_slow.shape != current_fast.shape:
            raise ValueError(
                "slow/fast feature 形状必须一致，"
                f"当前 {tuple(warped_slow.shape)} vs "
                f"{tuple(current_fast.shape)}"
            )
        if warped_slow.dim() != 5:
            raise ValueError(
                "attention 输入必须为5D (B,C,X,Y,Z)，"
                f"当前 {tuple(warped_slow.shape)}"
            )

        query = self.q_stem(current_fast.contiguous())
        key_value = self.kv_stem(warped_slow.contiguous())
        hidden = self.local0(self.block0(query, key_value))
        hidden = self.local1(self.block1(hidden, key_value))
        return self.head(hidden)
