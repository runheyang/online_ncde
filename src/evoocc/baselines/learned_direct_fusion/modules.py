from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from evoocc.models.decoder import DenseDecoder
from evoocc.utils.nn import resolve_group_norm_groups


class XYDownsampleEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        xy_downsample_factor: int = 2,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        factor = int(xy_downsample_factor)
        if factor <= 0:
            raise ValueError(f"xy_downsample_factor 必须为正整数，当前 {factor}")
        self.xy_downsample_factor = factor
        groups = resolve_group_norm_groups(out_channels, gn_groups)
        # Average pooling keeps low-res voxel centers aligned with the physical grid.
        self.pool = nn.AvgPool3d(
            kernel_size=(1, factor, factor),
            stride=(1, factor, factor),
        )
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 5:
            raise ValueError(
                f"encoder 输入必须为 5D (B,C,X,Y,Z)，当前 {tuple(logits.shape)}"
            )
        if (
            logits.shape[2] % self.xy_downsample_factor
            or logits.shape[3] % self.xy_downsample_factor
        ):
            raise ValueError(
                "XY 尺寸必须能被下采样倍数整除，"
                f"当前 X={logits.shape[2]}, Y={logits.shape[3]}"
            )
        hidden = logits.permute(0, 1, 4, 3, 2).contiguous()
        hidden = self.pool(hidden)
        hidden = self.act(self.norm(self.conv(hidden)))
        return hidden.permute(0, 1, 4, 3, 2).contiguous()


class _ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, gn_groups: int) -> None:
        super().__init__()
        groups = resolve_group_norm_groups(channels, gn_groups)
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.act(self.norm(self.conv(hidden)))


class DirectFusionNet(nn.Module):
    def __init__(
        self,
        feature_dim: int = 288,
        inner_dim: int = 104,
        body_dilations: Sequence[int] = (1, 3, 5),
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        dilations = tuple(int(value) for value in body_dilations)
        if not dilations or any(value <= 0 for value in dilations):
            raise ValueError(f"body_dilations 必须为非空正整数序列，当前 {dilations}")

        groups = resolve_group_norm_groups(inner_dim, gn_groups)
        self.stem = nn.Conv3d(2 * feature_dim, inner_dim, kernel_size=1, bias=False)
        self.stem_norm = nn.GroupNorm(groups, inner_dim)
        self.stem_act = nn.SiLU(inplace=True)
        self.body = nn.ModuleList(
            [
                _ResidualDilatedBlock(
                    channels=inner_dim,
                    dilation=dilation,
                    gn_groups=gn_groups,
                )
                for dilation in dilations
            ]
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
                f"当前 {tuple(warped_slow.shape)} vs {tuple(current_fast.shape)}"
            )
        if warped_slow.dim() != 5:
            raise ValueError(
                f"fusion 输入必须为 5D (B,C,X,Y,Z)，当前 {tuple(warped_slow.shape)}"
            )
        slow_zyx = warped_slow.permute(0, 1, 4, 3, 2).contiguous()
        fast_zyx = current_fast.permute(0, 1, 4, 3, 2).contiguous()
        hidden = torch.cat([slow_zyx, fast_zyx], dim=1)
        hidden = self.stem_act(self.stem_norm(self.stem(hidden)))
        for block in self.body:
            hidden = block(hidden)
        hidden = self.head(hidden)
        return hidden.permute(0, 1, 4, 3, 2).contiguous()


class XYUpsampleResidualDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 288,
        decoder_channels: int = 32,
        out_channels: int = 18,
        init_scale: float | None = 1.0e-6,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.project = nn.Conv3d(
            in_channels,
            decoder_channels,
            kernel_size=1,
            bias=False,
        )
        self.decoder = DenseDecoder(
            in_channels=decoder_channels,
            out_channels=out_channels,
            init_scale=init_scale,
            gn_groups=gn_groups,
        )

    def forward(
        self,
        latent: torch.Tensor,
        output_shape_xyz: tuple[int, int, int],
    ) -> torch.Tensor:
        if latent.dim() != 5:
            raise ValueError(
                f"decoder 输入必须为 5D (B,C,X,Y,Z)，当前 {tuple(latent.shape)}"
            )
        x_size, y_size, z_size = (int(value) for value in output_shape_xyz)
        latent_zyx = latent.permute(0, 1, 4, 3, 2).contiguous()
        hidden = self.project(latent_zyx)
        hidden = F.interpolate(
            hidden,
            size=(z_size, y_size, x_size),
            mode="trilinear",
            align_corners=False,
        )
        logits = self.decoder(hidden)
        return logits.permute(0, 1, 4, 3, 2).contiguous()
