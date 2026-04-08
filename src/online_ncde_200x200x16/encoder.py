"""全分辨率编码器（不下采样）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from online_ncde.utils.nn import resolve_group_norm_groups


class DenseEncoder200(nn.Module):
    """单层 3x3x3 卷积编码器，stride=(1,1,1)，保持 200x200x16 分辨率。"""

    def __init__(self, in_channels: int, out_channels: int, gn_groups: int = 8) -> None:
        super().__init__()
        resolved_groups = resolve_group_norm_groups(
            num_channels=out_channels, preferred_groups=gn_groups
        )
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False,
        )
        self.gn = nn.GroupNorm(resolved_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, C, X, Y, Z)
        输出: (B, C_out, X, Y, Z)  — 空间尺寸不变
        Conv3d 约定的空间维为 (D, H, W)，这里把 Z 放到 D。
        """
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        x = self.relu(self.gn(self.conv(x)))
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        return x
