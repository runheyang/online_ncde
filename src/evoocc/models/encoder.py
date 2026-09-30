from __future__ import annotations

import torch
import torch.nn as nn

from evoocc.utils.nn import resolve_group_norm_groups


class DenseEncoder(nn.Module):
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
        """x: (B,C,X,Y,Z) -> (B,C_out,X,Y,Z); Z is mapped to Conv3d depth."""
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        x = self.relu(self.gn(self.conv(x)))
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        return x
