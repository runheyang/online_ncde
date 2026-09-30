from __future__ import annotations

from typing import Optional

import torch.nn as nn

from evoocc.utils.nn import resolve_group_norm_groups


class DenseDecoder(nn.Module):
    """init_scale: None keeps default init; <=0 zeros the output head; >0 normal(std=init_scale), zero bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_scale: Optional[float] = 1.0e-6,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        groups_1 = resolve_group_norm_groups(
            num_channels=in_channels, preferred_groups=gn_groups
        )

        self.conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.gn = nn.GroupNorm(groups_1, in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.refine_dw = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            groups=in_channels, bias=False,
        )
        self.refine_gn = nn.GroupNorm(groups_1, in_channels)
        self.refine_relu = nn.ReLU(inplace=True)

        self.out_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self._init_output(init_scale)

    def _init_output(self, init_scale: Optional[float]) -> None:
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

    def forward(self, x):
        """x: (B, C, Z, Y, X); spatial size unchanged."""
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        x = self.refine_dw(x)
        x = self.refine_gn(x)
        x = self.refine_relu(x)

        x = self.out_conv(x)
        return x
