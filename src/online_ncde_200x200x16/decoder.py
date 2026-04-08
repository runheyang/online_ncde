"""全分辨率解码器（无上采样）。"""

from __future__ import annotations

import torch.nn as nn

from online_ncde.utils.nn import resolve_group_norm_groups


class DenseDecoder200(nn.Module):
    """全分辨率解码器：3x3x3 卷积 + 1x1x1 映射到 logits，无 interpolate。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_scale: float = 1.0e-6,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        resolved_groups = resolve_group_norm_groups(
            num_channels=in_channels, preferred_groups=gn_groups
        )

        # 3x3x3 特征细化
        self.conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.gn = nn.GroupNorm(resolved_groups, in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 1x1x1 映射到类别 logits delta
        self.out_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self._init_output(init_scale)

    def _init_output(self, init_scale: float) -> None:
        """残差范式下将输出头接近置零。"""
        if init_scale <= 0.0:
            nn.init.constant_(self.out_conv.weight, 0.0)
            if self.out_conv.bias is not None:
                nn.init.constant_(self.out_conv.bias, 0.0)
        else:
            nn.init.normal_(self.out_conv.weight, mean=0.0, std=init_scale)
            if self.out_conv.bias is not None:
                nn.init.constant_(self.out_conv.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: (B, C, Z, Y, X)
        Returns:
            (B, C_out, Z, Y, X) — 空间尺寸不变
        """
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x
