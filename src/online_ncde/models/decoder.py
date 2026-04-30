"""全分辨率解码器（无上采样）。"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from online_ncde.utils.nn import resolve_group_norm_groups


class DenseDecoder(nn.Module):
    """全分辨率解码器：3x3x3 卷积 → depthwise (1,3,3) 卷积 → 1x1x1 映射到 logits。

    三段式结构：语义增强 → XY 方向细化 → pointwise 输出。

    init_scale 语义：
      - None：输出头走 PyTorch Conv3d 默认初始化（Kaiming uniform），
        适用于 decoder 直接产出绝对 logits（无残差）的场景。
      - <= 0：权重/bias 全置零，残差范式下首步输出 = 0。
      - > 0：权重以 std=init_scale 的正态采样，bias 置零，残差范式下首步输出近 0。
    """

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

        # 3x3x3 语义增强
        self.conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.gn = nn.GroupNorm(groups_1, in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Depthwise (1,3,3)：仅在 XY 方向做局部融合，Z 方向不融合
        self.refine_dw = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            groups=in_channels, bias=False,
        )
        self.refine_gn = nn.GroupNorm(groups_1, in_channels)
        self.refine_relu = nn.ReLU(inplace=True)

        # 1x1x1 映射到类别 logits delta
        self.out_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self._init_output(init_scale)

    def _init_output(self, init_scale: Optional[float]) -> None:
        """init_scale=None 走 PyTorch 默认；<=0 全零；>0 小方差正态。"""
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
        """
        Args:
            x: (B, C, Z, Y, X)
        Returns:
            (B, C_out, Z, Y, X) — 空间尺寸不变
        """
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        x = self.refine_dw(x)
        x = self.refine_gn(x)
        x = self.refine_relu(x)

        x = self.out_conv(x)
        return x
