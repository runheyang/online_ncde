"""Dense Decoder：面向 online 场景的轻量读出头。"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from online_ncde.utils.nn import resolve_group_norm_groups


class DenseDecoder(nn.Module):
    """低分辨率 grouped 3x3x3 + pointwise 混合 + 上采样 + 高分辨率轻量细化。"""

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
        lowres_groups = 2
        if in_channels % lowres_groups != 0:
            raise ValueError(
                f"in_channels={in_channels} must be divisible by lowres_groups={lowres_groups}"
            )

        # 低分辨率语义增强：先分组 3x3x3，再用 pointwise 做组间融合。
        self.lowres_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=lowres_groups,
            bias=False,
        )
        self.lowres_pw = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.lowres_gn = nn.GroupNorm(resolved_groups, in_channels)
        self.lowres_relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.lowres_pw.weight, mode="fan_out", nonlinearity="relu")

        # 高分辨率轻量细化：Depthwise 3D 卷积（仅在 XY 方向做 3x3）
        self.refine_dw = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            groups=in_channels,
            bias=False,
        )
        self.refine_gn = nn.GroupNorm(resolved_groups, in_channels)
        self.refine_relu = nn.ReLU(inplace=True)

        # Pointwise 输出头，映射到类别 logits delta
        self.out_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self._init_output(init_scale)

    def _init_output(self, init_scale: float) -> None:
        # 残差范式下默认将最后 1x1 输出头接近置零
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
            (B, C_out, Z, 2Y, 2X)
        """
        x = self.lowres_conv(x)
        x = self.lowres_gn(x)
        x = self.lowres_relu(x)
        x = self.lowres_pw(x)
        x = self.lowres_gn(x)
        x = self.lowres_relu(x)

        x = F.interpolate(
            x,
            scale_factor=(1.0, 2.0, 2.0),
            mode="trilinear",
            align_corners=False,
        )

        x = self.refine_dw(x)
        x = self.refine_gn(x)
        x = self.refine_relu(x)

        x = self.out_conv(x)
        return x
