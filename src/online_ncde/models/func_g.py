"""NCDE 向量场函数 g（纯 Dense 3D 卷积重负载版）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from online_ncde.utils.nn import resolve_group_norm_groups


def _resolve_group_norm_groups(num_channels: int, preferred_groups: int) -> int:
    return resolve_group_norm_groups(num_channels, preferred_groups)


class _ResidualDilatedBlock(nn.Module):
    """单个残差膨胀卷积块：Conv3d(dilation=d) + GN + SiLU + Residual。"""

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


class FuncG(nn.Module):
    """FuncG-Heavy32x3：Stem(1x1) + 3个膨胀残差3x3块 + Head(1x1+tanh)。

    默认结构（对应推荐方案）：
      - Stem : 1x1 Conv 64 -> 32
      - Body : 3 层 3x3 Conv 32 -> 32，dilation=[1,2,3]，带 residual
      - Head : 1x1 Conv 32 -> 32 + tanh
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        inner_dim: int = 32,
        body_dilations: tuple[int, ...] = (1, 2, 3),
        gn_groups: int = 8,
    ) -> None:
        super().__init__()

        if not body_dilations:
            raise ValueError("body_dilations 不能为空，至少需要一个 dilation。")
        if any(int(d) <= 0 for d in body_dilations):
            raise ValueError(f"body_dilations 中每个 dilation 必须 > 0，当前: {body_dilations}")

        resolved_groups = _resolve_group_norm_groups(
            num_channels=inner_dim, preferred_groups=gn_groups
        )

        # Stem：1x1 降/升维到 inner_dim
        self.stem_conv = nn.Conv3d(in_channels, inner_dim, kernel_size=1, bias=False)
        self.stem_gn = nn.GroupNorm(num_groups=resolved_groups, num_channels=inner_dim)
        self.stem_act = nn.SiLU(inplace=True)

        # Body：多层膨胀残差卷积
        self.body = nn.ModuleList(
            [
                _ResidualDilatedBlock(
                    channels=inner_dim,
                    dilation=int(dilation),
                    gn_groups=resolved_groups,
                )
                for dilation in body_dilations
            ]
        )

        # Head：1x1 回到 hidden_channels
        self.head_conv = nn.Conv3d(inner_dim, hidden_channels, kernel_size=1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, z_tensor: torch.Tensor, fast_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_tensor: (C_h, X, Y, Z) 或 (N, C_h, X, Y, Z)
            fast_tensor: (C_f, X, Y, Z) 或 (N, C_f, X, Y, Z)
        Returns:
            out: 与输入同形状
        """
        # 兼容处理无 batch 维度的输入
        is_4d = False
        if z_tensor.dim() == 4:
            is_4d = True
            z_tensor = z_tensor.unsqueeze(0)
            fast_tensor = fast_tensor.unsqueeze(0)

        x = torch.cat([z_tensor, fast_tensor], dim=1)
        x = self.stem_conv(x)
        x = self.stem_gn(x)
        x = self.stem_act(x)

        for block in self.body:
            x = block(x)

        x = self.head_conv(x)
        out = self.tanh(x)

        if is_4d:
            out = out.squeeze(0)
            
        return out
