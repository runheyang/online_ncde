from __future__ import annotations

import torch
import torch.nn as nn

from evoocc.utils.nn import resolve_group_norm_groups


def _resolve_group_norm_groups(num_channels: int, preferred_groups: int) -> int:
    return resolve_group_norm_groups(num_channels, preferred_groups)


class _ResidualDilatedBlock(nn.Module):
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

        self.stem_conv = nn.Conv3d(in_channels, inner_dim, kernel_size=1, bias=False)
        self.stem_gn = nn.GroupNorm(num_groups=resolved_groups, num_channels=inner_dim)
        self.stem_act = nn.SiLU(inplace=True)

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

        self.head_conv = nn.Conv3d(inner_dim, hidden_channels, kernel_size=1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, z_tensor: torch.Tensor, fast_tensor: torch.Tensor) -> torch.Tensor:
        """z: (C_h,X,Y,Z) or (N,C_h,X,Y,Z); fast: (C_f,X,Y,Z) or (N,C_f,X,Y,Z); returns same shape as z."""
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
