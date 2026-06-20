"""StreamingFlow baseline 使用的轻量 2D building blocks。"""

from __future__ import annotations

import torch
import torch.nn as nn

from online_ncde.utils.nn import resolve_group_norm_groups


def _make_activation(name: str) -> nn.Module | None:
    name_l = str(name).lower()
    if name_l in {"none", "identity", ""}:
        return None
    if name_l == "silu":
        return nn.SiLU(inplace=True)
    if name_l == "gelu":
        return nn.GELU()
    if name_l == "relu":
        return nn.ReLU(inplace=True)
    if name_l == "tanh":
        return nn.Tanh()
    raise ValueError(f"未知 activation: {name!r}")


class ConvNormAct2d(nn.Module):
    """Conv2d + GroupNorm + activation。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
        gn_groups: int = 8,
        activation: str = "silu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (int(kernel_size) - 1) // 2
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            bias=bool(bias),
        )
        if use_norm:
            groups = resolve_group_norm_groups(
                num_channels=int(out_channels), preferred_groups=int(gn_groups)
            )
            self.norm: nn.Module | None = nn.GroupNorm(groups, int(out_channels))
        else:
            self.norm = None
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResBlock2D(nn.Module):
    """两层 3x3 Conv 的残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        gn_groups: int = 8,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        out_c = int(out_channels or in_channels)
        self.conv1 = ConvNormAct2d(
            int(in_channels), out_c, kernel_size=3, gn_groups=gn_groups, activation=activation
        )
        self.conv2 = ConvNormAct2d(
            out_c, out_c, kernel_size=3, gn_groups=gn_groups, activation="none"
        )
        self.proj = (
            nn.Conv2d(int(in_channels), out_c, kernel_size=1, bias=False)
            if int(in_channels) != out_c
            else None
        )
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.proj is None else self.proj(x)
        out = self.conv2(self.conv1(x))
        out = out + residual
        if self.act is not None:
            out = self.act(out)
        return out


class SELayer2D(nn.Module):
    """Squeeze-and-Excitation，用于 p_model。"""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(c // int(reduction), 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.avg_pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class TrustBottleBlock2D(nn.Module):
    """StreamingFlow trusting gate 使用的 Bottleblock 本地实现。"""

    def __init__(self, in_channels: int, out_channels: int, gn_groups: int = 8) -> None:
        super().__init__()
        in_c = int(in_channels)
        out_c = int(out_channels)
        bottleneck_c = max(in_c // 2, 1)
        self.conv7 = ConvNormAct2d(
            in_c,
            bottleneck_c,
            kernel_size=7,
            padding=3,
            bias=False,
            gn_groups=gn_groups,
            activation="gelu",
        )
        self.conv1 = ConvNormAct2d(
            bottleneck_c,
            bottleneck_c,
            kernel_size=1,
            padding=0,
            bias=False,
            gn_groups=gn_groups,
            activation="gelu",
        )
        self.conv3 = ConvNormAct2d(
            bottleneck_c,
            out_c,
            kernel_size=3,
            padding=1,
            bias=False,
            gn_groups=gn_groups,
            activation="gelu",
        )
        self.proj = (
            nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, bias=False), nn.GELU())
            if in_c != out_c
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3(self.conv1(self.conv7(x)))
        residual = x if self.proj is None else self.proj(x)
        return out + residual
