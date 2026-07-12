"""3D occupancy logits 与 StreamingFlow BEV latent 之间的桥接模块。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from evoocc.baselines.streamingflow.blocks import ConvNormAct2d, ResBlock2D


class LogitsToBEVAdapter(nn.Module):
    """Occ3D 3D logits -> BEV feature。"""

    def __init__(
        self,
        num_classes: int = 18,
        height_bins: int = 16,
        mid_channels: int = 128,
        out_channels: int = 64,
        stride_xy: int = 2,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.height_bins = int(height_bins)
        self.stride_xy = int(stride_xy)
        if self.stride_xy < 1:
            raise ValueError(f"stride_xy 必须 >= 1，当前 {stride_xy}")
        in_channels = self.num_classes * self.height_bins
        self.net = nn.Sequential(
            ConvNormAct2d(
                in_channels,
                int(mid_channels),
                kernel_size=3,
                stride=self.stride_xy,
                padding=1,
                gn_groups=gn_groups,
                activation="silu",
            ),
            ResBlock2D(int(mid_channels), int(mid_channels), gn_groups=gn_groups),
            ConvNormAct2d(
                int(mid_channels),
                int(out_channels),
                kernel_size=3,
                padding=1,
                gn_groups=gn_groups,
                activation="silu",
            ),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """输入 `(B, C, X, Y, Z)`，输出 `(B, C_bev, X/stride, Y/stride)`。"""
        if logits.dim() != 5:
            raise ValueError(f"logits 需要 5D (B,C,X,Y,Z)，当前: {tuple(logits.shape)}")
        b, c, x_size, y_size, z_size = logits.shape
        if c != self.num_classes or z_size != self.height_bins:
            raise ValueError(
                f"仅支持 Occ3D C={self.num_classes}, Z={self.height_bins}，"
                f"当前 C={c}, Z={z_size}"
            )
        logits_2d = logits.permute(0, 1, 4, 2, 3).contiguous()
        logits_2d = logits_2d.view(b, c * z_size, x_size, y_size)
        return self.net(logits_2d)


class LightNoPoolSmallEncoder(nn.Module):
    """不下采样的 SRVP-style pre-ODE transform。"""

    def __init__(self, channels: int = 64, num_blocks: int = 3, gn_groups: int = 8) -> None:
        super().__init__()
        blocks = [ResBlock2D(int(channels), int(channels), gn_groups=gn_groups) for _ in range(num_blocks)]
        blocks.append(
            ConvNormAct2d(
                int(channels),
                int(channels),
                kernel_size=3,
                padding=1,
                gn_groups=gn_groups,
                activation="tanh",
            )
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightNoPoolSmallDecoder(nn.Module):
    """不改变分辨率的 post-ODE BEV latent decode。"""

    def __init__(self, channels: int = 64, num_blocks: int = 3, gn_groups: int = 8) -> None:
        super().__init__()
        blocks = [ResBlock2D(int(channels), int(channels), gn_groups=gn_groups) for _ in range(num_blocks)]
        blocks.append(
            ConvNormAct2d(
                int(channels),
                int(channels),
                kernel_size=3,
                padding=1,
                gn_groups=gn_groups,
                activation="silu",
            )
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, s, c, h, w = x.shape
            y = self.net(x.reshape(b * s, c, h, w))
            return y.view(b, s, *y.shape[1:])
        return self.net(x)


class StreamingFlowSmallEncoder2D(nn.Module):
    """StreamingFlow 风格的 200x200 -> 100x100 SRVP encoder。"""

    def __init__(
        self,
        in_channels: int = 64,
        latent_channels: int = 96,
        filter_size: int = 24,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        nf = int(filter_size)
        self.blocks = nn.ModuleList(
            [
                ResBlock2D(int(in_channels), nf, gn_groups=gn_groups),
                ResBlock2D(nf, nf * 2, gn_groups=gn_groups),
                ResBlock2D(nf * 2, nf * 2, gn_groups=gn_groups),
                ResBlock2D(nf * 2, nf * 2, gn_groups=gn_groups),
                ResBlock2D(nf * 2, nf * 4, gn_groups=gn_groups),
            ]
        )
        self.last_conv = ConvNormAct2d(
            nf * 4,
            int(latent_channels),
            kernel_size=3,
            padding=1,
            gn_groups=gn_groups,
            activation="tanh",
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"StreamingFlowSmallEncoder2D 输入需为 4D，当前: {tuple(x.shape)}")
        h = x
        for idx, block in enumerate(self.blocks):
            if idx == 1:
                h = self.maxpool(h)
            h = block(h)
        return self.last_conv(h)


class StreamingFlowSmallDecoder2D(nn.Module):
    """StreamingFlow 风格的 100x100 -> 200x200 SRVP decoder。"""

    def __init__(
        self,
        latent_channels: int = 96,
        out_channels: int = 64,
        filter_size: int = 24,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        nf = int(filter_size)
        self.first_conv = ConvNormAct2d(
            int(latent_channels),
            nf * 4,
            kernel_size=3,
            padding=1,
            gn_groups=gn_groups,
            activation="silu",
        )
        self.blocks = nn.ModuleList(
            [
                ResBlock2D(nf * 4, nf * 2, gn_groups=gn_groups),
                ResBlock2D(nf * 2, nf * 2, gn_groups=gn_groups),
                ResBlock2D(nf * 2, nf * 2, gn_groups=gn_groups),
                ResBlock2D(nf * 2, nf, gn_groups=gn_groups),
                ResBlock2D(nf, nf, gn_groups=gn_groups),
            ]
        )
        self.last_conv = nn.Sequential(
            ConvNormAct2d(nf, nf, kernel_size=3, padding=1, gn_groups=gn_groups, activation="silu"),
            nn.Conv2d(nf, int(out_channels), kernel_size=3, padding=1, bias=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _decode_flat(self, x: torch.Tensor) -> torch.Tensor:
        h = self.first_conv(x)
        for idx, block in enumerate(self.blocks):
            h = block(h)
            if idx == 2:
                h = self.upsample(h)
        return self.last_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, s, c, h, w = x.shape
            y = self._decode_flat(x.reshape(b * s, c, h, w))
            return y.view(b, s, *y.shape[1:])
        if x.dim() == 4:
            return self._decode_flat(x)
        raise ValueError(f"StreamingFlowSmallDecoder2D 输入需为 4D/5D，当前: {tuple(x.shape)}")


class BEVTo3DDecoder(nn.Module):
    """BEV latent -> 200x200x16 absolute occupancy logits。"""

    def __init__(
        self,
        in_channels: int = 64,
        mid_channels: int = 96,
        high_channels: int = 128,
        upsample_scale: int = 2,
        num_classes: int = 18,
        height_bins: int = 16,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.height_bins = int(height_bins)
        self.upsample_scale = int(upsample_scale)
        if self.upsample_scale < 1:
            raise ValueError(f"upsample_scale 必须 >= 1，当前 {upsample_scale}")
        self.stem = ConvNormAct2d(
            int(in_channels), int(high_channels), kernel_size=3, padding=1, gn_groups=gn_groups
        )
        self.high_block = ResBlock2D(int(high_channels), int(high_channels), gn_groups=gn_groups)
        self.mid = ConvNormAct2d(
            int(high_channels), int(mid_channels), kernel_size=3, padding=1, gn_groups=gn_groups
        )
        self.mid_block = ResBlock2D(int(mid_channels), int(mid_channels), gn_groups=gn_groups)
        self.out_conv = nn.Conv2d(
            int(mid_channels), self.num_classes * self.height_bins, kernel_size=1, bias=True
        )

    def _decode_flat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.high_block(self.stem(x))
        if self.upsample_scale != 1:
            x = F.interpolate(
                x, scale_factor=self.upsample_scale, mode="bilinear", align_corners=False
            )
        x = self.mid_block(self.mid(x))
        x = self.out_conv(x)
        n, _, x_size, y_size = x.shape
        x = x.view(n, self.num_classes, self.height_bins, x_size, y_size)
        return x.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, s, c, h, w = x.shape
            y = self._decode_flat(x.reshape(b * s, c, h, w))
            return y.view(b, s, *y.shape[1:])
        if x.dim() == 4:
            return self._decode_flat(x)
        raise ValueError(f"BEVTo3DDecoder 输入需为 4D/5D，当前: {tuple(x.shape)}")


class SameTimeGatedFusion(nn.Module):
    """融合 t0 同时刻 slow / fast BEV observation。"""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.gate = nn.Conv2d(int(channels) * 2, int(channels), kernel_size=3, padding=1)

    def forward(self, slow_bev: torch.Tensor, fast_bev: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([slow_bev, fast_bev], dim=1)))
        return gate * slow_bev + (1.0 - gate) * fast_bev


class SpatialGRU2D(nn.Module):
    """StreamingFlow 风格的 BEV sequence spatial GRU refinement。"""

    def __init__(self, channels: int = 64, gru_bias_init: float = 0.0) -> None:
        super().__init__()
        c = int(channels)
        self.gru_bias_init = float(gru_bias_init)
        self.conv_update = nn.Conv2d(c + c, c, kernel_size=3, padding=1, bias=True)
        self.conv_reset = nn.Conv2d(c + c, c, kernel_size=3, padding=1, bias=True)
        self.conv_state_tilde = nn.Conv2d(c + c, c, kernel_size=3, padding=1, bias=True)
        self.conv_decoder = nn.Conv2d(c, c, kernel_size=1, bias=False)

    def _gru_cell(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = torch.sigmoid(self.conv_update(x_and_state) + self.gru_bias_init)
        reset_gate = torch.sigmoid(self.conv_reset(x_and_state) + self.gru_bias_init)
        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))
        return (1.0 - update_gate) * state + update_gate * state_tilde

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"SpatialGRU2D 输入需为 (B,S,C,H,W)，当前: {tuple(x.shape)}")
        b, steps, c, h, w = x.shape
        rnn_state = x.new_zeros((b, c, h, w)) if state is None else state
        outputs = []
        for idx in range(steps):
            rnn_state = self._gru_cell(x[:, idx], rnn_state)
            outputs.append(self.conv_decoder(rnn_state))
        return torch.stack(outputs, dim=1)


class SpatialGRURefiner2D(nn.Module):
    """可选的 target BEV sequence refinement。"""

    def __init__(
        self,
        channels: int = 64,
        num_gru_blocks: int = 1,
        num_res_layers: int = 1,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.grus = nn.ModuleList([SpatialGRU2D(channels) for _ in range(int(num_gru_blocks))])
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ResBlock2D(int(channels), int(channels), gn_groups=gn_groups)
                        for _ in range(int(num_res_layers))
                    ]
                )
                for _ in range(int(num_gru_blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.grus:
            return x
        hidden_state = x[:, 0]
        for gru, res in zip(self.grus, self.res_blocks):
            x = gru(x, hidden_state)
            b, s, c, h, w = x.shape
            x = res(x.reshape(b * s, c, h, w)).view(b, s, c, h, w)
        return x
