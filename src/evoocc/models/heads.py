from __future__ import annotations

import torch
import torch.nn as nn


class CtrlProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, delta_ctrl: torch.Tensor) -> torch.Tensor:
        is_4d = False
        if delta_ctrl.dim() == 4:
            is_4d = True
            delta_ctrl = delta_ctrl.unsqueeze(0)

        out = self.conv(delta_ctrl)

        if is_4d:
            out = out.squeeze(0)
        return out


