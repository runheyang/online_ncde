"""StreamingFlow-style Dual-GRU jump / GRU-ODE cell 本地实现。"""

from __future__ import annotations

import torch
import torch.nn as nn

from evoocc.baselines.streamingflow.blocks import (
    ConvNormAct2d,
    ResBlock2D,
    SELayer2D,
    TrustBottleBlock2D,
)


class DualGRUCell2D(nn.Module):
    """StreamingFlow observation jump 使用的 dual-branch GRU cell。"""

    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 64,
        gru_bias_init: float = 0.0,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.gru_bias_init = float(gru_bias_init)

        in_1 = self.input_size + self.hidden_size
        self.conv_update_1 = nn.Conv2d(in_1, self.hidden_size, kernel_size=3, padding=1, bias=True)
        self.conv_reset_1 = nn.Conv2d(in_1, self.hidden_size, kernel_size=3, padding=1, bias=True)
        self.conv_state_tilde_1 = nn.Conv2d(
            in_1, self.hidden_size, kernel_size=3, padding=1, bias=True
        )

        in_2 = self.hidden_size + self.hidden_size
        self.conv_update_2 = nn.Conv2d(in_2, self.hidden_size, kernel_size=3, padding=1, bias=True)
        self.conv_reset_2 = nn.Conv2d(in_2, self.hidden_size, kernel_size=3, padding=1, bias=True)
        self.conv_state_tilde_2 = nn.Conv2d(
            in_2, self.hidden_size, kernel_size=3, padding=1, bias=True
        )
        self.conv_decoder_2 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=True
        )

        self.trusting_gate = nn.Sequential(
            TrustBottleBlock2D(self.hidden_size * 2, self.hidden_size, gn_groups=gn_groups),
            nn.Conv2d(self.hidden_size, 2, kernel_size=1, bias=False),
        )

    def _gru_cell_1(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = torch.sigmoid(self.conv_update_1(x_and_state) + self.gru_bias_init)
        reset_gate = torch.sigmoid(self.conv_reset_1(x_and_state) + self.gru_bias_init)
        state_tilde = self.conv_state_tilde_1(
            torch.cat([x, (1.0 - reset_gate) * state], dim=1)
        )
        return (1.0 - update_gate) * state + update_gate * state_tilde

    def _gru_cell_2(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = torch.sigmoid(self.conv_update_2(x_and_state) + self.gru_bias_init)
        reset_gate = torch.sigmoid(self.conv_reset_2(x_and_state) + self.gru_bias_init)
        state_tilde = self.conv_state_tilde_2(
            torch.cat([x, (1.0 - reset_gate) * state], dim=1)
        )
        return (1.0 - update_gate) * state + update_gate * state_tilde

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or state.dim() != 4:
            raise ValueError(
                f"DualGRUCell2D 输入需为 4D，x={tuple(x.shape)}, state={tuple(state.shape)}"
            )
        if x.shape[1] != self.input_size or state.shape[1] != self.hidden_size:
            raise ValueError(
                f"channel mismatch: x={x.shape[1]}, state={state.shape[1]}, "
                f"expected {self.input_size}/{self.hidden_size}"
            )

        state_1 = self._gru_cell_1(x, state)
        state_2 = self._gru_cell_2(state, state)
        state_2 = self.conv_decoder_2(state_2)

        mixed = torch.cat([state_1, state_2], dim=1)
        gate = torch.softmax(self.trusting_gate(mixed), dim=1)
        return state_2 * gate[:, 0:1] + state_1 * gate[:, 1:2]


class DualGRUODECell2D(nn.Module):
    """StreamingFlow GRU-ODE vector field：DualGRU(x,h) - h。"""

    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 64,
        gru_bias_init: float = 0.0,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.cell = DualGRUCell2D(
            input_size=input_size,
            hidden_size=hidden_size,
            gru_bias_init=gru_bias_init,
            gn_groups=gn_groups,
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.cell(x, state) - state


class PModel2D(nn.Module):
    """从 hidden state 预测 imputed input 的 mean/log_sigma 参数。"""

    def __init__(self, channels: int = 64, gn_groups: int = 8) -> None:
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            ResBlock2D(c, c * 2, gn_groups=gn_groups),
            SELayer2D(c * 2),
            ResBlock2D(c * 2, c * 2, gn_groups=gn_groups),
            SELayer2D(c * 2),
            ConvNormAct2d(
                c * 2,
                c * 2,
                kernel_size=3,
                padding=1,
                bias=True,
                activation="silu",
                use_norm=False,
            ),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def infer_state_deterministic(q_params: torch.Tensor) -> torch.Tensor:
    """只取 mean 作为 imputed input，保留 log_sigma 但不采样。"""
    mean, _log_sigma = q_params.chunk(2, dim=1)
    return mean
