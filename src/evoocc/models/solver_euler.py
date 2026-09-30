from __future__ import annotations

import torch
import torch.nn as nn

from evoocc.models.func_g import FuncG
from evoocc.models.heads import CtrlProjector


class EulerNextFastSolver(nn.Module):
    def __init__(self, func_g: FuncG, ctrl_proj: CtrlProjector) -> None:
        super().__init__()
        self.func_g = func_g
        self.ctrl_proj = ctrl_proj

    def step(
        self,
        h_adv: torch.Tensor,
        f_prev_adv: torch.Tensor,  # noqa: ARG002
        f_t: torch.Tensor,
        delta_ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_scene = self.ctrl_proj(delta_ctrl)
        slope = self.func_g(h_adv, f_t)
        h_next = h_adv + slope * delta_scene
        return h_next, delta_scene
