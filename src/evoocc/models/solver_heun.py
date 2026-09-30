from __future__ import annotations

import torch
import torch.nn as nn

from evoocc.models.func_g import FuncG
from evoocc.models.heads import CtrlProjector


class HeunSolver(nn.Module):
    def __init__(self, func_g: FuncG, ctrl_proj: CtrlProjector) -> None:
        super().__init__()
        self.func_g = func_g
        self.ctrl_proj = ctrl_proj

    def step(
        self,
        h_adv: torch.Tensor,
        f_prev_adv: torch.Tensor,
        f_t: torch.Tensor,
        delta_ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_scene = self.ctrl_proj(delta_ctrl)

        s1 = self.func_g(h_adv, f_prev_adv)
        k1 = s1 * delta_scene
        h_hat = h_adv + k1

        s2 = self.func_g(h_hat, f_t)
        k2 = s2 * delta_scene

        h_next = h_adv + 0.5 * (k1 + k2)
        return h_next, delta_scene


