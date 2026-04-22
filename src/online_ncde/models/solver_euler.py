"""Euler + next-fast 更新器（对齐 legacy/train_online_ncde_euler_next_fast.py）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from online_ncde.models.func_g import FuncG
from online_ncde.models.heads import CtrlProjector


class EulerNextFastSolver(nn.Module):
    """Euler 单步更新：func_g 仅喂 next-fast 特征 f_t，忽略 f_prev_adv。

    step 签名与 HeunSolver 对齐以便在 aligner 中直接替换，但 f_prev_adv 不进入 func_g。
    """

    def __init__(self, func_g: FuncG, ctrl_proj: CtrlProjector) -> None:
        super().__init__()
        self.func_g = func_g
        self.ctrl_proj = ctrl_proj

    def step(
        self,
        h_adv: torch.Tensor,
        f_prev_adv: torch.Tensor,  # noqa: ARG002  # 保留签名一致，不使用
        f_t: torch.Tensor,
        delta_ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_scene = self.ctrl_proj(delta_ctrl)
        slope = self.func_g(h_adv, f_t)
        h_next = h_adv + slope * delta_scene
        return h_next, delta_scene
