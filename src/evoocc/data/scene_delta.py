"""场景控制增量构造。

Dense Cache + Sparse Dynamics 版本去掉了 gamma（warp 命中掩码）参数，
因为 backward trilinear warp 不再区分"命中"与"未命中"——所有体素均有完整特征。
"""

from __future__ import annotations

import torch


def build_ctrl_with_time(fast_feat: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
    """构造带时间戳的控制信号 X_ctrl = [F, τ] (全图 Dense 版)。

    fast_feat: (C_f, X, Y, Z) 或 (N, C_f, X, Y, Z)
    tau: 标量或单元素张量
    """
    if isinstance(tau, torch.Tensor):
        tau_t = tau.to(device=fast_feat.device, dtype=fast_feat.dtype).reshape(1)
    else:
        tau_t = torch.tensor([tau], device=fast_feat.device, dtype=fast_feat.dtype)

    is_4d = False
    if fast_feat.dim() == 4:
        is_4d = True
        fast_feat = fast_feat.unsqueeze(0)

    N, C_f, X, Y, Z = fast_feat.shape
    time_col = tau_t.reshape(1, 1, 1, 1, 1).expand(N, 1, X, Y, Z)
    
    out = torch.cat([fast_feat, time_col], dim=1)
    if is_4d:
        out = out.squeeze(0)
    return out


def build_scene_delta_ctrl(
    fast_curr: torch.Tensor,
    fast_prev_adv: torch.Tensor,
    tau_curr: torch.Tensor | float,
    tau_prev: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """构造场景控制增量：ΔX_ctrl = X_t_ctrl - X_{t-1→t}_ctrl (Dense 版)。

    Args:
        fast_curr    : (C_f, X, Y, Z) 或 (N, C_f, X, Y, Z)
        fast_prev_adv: (C_f, X, Y, Z) 或 (N, C_f, X, Y, Z)
        tau_curr     : t 帧累计时间戳
        tau_prev     : t-1 帧累计时间戳
    """
    x_curr = build_ctrl_with_time(fast_curr, tau_curr)
    x_prev_adv = build_ctrl_with_time(fast_prev_adv, tau_prev)
    delta = x_curr - x_prev_adv
    return {
        "x_curr_ctrl": x_curr,
        "x_prev_adv_ctrl": x_prev_adv,
        "delta_ctrl": delta,
    }
