from __future__ import annotations

import torch


def build_ctrl_with_time(fast_feat: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
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
    x_curr = build_ctrl_with_time(fast_curr, tau_curr)
    x_prev_adv = build_ctrl_with_time(fast_prev_adv, tau_prev)
    delta = x_curr - x_prev_adv
    return {
        "x_curr_ctrl": x_curr,
        "x_prev_adv_ctrl": x_prev_adv,
        "delta_ctrl": delta,
    }
