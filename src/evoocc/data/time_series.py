from __future__ import annotations

import torch


def compute_segment_dt(
    frame_timestamps: torch.Tensor | None,
    frame_dt: torch.Tensor | None,
    num_frames: int,
    eps: float = 1.0e-6,
    timestamp_scale: float = 1.0e-6,
) -> torch.Tensor:
    if num_frames <= 1:
        return torch.zeros((0,), dtype=torch.float32)

    if frame_dt is not None:
        # frame_dt is cumulative (length T); difference it to per-interval dt
        if frame_dt.numel() == num_frames:
            dt = (frame_dt[1:] - frame_dt[:-1]).float()
        elif frame_dt.numel() == num_frames - 1:
            dt = frame_dt.float()
        else:
            dt = frame_dt.reshape(-1)[: num_frames - 1].float()
        return dt.clamp_min(eps)

    if frame_timestamps is not None:
        # difference int64 timestamps before casting to float to avoid precision loss
        dt = (frame_timestamps[1:] - frame_timestamps[:-1]).float() * float(timestamp_scale)
        return dt.clamp_min(eps)

    return torch.ones((num_frames - 1,), dtype=torch.float32)


def cumulative_tau(dt: torch.Tensor) -> torch.Tensor:
    if dt.numel() == 0:
        return torch.zeros((1,), device=dt.device, dtype=torch.float32)
    prefix = torch.zeros((1,), device=dt.device, dtype=torch.float32)
    return torch.cat([prefix, torch.cumsum(dt.float(), dim=0)], dim=0)


