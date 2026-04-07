"""时间序列辅助函数。"""

from __future__ import annotations

import torch


def compute_segment_dt(
    frame_timestamps: torch.Tensor | None,
    frame_dt: torch.Tensor | None,
    num_frames: int,
    eps: float = 1.0e-6,
    timestamp_scale: float = 1.0e-6,
) -> torch.Tensor:
    """计算长度为 (T-1) 的时间间隔。
    
    frame_dt 假定为累积时间序列（长度 T），需做差分得到区间。
    frame_timestamps 为 int64 大数，需先差分再转 float 避免精度损失。
    """
    if num_frames <= 1:
        return torch.zeros((0,), dtype=torch.float32)

    if frame_dt is not None:
        # frame_dt 是累积时间序列，做差分得到区间 dt
        if frame_dt.numel() == num_frames:
            dt = (frame_dt[1:] - frame_dt[:-1]).float()
        elif frame_dt.numel() == num_frames - 1:
            # 已经是区间形式，直接使用
            dt = frame_dt.float()
        else:
            # fallback: 截断到 T-1 长度
            dt = frame_dt.reshape(-1)[: num_frames - 1].float()
        return dt.clamp_min(eps)

    if frame_timestamps is not None:
        # 先差分 int64 再转 float，避免大数精度损失
        dt = (frame_timestamps[1:] - frame_timestamps[:-1]).float() * float(timestamp_scale)
        return dt.clamp_min(eps)

    return torch.ones((num_frames - 1,), dtype=torch.float32)


def cumulative_tau(dt: torch.Tensor) -> torch.Tensor:
    """由 dt 生成时间通道 tau，长度为 T。"""
    if dt.numel() == 0:
        return torch.zeros((1,), device=dt.device, dtype=torch.float32)
    prefix = torch.zeros((1,), device=dt.device, dtype=torch.float32)
    return torch.cat([prefix, torch.cumsum(dt.float(), dim=0)], dim=0)


