from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_lidar_rays(device: torch.device | str = "cpu") -> torch.Tensor:
    from evoocc.ops.dvr.lidar_rays import generate_lidar_rays as _np_rays

    return torch.from_numpy(_np_rays()).to(device=device, dtype=torch.float32)


class RayLoss(nn.Module):
    def __init__(
        self,
        pc_range: Sequence[float],
        free_index: int,
        num_samples: int = 50,
        step_m: float = 0.4,
        window_voxels: int = 1,
        near_max_m: float = 10.0,
        mid_max_m: float = 20.0,
        near_weight: float = 2.0,
        mid_weight: float = 1.0,
        lambda_hit: float = 0.5,
        gt_dist_bias_m: float | None = None,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if len(pc_range) != 6:
            raise ValueError(f"pc_range 必须是 6 元组，实际 {pc_range}")
        self.pc_range: Tuple[float, float, float, float, float, float] = tuple(
            float(x) for x in pc_range
        )  # type: ignore[assignment]
        self.free_index = int(free_index)
        self.num_samples = int(num_samples)
        self.step_m = float(step_m)
        self.window_voxels = int(window_voxels)
        self.near_max_m = float(near_max_m)
        self.mid_max_m = float(mid_max_m)
        self.near_weight = float(near_weight)
        self.mid_weight = float(mid_weight)
        self.lambda_hit = float(lambda_hit)
        # DVR gives voxel exit distance; default bias 0.5*step_m maps it to a center-like distance (0.0 for center-semantics GT).
        self.gt_dist_bias_m = (
            0.5 * self.step_m if gt_dist_bias_m is None else float(gt_dist_bias_m)
        )
        self.eps = float(eps)
        self.ray_horizon_m = min(self.mid_max_m, self.num_samples * self.step_m)

        d = (torch.arange(self.num_samples, dtype=torch.float32) + 0.5) * self.step_m
        self.register_buffer("sample_depths", d, persistent=False)

    def _world_to_grid(self, xyz: torch.Tensor) -> torch.Tensor:
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        nx = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        ny = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
        nz = 2.0 * (z - z_min) / (z_max - z_min) - 1.0
        # grid_sample expects (W,H,D) = (z,y,x) for logits laid out (B,C,X,Y,Z).
        return torch.stack([nz, ny, nx], dim=-1)

    def forward(
        self,
        logits: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
        gt_dist: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        origin_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if logits.dim() != 5:
            raise ValueError(f"logits 必须是 5D (B,C,X,Y,Z)，实际 {tuple(logits.shape)}")
        B = logits.shape[0]
        device = logits.device
        dtype = logits.dtype

        if ray_origins.dim() != 3:
            raise ValueError(
                f"ray_origins 必须是 (B,K,3)，实际 {tuple(ray_origins.shape)}"
            )
        if gt_dist.dim() != 3:
            raise ValueError(
                f"gt_dist 必须是 (B,K,R)，实际 {tuple(gt_dist.shape)}"
            )

        K = ray_origins.shape[1]
        if gt_dist.shape[0] != B or gt_dist.shape[1] != K:
            raise ValueError(
                f"gt_dist shape {tuple(gt_dist.shape)} 与 ray_origins "
                f"shape {tuple(ray_origins.shape)} 不匹配"
            )
        if valid_mask is not None:
            if valid_mask.dim() != 3 or valid_mask.shape[:2] != (B, K):
                raise ValueError(
                    f"valid_mask 必须是 (B,K,R)，实际 {tuple(valid_mask.shape)}"
                )

        if ray_dirs.dim() == 2:
            R = ray_dirs.shape[0]
            dirs_base = ray_dirs.to(device=device, dtype=dtype).view(1, 1, R, 1, 3)
        elif ray_dirs.dim() == 3:
            if ray_dirs.shape[0] != B:
                raise ValueError(
                    f"ray_dirs batch 维 {ray_dirs.shape[0]} 与 logits batch {B} 不一致"
                )
            R = ray_dirs.shape[1]
            dirs_base = ray_dirs.to(device=device, dtype=dtype).view(B, 1, R, 1, 3)
        else:
            raise ValueError(f"ray_dirs 维度不合法：{tuple(ray_dirs.shape)}")
        if gt_dist.shape[2] != R:
            raise ValueError(
                f"gt_dist ray 维 {gt_dist.shape[2]} 与 ray_dirs ray 维 {R} 不一致"
            )
        N = self.num_samples

        ray_origins = ray_origins.to(device=device, dtype=dtype)
        gt_dist = gt_dist.to(device=device, dtype=dtype)

        d = self.sample_depths.to(device=device, dtype=dtype)
        origins_e = ray_origins.view(B, K, 1, 1, 3)
        d_e = d.view(1, 1, 1, N, 1)
        xyz = origins_e + d_e * dirs_base

        grid = self._world_to_grid(xyz)
        sample_valid = (grid.abs() <= 1.0).all(dim=-1)
        # Fold K*R*N into D_out so one grid_sample call covers all samples.
        grid_s = grid.reshape(B, K * R * N, 1, 1, 3)

        probs = F.softmax(logits, dim=1)
        p_free_vol = probs[:, self.free_index : self.free_index + 1]
        p_free = F.grid_sample(
            p_free_vol,
            grid_s,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        p_free = p_free.reshape(B, K, R, N)
        # Out-of-volume samples are forced free; border padding would otherwise create false first hits.
        p_free = torch.where(sample_valid, p_free, torch.ones_like(p_free))
        p_occ = (1.0 - p_free).clamp(max=1.0 - self.eps)

        log_one_minus_p = torch.log(
            (1.0 - p_occ).clamp(min=self.eps)
        )
        # Exclusive cumsum: log_trans_i = sum_{j<i} log(1-p_j); q_i = p_i * trans_i.
        cum = torch.cumsum(log_one_minus_p, dim=-1)
        log_trans = torch.cat(
            [torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1
        )
        trans = torch.exp(log_trans)
        q = p_occ * trans

        base_mask = torch.ones((B, K, R), device=device, dtype=torch.bool)
        if origin_mask is not None:
            if origin_mask.dim() != 2 or origin_mask.shape != (B, K):
                raise ValueError(
                    f"origin_mask 必须是 (B,K)，实际 {tuple(origin_mask.shape)}"
                )
            base_mask = base_mask & origin_mask.to(device=device).bool().unsqueeze(-1)
        if valid_mask is not None:
            base_mask = base_mask & valid_mask.to(device=device).bool()

        hit_mask_raw = torch.isfinite(gt_dist) & (gt_dist > 0) & base_mask
        hit_mask = hit_mask_raw & (gt_dist < self.ray_horizon_m)
        gt_dist_hit = torch.where(hit_mask, gt_dist, torch.zeros_like(gt_dist))

        base_w = torch.where(
            gt_dist_hit < self.near_max_m, self.near_weight, self.mid_weight
        )

        gt_dist_eff = gt_dist_hit - self.gt_dist_bias_m

        half_win_m = self.window_voxels * self.step_m
        d_broadcast = d.view(1, 1, 1, N)
        in_window = (d_broadcast - gt_dist_eff.unsqueeze(-1)).abs() <= half_win_m
        # Rays without any sample in the window are dropped to avoid a constant large penalty.
        has_window = in_window.any(dim=-1)
        hit_mask = hit_mask & has_window
        w_hit = base_w * hit_mask.to(dtype)
        hit_rays = hit_mask.sum()
        supervised_rays = hit_rays

        zero = logits.sum() * 0.0
        zero_count = torch.tensor(0, device=device, dtype=torch.long)
        if int(supervised_rays.item()) == 0:
            return {
                "total": zero,
                "hit": zero,
                "hit_raw": zero.detach(),
                "hit_rays": zero_count,
                "supervised_rays": zero_count,
                "valid_rays": zero_count,
            }

        def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            return (values * weights).sum() / weights.sum().clamp_min(self.eps)

        q_in_window = (q * in_window.to(q.dtype)).sum(dim=-1)
        nll = -torch.log(q_in_window + self.eps)
        hit_raw = _masked_mean(nll, w_hit)

        hit_weighted = self.lambda_hit * hit_raw
        return {
            "total": hit_weighted,
            "hit": hit_weighted,
            "hit_raw": hit_raw.detach(),
            "hit_rays": hit_rays,
            "supervised_rays": supervised_rays,
            "valid_rays": supervised_rays,
        }
