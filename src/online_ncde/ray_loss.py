"""可微的 ray first-hit + no-hit termination loss。

设计目标：针对 RayIoU 的 first-hit 召回率下降问题，直接在 ray 级别上监督
"沿着 GT 有 hit 的 ray，first-hit 概率质量应集中在 GT 深度附近"。

数据流：
    logits (B,C,X,Y,Z)
        → softmax → p_free 通道 → p_occ (B,1,X,Y,Z)
        → 沿 ray 等步长采样 N 个点
        → F.grid_sample 三线性插值到 (B,R,N)
        → first-hit 分布 q_i = p_i * Π_{j<i}(1-p_j)
        → L_hit：窗口 |d_i - d*| ≤ δ·step 内 q 求和的 NLL
        → L_empty：GT no-hit ray 上对 trans_end 做 NLL
        → L_depth：d_hat = Σ q·d + trans_end·d_max，与 d* 的非对称 SmoothL1

GT finite hit ray 监督到 hit/depth；GT inf ray 监督到 no-hit。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_lidar_rays(device: torch.device | str = "cpu") -> torch.Tensor:
    """生成全场景共享的 14040 条 lidar ray 单位方向（torch 版，对齐评估）。

    delegate 到 `online_ncde.ops.dvr.lidar_rays.generate_lidar_rays`，保证训练
    与 RayIoU 评估用的是同一份 numpy 实现，避免 pitch/azimuth 两侧漂移。
    """
    from online_ncde.ops.dvr.lidar_rays import generate_lidar_rays as _np_rays

    return torch.from_numpy(_np_rays()).to(device=device, dtype=torch.float32)


class RayLoss(nn.Module):
    """Ray first-hit + no-hit termination + asymmetric depth loss（可微）。

    Args:
        pc_range:        (x_min, y_min, z_min, x_max, y_max, z_max) ego 坐标系下的 bbox。
        free_index:      free 类在 logits 第 1 维的索引。
        num_samples:     每条 ray 沿深度方向采样点数（默认 50）。
        step_m:          采样步长（米），默认 0.4（与 voxel_size 对齐）。
        window_voxels:   L_hit 窗口半宽，以 step 为单位（δ=1 → 窗口 ±0.4m）。
        near_max_m/mid_max_m: 近场/中场的深度上界。
        near_weight/mid_weight: 两段的 ray 权重。
        lambda_hit/lambda_empty/lambda_depth: 三项 loss 的组合权重。
        depth_asym_far:  pred 比 GT 远时 SmoothL1 的权重。
        depth_asym_near: pred 比 GT 近时 SmoothL1 的权重。
        smooth_l1_beta:  SmoothL1 的切换阈值（米）。
        gt_dist_bias_m:  从 gt_dist 里减去的系统偏置。DVR 返回的是 hit voxel
                         的"出射边界距离"（约 center + 0.5 voxel），而差分 ray
                         marching 的 d_hat 在理想情形≈ voxel center，两者差
                         约 0.5 * step。None → 默认 0.5 * step_m；手动传 0.0
                         表示 GT 已经是 center 语义（如单元测试里人造的 GT）。
        eps:             数值稳定项。
    """

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
        lambda_empty: float = 0.5,
        lambda_depth: float = 0.2,
        depth_asym_far: float = 2.0,
        depth_asym_near: float = 1.0,
        smooth_l1_beta: float = 1.0,
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
        self.lambda_empty = float(lambda_empty)
        self.lambda_depth = float(lambda_depth)
        self.depth_asym_far = float(depth_asym_far)
        self.depth_asym_near = float(depth_asym_near)
        self.smooth_l1_beta = float(smooth_l1_beta)
        # DVR 输出的是 voxel 出射距离，差分 ray marching 的 d_hat 接近 center，
        # 默认补偿 0.5 * step_m。测试/纯 center 语义的 GT 传 0.0 关掉。
        self.gt_dist_bias_m = (
            0.5 * self.step_m if gt_dist_bias_m is None else float(gt_dist_bias_m)
        )
        self.eps = float(eps)
        self.ray_horizon_m = min(self.mid_max_m, self.num_samples * self.step_m)

        # 预计算采样深度 d_i = (i + 0.5) * step
        d = (torch.arange(self.num_samples, dtype=torch.float32) + 0.5) * self.step_m
        self.register_buffer("sample_depths", d, persistent=False)

    # ------------------------------------------------------------------
    # 坐标变换
    # ------------------------------------------------------------------

    def _world_to_grid(self, xyz: torch.Tensor) -> torch.Tensor:
        """ego 世界坐标 (..., 3) → grid_sample 归一化坐标 (..., 3)。

        体素布局约定：logits shape (B, C, X, Y, Z)，对应 grid_sample 的
        (N, C, D=X, H=Y, W=Z)，其 grid 最后维顺序为 (W, H, D) = (z, y, x)。
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        nx = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        ny = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
        nz = 2.0 * (z - z_min) / (z_max - z_min) - 1.0
        return torch.stack([nz, ny, nx], dim=-1)

    # ------------------------------------------------------------------
    # 主 forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
        gt_dist: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        origin_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算 L_hit + L_empty + L_depth（多原点）。

        Args:
            logits:      (B, C, X, Y, Z) 模型输出 logits。
            ray_origins: (B, K, 3) ego 系下的 lidar origin。
            ray_dirs:    (R, 3) 或 (B, R, 3) 单位方向向量。所有 K 个原点共用同一
                         套方向。
            gt_dist:     (B, K, R) GT ray 监督。
                         finite > 0 = first-hit 距离（米）
                         inf        = 监督视野内 no-hit
                         NaN        = ignore
            valid_mask:  (B, K, R) bool 可选；false 表示忽略该 ray。
            origin_mask: (B, K) bool 可选；false 表示该原点是 pad，不贡献 loss。

        Returns:
            dict 包含:
                total:      lambda_hit * hit + lambda_empty * empty + lambda_depth * depth
                hit:        加权后 hit loss（参与 total）
                empty:      加权后 empty loss（参与 total）
                depth:      加权后 depth loss（参与 total）
                hit_raw:    未加权 hit loss（便于日志）
                empty_raw:  未加权 empty loss
                depth_raw:  未加权 depth loss
                hit_rays:   本 batch 参与 hit/depth 的 ray 数
                empty_rays: 本 batch 参与 empty 的 ray 数
                valid_rays: 本 batch 参与计算的 ray 数量（跨 K × R 求和）
        """
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

        # --- 1. 构造采样点 (B, K, R, N, 3) ---
        d = self.sample_depths.to(device=device, dtype=dtype)  # (N,)
        origins_e = ray_origins.view(B, K, 1, 1, 3)
        d_e = d.view(1, 1, 1, N, 1)                             # (1,1,1,N,1)
        xyz = origins_e + d_e * dirs_base                       # (B,K,R,N,3)

        # --- 2. 归一化到 grid_sample 坐标 ---
        grid = self._world_to_grid(xyz)                         # (B,K,R,N,3)
        # 采样点是否落在体积内（三个归一化坐标都在 [-1,1]）
        sample_valid = (grid.abs() <= 1.0).all(dim=-1)          # (B,K,R,N) bool
        # grid_sample 5D: input (B,1,X,Y,Z)，grid 需要 (B, D_out, H_out, W_out, 3)。
        # 把 (K*R*N) 压到 D_out，另外两个维度置 1，一次 kernel 调用搞定。
        grid_s = grid.reshape(B, K * R * N, 1, 1, 3)

        # --- 3. p_free → p_occ 三线性插值 ---
        probs = F.softmax(logits, dim=1)                        # (B,C,X,Y,Z)
        p_free_vol = probs[:, self.free_index : self.free_index + 1]  # (B,1,X,Y,Z)
        p_free = F.grid_sample(
            p_free_vol,
            grid_s,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # (B, 1, K*R*N, 1, 1)
        p_free = p_free.reshape(B, K, R, N)
        # 完全越界的 sample：border padding 会复用边界值，这里强制当作 free
        # (p_free=1 → p_occ=0)，避免边界外 ray 沿用边界 p_free 产生假 first-hit。
        # 体积内、靠近边界的 sample 仍走 border padding，避免 zeros 在边界内侧把
        # p_free 低估 / p_occ 高估。
        p_free = torch.where(sample_valid, p_free, torch.ones_like(p_free))
        p_occ = (1.0 - p_free).clamp(max=1.0 - self.eps)  # (B,K,R,N)

        # --- 4. first-hit 分布 q_i ---
        # trans_i = Π_{j<i}(1 - p_j)，q_i = p_i * trans_i
        log_one_minus_p = torch.log(
            (1.0 - p_occ).clamp(min=self.eps)
        )  # (B,K,R,N)
        cum = torch.cumsum(log_one_minus_p, dim=-1)             # (B,K,R,N)
        # exclusive cumsum: log_trans_i = sum_{j<i} log(1 - p_j)
        log_trans = torch.cat(
            [torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1
        )
        trans = torch.exp(log_trans)                            # (B,K,R,N)
        q = p_occ * trans                                       # (B,K,R,N)
        trans_end = torch.exp(cum[..., -1])                     # (B,K,R) 完全打不到的概率

        # --- 5. GT ray 拆成 hit / empty / ignore ---
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
        empty_mask = (gt_dist == float("inf")) & base_mask
        hit_mask = hit_mask_raw & (gt_dist < self.ray_horizon_m)
        empty_mask = empty_mask | (hit_mask_raw & (gt_dist >= self.ray_horizon_m))

        gt_dist_hit = torch.where(hit_mask, gt_dist, torch.zeros_like(gt_dist))

        # 近/中场权重（DVR 原生语义，与 sidecar / eval 指标口径一致）
        base_w = torch.where(
            gt_dist_hit < self.near_max_m, self.near_weight, self.mid_weight
        )

        # DVR 出射距离 → center-like 距离，对齐 d_hat 的理想位置
        gt_dist_eff = gt_dist_hit - self.gt_dist_bias_m

        half_win_m = self.window_voxels * self.step_m
        d_broadcast = d.view(1, 1, 1, N)
        in_window = (d_broadcast - gt_dist_eff.unsqueeze(-1)).abs() <= half_win_m  # (B,K,R,N)
        # 窗口内必须至少有一个 sample，否则产生恒定大常数惩罚污染均值
        has_window = in_window.any(dim=-1)                      # (B,K,R)
        hit_mask = hit_mask & has_window
        gt_dist_eff = torch.where(hit_mask, gt_dist - self.gt_dist_bias_m, torch.zeros_like(gt_dist))
        w_hit = base_w * hit_mask.to(dtype)                     # (B,K,R)
        w_empty = empty_mask.to(dtype)                          # (B,K,R)
        hit_rays = hit_mask.sum()
        empty_rays = empty_mask.sum()
        supervised_rays = hit_rays + empty_rays

        zero = logits.sum() * 0.0
        zero_count = torch.tensor(0, device=device, dtype=torch.long)
        if int(supervised_rays.item()) == 0:
            return {
                "total": zero,
                "hit": zero,
                "empty": zero,
                "depth": zero,
                "hit_raw": zero.detach(),
                "empty_raw": zero.detach(),
                "depth_raw": zero.detach(),
                "hit_rays": zero_count,
                "empty_rays": zero_count,
                "supervised_rays": zero_count,
                "valid_rays": zero_count,
            }

        def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            return (values * weights).sum() / weights.sum().clamp_min(self.eps)

        # --- 6. L_hit：窗口内 q 之和的 NLL ---
        q_in_window = (q * in_window.to(q.dtype)).sum(dim=-1)    # (B,K,R)
        nll = -torch.log(q_in_window + self.eps)                 # (B,K,R)
        hit_raw = _masked_mean(nll, w_hit)

        # --- 7. L_empty：no-hit 概率的 NLL ---
        empty_nll = -torch.log(trans_end + self.eps)             # (B,K,R)
        empty_raw = _masked_mean(empty_nll, w_empty)

        # --- 8. L_depth：非对称 SmoothL1 on expected first-hit depth ---
        d_max = float(N * self.step_m)
        d_hat = (q * d_broadcast).sum(dim=-1) + trans_end * d_max  # (B,K,R)
        err = d_hat - gt_dist_eff                                   # 正 = 预测更远
        pos_err = err.clamp(min=0.0)
        neg_err = (-err).clamp(min=0.0)

        def _sl1(x: torch.Tensor) -> torch.Tensor:
            return F.smooth_l1_loss(
                x, torch.zeros_like(x), reduction="none", beta=self.smooth_l1_beta
            )

        depth_per_ray = (
            self.depth_asym_far * _sl1(pos_err)
            + self.depth_asym_near * _sl1(neg_err)
        )
        depth_raw = _masked_mean(depth_per_ray, w_hit)

        # --- 9. 汇总 ---
        hit_weighted = self.lambda_hit * hit_raw
        empty_weighted = self.lambda_empty * empty_raw
        depth_weighted = self.lambda_depth * depth_raw
        total = hit_weighted + empty_weighted + depth_weighted
        return {
            "total": total,
            "hit": hit_weighted,
            "empty": empty_weighted,
            "depth": depth_weighted,
            "hit_raw": hit_raw.detach(),
            "empty_raw": empty_raw.detach(),
            "depth_raw": depth_raw.detach(),
            "hit_rays": hit_rays,
            "empty_rays": empty_rays,
            "supervised_rays": supervised_rays,
            "valid_rays": supervised_rays,
        }
