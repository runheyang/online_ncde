"""online_ncde 损失函数。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.lovasz_losses import lovasz_softmax, lovasz_softmax_flat  # type: ignore[import-not-found]

from online_ncde.ray_loss import RayLoss, generate_lidar_rays


def resize_labels_and_mask_to_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """当输出分辨率与 GT 不一致时，对标签和 mask 做最近邻对齐。"""
    target_shape = logits.shape[-3:]
    if tuple(labels.shape[-3:]) == tuple(target_shape):
        return labels, mask

    labels_rs = (
        F.interpolate(labels.unsqueeze(1).float(), size=target_shape, mode="nearest")
        .squeeze(1)
        .to(torch.long)
    )
    if mask is None:
        return labels_rs, None
    mask_rs = (
        F.interpolate(mask.unsqueeze(1).float(), size=target_shape, mode="nearest")
        .squeeze(1)
        .to(mask.dtype)
    )
    return labels_rs, mask_rs


class FocalLoss(nn.Module):
    """多类 Focal Loss（sigmoid BCE 形式）。"""

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.class_weights = class_weights
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        pixel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        probs = torch.sigmoid(logits)
        pt = probs * one_hot + (1.0 - probs) * (1.0 - one_hot)
        ce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
        loss = (1.0 - pt).pow(self.gamma) * ce

        if self.class_weights is not None:
            weights = torch.tensor(
                self.class_weights,
                device=logits.device,
                dtype=logits.dtype,
            ).view(1, -1, 1, 1, 1)
            loss = loss * weights

        if pixel_weights is not None:
            # 全区域监督，按权重缩放
            loss = loss * pixel_weights.unsqueeze(1)
            denom = pixel_weights.sum().clamp_min(self.eps) * self.num_classes
        elif mask is not None:
            # 二值 mask，mask 外忽略
            loss = loss * mask.unsqueeze(1)
            denom = mask.sum().clamp_min(self.eps) * self.num_classes
        else:
            denom = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)
        return loss.sum() / denom


class OnlineNcdeLoss(nn.Module):
    """手动调权 Focal + Lovasz。"""

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        lambda_focal: float = 1.0,
        lambda_lovasz: float = 1.0,
        ignore_index: int = -1,
        focal_mask_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(
            num_classes=num_classes,
            gamma=gamma,
            class_weights=class_weights,
        )
        self.lambda_focal = float(lambda_focal)
        self.lambda_lovasz = float(lambda_lovasz)
        self.ignore_index = ignore_index
        # focal_mask_weight 不为 None 时，Focal 对全区域监督：
        # mask 内 ×focal_mask_weight，mask 外 ×1.0；Lovász 仍只算 mask 内。
        self.focal_mask_weight = focal_mask_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        targets, mask = resize_labels_and_mask_to_logits(logits, targets, mask)

        if self.focal_mask_weight is not None and mask is not None:
            # Focal 全区域监督，mask 内高权重
            pixel_weights = torch.where(
                mask > 0.5,
                torch.tensor(self.focal_mask_weight, device=logits.device, dtype=logits.dtype),
                torch.tensor(1.0, device=logits.device, dtype=logits.dtype),
            )
            focal = self.focal(logits, targets, pixel_weights=pixel_weights)
        else:
            # 原始行为：只算 mask 内
            focal = self.focal(logits, targets, mask)
        probs = F.softmax(logits, dim=1)
        if mask is not None:
            targets_lovasz = targets.masked_fill(mask == 0, self.ignore_index)
            ignore = self.ignore_index
        else:
            targets_lovasz = targets
            ignore = None

        if probs.dim() == 5:
            probas_flat = probs.permute(0, 2, 3, 4, 1).reshape(-1, probs.shape[1])
            labels_flat = targets_lovasz.reshape(-1)
            if ignore is not None:
                valid = labels_flat != ignore
                probas_flat = probas_flat[valid]
                labels_flat = labels_flat[valid]
            lovasz = lovasz_softmax_flat(probas_flat, labels_flat, classes="present")
        else:
            lovasz = lovasz_softmax(
                probs,
                targets_lovasz,
                classes="present",
                per_image=False,
                ignore=ignore,
            )

        focal_weighted = self.lambda_focal * focal
        lovasz_weighted = self.lambda_lovasz * lovasz
        total = focal_weighted + lovasz_weighted
        return {
            "total": total,
            # 统计/显示口径与 total 保持一致，返回加权后的分项。
            "focal": focal_weighted,
            "aux": lovasz_weighted,
            # 同时保留未加权值，便于需要时单独分析。
            "focal_raw": focal,
            "aux_raw": lovasz,
        }


class SoftDiceLoss(nn.Module):
    """多类加权 Soft Dice Loss，支持逐体素权重。"""

    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pixel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        if pixel_weights is not None:
            w = pixel_weights.unsqueeze(1)
        else:
            w = torch.ones(1, device=logits.device, dtype=logits.dtype)

        # 按类别计算加权 dice
        intersection = (probs * one_hot * w).sum(dim=(0, 2, 3, 4))
        cardinality = ((probs + one_hot) * w).sum(dim=(0, 2, 3, 4))
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # 只对 present classes 求均值
        present = one_hot.sum(dim=(0, 2, 3, 4)) > 0
        if present.any():
            return 1.0 - dice_per_class[present].mean()
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)


class OnlineNcdeFocalDiceLoss(nn.Module):
    """Focal + Dice，全区域监督，mask 内高权重。"""

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        mask_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(
            num_classes=num_classes,
            gamma=gamma,
            class_weights=class_weights,
        )
        self.dice = SoftDiceLoss(num_classes=num_classes)
        self.lambda_focal = float(lambda_focal)
        self.lambda_dice = float(lambda_dice)
        self.mask_weight = float(mask_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        targets, mask = resize_labels_and_mask_to_logits(logits, targets, mask)

        # 构建逐体素权重：mask 内 mask_weight，mask 外 1.0
        if mask is not None:
            pixel_weights = torch.where(
                mask > 0.5,
                torch.tensor(self.mask_weight, device=logits.device, dtype=logits.dtype),
                torch.tensor(1.0, device=logits.device, dtype=logits.dtype),
            )
        else:
            pixel_weights = None

        focal = self.focal(logits, targets, pixel_weights=pixel_weights)
        dice = self.dice(logits, targets, pixel_weights=pixel_weights)

        focal_weighted = self.lambda_focal * focal
        dice_weighted = self.lambda_dice * dice
        total = focal_weighted + dice_weighted
        return {
            "total": total,
            "focal": focal_weighted,
            "aux": dice_weighted,
            "focal_raw": focal,
            "aux_raw": dice,
        }


class SegAndRayLoss(nn.Module):
    """seg loss + ray first-hit/depth loss 的组合包装。

    - seg loss 照旧接收 (logits, targets, mask)，返回 dict（必含 total/focal/aux）。
    - ray loss 仅在 forward 的 kwargs 里同时给出 ray_origins / gt_dist 时才会计算；
      否则只跑 seg loss（非多帧路径或 sidecar 缺失的 fallback）。
    - 返回 dict 保持 seg loss 的 key 兼容，额外带 ray_* 字段，便于日志。
    """

    def __init__(
        self,
        seg_loss: nn.Module,
        ray_loss: "RayLoss",
        lambda_ray: float = 0.5,
    ) -> None:
        super().__init__()
        self.seg = seg_loss
        self.ray = ray_loss
        self.lambda_ray = float(lambda_ray)
        # 14040 条固定 ray 方向作为 buffer，自动跟 model 同 device/dtype。
        self.register_buffer("ray_dirs", generate_lidar_rays("cpu"), persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        ray_origins: torch.Tensor | None = None,
        gt_dist: torch.Tensor | None = None,
        ray_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        seg_out = self.seg(logits, targets, mask)

        if ray_origins is None or gt_dist is None:
            zero = logits.sum() * 0.0
            seg_out["ray_total"] = zero.detach()
            seg_out["ray_hit"] = zero.detach()
            seg_out["ray_depth"] = zero.detach()
            seg_out["ray_valid_rays"] = torch.tensor(0, device=logits.device)
            return seg_out

        ray_out = self.ray(
            logits=logits,
            ray_origins=ray_origins,
            ray_dirs=self.ray_dirs,
            gt_dist=gt_dist,
            valid_mask=None,
        )
        ray_total = ray_out["total"]
        seg_out["total"] = seg_out["total"] + self.lambda_ray * ray_total
        seg_out["ray_total"] = (self.lambda_ray * ray_total).detach()
        seg_out["ray_hit"] = ray_out["hit_raw"]
        seg_out["ray_depth"] = ray_out["depth_raw"]
        seg_out["ray_valid_rays"] = ray_out["valid_rays"]
        return seg_out


def build_loss(loss_cfg: dict, num_classes: int) -> nn.Module:
    """根据配置构建 loss 函数。"""
    loss_type = loss_cfg.get("type", "focal_lovasz")
    if loss_type == "focal_lovasz":
        seg = OnlineNcdeLoss(
            num_classes=num_classes,
            gamma=loss_cfg.get("gamma", 2.0),
            class_weights=loss_cfg.get("class_weights", None),
            lambda_focal=loss_cfg.get("lambda_focal", 1.0),
            lambda_lovasz=loss_cfg.get("lambda_lovasz", 1.0),
            focal_mask_weight=loss_cfg.get("focal_mask_weight", None),
        )
    elif loss_type == "focal_dice":
        seg = OnlineNcdeFocalDiceLoss(
            num_classes=num_classes,
            gamma=loss_cfg.get("gamma", 2.0),
            class_weights=loss_cfg.get("class_weights", None),
            lambda_focal=loss_cfg.get("lambda_focal", 1.0),
            lambda_dice=loss_cfg.get("lambda_dice", 1.0),
            mask_weight=loss_cfg.get("mask_weight", 5.0),
        )
    else:
        raise ValueError(f"未知 loss type: {loss_type!r}，支持 'focal_lovasz' 或 'focal_dice'")

    ray_cfg = loss_cfg.get("ray", None)
    if not ray_cfg:
        return seg

    # pc_range / free_index 由 train 脚本从 data 配置注入；其余参数 yaml 可覆盖，
    # 未填字段 fallback 到 RayLoss 默认值。
    pc_range = ray_cfg.get("pc_range") or loss_cfg.get("pc_range")
    free_index = ray_cfg.get("free_index")
    if pc_range is None or free_index is None:
        raise ValueError(
            "build_loss(ray): 需要 pc_range 与 free_index（由 train 脚本注入 loss_cfg）。"
        )
    ray_kwargs: dict = {}
    # 显式列出所有可调参数，避免 yaml 里混入无关 key 被误传给 RayLoss
    for key in (
        "num_samples",
        "step_m",
        "window_voxels",
        "near_max_m",
        "mid_max_m",
        "near_weight",
        "mid_weight",
        "lambda_hit",
        "lambda_depth",
        "depth_asym_far",
        "depth_asym_near",
        "smooth_l1_beta",
        "gt_dist_bias_m",
    ):
        if key in ray_cfg:
            ray_kwargs[key] = ray_cfg[key]
    ray = RayLoss(
        pc_range=pc_range,
        free_index=int(free_index),
        **ray_kwargs,
    )
    return SegAndRayLoss(
        seg_loss=seg,
        ray_loss=ray,
        lambda_ray=float(ray_cfg.get("lambda_ray", 0.5)),
    )

