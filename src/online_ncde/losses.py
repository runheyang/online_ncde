"""online_ncde 损失函数。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.lovasz_losses import lovasz_softmax, lovasz_softmax_flat  # type: ignore[import-not-found]


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


def build_loss(loss_cfg: dict, num_classes: int) -> nn.Module:
    """根据配置构建 loss 函数。"""
    loss_type = loss_cfg.get("type", "focal_lovasz")
    if loss_type == "focal_lovasz":
        return OnlineNcdeLoss(
            num_classes=num_classes,
            gamma=loss_cfg.get("gamma", 2.0),
            class_weights=loss_cfg.get("class_weights", None),
            lambda_focal=loss_cfg.get("lambda_focal", 1.0),
            lambda_lovasz=loss_cfg.get("lambda_lovasz", 1.0),
            focal_mask_weight=loss_cfg.get("focal_mask_weight", None),
        )
    elif loss_type == "focal_dice":
        return OnlineNcdeFocalDiceLoss(
            num_classes=num_classes,
            gamma=loss_cfg.get("gamma", 2.0),
            class_weights=loss_cfg.get("class_weights", None),
            lambda_focal=loss_cfg.get("lambda_focal", 1.0),
            lambda_dice=loss_cfg.get("lambda_dice", 1.0),
            mask_weight=loss_cfg.get("mask_weight", 5.0),
        )
    else:
        raise ValueError(f"未知 loss type: {loss_type!r}，支持 'focal_lovasz' 或 'focal_dice'")

