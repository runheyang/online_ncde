from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from evoocc.ray_loss import RayLoss, generate_lidar_rays
from evoocc.utils.lovasz_losses import lovasz_softmax, lovasz_softmax_flat


def resize_labels_and_mask_to_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        if class_weights is not None and len(class_weights) != int(num_classes):
            raise ValueError(
                f"class_weights 长度必须等于 num_classes，"
                f"当前 {len(class_weights)} vs {num_classes}"
            )
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
            loss = loss * pixel_weights.unsqueeze(1)
            denom = pixel_weights.sum().clamp_min(self.eps) * self.num_classes
        elif mask is not None:
            loss = loss * mask.unsqueeze(1)
            denom = mask.sum().clamp_min(self.eps) * self.num_classes
        else:
            denom = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)
        return loss.sum() / denom


class EvoOccLoss(nn.Module):
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
        self.focal_mask_weight = focal_mask_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        targets, mask = resize_labels_and_mask_to_logits(logits, targets, mask)

        if self.focal_mask_weight is not None and mask is not None:
            # With focal_mask_weight, Focal covers all voxels (in-mask weighted up); Lovasz stays in-mask only.
            pixel_weights = torch.where(
                mask > 0.5,
                torch.tensor(self.focal_mask_weight, device=logits.device, dtype=logits.dtype),
                torch.tensor(1.0, device=logits.device, dtype=logits.dtype),
            )
            focal = self.focal(logits, targets, pixel_weights=pixel_weights)
        else:
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
            "focal": focal_weighted,
            "aux": lovasz_weighted,
            "focal_raw": focal,
            "aux_raw": lovasz,
        }


class OccupancyDiceSemCeLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        free_index: int,
        class_weights: list[float] | None = None,
        lambda_occ_bce: float = 1.0,
        lambda_occ_dice: float = 1.0,
        lambda_sem_ce: float = 1.0,
        lambda_sem_dice: float = 0.0,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.free_index = int(free_index)
        if not 0 <= self.free_index < self.num_classes:
            raise ValueError(
                f"free_index 必须在 [0, {self.num_classes}) 内，当前 {self.free_index}"
            )
        if class_weights is not None and len(class_weights) not in (
            self.num_classes,
            self.num_classes - 1,
        ):
            raise ValueError(
                "class_weights 长度必须等于 num_classes 或 num_classes-1，"
                f"当前 {len(class_weights)} vs {self.num_classes}"
            )
        self.class_weights = class_weights
        self.lambda_occ_bce = float(lambda_occ_bce)
        self.lambda_occ_dice = float(lambda_occ_dice)
        self.lambda_sem_ce = float(lambda_sem_ce)
        self.lambda_sem_dice = float(lambda_sem_dice)
        self.eps = float(eps)

    def _nonfree_logits(self, logits: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.free_index > 0:
            parts.append(logits[:, : self.free_index])
        if self.free_index + 1 < self.num_classes:
            parts.append(logits[:, self.free_index + 1 :])
        if not parts:
            raise ValueError("num_classes 至少需要包含一个非 free 类")
        return torch.cat(parts, dim=1)

    def _semantic_targets(self, targets: torch.Tensor) -> torch.Tensor:
        sem_targets = targets.clone()
        sem_targets = torch.where(
            sem_targets > self.free_index,
            sem_targets - 1,
            sem_targets,
        )
        return sem_targets

    def _semantic_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.class_weights is None:
            return None
        weights = torch.tensor(self.class_weights, device=device, dtype=dtype)
        if weights.numel() == self.num_classes:
            keep = torch.ones(self.num_classes, device=device, dtype=torch.bool)
            keep[self.free_index] = False
            weights = weights[keep]
        return weights

    def _semantic_dice(
        self,
        sem_logits: torch.Tensor,
        sem_targets: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(sem_logits, dim=1)
        one_hot = F.one_hot(
            sem_targets,
            num_classes=self.num_classes - 1,
        ).to(dtype=probs.dtype)
        present = one_hot.sum(dim=0) > 0
        if not bool(present.any()):
            return sem_logits.sum() * 0.0
        probs = probs[:, present]
        one_hot = one_hot[:, present]
        inter = (probs * one_hot).sum(dim=0)
        denom = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        return dice.mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        targets, mask = resize_labels_and_mask_to_logits(logits, targets, mask)
        valid = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask > 0.5
        valid_f = valid.to(dtype=logits.dtype)
        denom = valid_f.sum().clamp_min(self.eps)

        nonfree_logits = self._nonfree_logits(logits)
        free_logits = logits[:, self.free_index]
        occ_logits = torch.logsumexp(nonfree_logits, dim=1) - free_logits
        target_occ = (targets != self.free_index).to(dtype=logits.dtype)

        occ_bce_raw = F.binary_cross_entropy_with_logits(
            occ_logits,
            target_occ,
            reduction="none",
        )
        occ_bce = (occ_bce_raw * valid_f).sum() / denom

        occ_prob = torch.sigmoid(occ_logits)
        occ_prob_valid = occ_prob * valid_f
        target_occ_valid = target_occ * valid_f
        inter = (occ_prob_valid * target_occ_valid).sum()
        occ_dice = 1.0 - (2.0 * inter + self.eps) / (
            occ_prob_valid.sum() + target_occ_valid.sum() + self.eps
        )

        sem_valid = valid & (targets != self.free_index)
        if bool(sem_valid.any()):
            sem_logits = nonfree_logits.permute(0, 2, 3, 4, 1)[sem_valid]
            sem_targets = self._semantic_targets(targets)[sem_valid].long()
            sem_ce = F.cross_entropy(
                sem_logits,
                sem_targets,
                weight=self._semantic_weights(logits.device, logits.dtype),
            )
            sem_dice = self._semantic_dice(sem_logits, sem_targets)
        else:
            sem_ce = logits.sum() * 0.0
            sem_dice = logits.sum() * 0.0

        occ_total = self.lambda_occ_bce * occ_bce + self.lambda_occ_dice * occ_dice
        sem_total = self.lambda_sem_ce * sem_ce + self.lambda_sem_dice * sem_dice
        total = occ_total + sem_total
        return {
            "total": total,
            "focal": occ_total,
            "aux": sem_total,
            "occ_bce_raw": occ_bce,
            "occ_dice_raw": occ_dice,
            "sem_ce_raw": sem_ce,
            "sem_dice_raw": sem_dice,
            "focal_raw": occ_total,
            "aux_raw": sem_ce,
        }


class SegAndRayLoss(nn.Module):
    accepts_ray_kwargs: bool = True

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
        self.register_buffer("ray_dirs", generate_lidar_rays("cpu"), persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        ray_origins: torch.Tensor | None = None,
        gt_dist: torch.Tensor | None = None,
        origin_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        seg_out = self.seg(logits, targets, mask)

        if ray_origins is None or gt_dist is None:
            zero = logits.sum() * 0.0
            seg_out["ray_total"] = zero.detach()
            seg_out["ray_hit"] = zero.detach()
            seg_out["ray_hit_rays"] = torch.tensor(0, device=logits.device)
            seg_out["ray_valid_rays"] = torch.tensor(0, device=logits.device)
            return seg_out

        ray_out = self.ray(
            logits=logits,
            ray_origins=ray_origins,
            ray_dirs=self.ray_dirs,
            gt_dist=gt_dist,
            # Per-ray filtering is not wired; build valid_mask here if FOV masking is needed.
            valid_mask=None,
            origin_mask=origin_mask,
        )
        ray_total = ray_out["total"]
        seg_out["total"] = seg_out["total"] + self.lambda_ray * ray_total
        seg_out["ray_total"] = (self.lambda_ray * ray_total).detach()
        seg_out["ray_hit"] = ray_out["hit_raw"]
        seg_out["ray_hit_rays"] = ray_out["hit_rays"]
        seg_out["ray_valid_rays"] = ray_out["valid_rays"]
        return seg_out


def build_loss(loss_cfg: dict, num_classes: int) -> nn.Module:
    loss_type = loss_cfg.get("type", "focal_lovasz")
    if loss_type == "focal_lovasz":
        seg = EvoOccLoss(
            num_classes=num_classes,
            gamma=loss_cfg.get("gamma", 2.0),
            class_weights=loss_cfg.get("class_weights", None),
            lambda_focal=loss_cfg.get("lambda_focal", 1.0),
            lambda_lovasz=loss_cfg.get("lambda_lovasz", 1.0),
            focal_mask_weight=loss_cfg.get("focal_mask_weight", None),
        )
    elif loss_type == "occ_bce_dice_sem_ce":
        seg = OccupancyDiceSemCeLoss(
            num_classes=num_classes,
            free_index=int(loss_cfg.get("free_index", num_classes - 1)),
            class_weights=loss_cfg.get("class_weights", None),
            lambda_occ_bce=loss_cfg.get("lambda_occ_bce", 1.0),
            lambda_occ_dice=loss_cfg.get("lambda_occ_dice", 1.0),
            lambda_sem_ce=loss_cfg.get("lambda_sem_ce", 1.0),
            lambda_sem_dice=loss_cfg.get("lambda_sem_dice", 0.0),
            eps=loss_cfg.get("eps", 1.0e-6),
        )
    else:
        raise ValueError(
            f"未知 loss type: {loss_type!r}，"
            "支持 'focal_lovasz' / 'occ_bce_dice_sem_ce'"
        )

    ray_cfg = loss_cfg.get("ray", None)
    if not ray_cfg:
        return seg

    pc_range = ray_cfg.get("pc_range") or loss_cfg.get("pc_range")
    free_index = ray_cfg.get("free_index")
    if pc_range is None or free_index is None:
        raise ValueError(
            "build_loss(ray): 需要 pc_range 与 free_index（由 train 脚本注入 loss_cfg）。"
        )
    ray_kwargs: dict = {}
    for key in (
        "num_samples",
        "step_m",
        "window_voxels",
        "near_max_m",
        "mid_max_m",
        "near_weight",
        "mid_weight",
        "lambda_hit",
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
