"""评估指标实现（与 spconv_ncde 对齐）。"""

from __future__ import annotations

import numpy as np
import torch

OCC3D_DYNAMIC_OBJECT_IDX = (2, 3, 4, 5, 6, 7, 9, 10)


def apply_free_threshold(logits: torch.Tensor, free_index: int, conf_thresh: float) -> torch.Tensor:
    """按最大 logit 的 sigmoid 置信度阈值筛为 free。"""
    if logits.dim() == 5:
        max_logits, preds = logits.max(dim=1)
        conf = torch.sigmoid(max_logits)
        preds = preds.clone()
        preds[conf < conf_thresh] = free_index
        return preds
    max_logits, preds = logits.max(dim=0)
    conf = torch.sigmoid(max_logits)
    preds = preds.clone()
    preds[conf < conf_thresh] = free_index
    return preds


def compute_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    free_index: int,
) -> dict:
    """
    计算每类 IoU 和 mIoU（排除 free 类）。
    logits: (B, C, X, Y, Z)
    targets: (B, X, Y, Z)
    mask: (B, X, Y, Z)
    """
    preds = logits.argmax(dim=1)
    valid = mask > 0

    ious = []
    for cls in range(num_classes):
        pred_c = (preds == cls) & valid
        tgt_c = (targets == cls) & valid
        inter = (pred_c & tgt_c).sum().item()
        union = (pred_c | tgt_c).sum().item()
        iou = inter / union if union > 0 else 0.0
        ious.append(iou)

    miou = (
        sum(iou for i, iou in enumerate(ious) if i != free_index)
        / max(num_classes - 1, 1)
    )
    miou_all = sum(ious) / max(num_classes, 1)
    return dict(ious=ious, miou=miou, miou_all=miou_all)


class MetricMiouOcc3D:
    """与 OPUS / spconv_ncde 一致的 Occ3D mIoU 统计方式。"""

    def __init__(
        self,
        num_classes: int = 18,
        use_lidar_mask: bool = False,
        use_image_mask: bool = True,
        dynamic_object_idx: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if num_classes == 18:
            self.class_names = [
                "others",
                "barrier",
                "bicycle",
                "bus",
                "car",
                "construction_vehicle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "trailer",
                "truck",
                "driveable_surface",
                "other_flat",
                "sidewalk",
                "terrain",
                "manmade",
                "vegetation",
                "free",
            ]
        elif num_classes == 2:
            self.class_names = ["non-free", "free"]
        else:
            self.class_names = [f"class_{i}" for i in range(num_classes)]

        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        if dynamic_object_idx is None and num_classes == 18:
            dynamic_object_idx = list(OCC3D_DYNAMIC_OBJECT_IDX)
        dynamic_indices = [] if dynamic_object_idx is None else list(dynamic_object_idx)
        self.dynamic_object_idx = np.asarray(
            [
                int(idx)
                for idx in dynamic_indices
                if 0 <= int(idx) < self.num_classes - 1
            ],
            dtype=np.int64,
        )
        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        self.cnt = 0

    def hist_info(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """构建混淆矩阵（与 OPUS 逻辑一致）。"""
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < self.num_classes)
        return np.bincount(
            self.num_classes * gt[k].astype(int) + pred[k].astype(int),
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

    def per_class_iu(self, hist: np.ndarray) -> np.ndarray:
        """计算每类 IoU，空类置为 NaN。"""
        denom = hist.sum(1) + hist.sum(0) - np.diag(hist)
        result = np.full(hist.shape[0], np.nan, dtype=np.float64)
        valid = denom > 0
        result[valid] = np.diag(hist)[valid] / denom[valid]
        return result

    def add_batch(
        self,
        semantics_pred: np.ndarray,
        semantics_gt: np.ndarray,
        mask_lidar: np.ndarray | None = None,
        mask_camera: np.ndarray | None = None,
    ) -> None:
        """累计单个样本的统计量。"""
        self.cnt += 1
        if self.use_image_mask and mask_camera is not None:
            mask = mask_camera.astype(bool)
            semantics_pred = semantics_pred[mask]
            semantics_gt = semantics_gt[mask]
        elif self.use_lidar_mask and mask_lidar is not None:
            mask = mask_lidar.astype(bool)
            semantics_pred = semantics_pred[mask]
            semantics_gt = semantics_gt[mask]

        if self.num_classes == 2:
            semantics_pred = np.copy(semantics_pred)
            semantics_gt = np.copy(semantics_gt)
            semantics_pred[semantics_pred < 17] = 0
            semantics_pred[semantics_pred == 17] = 1
            semantics_gt[semantics_gt < 17] = 0
            semantics_gt[semantics_gt == 17] = 1

        self.hist += self.hist_info(semantics_pred.flatten(), semantics_gt.flatten())

    def count_miou(self, verbose: bool = True) -> float:
        """输出并返回 mIoU（百分比）。"""
        mIoU = self.per_class_iu(self.hist)
        miou = float(round(np.nanmean(mIoU[: self.num_classes - 1]) * 100, 2))
        miou_d = self.count_miou_d(verbose=False, class_iou=mIoU)
        if verbose:
            print(f"===> per class IoU of {self.cnt} samples:")
            for ind_class in range(self.num_classes - 1):
                name = self.class_names[ind_class]
                print(f"===> {name} - IoU = " + str(round(mIoU[ind_class] * 100, 2)))
            print(f"===> mIoU of {self.cnt} samples: {miou}")
            if np.isfinite(miou_d):
                print(f"===> mIoU_D = {miou_d}")
        return miou

    def count_miou_d(self, verbose: bool = True, class_iou: np.ndarray | None = None) -> float:
        """输出并返回动态类 mIoU_D（百分比）。"""
        if class_iou is None:
            class_iou = self.per_class_iu(self.hist)
        if self.dynamic_object_idx.size == 0:
            miou_d = float("nan")
        else:
            miou_d = float(round(np.nanmean(class_iou[self.dynamic_object_idx]) * 100, 2))
        if verbose and np.isfinite(miou_d):
            print(f"===> mIoU_D = {miou_d}")
        return miou_d

    def get_per_class_iou(self) -> np.ndarray:
        """返回每类 IoU（百分比，空类为 NaN）。"""
        mIoU = self.per_class_iu(self.hist) * 100.0
        return mIoU
