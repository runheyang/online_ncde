"""统一的 dense occupancy 内存评估协议。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from online_ncde.data.labels_io import load_labels_npz
from online_ncde.metrics import build_miou_metric


@dataclass
class DenseOccPrediction:
    """单个 dense occupancy 预测记录。"""

    pred: np.ndarray
    token: str
    scene_name: str = ""
    step_idx: int | None = None
    gt: np.ndarray | None = None
    mask_camera: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pred": self.pred,
            "token": self.token,
            "scene_name": self.scene_name,
            "step_idx": self.step_idx,
            "gt": self.gt,
            "mask_camera": self.mask_camera,
        }


DenseOccRecord = DenseOccPrediction | Mapping[str, Any]


def make_dense_occ_prediction(
    *,
    pred: np.ndarray,
    token: str,
    scene_name: str = "",
    step_idx: int | None = None,
    gt: np.ndarray | None = None,
    mask_camera: np.ndarray | None = None,
) -> dict[str, Any]:
    """构造向后兼容的 dict 预测记录。"""
    return DenseOccPrediction(
        pred=np.asarray(pred).astype(np.uint8, copy=False),
        token=str(token),
        scene_name=str(scene_name),
        step_idx=None if step_idx is None else int(step_idx),
        gt=None if gt is None else np.asarray(gt).astype(np.uint8, copy=False),
        mask_camera=None if mask_camera is None else np.asarray(mask_camera),
    ).to_dict()


def _record_to_dict(record: DenseOccRecord) -> dict[str, Any]:
    if isinstance(record, DenseOccPrediction):
        return record.to_dict()
    return dict(record)


def _to_json_number(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def _validate_shape(name: str, arr: np.ndarray, grid_size: tuple[int, int, int]) -> None:
    if tuple(arr.shape) != grid_size:
        raise ValueError(f"{name} shape={arr.shape} 与 grid_size={grid_size} 不一致")


def attach_occ3d_targets(
    records: list[DenseOccRecord],
    *,
    gt_root: str,
    gt_mask_key: str = "mask_camera",
    grid_size: tuple[int, int, int] = (200, 200, 16),
) -> tuple[list[dict[str, Any]], int]:
    """为预测记录补齐 GT/mask；已有 GT 的记录会直接复用。"""
    attached: list[dict[str, Any]] = []
    missing_gt_count = 0
    for record in records:
        item = _record_to_dict(record)
        pred = np.asarray(item["pred"]).astype(np.uint8, copy=False)
        _validate_shape("pred", pred, grid_size)
        item["pred"] = pred

        gt = item.get("gt", None)
        mask = item.get("mask_camera", None)
        if gt is None:
            scene_name = str(item.get("scene_name", ""))
            token = str(item.get("token", ""))
            gt_path = os.path.join(gt_root, scene_name, token, "labels.npz")
            if not scene_name or not token or not os.path.exists(gt_path):
                missing_gt_count += 1
                continue
            gt_npz = load_labels_npz(gt_path)
            gt = gt_npz["semantics"]
            mask = gt_npz.get(gt_mask_key, np.ones(gt.shape, dtype=np.float32))

        gt_u8 = np.asarray(gt).astype(np.uint8, copy=False)
        _validate_shape("gt", gt_u8, grid_size)
        item["gt"] = gt_u8
        if mask is not None:
            mask_np = np.asarray(mask)
            _validate_shape("mask_camera", mask_np, grid_size)
            item["mask_camera"] = mask_np
        else:
            item["mask_camera"] = None
        attached.append(item)
    return attached, missing_gt_count


def _metric_payload(metric: Any, class_names: list[str]) -> dict[str, Any]:
    if metric.cnt <= 0:
        return {
            "num_keyframes": 0,
            "miou": None,
            "miou_d": None,
            "per_class_iou": [],
            "class_names": class_names,
        }
    miou = float(metric.count_miou(verbose=False))
    miou_d = float(metric.count_miou_d(verbose=False))
    per_class = np.nan_to_num(metric.get_per_class_iou(), nan=0.0).tolist()
    return {
        "num_keyframes": int(metric.cnt),
        "miou": _to_json_number(miou),
        "miou_d": _to_json_number(miou_d),
        "per_class_iou": [float(v) for v in per_class],
        "class_names": class_names,
    }


def compute_dense_miou(
    records: list[DenseOccRecord],
    *,
    num_classes: int,
    use_image_mask: bool = True,
    use_lidar_mask: bool = False,
) -> dict[str, Any]:
    """从内存 dense 预测记录计算 all/per-step mIoU。"""
    metric_all = build_miou_metric(
        num_classes=num_classes,
        use_image_mask=use_image_mask,
        use_lidar_mask=use_lidar_mask,
    )
    per_step_metrics: dict[int, Any] = {}
    class_names = build_miou_metric(num_classes=num_classes).class_names

    for record in records:
        item = _record_to_dict(record)
        if item.get("gt", None) is None:
            continue
        pred = np.asarray(item["pred"])
        gt = np.asarray(item["gt"])
        mask_camera = item.get("mask_camera", None)
        metric_all.add_batch(pred, gt, mask_lidar=None, mask_camera=mask_camera)
        step_idx = item.get("step_idx", None)
        if step_idx is not None:
            step = int(step_idx)
            metric = per_step_metrics.setdefault(
                step,
                build_miou_metric(
                    num_classes=num_classes,
                    use_image_mask=use_image_mask,
                    use_lidar_mask=use_lidar_mask,
                ),
            )
            metric.add_batch(pred, gt, mask_lidar=None, mask_camera=mask_camera)

    return {
        "all": _metric_payload(metric_all, class_names),
        "per_step": {
            str(step): _metric_payload(metric, class_names)
            for step, metric in sorted(per_step_metrics.items())
        },
    }


def compute_dense_rayiou(
    records: list[DenseOccRecord],
    *,
    sweep_pkl: str,
    print_table_all: bool = True,
) -> dict[str, Any]:
    """从内存 dense 预测记录计算 all/per-step RayIoU。"""
    from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
    from online_ncde.ops.dvr.ray_metrics import RayIouAccumulator

    origins_by_token = load_origins_from_sweep_pkl(sweep_pkl)
    per_step_ray: dict[int, Any] = {}
    ray_all = RayIouAccumulator()
    missing_origin_count = 0
    skipped_no_gt_count = 0

    for record in records:
        item = _record_to_dict(record)
        gt = item.get("gt", None)
        if gt is None:
            skipped_no_gt_count += 1
            continue
        token = str(item.get("token", ""))
        origin = origins_by_token.get(token, None)
        if origin is None:
            missing_origin_count += 1
            continue
        pred = np.asarray(item["pred"])
        gt_np = np.asarray(gt)
        ray_all.add_sample(pred, gt_np, origin)
        step_idx = item.get("step_idx", None)
        if step_idx is not None:
            step = int(step_idx)
            ray_acc = per_step_ray.setdefault(step, RayIouAccumulator())
            ray_acc.add_sample(pred, gt_np, origin)

    all_result = ray_all.finalize(print_table=print_table_all) if ray_all.num_samples > 0 else None
    return {
        "all": all_result,
        "per_step": {
            str(step): ray_acc.finalize(print_table=False)
            for step, ray_acc in sorted(per_step_ray.items())
            if ray_acc.num_samples > 0
        },
        "missing_origin_count": int(missing_origin_count),
        "skipped_no_gt_count": int(skipped_no_gt_count),
        "origins_count": int(len(origins_by_token)),
    }


def evaluate_dense_occ(
    records: list[DenseOccRecord],
    *,
    num_classes: int,
    enable_rayiou: bool = False,
    sweep_pkl: str | None = None,
    print_rayiou_table: bool = True,
) -> dict[str, Any]:
    """统一计算 dense occupancy 的 mIoU/RayIoU，并合并为脚本友好的结构。"""
    miou_result = compute_dense_miou(records, num_classes=num_classes)
    ray_result = None
    if enable_rayiou:
        if not sweep_pkl:
            raise ValueError("enable_rayiou=True 时必须提供 sweep_pkl")
        ray_result = compute_dense_rayiou(
            records,
            sweep_pkl=sweep_pkl,
            print_table_all=print_rayiou_table,
        )

    all_payload = dict(miou_result["all"])
    all_payload["rayiou"] = None if ray_result is None else ray_result["all"]

    step_keys = set(miou_result["per_step"].keys())
    if ray_result is not None:
        step_keys.update(ray_result["per_step"].keys())
    per_step = {}
    for step in sorted(step_keys, key=lambda x: int(x)):
        payload = dict(miou_result["per_step"].get(step, {}))
        payload["rayiou"] = None if ray_result is None else ray_result["per_step"].get(step, None)
        per_step[step] = payload

    return {
        "all": all_payload,
        "per_step": per_step,
        "rayiou_enabled": bool(enable_rayiou),
        "rayiou_meta": None if ray_result is None else {
            "missing_origin_count": int(ray_result["missing_origin_count"]),
            "skipped_no_gt_count": int(ray_result["skipped_no_gt_count"]),
            "origins_count": int(ray_result["origins_count"]),
        },
        "num_predictions": int(len(records)),
    }
