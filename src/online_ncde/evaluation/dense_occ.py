"""统一的 dense occupancy 内存评估协议。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Union

import numpy as np

from online_ncde.data.labels_io import load_labels_npz
from online_ncde.metrics import build_miou_metric


_SURROUNDOCC_LIDAR_INDEX_CACHE: dict[tuple[str, str], dict[str, str]] = {}


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


DenseOccRecord = Union[DenseOccPrediction, Mapping[str, Any]]


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


def _finalize_attached_item(
    item: dict[str, Any],
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None,
    grid_size: tuple[int, int, int],
) -> dict[str, Any]:
    gt_u8 = np.asarray(gt).astype(np.uint8, copy=False)
    _validate_shape("gt", gt_u8, grid_size)
    item["pred"] = pred
    item["gt"] = gt_u8
    if mask is not None:
        mask_np = np.asarray(mask)
        _validate_shape("mask_camera", mask_np, grid_size)
        item["mask_camera"] = mask_np
    else:
        item["mask_camera"] = None
    return item


def attach_occ3d_targets(
    records: list[DenseOccRecord],
    *,
    gt_root: str,
    gt_mask_key: str = "mask_camera",
    grid_size: tuple[int, int, int] = (200, 200, 16),
    gt_cache: dict[tuple[str, str], tuple[np.ndarray | None, np.ndarray | None]] | None = None,
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
            cache_key = (scene_name, token)
            if gt_cache is not None and cache_key in gt_cache:
                gt, mask = gt_cache[cache_key]
            elif not scene_name or not token:
                missing_gt_count += 1
                continue
            else:
                gt_path = os.path.join(gt_root, scene_name, token, "labels.npz")
                if not os.path.exists(gt_path):
                    if gt_cache is not None:
                        gt_cache[cache_key] = (None, None)
                    missing_gt_count += 1
                    continue
                gt_npz = load_labels_npz(gt_path)
                gt = gt_npz["semantics"]
                mask = gt_npz.get(gt_mask_key, np.ones(gt.shape, dtype=np.float32))
                if gt_cache is not None:
                    gt_cache[cache_key] = (gt, mask)
            if gt is None:
                missing_gt_count += 1
                continue

        attached.append(
            _finalize_attached_item(
                item,
                pred=pred,
                gt=np.asarray(gt),
                mask=None if mask is None else np.asarray(mask),
                grid_size=grid_size,
            )
        )
    return attached, missing_gt_count


def _build_surroundocc_lidar_filename_index(
    *,
    nuscenes_root: str,
    nuscenes_version: str,
) -> dict[str, str]:
    """建立 sample_token -> LIDAR_TOP 文件名索引。"""
    cache_key = (os.path.abspath(nuscenes_root), str(nuscenes_version))
    cached = _SURROUNDOCC_LIDAR_INDEX_CACHE.get(cache_key, None)
    if cached is not None:
        return cached

    table_dir = os.path.join(nuscenes_root, nuscenes_version)
    sample_path = os.path.join(table_dir, "sample.json")
    sample_data_path = os.path.join(table_dir, "sample_data.json")
    calibrated_sensor_path = os.path.join(table_dir, "calibrated_sensor.json")
    sensor_path = os.path.join(table_dir, "sensor.json")
    for table_path in (sample_path, sample_data_path, calibrated_sensor_path, sensor_path):
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"缺少 nuScenes table: {table_path}")

    with open(sample_path, "r", encoding="utf-8") as f:
        sample = json.load(f)
    with open(sample_data_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)
    with open(calibrated_sensor_path, "r", encoding="utf-8") as f:
        calibrated_sensor = json.load(f)
    with open(sensor_path, "r", encoding="utf-8") as f:
        sensor = json.load(f)

    sensor_by_token = {str(rec["token"]): rec for rec in sensor}
    calibrated_by_token = {str(rec["token"]): rec for rec in calibrated_sensor}
    sample_ts_by_token = {str(rec["token"]): int(rec["timestamp"]) for rec in sample}

    candidates: dict[str, list[tuple[int, bool, str]]] = {}
    for rec in sample_data:
        calibrated = calibrated_by_token.get(str(rec.get("calibrated_sensor_token", "")), {})
        sensor_rec = sensor_by_token.get(str(calibrated.get("sensor_token", "")), {})
        if sensor_rec.get("channel") != "LIDAR_TOP":
            continue
        sample_token = str(rec.get("sample_token", ""))
        filename = str(rec.get("filename", ""))
        if not sample_token or not filename:
            continue
        target_ts = sample_ts_by_token.get(sample_token, int(rec.get("timestamp", 0)))
        dt = abs(int(rec.get("timestamp", 0)) - int(target_ts))
        candidates.setdefault(sample_token, []).append(
            (dt, bool(rec.get("is_key_frame", False)), os.path.basename(filename))
        )

    index: dict[str, str] = {}
    for sample_token, items in candidates.items():
        items_sorted = sorted(items, key=lambda item: (not item[1], item[0]))
        index[sample_token] = items_sorted[0][2]
    if not index:
        raise RuntimeError(f"未能从 {sample_data_path} 建立 LIDAR_TOP 索引")
    _SURROUNDOCC_LIDAR_INDEX_CACHE[cache_key] = index
    return index


def _load_surroundocc_dense_gt(
    *,
    gt_root: str,
    lidar_filename: str,
    grid_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    gt_path = os.path.join(gt_root, "samples", f"{lidar_filename}.npy")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(gt_path)
    sparse = np.load(gt_path)
    if sparse.ndim != 2 or sparse.shape[1] < 4:
        raise ValueError(f"SurroundOcc GT 形状异常: {gt_path}, shape={sparse.shape}")

    coords = sparse[:, :3].astype(np.int64, copy=False)
    labels = sparse[:, 3].astype(np.uint8, copy=False)
    bounds = np.asarray(grid_size, dtype=np.int64)
    valid = np.all((coords >= 0) & (coords < bounds), axis=1)
    if not bool(np.all(valid)):
        bad_count = int((~valid).sum())
        raise ValueError(f"SurroundOcc GT 坐标越界: {gt_path}, bad={bad_count}")

    occ = np.zeros(grid_size, dtype=np.uint8)
    occ[coords[:, 0], coords[:, 1], coords[:, 2]] = labels
    # 对齐 OccStudio: dense 中 label 0 视为 free，再压缩 1..17 -> 0..16。
    occ[occ == 0] = 17
    occ = occ - 1
    mask = np.ones(grid_size, dtype=np.float32)
    return occ.astype(np.uint8, copy=False), mask


def attach_surroundocc_targets(
    records: list[DenseOccRecord],
    *,
    gt_root: str,
    nuscenes_root: str,
    nuscenes_version: str = "v1.0-trainval",
    grid_size: tuple[int, int, int] = (200, 200, 16),
    gt_cache: dict[tuple[str, str], tuple[np.ndarray | None, np.ndarray | None]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """为预测记录补齐 SurroundOcc GT/mask；record token 必须是 sample_token。"""
    lidar_index = _build_surroundocc_lidar_filename_index(
        nuscenes_root=nuscenes_root,
        nuscenes_version=nuscenes_version,
    )
    attached: list[dict[str, Any]] = []
    missing_gt_count = 0
    for record in records:
        item = _record_to_dict(record)
        pred = np.asarray(item["pred"]).astype(np.uint8, copy=False)
        _validate_shape("pred", pred, grid_size)

        gt = item.get("gt", None)
        mask = item.get("mask_camera", None)
        if gt is None:
            token = str(item.get("token", ""))
            cache_key = ("surroundocc", token)
            if gt_cache is not None and cache_key in gt_cache:
                gt, mask = gt_cache[cache_key]
            elif not token:
                missing_gt_count += 1
                continue
            else:
                lidar_filename = lidar_index.get(token, "")
                if not lidar_filename:
                    if gt_cache is not None:
                        gt_cache[cache_key] = (None, None)
                    missing_gt_count += 1
                    continue
                try:
                    gt, mask = _load_surroundocc_dense_gt(
                        gt_root=gt_root,
                        lidar_filename=lidar_filename,
                        grid_size=grid_size,
                    )
                except FileNotFoundError:
                    if gt_cache is not None:
                        gt_cache[cache_key] = (None, None)
                    missing_gt_count += 1
                    continue
                if gt_cache is not None:
                    gt_cache[cache_key] = (gt, mask)
            if gt is None:
                missing_gt_count += 1
                continue

        attached.append(
            _finalize_attached_item(
                item,
                pred=pred,
                gt=np.asarray(gt),
                mask=None if mask is None else np.asarray(mask),
                grid_size=grid_size,
            )
        )
    return attached, missing_gt_count


def attach_dense_occ_targets(
    records: list[DenseOccRecord],
    *,
    dataset_variant: str,
    gt_root: str,
    gt_mask_key: str = "mask_camera",
    grid_size: tuple[int, int, int] = (200, 200, 16),
    nuscenes_root: str | None = None,
    nuscenes_version: str = "v1.0-trainval",
    gt_cache: dict[tuple[str, str], tuple[np.ndarray | None, np.ndarray | None]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """按数据集类型为 dense 预测补齐 GT。"""
    variant = str(dataset_variant).strip().lower()
    if variant == "occ3d":
        return attach_occ3d_targets(
            records,
            gt_root=gt_root,
            gt_mask_key=gt_mask_key,
            grid_size=grid_size,
            gt_cache=gt_cache,
        )
    if variant == "surroundocc":
        if not nuscenes_root:
            raise ValueError("attach_dense_occ_targets(surroundocc) 需要 nuscenes_root")
        return attach_surroundocc_targets(
            records,
            gt_root=gt_root,
            nuscenes_root=nuscenes_root,
            nuscenes_version=nuscenes_version,
            grid_size=grid_size,
            gt_cache=gt_cache,
        )
    raise ValueError(f"未知 dataset_variant: {dataset_variant!r}")


def _metric_payload(metric: Any, class_names: list[str]) -> dict[str, Any]:
    if metric.cnt <= 0:
        payload = {
            "num_keyframes": 0,
            "miou": None,
            "miou_d": None,
            "per_class_iou": [],
            "class_names": class_names,
        }
        if hasattr(metric, "count_occupied_iou"):
            payload["occupied_iou"] = None
        return payload
    miou = float(metric.count_miou(verbose=False))
    miou_d = float(metric.count_miou_d(verbose=False))
    per_class = np.nan_to_num(metric.get_per_class_iou(), nan=0.0).tolist()
    payload = {
        "num_keyframes": int(metric.cnt),
        "miou": _to_json_number(miou),
        "miou_d": _to_json_number(miou_d),
        "per_class_iou": [float(v) for v in per_class],
        "class_names": class_names,
    }
    if hasattr(metric, "count_occupied_iou"):
        occupied_iou = float(metric.count_occupied_iou(verbose=False))
        payload["occupied_iou"] = _to_json_number(occupied_iou)
    return payload


def compute_dense_miou(
    records: list[DenseOccRecord],
    *,
    num_classes: int,
    use_image_mask: bool = True,
    use_lidar_mask: bool = False,
    metric_variant: str = "occ3d",
) -> dict[str, Any]:
    """从内存 dense 预测记录计算 all/per-step mIoU。"""
    metric_all = build_miou_metric(
        num_classes=num_classes,
        use_image_mask=use_image_mask,
        use_lidar_mask=use_lidar_mask,
        variant=metric_variant,
    )
    per_step_metrics: dict[int, Any] = {}
    class_names = build_miou_metric(
        num_classes=num_classes,
        variant=metric_variant,
    ).class_names

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
                    variant=metric_variant,
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


def _load_or_use_origins(
    *,
    sweep_pkl: str | None,
    origins_by_token: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if origins_by_token is not None:
        return origins_by_token
    if not sweep_pkl:
        raise ValueError("计算 RayIoU 时必须提供 sweep_pkl 或 origins_by_token")
    from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl

    return load_origins_from_sweep_pkl(sweep_pkl)


def _prepare_rayiou_items(
    records: list[DenseOccRecord],
    *,
    sweep_pkl: str | None,
    origins_by_token: Mapping[str, Any] | None = None,
) -> tuple[list[tuple[dict[str, Any], np.ndarray, np.ndarray, Any]], dict[str, int]]:
    origins = _load_or_use_origins(sweep_pkl=sweep_pkl, origins_by_token=origins_by_token)
    ray_items: list[tuple[dict[str, Any], np.ndarray, np.ndarray, Any]] = []
    missing_origin_count = 0
    skipped_no_gt_count = 0

    for record in records:
        item = _record_to_dict(record)
        gt = item.get("gt", None)
        if gt is None:
            skipped_no_gt_count += 1
            continue
        token = str(item.get("token", ""))
        origin = origins.get(token, None)
        if origin is None:
            missing_origin_count += 1
            continue
        ray_items.append((item, np.asarray(item["pred"]), np.asarray(gt), origin))

    return ray_items, {
        "missing_origin_count": int(missing_origin_count),
        "skipped_no_gt_count": int(skipped_no_gt_count),
        "origins_count": int(len(origins)),
    }


def compute_dense_rayiou(
    records: list[DenseOccRecord],
    *,
    sweep_pkl: str | None = None,
    origins_by_token: Mapping[str, Any] | None = None,
    print_table_all: bool = True,
) -> dict[str, Any]:
    """从内存 dense 预测记录计算 all/per-step RayIoU。"""
    from online_ncde.ops.dvr.ray_metrics import RayIouAccumulator

    ray_items, meta = _prepare_rayiou_items(
        records,
        sweep_pkl=sweep_pkl,
        origins_by_token=origins_by_token,
    )
    per_step_ray: dict[int, Any] = {}
    ray_all = RayIouAccumulator()

    for item, pred, gt_np, origin in ray_items:
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
        **meta,
    }


def compute_dense_rayiou_with_pcds(
    records: list[DenseOccRecord],
    *,
    sweep_pkl: str | None = None,
    origins_by_token: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[Any], list[Any], dict[str, int]]:
    """计算 RayIoU 并返回 raw pcd，供分箱 Ray 统计复用。"""
    from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou

    ray_items, meta = _prepare_rayiou_items(
        records,
        sweep_pkl=sweep_pkl,
        origins_by_token=origins_by_token,
    )
    if not ray_items:
        return None, [], [], meta

    sem_pred_list = [pred for _, pred, _, _ in ray_items]
    sem_gt_list = [gt for _, _, gt, _ in ray_items]
    lidar_origin_list = [origin for _, _, _, origin in ray_items]
    rayiou_result, raw_pcd_pred, raw_pcd_gt = calc_rayiou(
        sem_pred_list,
        sem_gt_list,
        lidar_origin_list,
        return_pcds=True,
    )
    rayiou_result = dict(rayiou_result)
    rayiou_result.setdefault("num_samples", int(len(ray_items)))
    return rayiou_result, raw_pcd_pred, raw_pcd_gt, meta


def evaluate_dense_occ(
    records: list[DenseOccRecord],
    *,
    num_classes: int,
    metric_variant: str = "occ3d",
    enable_rayiou: bool = False,
    sweep_pkl: str | None = None,
    origins_by_token: Mapping[str, Any] | None = None,
    print_rayiou_table: bool = True,
) -> dict[str, Any]:
    """统一计算 dense occupancy 的 mIoU/RayIoU，并合并为脚本友好的结构。"""
    miou_result = compute_dense_miou(
        records,
        num_classes=num_classes,
        metric_variant=metric_variant,
    )
    ray_result = None
    if enable_rayiou:
        ray_result = compute_dense_rayiou(
            records,
            sweep_pkl=sweep_pkl,
            origins_by_token=origins_by_token,
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
