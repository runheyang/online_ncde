#!/usr/bin/env python3
"""多线程评估已保存的 SurroundOcc ALOcc dense top-k logits。

默认评估 alocc3d_surroundocc 的 val split：

    conda run -n neural_ode python tests/evoocc/eval_surroundocc_saved_logits_miou.py

仅快速检查前 100 个样本：

    conda run -n neural_ode python tests/evoocc/eval_surroundocc_saved_logits_miou.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from evoocc.config import resolve_path  # noqa: E402
from evoocc.evaluation.dense_occ import (  # noqa: E402
    _build_surroundocc_lidar_filename_index,
    _load_surroundocc_dense_gt,
)
from evoocc.metrics import (  # noqa: E402
    SURROUNDOCC_CLASS_NAMES,
    SURROUNDOCC_DYNAMIC_OBJECT_IDX,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved SurroundOcc ALOcc logits with multithreaded IO."
    )
    parser.add_argument(
        "--canonical-info",
        default="configs/evoocc/canonical_infos_val_full.pkl",
        help="canonical infos pkl，默认 val_full",
    )
    parser.add_argument(
        "--pred-root",
        default="data/alocc3d_surroundocc_logits",
        help="预测 logits 根目录，结构为 <scene>/<sample_token>/logits.npz",
    )
    parser.add_argument(
        "--gt-root",
        default="data/nuscenes/gts_surroundocc",
        help="SurroundOcc GT 根目录",
    )
    parser.add_argument(
        "--nuscenes-root",
        default="data/nuscenes",
        help="nuScenes dataroot，用于 sample_token -> LIDAR_TOP filename 映射",
    )
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--free-index", type=int, default=16)
    parser.add_argument("--grid-size", nargs=3, type=int, default=(200, 200, 16))
    parser.add_argument(
        "--label-id-offset",
        type=int,
        default=-1,
        help="topk_indices 原始 label id 到内部 label id 的偏移，SurroundOcc 默认 1..17 -> 0..16",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="ThreadPoolExecutor worker 数",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只评估前 N 个 valid 样本，0 表示全量",
    )
    parser.add_argument(
        "--min-history-completeness",
        type=int,
        default=0,
        help="可选过滤 history_completeness；0 表示不按历史长度过滤",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="每处理多少个样本打印一次进度",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="可选：保存汇总结果 JSON",
    )
    return parser.parse_args()


def _load_infos(path: str) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    return list(infos)


def _filter_infos(
    infos: list[dict[str, Any]],
    *,
    min_history_completeness: int,
    limit: int,
) -> list[dict[str, Any]]:
    filtered = [info for info in infos if info.get("valid", True)]
    if min_history_completeness > 0:
        filtered = [
            info
            for info in filtered
            if int(info.get("history_completeness", info.get("history_keyframes", 0)))
            >= min_history_completeness
        ]
    if limit > 0:
        filtered = filtered[:limit]
    return filtered


def _load_pred_from_topk(
    pred_path: str,
    *,
    label_id_offset: int,
    num_classes: int,
    grid_size: tuple[int, int, int],
) -> np.ndarray:
    with np.load(pred_path, allow_pickle=False) as data:
        topk_indices = data["topk_indices"]
        if "topk_values" in data.files:
            topk_values = data["topk_values"]
            top1_pos = np.argmax(topk_values, axis=-1, keepdims=True)
            pred = np.take_along_axis(topk_indices, top1_pos, axis=-1)[..., 0]
        else:
            pred = topk_indices[..., 0]

    if tuple(pred.shape) != tuple(grid_size):
        raise ValueError(f"pred shape={pred.shape} 与 grid_size={grid_size} 不一致: {pred_path}")
    pred = pred.astype(np.int64, copy=False) + int(label_id_offset)
    min_pred = int(pred.min())
    max_pred = int(pred.max())
    if min_pred < 0 or max_pred >= num_classes:
        raise ValueError(
            f"pred label 越界: min={min_pred}, max={max_pred}, "
            f"num_classes={num_classes}, path={pred_path}"
        )
    return pred.astype(np.uint8, copy=False)


def _hist_from_pred_gt(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    *,
    num_classes: int,
    free_index: int,
) -> tuple[np.ndarray, int, int]:
    valid = mask.astype(bool)
    pred_v = pred[valid].astype(np.int64, copy=False)
    gt_v = gt[valid].astype(np.int64, copy=False)
    keep = (gt_v >= 0) & (gt_v < num_classes)
    pred_v = pred_v[keep]
    gt_v = gt_v[keep]
    hist = np.bincount(
        num_classes * gt_v + pred_v,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    pred_occ = pred_v != free_index
    gt_occ = gt_v != free_index
    occ_inter = int(np.logical_and(pred_occ, gt_occ).sum())
    occ_union = int(np.logical_or(pred_occ, gt_occ).sum())
    return hist.astype(np.float64, copy=False), occ_inter, occ_union


def _eval_one(
    info: dict[str, Any],
    *,
    pred_root: str,
    gt_root: str,
    lidar_index: dict[str, str],
    grid_size: tuple[int, int, int],
    num_classes: int,
    free_index: int,
    label_id_offset: int,
) -> dict[str, Any]:
    scene_name = str(info.get("scene_name", ""))
    token = str(info.get("token", ""))
    if not scene_name or not token:
        return {"status": "bad_info", "token": token}

    pred_path = os.path.join(pred_root, scene_name, token, "logits.npz")
    if not os.path.exists(pred_path):
        return {"status": "missing_pred", "token": token, "path": pred_path}

    lidar_filename = lidar_index.get(token, "")
    if not lidar_filename:
        return {"status": "missing_lidar_index", "token": token}

    try:
        pred = _load_pred_from_topk(
            pred_path,
            label_id_offset=label_id_offset,
            num_classes=num_classes,
            grid_size=grid_size,
        )
        gt, mask = _load_surroundocc_dense_gt(
            gt_root=gt_root,
            lidar_filename=lidar_filename,
            grid_size=grid_size,
        )
        hist, occ_inter, occ_union = _hist_from_pred_gt(
            pred,
            gt,
            mask,
            num_classes=num_classes,
            free_index=free_index,
        )
    except FileNotFoundError as exc:
        return {"status": "missing_gt", "token": token, "path": str(exc)}
    except Exception as exc:  # noqa: BLE001 - 评估脚本需要不中断地统计坏样本
        return {"status": "error", "token": token, "error": repr(exc)}

    return {
        "status": "ok",
        "hist": hist,
        "occ_inter": occ_inter,
        "occ_union": occ_union,
    }


def _per_class_iou(hist: np.ndarray) -> np.ndarray:
    denom = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    iou = np.full(hist.shape[0], np.nan, dtype=np.float64)
    valid = denom > 0
    iou[valid] = np.diag(hist)[valid] / denom[valid]
    return iou


def _json_number(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def main() -> None:
    args = parse_args()
    root_path = str(ROOT)
    canonical_info = resolve_path(root_path, args.canonical_info)
    pred_root = resolve_path(root_path, args.pred_root)
    gt_root = resolve_path(root_path, args.gt_root)
    nuscenes_root = resolve_path(root_path, args.nuscenes_root)
    grid_size = tuple(int(v) for v in args.grid_size)

    infos = _load_infos(canonical_info)
    eval_infos = _filter_infos(
        infos,
        min_history_completeness=int(args.min_history_completeness),
        limit=int(args.limit),
    )
    print(f"[infos] total={len(infos)} eval={len(eval_infos)}")
    print(f"[pred] {pred_root}")
    print(f"[gt] {gt_root}")
    print(f"[workers] threads={args.workers}")

    lidar_index = _build_surroundocc_lidar_filename_index(
        nuscenes_root=nuscenes_root,
        nuscenes_version=str(args.nuscenes_version),
    )

    hist_total = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    counts = {
        "ok": 0,
        "missing_pred": 0,
        "missing_gt": 0,
        "missing_lidar_index": 0,
        "bad_info": 0,
        "error": 0,
    }
    occ_inter_total = 0
    occ_union_total = 0
    first_errors: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
        futures = [
            pool.submit(
                _eval_one,
                info,
                pred_root=pred_root,
                gt_root=gt_root,
                lidar_index=lidar_index,
                grid_size=grid_size,
                num_classes=int(args.num_classes),
                free_index=int(args.free_index),
                label_id_offset=int(args.label_id_offset),
            )
            for info in eval_infos
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = str(result.get("status", "error"))
            counts[status] = counts.get(status, 0) + 1
            if status == "ok":
                hist_total += result["hist"]
                occ_inter_total += int(result["occ_inter"])
                occ_union_total += int(result["occ_union"])
            elif len(first_errors) < 10:
                first_errors.append(result)
            if args.log_interval > 0 and (idx % args.log_interval == 0 or idx == len(futures)):
                print(
                    f"[progress] {idx}/{len(futures)} "
                    f"ok={counts.get('ok', 0)} missing_pred={counts.get('missing_pred', 0)} "
                    f"missing_gt={counts.get('missing_gt', 0)} error={counts.get('error', 0)}"
                )

    class_iou = _per_class_iou(hist_total) * 100.0
    nonfree_indices = [i for i in range(args.num_classes) if i != args.free_index]
    dynamic_indices = [
        int(i)
        for i in SURROUNDOCC_DYNAMIC_OBJECT_IDX
        if 0 <= int(i) < args.num_classes and int(i) != args.free_index
    ]
    miou = float(np.nanmean(class_iou[nonfree_indices]))
    miou_d = float(np.nanmean(class_iou[dynamic_indices])) if dynamic_indices else float("nan")
    occupied_iou = (
        float(occ_inter_total / occ_union_total * 100.0)
        if occ_union_total > 0
        else float("nan")
    )

    print("\n===> SurroundOcc saved logits evaluation")
    print(f"===> evaluated = {counts.get('ok', 0)} / {len(eval_infos)}")
    print(f"===> missing_pred = {counts.get('missing_pred', 0)}")
    print(f"===> missing_gt = {counts.get('missing_gt', 0)}")
    print(f"===> missing_lidar_index = {counts.get('missing_lidar_index', 0)}")
    print(f"===> error = {counts.get('error', 0)}")
    print(f"===> mIoU = {round(miou, 4)}")
    print(f"===> mIoU_D = {round(miou_d, 4)}")
    print(f"===> occupied_iou = {round(occupied_iou, 4)}")
    print("===> per class IoU:")
    for idx, name in enumerate(SURROUNDOCC_CLASS_NAMES[: args.num_classes]):
        value = class_iou[idx]
        text = "nan" if not np.isfinite(value) else str(round(float(value), 4))
        print(f"===> {name} - IoU = {text}")

    if first_errors:
        print("\n[first bad samples]")
        for item in first_errors:
            print(item)

    if args.output_json:
        output_json = resolve_path(root_path, args.output_json)
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        payload = {
            "canonical_info": canonical_info,
            "pred_root": pred_root,
            "gt_root": gt_root,
            "num_requested": len(eval_infos),
            "counts": {k: int(v) for k, v in counts.items()},
            "mIoU": _json_number(miou),
            "mIoU_D": _json_number(miou_d),
            "occupied_iou": _json_number(occupied_iou),
            "class_names": SURROUNDOCC_CLASS_NAMES[: args.num_classes],
            "per_class_iou": [_json_number(float(v)) for v in class_iou],
            "first_errors": first_errors,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[json] saved -> {output_json}")


if __name__ == "__main__":
    main()
