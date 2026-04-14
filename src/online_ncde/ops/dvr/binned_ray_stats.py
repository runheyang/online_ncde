#!/usr/bin/env python3
"""分箱 ray 统计：从 raycasting pcd 数据计算 0-10m / 10-20m / 20-40m 的
RayIoU、hit/no-hit 四格表、深度误差统计。

设计为与 ray_metrics.main(return_pcds=True) 配合使用，避免二次 raycasting。
"""

from __future__ import annotations

import numpy as np

# 与 ray_metrics.py 保持一致
_NUM_CLASSES = 18
_FREE_ID = 17
_THRESHOLDS = [1, 2, 4]
_BINS = {"0-10m": (0, 10), "10-20m": (10, 20), "20-40m": (20, 40)}


def compute_binned_ray_stats(
    raw_pcd_pred_list: list[np.ndarray],
    raw_pcd_gt_list: list[np.ndarray],
) -> dict:
    """从未过滤的 per-sample pcd 数据计算分箱统计。

    Args:
        raw_pcd_pred_list: 每个元素 shape (N_rays, 2)，列 = [class_id, distance]
        raw_pcd_gt_list: 同上，GT 数据

    Returns:
        dict，键为 "all" / "0-10m" / "10-20m" / "20-40m"，值为指标字典。
    """
    bin_keys = list(_BINS.keys()) + ["all"]
    # 初始化累加器
    acc = {}
    for bk in bin_keys:
        acc[bk] = _new_accumulator()

    for pcd_pred, pcd_gt in zip(raw_pcd_pred_list, raw_pcd_gt_list):
        gt_class = pcd_gt[:, 0].astype(np.int32)
        gt_dist = pcd_gt[:, 1]
        pred_class = pcd_pred[:, 0].astype(np.int32)
        pred_dist = pcd_pred[:, 1]

        gt_hit = gt_class != _FREE_ID
        pred_hit = pred_class != _FREE_ID

        # 对每个 bin 累加
        for bk in bin_keys:
            if bk == "all":
                mask = np.ones(len(gt_class), dtype=bool)
            else:
                lo, hi = _BINS[bk]
                # 用 GT distance 分桶（保证 fast/aligned 对比同批 ray）
                mask = (gt_dist >= lo) & (gt_dist < hi)
            if not mask.any():
                continue
            _accumulate(acc[bk], gt_class[mask], gt_dist[mask],
                        pred_class[mask], pred_dist[mask],
                        gt_hit[mask], pred_hit[mask])

    # 汇总
    return {bk: _summarize(acc[bk]) for bk in bin_keys}


def _new_accumulator() -> dict:
    return {
        "gt_cnt": np.zeros(_NUM_CLASSES),
        "pred_cnt": np.zeros(_NUM_CLASSES),
        "tp_cnt": np.zeros([len(_THRESHOLDS), _NUM_CLASSES]),
        "total_rays": 0,
        # hit/no-hit 四格表
        "gt_hit_pred_hit": 0,
        "gt_hit_pred_nohit": 0,
        "gt_nohit_pred_hit": 0,
        "gt_nohit_pred_nohit": 0,
        # 深度误差（仅 GT-hit ray）
        "n_gt_hit": 0,
        "signed_err_sum": 0.0,
        "abs_err_sum": 0.0,
        "closer_count": 0,
        "farther_count": 0,
    }


def _accumulate(
    s: dict,
    gt_class: np.ndarray,
    gt_dist: np.ndarray,
    pred_class: np.ndarray,
    pred_dist: np.ndarray,
    gt_hit: np.ndarray,
    pred_hit: np.ndarray,
) -> None:
    n = len(gt_class)
    s["total_rays"] += n

    # hit/no-hit 四格表
    s["gt_hit_pred_hit"] += int((gt_hit & pred_hit).sum())
    s["gt_hit_pred_nohit"] += int((gt_hit & ~pred_hit).sum())
    s["gt_nohit_pred_hit"] += int((~gt_hit & pred_hit).sum())
    s["gt_nohit_pred_nohit"] += int((~gt_hit & ~pred_hit).sum())

    # 仅对 GT-hit ray 计算深度误差和 RayIoU
    if not gt_hit.any():
        return
    gc = gt_class[gt_hit]
    gd = gt_dist[gt_hit]
    pc = pred_class[gt_hit]
    pd = pred_dist[gt_hit]
    n_hit = len(gc)
    s["n_gt_hit"] += n_hit

    signed_err = pd - gd
    s["signed_err_sum"] += float(signed_err.sum())
    s["abs_err_sum"] += float(np.abs(signed_err).sum())
    s["closer_count"] += int((signed_err < 0).sum())
    s["farther_count"] += int((signed_err > 0).sum())

    # RayIoU 累加
    s["gt_cnt"] += np.bincount(gc, minlength=_NUM_CLASSES).astype(float)
    s["pred_cnt"] += np.bincount(pc, minlength=_NUM_CLASSES).astype(float)
    abs_err = np.abs(pd - gd)
    for j, t in enumerate(_THRESHOLDS):
        tp_mask = (pc == gc) & (abs_err < t)
        s["tp_cnt"][j] += np.bincount(gc[tp_mask], minlength=_NUM_CLASSES).astype(float)


def _summarize(s: dict) -> dict:
    """将累加器转为最终指标字典。"""
    n_hit = max(s["n_gt_hit"], 1)
    total = max(s["total_rays"], 1)

    # 分阈值 RayIoU（排除 free 类）
    iou_per_thresh = []
    for j in range(len(_THRESHOLDS)):
        denom = s["gt_cnt"] + s["pred_cnt"] - s["tp_cnt"][j]
        iou = s["tp_cnt"][j] / np.maximum(denom, 1)
        iou_per_thresh.append(float(np.nanmean(iou[:_FREE_ID])))

    gt_hit_total = s["gt_hit_pred_hit"] + s["gt_hit_pred_nohit"]
    gt_nohit_total = s["gt_nohit_pred_hit"] + s["gt_nohit_pred_nohit"]

    return {
        "total_rays": s["total_rays"],
        # RayIoU
        "RayIoU@1": iou_per_thresh[0],
        "RayIoU@2": iou_per_thresh[1],
        "RayIoU@4": iou_per_thresh[2],
        "RayIoU": float(np.mean(iou_per_thresh)),
        # 深度误差（仅 GT-hit ray）
        "mean_signed_err": s["signed_err_sum"] / n_hit,
        "mean_abs_err": s["abs_err_sum"] / n_hit,
        "closer_ratio": s["closer_count"] / n_hit,
        "farther_ratio": s["farther_count"] / n_hit,
        # hit/no-hit 四格表
        "gt_hit_pred_hit": s["gt_hit_pred_hit"],
        "gt_hit_pred_nohit": s["gt_hit_pred_nohit"],
        "gt_nohit_pred_hit": s["gt_nohit_pred_hit"],
        "gt_nohit_pred_nohit": s["gt_nohit_pred_nohit"],
        "miss_rate": s["gt_hit_pred_nohit"] / max(gt_hit_total, 1),
        "false_hit_rate": s["gt_nohit_pred_hit"] / max(gt_nohit_total, 1),
    }
