#!/usr/bin/env python3
"""评估 ALOCC 预测 logits 的 mIoU：基于 canonical_infos_val.pkl 的最后一帧 (step 12) keyframe。

用法示例：
    # 评估 alocc3d（sample_token 组织）
    python tests/online_ncde/eval_alocc_logits_miou.py \
        --pred-root data/alocc3d \
        --token-type sample

    # 评估 alocc2d_mini（frame_token 组织）
    python tests/online_ncde/eval_alocc_logits_miou.py \
        --pred-root data/alocc2d_mini \
        --token-type frame

    # 使用 lidar mask
    python tests/online_ncde/eval_alocc_logits_miou.py \
        --pred-root data/alocc3d \
        --token-type sample \
        --mask-type lidar
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 ALOCC logits (top-3) 在 canonical_infos_val 最后一帧上的 mIoU"
    )
    parser.add_argument(
        "--canonical-info",
        default=str(ROOT / "configs/online_ncde/canonical_infos_val.pkl"),
        help="canonical_infos_val.pkl 路径",
    )
    parser.add_argument(
        "--pred-root",
        required=True,
        help="ALOCC logits 根目录，例如 data/alocc3d 或 data/alocc2d_mini",
    )
    parser.add_argument(
        "--token-type",
        choices=("sample", "frame"),
        required=True,
        help="logits 目录使用的 token 类型：sample=sample_token, frame=frame_token",
    )
    parser.add_argument(
        "--gt-root",
        default=str(ROOT / "data/nuscenes/gts"),
        help="GT labels 根目录",
    )
    parser.add_argument(
        "--mask-type",
        choices=("camera", "lidar"),
        default="camera",
        help="评估时使用的 mask 类型（默认 camera）",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=18,
        help="类别数（默认 18）",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="可选：将结果写入 json 文件",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评估前 N 个有效样本，0 表示全量",
    )
    return parser.parse_args()


def topk_to_pred(topk_indices: np.ndarray) -> np.ndarray:
    """将 top-3 logits 转为语义预测：取 top-1 类别。

    Args:
        topk_indices: (X, Y, Z, 3) uint8, top-3 类别 id

    Returns:
        pred: (X, Y, Z) uint8, 语义预测
    """
    return topk_indices[..., 0]


def main() -> None:
    args = parse_args()
    root_path = str(ROOT)

    # 解析 pred_root 为绝对路径
    pred_root = args.pred_root
    if not os.path.isabs(pred_root):
        pred_root = os.path.join(root_path, pred_root)

    gt_root = args.gt_root
    if not os.path.isabs(gt_root):
        gt_root = os.path.join(root_path, gt_root)

    # 加载 canonical info
    with open(args.canonical_info, "rb") as f:
        canonical_data = pickle.load(f)
    infos = canonical_data["infos"]

    # 筛选 valid 样本
    valid_infos = [info for info in infos if info.get("valid", False)]
    print(f"总样本数: {len(infos)}, 有效样本数: {len(valid_infos)}")

    if args.limit > 0:
        valid_infos = valid_infos[: args.limit]
        print(f"限制评估前 {args.limit} 个有效样本")

    # 初始化 metric
    use_lidar = args.mask_type == "lidar"
    use_camera = args.mask_type == "camera"
    metric = MetricMiouOcc3D(
        num_classes=args.num_classes,
        use_lidar_mask=use_lidar,
        use_image_mask=use_camera,
    )

    mask_key = "mask_lidar" if use_lidar else "mask_camera"
    missing_pred = 0
    missing_gt = 0
    evaluated = 0

    iterator = tqdm(valid_infos, desc="Evaluating") if tqdm else valid_infos
    for info in iterator:
        scene_name = info["scene_name"]

        # 确定预测 logits 路径
        if args.token_type == "sample":
            pred_token = info["token"]  # sample_token
        else:
            pred_token = info["frame_tokens"][12]  # frame_token (step 12)

        pred_path = os.path.join(pred_root, scene_name, pred_token, "logits.npz")
        if not os.path.exists(pred_path):
            missing_pred += 1
            continue

        # GT 路径：使用 curr_gt_rel_path 或 sample_token
        gt_rel = info.get("curr_gt_rel_path", "")
        if gt_rel:
            gt_path = os.path.join(gt_root, gt_rel) if not os.path.isabs(gt_rel) else gt_rel
        else:
            gt_path = os.path.join(gt_root, scene_name, info["token"], "labels.npz")

        if not os.path.exists(gt_path):
            missing_gt += 1
            continue

        # 加载预测
        pred_data = np.load(pred_path)
        topk_indices = pred_data["topk_indices"]  # (200, 200, 16, 3) uint8
        pred_sem = topk_to_pred(topk_indices)  # (200, 200, 16)

        # 加载 GT
        gt_data = np.load(gt_path)
        gt_semantics = gt_data["semantics"].astype(np.uint8)  # (200, 200, 16)
        mask_lidar = gt_data.get("mask_lidar", None)
        mask_camera = gt_data.get("mask_camera", None)

        # 累计统计
        metric.add_batch(
            semantics_pred=pred_sem,
            semantics_gt=gt_semantics,
            mask_lidar=mask_lidar,
            mask_camera=mask_camera,
        )
        evaluated += 1

    # 输出结果
    print(f"\n评估完成: {evaluated} 个样本")
    print(f"缺失预测: {missing_pred}, 缺失 GT: {missing_gt}")
    print(f"Mask 类型: {args.mask_type}")
    print()

    miou = metric.count_miou(verbose=True)

    # 可选输出 json
    if args.output_json:
        per_class_iou = metric.get_per_class_iou()
        result = {
            "pred_root": pred_root,
            "token_type": args.token_type,
            "mask_type": args.mask_type,
            "num_evaluated": evaluated,
            "num_missing_pred": missing_pred,
            "num_missing_gt": missing_gt,
            "mIoU": miou,
            "per_class_iou": {
                name: float(per_class_iou[i])
                for i, name in enumerate(metric.class_names)
            },
        }
        output_path = args.output_json
        if not os.path.isabs(output_path):
            output_path = os.path.join(root_path, output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
