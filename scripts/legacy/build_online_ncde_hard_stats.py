#!/usr/bin/env python3
"""离线统计 Online NCDE hard 过采样所需样本信息。"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402

HARD_CLASS_IDS = (8, 7, 6, 3)  # traffic_cone, pedestrian, motorcycle, bus
HARD_VOXEL_THRESHOLD = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线统计 hard 过采样样本。")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/online_ncde/fast_opusv1t__slow_opusv2l/train.yaml"),
        help="配置文件路径（用于读取 root_path/data.info_path/data.gt_root）。",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "configs/online_ncde/hard_oversample_train_stats.json"),
        help="输出 JSON 路径。",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="每处理多少个样本打印一次进度。",
    )
    return parser.parse_args()


def sample_key_from_info(info: dict) -> str:
    scene_name = str(info.get("scene_name", "")).strip()
    token = str(info.get("token", "")).strip()
    if not scene_name or not token:
        raise KeyError("样本缺少 scene_name 或 token。")
    return f"{scene_name}/{token}"


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    root_path = cfg["root_path"]
    data_cfg = cfg["data"]

    info_path = resolve_path(root_path, data_cfg["info_path"])
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(info_path, "rb") as f:
        payload = pickle.load(f)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    infos = [info for info in infos if info.get("valid", True)]

    hard_voxel_counts: dict[str, int] = {}
    hard_count = 0
    total = len(infos)
    for idx, info in enumerate(infos, start=1):
        key = sample_key_from_info(info)
        scene_name, token = key.split("/", maxsplit=1)
        gt_path = os.path.join(gt_root, scene_name, token, "labels.npz")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT 文件不存在: {gt_path}")
        with np.load(gt_path, allow_pickle=False) as data:
            semantics = data["semantics"]
        hard_voxels = int(np.isin(semantics, HARD_CLASS_IDS).sum())
        hard_voxel_counts[key] = hard_voxels
        hard_count += int(hard_voxels >= HARD_VOXEL_THRESHOLD)

        if args.log_interval > 0 and idx % args.log_interval == 0:
            print(f"[build-hard-stats] progress: {idx}/{total}")

    output_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(Path(args.config).expanduser().resolve()),
        "info_path": info_path,
        "gt_root": gt_root,
        "hard_class_ids": list(HARD_CLASS_IDS),
        "hard_voxel_threshold": HARD_VOXEL_THRESHOLD,
        "num_samples": total,
        "num_hard_samples": hard_count,
        "hard_ratio": (hard_count / total) if total > 0 else 0.0,
        "hard_voxel_counts": hard_voxel_counts,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"[build-hard-stats] written: {output_path}")
    print(
        "[build-hard-stats] summary:"
        f" total={total}, hard={hard_count}, ratio={hard_count / total:.4f}"
    )


if __name__ == "__main__":
    main()
