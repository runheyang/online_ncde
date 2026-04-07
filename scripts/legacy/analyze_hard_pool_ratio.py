#!/usr/bin/env python3
"""统计 Occ3D GT 中“难类池”样本占比。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CLASS_TO_ID = {
    "others": 0,
    "barrier": 1,
    "bicycle": 2,
    "bus": 3,
    "car": 4,
    "construction_vehicle": 5,
    "motorcycle": 6,
    "pedestrian": 7,
    "traffic_cone": 8,
    "trailer": 9,
    "truck": 10,
    "driveable_surface": 11,
    "other_flat": 12,
    "sidewalk": 13,
    "terrain": 14,
    "manmade": 15,
    "vegetation": 16,
    "free": 17,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze hard/easy sample ratio from Occ3D GT.")
    parser.add_argument(
        "--gt-root",
        type=str,
        default="data/nuscenes/gts",
        help="GT root directory. Expected layout: <gt-root>/<scene>/<token>/labels.npz",
    )
    parser.add_argument(
        "--hard-classes",
        type=str,
        default="traffic_cone,pedestrian,motorcycle,bus,others",
        help="Comma-separated class names used to define hard samples.",
    )
    parser.add_argument(
        "--per-scene",
        type=int,
        default=3,
        help="Number of frames sampled per scene (without replacement).",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="1,10,50,100,500,1000",
        help="Comma-separated voxel-count thresholds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260226,
        help="Random seed for per-scene sampling.",
    )
    return parser.parse_args()


def parse_csv_items(text: str) -> list[str]:
    items = [x.strip() for x in text.split(",")]
    return [x for x in items if x]


def parse_thresholds(text: str) -> list[int]:
    values = []
    for item in parse_csv_items(text):
        v = int(item)
        if v <= 0:
            raise ValueError(f"threshold must be > 0, got: {v}")
        values.append(v)
    if not values:
        raise ValueError("thresholds is empty.")
    return sorted(set(values))


def get_scene_dirs(gt_root: Path) -> list[Path]:
    return sorted([p for p in gt_root.iterdir() if p.is_dir()])


def get_token_dirs(scene_dir: Path) -> list[Path]:
    return sorted([p for p in scene_dir.iterdir() if p.is_dir() and (p / "labels.npz").exists()])


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root).expanduser().resolve()
    if not gt_root.exists():
        raise FileNotFoundError(f"gt root not found: {gt_root}")

    hard_class_names = parse_csv_items(args.hard_classes)
    unknown = [x for x in hard_class_names if x not in CLASS_TO_ID]
    if unknown:
        raise ValueError(f"unknown class names in --hard-classes: {unknown}")
    hard_ids = [CLASS_TO_ID[x] for x in hard_class_names]
    thresholds = parse_thresholds(args.thresholds)

    scene_dirs = get_scene_dirs(gt_root)
    rng = np.random.default_rng(args.seed)
    per_scene = int(args.per_scene)
    if per_scene <= 0:
        raise ValueError(f"--per-scene must be > 0, got {per_scene}")

    hard_counts = {th: 0 for th in thresholds}
    class_presence_counts = {name: 0 for name in hard_class_names}

    sampled_frames = 0
    scene_used = 0
    scene_empty = 0

    for scene_dir in scene_dirs:
        token_dirs = get_token_dirs(scene_dir)
        if not token_dirs:
            scene_empty += 1
            continue
        scene_used += 1

        if len(token_dirs) <= per_scene:
            chosen = token_dirs
        else:
            idx = rng.choice(len(token_dirs), size=per_scene, replace=False)
            chosen = [token_dirs[i] for i in idx]

        for token_dir in chosen:
            labels_path = token_dir / "labels.npz"
            with np.load(labels_path, allow_pickle=False) as data:
                semantics = data["semantics"]

            sampled_frames += 1

            hard_voxels = int(np.isin(semantics, hard_ids).sum())
            for th in thresholds:
                hard_counts[th] += int(hard_voxels >= th)

            for name in hard_class_names:
                cid = CLASS_TO_ID[name]
                class_presence_counts[name] += int(np.any(semantics == cid))

    if sampled_frames == 0:
        raise RuntimeError("No frames sampled. Please check --gt-root.")

    print(f"gt_root={gt_root}")
    print(f"scene_total={len(scene_dirs)}, scene_used={scene_used}, scene_empty={scene_empty}")
    print(f"sampled_frames={sampled_frames}, per_scene={per_scene}, seed={args.seed}")
    print(f"hard_classes={hard_class_names}")
    print("")
    print("threshold | hard_count | hard_ratio | easy_count | easy_ratio")
    for th in thresholds:
        hard = hard_counts[th]
        easy = sampled_frames - hard
        print(
            f"{th:9d} | {hard:10d} | {hard / sampled_frames:10.4f} | "
            f"{easy:10d} | {easy / sampled_frames:10.4f}"
        )

    print("")
    print("class presence rate (sample-level):")
    for name in hard_class_names:
        cnt = class_presence_counts[name]
        print(f"  {name:16s}: {cnt:6d}/{sampled_frames} = {cnt / sampled_frames:.4f}")


if __name__ == "__main__":
    main()
