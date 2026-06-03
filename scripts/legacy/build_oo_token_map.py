"""一次性生成 sample_token → (scene_token, lidar_sd_token) 映射 pkl。

OpenOccupancy GT 命名为 `scene_<scene_token>/occupancy/<LIDAR_TOP sd_token>.npy`，
但现有 canonical_infos pkl 只存 sample_token，没存 LIDAR_TOP 的 sample_data token。
本脚本用 nuScenes devkit 一次性扫所有 sample，dump 一个小 pkl，OO dataset 在
__init__ 时直接读，避免每个 worker 重复加载 nuScenes。

输出：
  configs/online_ncde/oo_token_map.pkl  —— {sample_token (str): {"scene_token": str, "lidar_token": str}}

用法：
  conda run -n neural_ode python scripts/build_oo_token_map.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="data/nuscenes", type=Path)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--out", default="configs/online_ncde/oo_token_map.pkl", type=Path)
    args = parser.parse_args()

    from nuscenes import NuScenes

    print(f"[build] loading NuScenes ({args.version} @ {args.dataroot})...")
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)

    mapping: dict[str, dict[str, str]] = {}
    for sample in nusc.sample:
        mapping[sample["token"]] = {
            "scene_token": sample["scene_token"],
            "lidar_token": sample["data"]["LIDAR_TOP"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(mapping, f)
    print(f"[build] {len(mapping)} samples → {args.out}")


if __name__ == "__main__":
    main()
