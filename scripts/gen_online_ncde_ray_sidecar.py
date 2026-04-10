#!/usr/bin/env python3
"""生成 Online NCDE ray-first-hit sidecar（供 RayLoss 训练使用）。

对每个样本的 4 个监督时刻（t-1.5/t-1.0/t-0.5/t）用 DVR 对该时刻的 GT 体素做一次
raycast，得到 14040 条 lidar ray 的 first-hit 距离，存到磁盘。

输出布局（一个目录，便于 mmap 加载）：
    <out_dir>/
      dist.npy      shape (N, 4, R) float16   NaN = 无效 ray
      origin.npy    shape (N, 4, 3) float32   lidar origin（无效 sup 填零）
      sup_mask.npy  shape (N, 4)    uint8     1=该 sup 有效
      meta.pkl      {"token_to_idx", "supervision_labels", "pc_range",
                     "voxel_size", "num_rays", "ray_dirs", ...}

用法:
    python scripts/gen_online_ncde_ray_sidecar.py \
        --info-path configs/online_ncde/ncde_align_infos_train.pkl \
        --sup4-sidecar configs/online_ncde/ncde_align_infos_train_sup4_sidecar.pkl \
        --out-dir configs/online_ncde/ncde_align_infos_train_ray_sidecar \
        [--limit 100]
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_ncde.data.labels_io import load_labels_npz  # noqa: E402
from online_ncde.ray_loss import generate_lidar_rays  # noqa: E402


# ---------------------------------------------------------------------------
# DVR 延迟加载（与 eval 保持一致）
# ---------------------------------------------------------------------------

_dvr = None


def _get_dvr():
    global _dvr
    if _dvr is None:
        from torch.utils.cpp_extension import load

        dvr_dir = ROOT / "src" / "online_ncde" / "ops" / "dvr"
        _dvr = load(
            "dvr",
            sources=[str(dvr_dir / "dvr.cpp"), str(dvr_dir / "dvr.cu")],
            verbose=True,
            extra_cuda_cflags=["-allow-unsupported-compiler"],
        )
    return _dvr


# ---------------------------------------------------------------------------
# 单样本 raycast
# ---------------------------------------------------------------------------


def _raycast_gt(
    sem_gt: np.ndarray,
    origin_xyz: np.ndarray,
    lidar_rays: torch.Tensor,
    pc_range: tuple[float, ...],
    voxel_size: float,
    free_index: int,
    grid_size: tuple[int, int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """对单份 GT 体素 raycast，返回 (per-ray dist m, per-ray class id)。

    origin_xyz: (3,) ego 坐标系下的 lidar origin。
    """
    dvr = _get_dvr()

    # (X, Y, Z) int → (Z, Y, X) binary float
    occ = np.zeros_like(sem_gt, dtype=np.uint8)
    occ[sem_gt != free_index] = 1
    occ_t = torch.from_numpy(occ).permute(2, 1, 0)  # (Z, Y, X)
    occ_t = occ_t[None, None, :].contiguous().float().to(device)

    offset = torch.tensor(pc_range[:3], dtype=torch.float32, device=device).view(1, 1, 3)
    scaler = torch.tensor([voxel_size] * 3, dtype=torch.float32, device=device).view(1, 1, 3)

    origin = torch.tensor(origin_xyz, dtype=torch.float32, device=device).view(1, 1, 3)  # (1,1,3)
    endpts = lidar_rays.to(device).unsqueeze(0) + origin                                    # (1,R,3)
    origin_render = ((origin - offset) / scaler).float()
    points_render = ((endpts - offset) / scaler).float()
    lidar_tindex = torch.zeros([1, lidar_rays.shape[0]], device=device)

    with torch.no_grad():
        pred_dist, _, coord_index = dvr.render_forward(
            occ_t,
            origin_render,
            points_render,
            lidar_tindex,
            [1, int(grid_size[2]), int(grid_size[1]), int(grid_size[0])],
            "test",
        )
        pred_dist = pred_dist * voxel_size

    coord_index = coord_index[0].int().cpu().numpy()  # (R, 3) int
    dist = pred_dist[0].cpu().numpy().astype(np.float32)  # (R,)
    cls = sem_gt[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]].astype(np.int32)
    return dist, cls


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-path", required=True)
    parser.add_argument("--sup4-sidecar", required=True)
    parser.add_argument("--gt-root", default="data/nuscenes/gts")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个样本（调试用）")
    parser.add_argument("--free-index", type=int, default=17)
    parser.add_argument(
        "--pc-range",
        nargs=6,
        type=float,
        default=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
    )
    parser.add_argument("--voxel-size", type=float, default=0.4)
    parser.add_argument(
        "--grid-size",
        nargs=3,
        type=int,
        default=[200, 200, 16],
    )
    parser.add_argument(
        "--max-valid-m",
        type=float,
        default=60.0,
        help="超过此距离视为无效（即 DVR 跑到边界都没命中）",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else (ROOT / p).resolve())


def _load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    args = parse_args()
    info_path = _resolve(args.info_path)
    sup4_path = _resolve(args.sup4_sidecar)
    gt_root = _resolve(args.gt_root)
    out_dir = Path(_resolve(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    pc_range = tuple(args.pc_range)
    voxel_size = float(args.voxel_size)
    grid_size = tuple(args.grid_size)  # type: ignore[assignment]
    free_index = int(args.free_index)

    print(f"[load] info={info_path}")
    info_payload = _load_pkl(info_path)
    infos = info_payload["infos"] if isinstance(info_payload, dict) and "infos" in info_payload else info_payload
    print(f"[load] sup4_sidecar={sup4_path}")
    sup4 = _load_pkl(sup4_path)
    entries = sup4["entries"]
    if len(entries) != len(infos):
        raise ValueError(
            f"sup4 entries ({len(entries)}) 与 infos ({len(infos)}) 数量不一致"
        )
    N = len(entries)
    if args.limit > 0:
        N = min(N, args.limit)
        print(f"[limit] 只处理前 {N} 个样本")

    num_sup = 4
    lidar_rays = generate_lidar_rays("cpu")  # (R, 3)
    R = int(lidar_rays.shape[0])
    print(f"[rays] R={R}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 预分配输出：dist 直接 memmap 到磁盘，省去 N*num_sup*R*2B 的 peak RAM。
    dist_path = out_dir / "dist.npy"
    dist_arr = np.lib.format.open_memmap(
        dist_path, mode="w+", dtype=np.float16, shape=(N, num_sup, R)
    )
    dist_arr[...] = np.float16(np.nan)
    origin_arr = np.zeros((N, num_sup, 3), dtype=np.float32)
    sup_mask_arr = np.zeros((N, num_sup), dtype=np.uint8)
    token_to_idx: dict[str, int] = {}

    t0 = time.time()
    num_valid_entries = 0
    num_raycasts = 0
    num_skipped_missing_gt = 0

    def _load_semantics(rel_path: str) -> np.ndarray | None:
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(gt_root, rel_path)
        try:
            data = load_labels_npz(abs_path)
        except FileNotFoundError:
            return None
        return data["semantics"].astype(np.int32)

    for idx in range(N):
        entry = entries[idx]
        token = str(entry.get("token", ""))
        if token:
            token_to_idx[token] = idx

        sup_mask = entry.get("supervision_mask", [0] * num_sup)
        sup_rel_paths = entry.get("supervision_gt_rel_paths", [""] * num_sup)
        lidar2ego_t = np.asarray(
            entry.get("lidar2ego_translation", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        if lidar2ego_t.shape != (3,):
            lidar2ego_t = lidar2ego_t.reshape(3)

        any_sup = False
        for sup_i in range(num_sup):
            if int(sup_mask[sup_i]) <= 0:
                continue
            rel = str(sup_rel_paths[sup_i])
            if not rel:
                continue
            sem_gt = _load_semantics(rel)
            if sem_gt is None:
                num_skipped_missing_gt += 1
                continue
            if tuple(sem_gt.shape) != tuple(grid_size):
                raise ValueError(
                    f"GT shape {sem_gt.shape} 与 grid_size {grid_size} 不一致: {rel}"
                )

            dist, cls = _raycast_gt(
                sem_gt=sem_gt,
                origin_xyz=lidar2ego_t,
                lidar_rays=lidar_rays,
                pc_range=pc_range,
                voxel_size=voxel_size,
                free_index=free_index,
                grid_size=grid_size,
                device=device,
            )
            # 过滤无效：DVR 遍历到 free 类 / 越界 / dist 异常
            invalid = (cls == free_index) | (dist <= 0) | (dist >= args.max_valid_m)
            dist[invalid] = np.nan

            dist_arr[idx, sup_i] = dist.astype(np.float16)
            origin_arr[idx, sup_i] = lidar2ego_t
            sup_mask_arr[idx, sup_i] = 1
            num_raycasts += 1
            any_sup = True

        if any_sup:
            num_valid_entries += 1

        if (idx + 1) % 1000 == 0 or (idx + 1) == N:
            spd = (idx + 1) / max(time.time() - t0, 1e-6)
            eta = (N - idx - 1) / max(spd, 1e-6)
            print(
                f"  [{idx + 1}/{N}] valid_entries={num_valid_entries} "
                f"raycasts={num_raycasts} {spd:.1f} it/s ETA={eta / 60:.1f}min"
            )

    print("[save] dist.npy (memmap flush)")
    dist_arr.flush()
    del dist_arr
    print("[save] origin.npy")
    np.save(out_dir / "origin.npy", origin_arr)
    print("[save] sup_mask.npy")
    np.save(out_dir / "sup_mask.npy", sup_mask_arr)

    meta = {
        "schema_version": "online_ncde_ray_sidecar_v1",
        "source_info_path": info_path,
        "source_sup4_sidecar_path": sup4_path,
        "gt_root": gt_root,
        "pc_range": list(pc_range),
        "voxel_size": voxel_size,
        "grid_size": list(grid_size),
        "free_index": free_index,
        "num_rays": R,
        "ray_dirs": lidar_rays.cpu().numpy().astype(np.float32),
        "supervision_labels": ["t-1.5", "t-1.0", "t-0.5", "t"],
        "num_entries": int(N),
        "num_valid_entries": int(num_valid_entries),
        "num_raycasts": int(num_raycasts),
        "num_skipped_missing_gt": int(num_skipped_missing_gt),
        "max_valid_m": float(args.max_valid_m),
        "token_to_idx": token_to_idx,
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"[save] meta.pkl ({len(token_to_idx)} tokens)")

    dist_bytes = N * num_sup * R * np.dtype(np.float16).itemsize
    total_bytes = dist_bytes + origin_arr.nbytes + sup_mask_arr.nbytes
    print(
        f"[done] N={N} valid={num_valid_entries} raycasts={num_raycasts} "
        f"skipped_gt={num_skipped_missing_gt} "
        f"size={total_bytes / 1024 / 1024:.1f} MB"
    )


if __name__ == "__main__":
    main()
