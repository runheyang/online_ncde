#!/usr/bin/env python3
"""生成 Online NCDE ray-first-hit sidecar（供 RayLoss 训练使用）。

对每个样本的 4 个监督时刻（t-1.5/t-1.0/t-0.5/t）用 DVR 对该时刻的 GT 体素做
raycast。每个监督时刻以 `supervision_gt_tokens[sup_i]` 为参考帧，从 nuScenes
sweep pkl 里用 `compute_lidar_origins` 计算最多 K 个 lidar origin（与 RayIoU
评估完全对齐：取同 scene 下过去+未来的 ego 位置，范围过滤 + 线性下采样到 K），
对每个 origin 独立发射 14040 条 lidar ray，记录 first-hit 距离。

输出布局（一个目录，便于 mmap 加载）：
    <out_dir>/
      <split>_dist.npy        (N, 4, K, R)  float16   NaN = 无效 ray
      <split>_origin.npy      (N, 4, K, 3)  float32   lidar origin（pad 填零）
      <split>_origin_mask.npy (N, 4, K)     uint8     1 = 该 origin 有效
      <split>_sup_mask.npy    (N, 4)        uint8     1 = 该 sup 有效
      <split>_meta.pkl        {"token_to_idx", "supervision_labels",
                               "num_origins", "schema_version": v2, ...}

用法:
    python scripts/gen_online_ncde_ray_sidecar.py \
        --info-path configs/online_ncde/ncde_align_infos_train.pkl \
        --sup4-sidecar configs/online_ncde/ncde_align_infos_train_sup4_sidecar.pkl \
        --sweep-pkl data/nuscenes/nuscenes_infos_train_sweep.pkl \
        --split train \
        --out-dir data/online_ncde_ray_sidecar \
        [--num-origins 8] [--limit 100]
"""

from __future__ import annotations

import argparse
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
from online_ncde.data.ray_sidecar import _SCHEMA_V2  # noqa: E402
from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl  # noqa: E402
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
# 单原点 raycast
# ---------------------------------------------------------------------------


def _raycast_gt(
    occ_t: torch.Tensor,             # (1,1,Z,Y,X) float on device
    sem_gt: np.ndarray,              # (X,Y,Z) int
    origin_xyz: np.ndarray,          # (3,)
    lidar_rays: torch.Tensor,        # (R,3) on device
    offset: torch.Tensor,            # (1,1,3) on device
    scaler: torch.Tensor,            # (1,1,3) on device
    grid_size: tuple[int, int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """对单份 GT 体素 raycast，返回 (per-ray dist m float32, per-ray class id int32)。"""
    dvr = _get_dvr()

    origin = torch.from_numpy(origin_xyz).to(device=device, dtype=torch.float32).view(1, 1, 3)
    endpts = lidar_rays.unsqueeze(0) + origin                                   # (1,R,3)
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
        pred_dist = pred_dist * float(scaler[0, 0, 0].item())

    coord_index = coord_index[0].int().cpu().numpy()                             # (R,3)
    dist = pred_dist[0].cpu().numpy().astype(np.float32)                         # (R,)
    cls = sem_gt[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]].astype(np.int32)
    return dist, cls


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-path", required=True)
    parser.add_argument("--sup4-sidecar", required=True)
    parser.add_argument(
        "--sweep-pkl",
        required=True,
        help="nuscenes_infos_*_sweep.pkl，供 compute_lidar_origins 使用",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "trainval", "test", "mini_train", "mini_val"],
        help="输出文件名前缀（train/val/...）",
    )
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
        "--num-origins",
        type=int,
        default=8,
        help="每个 sup step 最多保留的 lidar origin 数量（与 RayIoU 评估对齐）",
    )
    parser.add_argument(
        "--origin-range-limit",
        type=float,
        default=39.0,
        help="origin 在 ego 系下 |x|,|y| 的最大范围（与评估口径一致，默认 39.0）",
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
    sweep_path = _resolve(args.sweep_pkl)
    gt_root = _resolve(args.gt_root)
    out_dir = Path(_resolve(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    split = str(args.split)
    prefix = f"{split}_"

    pc_range = tuple(args.pc_range)
    voxel_size = float(args.voxel_size)
    grid_size = tuple(args.grid_size)  # type: ignore[assignment]
    free_index = int(args.free_index)
    K = int(args.num_origins)
    if K < 1:
        raise ValueError(f"--num-origins 必须 >=1，实际 {K}")

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
    lidar_rays_cpu = generate_lidar_rays("cpu")                                   # (R,3)
    R = int(lidar_rays_cpu.shape[0])
    print(f"[rays] R={R}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    lidar_rays = lidar_rays_cpu.to(device)
    offset = torch.tensor(pc_range[:3], dtype=torch.float32, device=device).view(1, 1, 3)
    scaler = torch.tensor([voxel_size] * 3, dtype=torch.float32, device=device).view(1, 1, 3)

    # ------------------------------------------------------------------
    # 多原点：一次性从 sweep pkl 计算 origins_by_token
    # ------------------------------------------------------------------
    print(f"[load] sweep_pkl={sweep_path}")
    origins_by_token = load_origins_from_sweep_pkl(
        sweep_path,
        max_origins=K,
        range_limit=float(args.origin_range_limit),
    )
    print(f"[origins] got origins for {len(origins_by_token)} tokens (K={K})")

    # ------------------------------------------------------------------
    # 预分配输出（新 schema v2，文件名带 split 前缀）
    # ------------------------------------------------------------------
    dist_path = out_dir / f"{prefix}dist.npy"
    origin_path = out_dir / f"{prefix}origin.npy"
    origin_mask_path = out_dir / f"{prefix}origin_mask.npy"
    sup_mask_path = out_dir / f"{prefix}sup_mask.npy"
    meta_path = out_dir / f"{prefix}meta.pkl"

    # dist memmap 到磁盘，避免 peak RAM 膨胀（(N,4,K,R)*2B 量级可达 10+ GB）
    dist_arr = np.lib.format.open_memmap(
        dist_path, mode="w+", dtype=np.float16, shape=(N, num_sup, K, R)
    )
    dist_arr[...] = np.float16(np.nan)
    origin_arr = np.zeros((N, num_sup, K, 3), dtype=np.float32)
    origin_mask_arr = np.zeros((N, num_sup, K), dtype=np.uint8)
    sup_mask_arr = np.zeros((N, num_sup), dtype=np.uint8)
    token_to_idx: dict[str, int] = {}

    t0 = time.time()
    num_valid_entries = 0
    num_raycasts = 0
    num_skipped_missing_gt = 0
    num_skipped_missing_origins = 0
    origin_count_hist: list[int] = []

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
        sup_gt_tokens = entry.get("supervision_gt_tokens", [""] * num_sup)

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

            # origins 以 sup_gt_token 为参考帧查表，与 RayIoU 评估严格一致
            sup_token = str(sup_gt_tokens[sup_i]) if sup_gt_tokens and sup_gt_tokens[sup_i] else ""
            origins_ref = origins_by_token.get(sup_token) if sup_token else None
            if origins_ref is None:
                num_skipped_missing_origins += 1
                continue

            origins_tensor = origins_ref
            if origins_tensor.dim() == 3:
                origins_tensor = origins_tensor[0]
            origins_np = origins_tensor.detach().cpu().numpy().astype(np.float32)

            T_ref = int(origins_np.shape[0])
            if T_ref == 0:
                continue
            if T_ref > K:
                origins_np = origins_np[:K]
                T_ref = K
            origin_count_hist.append(T_ref)

            occ = np.zeros_like(sem_gt, dtype=np.uint8)
            occ[sem_gt != free_index] = 1
            occ_t = torch.from_numpy(occ).permute(2, 1, 0)                          # (Z,Y,X)
            occ_t = occ_t[None, None, :].contiguous().float().to(device)

            for k in range(T_ref):
                dist_k, cls_k = _raycast_gt(
                    occ_t=occ_t,
                    sem_gt=sem_gt,
                    origin_xyz=origins_np[k],
                    lidar_rays=lidar_rays,
                    offset=offset,
                    scaler=scaler,
                    grid_size=grid_size,
                    device=device,
                )
                invalid = (
                    (cls_k == free_index)
                    | (dist_k <= 0)
                    | (dist_k >= args.max_valid_m)
                )
                dist_k[invalid] = np.nan

                dist_arr[idx, sup_i, k] = dist_k.astype(np.float16)
                origin_arr[idx, sup_i, k] = origins_np[k]
                origin_mask_arr[idx, sup_i, k] = 1
                num_raycasts += 1

            sup_mask_arr[idx, sup_i] = 1
            any_sup = True

        if any_sup:
            num_valid_entries += 1

        if (idx + 1) % 500 == 0 or (idx + 1) == N:
            spd = (idx + 1) / max(time.time() - t0, 1e-6)
            eta = (N - idx - 1) / max(spd, 1e-6)
            print(
                f"  [{idx + 1}/{N}] valid_entries={num_valid_entries} "
                f"raycasts={num_raycasts} {spd:.1f} it/s ETA={eta / 60:.1f}min"
            )

    print("[save] dist.npy (memmap flush)")
    dist_arr.flush()
    del dist_arr
    print(f"[save] {origin_path.name}")
    np.save(origin_path, origin_arr)
    print(f"[save] {origin_mask_path.name}")
    np.save(origin_mask_path, origin_mask_arr)
    print(f"[save] {sup_mask_path.name}")
    np.save(sup_mask_path, sup_mask_arr)

    if origin_count_hist:
        origin_cnt_arr = np.asarray(origin_count_hist, dtype=np.int32)
        origin_cnt_mean = float(origin_cnt_arr.mean())
        origin_cnt_min = int(origin_cnt_arr.min())
        origin_cnt_max = int(origin_cnt_arr.max())
    else:
        origin_cnt_mean = 0.0
        origin_cnt_min = 0
        origin_cnt_max = 0

    meta = {
        "schema_version": _SCHEMA_V2,
        "split": split,
        "source_info_path": info_path,
        "source_sup4_sidecar_path": sup4_path,
        "source_sweep_pkl_path": sweep_path,
        "gt_root": gt_root,
        "pc_range": list(pc_range),
        "voxel_size": voxel_size,
        "grid_size": list(grid_size),
        "free_index": free_index,
        "num_rays": R,
        "num_origins": K,
        "origin_range_limit": float(args.origin_range_limit),
        "ray_dirs": lidar_rays_cpu.cpu().numpy().astype(np.float32),
        "supervision_labels": ["t-1.5", "t-1.0", "t-0.5", "t"],
        "num_entries": int(N),
        "num_valid_entries": int(num_valid_entries),
        "num_raycasts": int(num_raycasts),
        "num_skipped_missing_gt": int(num_skipped_missing_gt),
        "num_skipped_missing_origins": int(num_skipped_missing_origins),
        "origin_count_mean": origin_cnt_mean,
        "origin_count_min": origin_cnt_min,
        "origin_count_max": origin_cnt_max,
        "max_valid_m": float(args.max_valid_m),
        "token_to_idx": token_to_idx,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[save] {meta_path.name} ({len(token_to_idx)} tokens)")

    dist_bytes = N * num_sup * K * R * np.dtype(np.float16).itemsize
    total_bytes = dist_bytes + origin_arr.nbytes + origin_mask_arr.nbytes + sup_mask_arr.nbytes
    print(
        f"[done] split={split} N={N} K={K} valid={num_valid_entries} "
        f"raycasts={num_raycasts} skipped_gt={num_skipped_missing_gt} "
        f"skipped_origins={num_skipped_missing_origins} "
        f"origin_cnt=(min={origin_cnt_min}, mean={origin_cnt_mean:.2f}, max={origin_cnt_max}) "
        f"size={total_bytes / 1024 / 1024:.1f} MB"
    )


if __name__ == "__main__":
    main()
