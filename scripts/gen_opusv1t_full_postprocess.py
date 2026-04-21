#!/usr/bin/env python3
"""对 OPUS 稀疏 full logits 做形态学 close 后处理，产出独立目录。

输入:
    <src-root>/<scene>/<frame_token>/logits.npz
    字段: sparse_coords (N, 3) uint8, sparse_values (N, 17) float16

输出:
    <dst-root>/<scene>/<frame_token>/logits.npz
    字段同上，sparse_coords 与 sparse_values 为 close 后新稀疏集

算法对齐 third_party/OPUS/tests/eval_single_frame_full_logits.py 的 close：
  1. sparse → dense score volume (17, X, Y, Z)，背景 background_logit
  2. close = dilation(max_pool3d) → erosion(-max_pool3d(-·))
  3. 保护原始 (occ > logit_thr).any(C) 体素
  4. 过滤 (eroded > logit_thr).any(C) 得新稀疏集
  5. 取新体素 close 后的 17 维 logits 作为新 sparse_values

sigmoid 单调，close 在 logit 空间与 score 空间等价，省 sigmoid 往返；
score_thr=0.5 对应 logit_thr=0。
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

# worker 进程本地状态：由 _worker_init 设置
_WORKER_DEVICE: Optional[torch.device] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        default=str(ROOT / "data/logits_opusv1t_full"),
        help="源目录，包含 <scene>/<frame_token>/logits.npz",
    )
    parser.add_argument(
        "--dst-root",
        default=str(ROOT / "data/logits_opusv1t_full_postprocess"),
    )
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--background-logit", type=float, default=-20.0)
    parser.add_argument(
        "--grid-size",
        default="200,200,16",
        help="逗号分隔的 X,Y,Z 网格尺寸",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="worker 进程数；GPU 模式下每卡 4-8 个 worker 吞吐最佳",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=32,
        help="Pool.imap_unordered 的 chunksize，大 chunksize 减少 IPC overhead",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=-1,
        help="使用的 GPU 数，-1=自动用全部可用卡，0=强制 CPU",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 个文件（冒烟/测试用），<=0 处理全部",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="目标文件已存在则跳过（默认开启）",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
    )
    parser.add_argument(
        "--save-uncompressed",
        action="store_true",
        help="np.savez 而非 savez_compressed，更快但文件大",
    )
    return parser.parse_args()


def apply_morph_close(
    sparse_coords: np.ndarray,
    sparse_values: np.ndarray,
    grid_size: Tuple[int, int, int],
    score_thr: float,
    kernel_size: int,
    background_logit: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (new_coords uint8, new_values float16)，所有张量运算跑在 device 上。"""
    num_sem = int(sparse_values.shape[1])
    if sparse_coords.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0, num_sem), dtype=np.float16),
        )
    X, Y, Z = grid_size
    pad = kernel_size // 2
    logit_thr = math.log(score_thr / (1.0 - score_thr))

    coords_t = torch.from_numpy(sparse_coords.astype(np.int64, copy=False)).to(
        device, non_blocking=True
    )
    values_t = torch.from_numpy(sparse_values.astype(np.float32, copy=False)).to(
        device, non_blocking=True
    )
    occ = torch.full(
        (num_sem, X, Y, Z),
        fill_value=float(background_logit),
        dtype=torch.float32,
        device=device,
    )
    occ[:, coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = values_t.T

    occ_t = occ.unsqueeze(0)
    dilated = F.max_pool3d(occ_t, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool3d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)
    orig_mask = (occ_t > logit_thr).any(dim=1, keepdim=True)
    eroded = torch.where(orig_mask.expand_as(eroded), occ_t, eroded)
    eroded = eroded.squeeze(0)
    keep = (eroded > logit_thr).any(dim=0)
    if not bool(keep.any()):
        return (
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0, num_sem), dtype=np.float16),
        )
    keep_idx = torch.nonzero(keep, as_tuple=False)
    new_values = eroded[:, keep_idx[:, 0], keep_idx[:, 1], keep_idx[:, 2]].T
    return (
        keep_idx.to(torch.uint8).cpu().numpy(),
        new_values.to(torch.float16).cpu().numpy(),
    )


def _worker_init(num_gpus: int) -> None:
    """worker 进程本地：限 torch 线程 + 绑定分到的 GPU（按 worker 序号轮转）。"""
    global _WORKER_DEVICE
    torch.set_num_threads(1)
    if num_gpus > 0:
        name = mp.current_process().name  # e.g. "SpawnPoolWorker-3"
        try:
            rank = int(name.rsplit("-", 1)[-1]) - 1
        except (ValueError, IndexError):
            rank = 0
        device_idx = rank % num_gpus
        torch.cuda.set_device(device_idx)
        _WORKER_DEVICE = torch.device(f"cuda:{device_idx}")
    else:
        _WORKER_DEVICE = torch.device("cpu")


def _process_one(task):
    (src_path, dst_path, grid_size, score_thr, kernel_size,
     background_logit, skip_existing, save_uncompressed) = task
    if skip_existing and os.path.exists(dst_path):
        return (src_path, None, "skip")
    device = _WORKER_DEVICE if _WORKER_DEVICE is not None else torch.device("cpu")
    try:
        with np.load(src_path, allow_pickle=False) as data:
            sparse_coords = np.ascontiguousarray(data["sparse_coords"])
            sparse_values = np.ascontiguousarray(data["sparse_values"])
        new_coords, new_values = apply_morph_close(
            sparse_coords=sparse_coords,
            sparse_values=sparse_values,
            grid_size=grid_size,
            score_thr=score_thr,
            kernel_size=kernel_size,
            background_logit=background_logit,
            device=device,
        )
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        saver = np.savez if save_uncompressed else np.savez_compressed
        saver(dst_path, sparse_coords=new_coords, sparse_values=new_values)
        return (
            src_path,
            (int(sparse_coords.shape[0]), int(new_coords.shape[0])),
            "ok",
        )
    except Exception as e:
        return (src_path, None, f"error: {e}")


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    grid_size = tuple(int(x) for x in args.grid_size.split(","))
    if len(grid_size) != 3:
        raise ValueError(f"--grid-size 需 3 个整数，得到 {grid_size}")

    # 解析 GPU 分配
    cuda_available = torch.cuda.is_available()
    if args.num_gpus < 0:
        num_gpus = torch.cuda.device_count() if cuda_available else 0
    else:
        num_gpus = min(args.num_gpus, torch.cuda.device_count() if cuda_available else 0)

    print(f"[src] {src_root}")
    print(f"[dst] {dst_root}")
    print(
        f"[morph] score_thr={args.score_thr} kernel={args.kernel_size} "
        f"bg_logit={args.background_logit} grid={grid_size}"
    )
    print(
        f"[io] skip_existing={args.skip_existing} workers={args.num_workers} "
        f"num_gpus={num_gpus} (cuda_available={cuda_available})"
    )

    t_scan = time.time()
    src_files = sorted(src_root.glob("*/*/logits.npz"))
    print(f"[scan] {len(src_files)} files in {time.time() - t_scan:.1f}s")
    if args.limit > 0:
        src_files = src_files[: args.limit]
        print(f"[limit] 只处理前 {len(src_files)} 个")

    tasks = []
    for src in src_files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        tasks.append(
            (
                str(src), str(dst), grid_size,
                args.score_thr, args.kernel_size, args.background_logit,
                args.skip_existing, args.save_uncompressed,
            )
        )

    ok = skipped = failed = 0
    n_orig_total = n_new_total = 0
    t0 = time.time()

    def _log_progress(i: int) -> None:
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (len(tasks) - (i + 1)) / max(rate, 1e-6)
        print(f"  [{i + 1}/{len(tasks)}] rate={rate:.1f}/s eta={eta:.0f}s")

    def _handle(result) -> None:
        nonlocal ok, skipped, failed, n_orig_total, n_new_total
        src, stat, status = result
        if status == "ok":
            ok += 1
            n_orig, n_new = stat
            n_orig_total += n_orig
            n_new_total += n_new
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
            print(f"[fail] {src}: {status}")

    if args.num_workers <= 1:
        _worker_init(num_gpus)
        for i, task in enumerate(tasks):
            _handle(_process_one(task))
            if (i + 1) % 500 == 0:
                _log_progress(i)
    else:
        # GPU 模式必须用 spawn（fork 后 CUDA 初始化不安全）；CPU 模式 fork 更快
        ctx = mp.get_context("spawn" if num_gpus > 0 else "fork")
        with ctx.Pool(
            args.num_workers,
            initializer=_worker_init,
            initargs=(num_gpus,),
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_process_one, tasks, chunksize=args.chunksize)
            ):
                _handle(result)
                if (i + 1) % 500 == 0:
                    _log_progress(i)

    elapsed = time.time() - t0
    print(f"[done] ok={ok} skipped={skipped} failed={failed} in {elapsed:.1f}s")
    if ok > 0:
        avg_orig = n_orig_total / ok
        avg_new = n_new_total / ok
        ratio = avg_new / max(avg_orig, 1.0)
        print(
            f"[stats] 平均每帧: 原始 {avg_orig:.0f} 体素 → "
            f"morph 后 {avg_new:.0f} 体素 (x{ratio:.2f})"
        )


if __name__ == "__main__":
    main()
