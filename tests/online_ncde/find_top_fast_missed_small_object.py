#!/usr/bin/env python3
"""逐 sample 在指定**小物体类**上比较 fast(curr) vs slow(curr) 的 per-class IoU，
找 "fast 漏检 / slow 没漏" 的极端样本：score = mean_c(slow_iou_c - fast_iou_c)
排序，输出 top-K 个 sample 以及具体到哪个类被漏。

为什么要逐类而不是整体 mIoU：
  小物体（行人、路锥、自行车…）voxel 数量很少，被路面 / 建筑等大类的 IoU
  淹没——fast 把行人完全漏检，整体 mIoU 也只动 1-2 个百分点，按整体 mIoU
  排序根本捞不到这种 case。所以这里只在 --target-classes 指定的类内做对比，
  并要求 GT 里至少有 --min-gt-voxels 个该类 voxel 才算"GT 真存在"。

约定（和 find_top_fast_slow_miou_gap.py / eval_online_ncde_evolution_times.py
对齐）：
  - sample 当前帧 = end keyframe = evolve_keyframe_sample_tokens[-1]
  - fast(curr)  = frame_rel_paths[num_real-1] 那帧 logits.argmax
  - slow(curr)  = end keyframe 自己的 slow logits.argmax
  - 评估区域 = mask_camera ∧ near_mask（默认 20m 圆，小物体几乎都在近景）
  - per-sample score = 该 sample 中 GT 真实存在的 target 类的
    (slow_iou_c - fast_iou_c) 平均；GT 不含任何 target 类的 sample 跳过

OCC3D 类索引快速参考：
  2 bicycle  3 bus  4 car  5 construction_vehicle  6 motorcycle
  7 pedestrian  8 traffic_cone  9 trailer  10 truck

用法（多进程）：
  conda run -n neural_ode python tests/online_ncde/find_top_fast_missed_small_object.py \\
      --config configs/online_ncde/fast_alocc2dmini__slow_alocc3d/base.yaml \\
      --info-path configs/online_ncde/evolve_infos_val_2s.pkl \\
      --output tests/online_ncde/top100_fast_missed_small_object.txt \\
      --target-classes 7,8,2,6 --min-gt-voxels 5 \\
      --top-k 100 --num-workers 8
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402

# 与 viewer / metrics 一致的类名（仅打印用）
OCC3D_CLASS_NAMES = [
    "others", "barrier", "bicycle", "bus", "car",
    "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone",
    "trailer", "truck", "driveable_surface", "other_flat", "sidewalk",
    "terrain", "manmade", "vegetation", "free",
]


# ────────────────────────── 近景 mask ──────────────────────────


def build_near_mask(
    pc_range: tuple, voxel_size: tuple, grid_size: tuple, near_radius_m: float,
) -> np.ndarray:
    X, Y, Z = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    x_min, y_min = float(pc_range[0]), float(pc_range[1])
    vx, vy = float(voxel_size[0]), float(voxel_size[1])
    ii, jj = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    xx = x_min + (ii + 0.5) * vx
    yy = y_min + (jj + 0.5) * vy
    near_xy = (xx * xx + yy * yy) <= float(near_radius_m) ** 2
    return np.broadcast_to(near_xy[:, :, None], (X, Y, Z)).copy()


# ────────────────────────── worker 状态 ──────────────────────────

_LOADER = None
_GT_ROOT: str = ""
_GT_MASK_KEY: str = "mask_camera"
_NEAR_MASK: np.ndarray | None = None
_TARGET_CLASSES: tuple[int, ...] = ()
_MIN_GT_VOXELS: int = 5


def _init_worker(
    data_cfg: Dict[str, Any], root_path: str, gt_root: str, gt_mask_key: str,
    near_mask: np.ndarray | None,
    target_classes: tuple[int, ...], min_gt_voxels: int,
) -> None:
    global _LOADER, _GT_ROOT, _GT_MASK_KEY, _NEAR_MASK
    global _TARGET_CLASSES, _MIN_GT_VOXELS
    _LOADER = build_logits_loader(data_cfg, root_path)
    _GT_ROOT = gt_root
    _GT_MASK_KEY = gt_mask_key
    _NEAR_MASK = near_mask
    _TARGET_CLASSES = tuple(int(c) for c in target_classes)
    _MIN_GT_VOXELS = int(min_gt_voxels)


# ────────────────────────── per-class IoU ──────────────────────────


def per_class_iou(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
    target_classes: tuple[int, ...], min_gt_voxels: int,
) -> dict[int, tuple[int, float]]:
    """对每个 target 类返回 (gt_voxel_count, iou)；GT 不足 min_gt_voxels 的类不返回。
    valid 已是 (mask_camera ∧ near_mask) 的 bool。
    """
    p = pred[valid].astype(np.int64).ravel()
    g = gt[valid].astype(np.int64).ravel()
    out: dict[int, tuple[int, float]] = {}
    for c in target_classes:
        gc = g == c
        gt_count = int(gc.sum())
        if gt_count < min_gt_voxels:
            continue
        pc = p == c
        union = int(np.logical_or(pc, gc).sum())
        if union == 0:
            continue
        inter = int(np.logical_and(pc, gc).sum())
        out[int(c)] = (gt_count, inter / union)
    return out


# ────────────────────────── 单 sample ──────────────────────────


def _process_one(idx: int, info: Dict[str, Any]) -> tuple:
    """返回 (idx, score, n_classes, fast_per_class, slow_per_class, scene, end_token, err)。
    fast/slow_per_class 是 dict[class_id, (gt_count, iou)]；err 非空表示跳过原因。
    """
    try:
        scene_name = str(info.get("scene_name", ""))
        ek_tokens = list(info.get("evolve_keyframe_sample_tokens", []))
        ek_gt_exists = list(info.get("evolve_keyframe_gt_exists", []))
        if not ek_tokens:
            return idx, np.nan, 0, {}, {}, scene_name, "", "no_ek_tokens"
        end_token = str(ek_tokens[-1])
        if ek_gt_exists and not int(ek_gt_exists[-1]):
            return idx, np.nan, 0, {}, {}, scene_name, end_token, "no_end_gt"

        gt_path = os.path.join(_GT_ROOT, scene_name, end_token, "labels.npz")
        if not os.path.exists(gt_path):
            return idx, np.nan, 0, {}, {}, scene_name, end_token, "gt_missing"
        with np.load(gt_path, allow_pickle=False) as g:
            gt = g["semantics"]
            mask = (
                g[_GT_MASK_KEY] if _GT_MASK_KEY in g.files
                else np.ones_like(gt, dtype=np.uint8)
            )

        valid = mask > 0
        if _NEAR_MASK is not None:
            valid = np.logical_and(valid, _NEAR_MASK)
        if not valid.any():
            return idx, np.nan, 0, {}, {}, scene_name, end_token, "empty_valid_mask"

        # 先粗筛：GT 在 valid 区域内是否含任何 target 类至少 min_gt_voxels 个
        g_valid = gt[valid]
        present_classes: list[int] = []
        for c in _TARGET_CLASSES:
            if int((g_valid == c).sum()) >= _MIN_GT_VOXELS:
                present_classes.append(int(c))
        if not present_classes:
            return idx, np.nan, 0, {}, {}, scene_name, end_token, "no_target_in_gt"

        # fast(curr)
        num_real = int(info.get("num_real_frames", len(info.get("frame_rel_paths", []))))
        frame_rel_paths = list(info.get("frame_rel_paths", []))
        if num_real <= 0 or num_real > len(frame_rel_paths):
            return idx, np.nan, 0, {}, {}, scene_name, end_token, "bad_num_real"
        fast_rel = frame_rel_paths[num_real - 1]
        if not fast_rel:
            return idx, np.nan, 0, {}, {}, scene_name, end_token, "fast_rel_empty"
        fast_logits = _LOADER.load_fast_logits(
            {"frame_rel_paths": [fast_rel]}, torch.device("cpu"),
        )
        fast_pred = fast_logits[0].argmax(0).numpy()

        # slow(curr)
        slow_rel = f"{scene_name}/{end_token}/logits.npz"
        slow_logits = _LOADER.load_slow_logits(
            {"slow_logit_path": slow_rel}, torch.device("cpu"),
        )
        slow_pred = slow_logits.argmax(0).numpy()

        target_set = tuple(present_classes)
        f_per = per_class_iou(fast_pred, gt, valid, target_set, _MIN_GT_VOXELS)
        s_per = per_class_iou(slow_pred, gt, valid, target_set, _MIN_GT_VOXELS)

        # 取 GT 真存在的类（present_classes）做平均；没出现在 f_per/s_per 的类
        # （union=0）当作 IoU=0
        diffs: list[float] = []
        for c in present_classes:
            f_iou = f_per.get(c, (0, 0.0))[1]
            s_iou = s_per.get(c, (0, 0.0))[1]
            diffs.append(s_iou - f_iou)
        if not diffs:
            return idx, np.nan, 0, {}, {}, scene_name, end_token, "no_diff"
        score = float(np.mean(diffs))
        return (idx, score, len(present_classes),
                {c: f_per.get(c, ((g_valid == c).sum(), 0.0)) for c in present_classes},
                {c: s_per.get(c, ((g_valid == c).sum(), 0.0)) for c in present_classes},
                scene_name, end_token, "")
    except Exception as e:
        return idx, np.nan, 0, {}, {}, "", "", f"exc:{type(e).__name__}:{e}"


# ────────────────────────── main ──────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--info-path", required=True,
                    help="evolve_infos pkl (schema=online_ncde_evolve_infos_v1)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--target-classes", default="7,8,2,6",
                    help="逗号分隔的 OCC3D class id；默认 7,8,2,6 = "
                         "pedestrian, traffic_cone, bicycle, motorcycle")
    ap.add_argument("--min-gt-voxels", type=int, default=5,
                    help="GT 中该类至少 N 个 voxel 才算 'GT 真存在'（避免噪点）")
    ap.add_argument("--near-radius-m", type=float, default=20.0,
                    help="评估区域半径 m，sqrt(x²+y²)≤此值；<=0 不限制")
    ap.add_argument("--sort-by",
                    choices=["slow_minus_fast", "abs", "fast_minus_slow"],
                    default="slow_minus_fast",
                    help="默认 slow-fast 大→小（找 slow 看到 fast 漏的）")
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--log-interval", type=int, default=500)
    args = ap.parse_args()

    target_classes = tuple(int(s) for s in args.target_classes.split(",") if s.strip())
    print(f"[target] classes={list(target_classes)} "
          f"({[OCC3D_CLASS_NAMES[c] for c in target_classes]})  "
          f"min_gt_voxels={args.min_gt_voxels}")

    cfg = load_config_with_base(args.config)
    data_cfg = cfg["data"]
    root_path = cfg["root_path"]
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    gt_mask_key = str(data_cfg.get("gt_mask_key", "mask_camera"))
    pc_range = tuple(data_cfg["pc_range"])
    voxel_size = tuple(data_cfg["voxel_size"])
    grid_size = tuple(data_cfg["grid_size"])

    if args.near_radius_m > 0:
        near_mask = build_near_mask(pc_range, voxel_size, grid_size, args.near_radius_m)
        n_near = int(near_mask.sum()); n_total = int(np.prod(grid_size))
        print(f"[near] radius={args.near_radius_m:.1f}m  "
              f"near_voxels={n_near}/{n_total} ({100*n_near/n_total:.1f}%)")
    else:
        near_mask = None
        print("[near] disabled")

    info_abs = resolve_path(root_path, args.info_path)
    with open(info_abs, "rb") as f:
        payload = pickle.load(f)
    md = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    schema = str(md.get("schema_version", ""))
    if schema != "online_ncde_evolve_infos_v1":
        print(f"[warn] schema={schema!r}，期望 evolve_infos；"
              "fast/slow 的 'curr' 帧定义可能不一致。", file=sys.stderr)
    infos = payload["infos"] if isinstance(payload, dict) else payload

    items = [(i, info) for i, info in enumerate(infos) if info.get("valid", True)]
    if args.max_samples > 0:
        items = items[: args.max_samples]
    print(f"[info] total={len(infos)} valid={len(items)} workers={args.num_workers}")

    init_args = (
        data_cfg, root_path, gt_root, gt_mask_key, near_mask,
        target_classes, args.min_gt_voxels,
    )

    t0 = time.time()
    results: list[tuple] = []
    err_counter: dict[str, int] = {}
    with ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=_init_worker, initargs=init_args,
    ) as exe:
        futures = {exe.submit(_process_one, i, info): i for i, info in items}
        total = len(futures); done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r); done += 1
            err = r[7]
            if err:
                err_counter[err] = err_counter.get(err, 0) + 1
            if done % args.log_interval == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-9)
                print(f"[progress] {done}/{total}  ({rate:.1f}/s, "
                      f"elapsed {elapsed:.0f}s)", file=sys.stderr, flush=True)

    if err_counter:
        print(f"[skip reasons] {err_counter}", file=sys.stderr)

    valid = [r for r in results if not np.isnan(r[1])]
    print(f"[info] scored={len(valid)} / requested={len(results)}")

    if args.sort_by == "abs":
        valid.sort(key=lambda r: -abs(r[1]))
    elif args.sort_by == "slow_minus_fast":
        valid.sort(key=lambda r: -r[1])
    else:
        valid.sort(key=lambda r: r[1])

    top = valid[: args.top_k]

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt_breakdown(present_classes, f_per, s_per) -> str:
        chunks: list[str] = []
        for c in present_classes:
            name = OCC3D_CLASS_NAMES[c][:4]
            f_ct, f_iou = f_per.get(c, (0, 0.0))
            s_ct, s_iou = s_per.get(c, (0, 0.0))
            chunks.append(f"{name}:gt{f_ct}:f{f_iou:.2f}->s{s_iou:.2f}")
        return "  ".join(chunks)

    with open(out_path, "w") as f:
        f.write(f"# source_pkl: {args.info_path}\n")
        f.write(f"# schema: {schema}\n")
        f.write(f"# target_classes: {list(target_classes)} "
                f"({[OCC3D_CLASS_NAMES[c] for c in target_classes]})\n")
        f.write(f"# min_gt_voxels: {args.min_gt_voxels}\n")
        f.write(f"# near_radius_m: {args.near_radius_m}\n")
        f.write(f"# sort_by: {args.sort_by}    "
                "score = mean_c(slow_iou_c - fast_iou_c) over GT-present target classes\n")
        f.write(f"# top_k: {len(top)} / scored={len(valid)} / total={len(items)}\n")
        f.write("# columns: idx  score  n_present_classes  scene  end_token  "
                "<class:gtN:fXX->sYY for each present class>\n")
        for idx, score, n_cls, f_per, s_per, scene, tok, _err in top:
            classes_sorted = sorted(f_per.keys())
            br = _fmt_breakdown(classes_sorted, f_per, s_per)
            f.write(f"{idx}\t{score:+.4f}\t{n_cls}\t{scene}\t{tok}\t{br}\n")
    print(f"[saved] {out_path}  (top-{len(top)})")


if __name__ == "__main__":
    main()
