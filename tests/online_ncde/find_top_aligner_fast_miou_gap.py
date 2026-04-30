#!/usr/bin/env python3
"""逐 sample 跑 NCDE aligner，比较 aligner(end_kf) vs fast(curr) 相对 GT 的
mIoU 差距（**仅近景区域**），按 (aligned_miou - fast_miou) 排序，输出差距
最大的 top-K idx。

用途：从 evolve_infos pkl 里挑出 aligner 在近景明显优于 fast 的样本，
方便论文挑可视化对比图——例如"fast 漏检 / aligner 借助 slow 演化补出"
这类极端 case。远处 voxel 体积大、含语义少，提升对最终视觉对比意义不大，
所以默认只看自车周围 20m 圆。

约定（和 scripts/eval_online_ncde_evolution_times.py 对齐）：
  - sample 当前帧 = end keyframe = K_{j+max_evolve}
  - aligner 在 end keyframe 的预测 = forward_stepwise_eval 输出中
    step_index = max_evolve * steps_per_kf 那一帧的 logits.argmax
  - fast(curr) = fast_logits[:, num_real_frames - 1].argmax (即 end keyframe
    对应的 fast frame）
  - per-sample mIoU：(mask_camera ∧ near_mask) 区域内对目标类算 IoU 后平均，
    跳过 union==0 的类（per-image 而非全集统计）
  - 默认 metric = miou_d，只统计 OCC3D 动态物体类（车 / 行人 / 自行车等
    8 类，id=2,3,4,5,6,7,9,10），因为 aligner 的核心价值就是补出动态目标。
    --metric miou 退化为评估全部非 free 类的 mIoU。
  - near_mask = sqrt(x² + y²) ≤ near_radius_m 的 voxel；x, y 取 voxel 中心
    在 ego frame 下的坐标，z 不限制；默认 20m

DataLoader 多 worker 加载 + 单 GPU 推理 + CPU 算 mIoU；--num-workers 控制
DataLoader worker 数。

用法：
  conda run -n neural_ode python tests/online_ncde/find_top_aligner_fast_miou_gap.py \\
      --config configs/online_ncde/fast_alocc2dmini__slow_alocc3d/base.yaml \\
      --checkpoint ckpts/.../epoch_9.pth \\
      --info-path configs/online_ncde/evolve_infos_val_2s.pkl \\
      --output tests/online_ncde/top100_aligner_fast_gap.txt \\
      --top-k 100 --num-workers 4 --batch-size 1
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.metrics import OCC3D_DYNAMIC_OBJECT_IDX  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402

EXPECTED_SCHEMA = "online_ncde_evolve_infos_v1"


# ────────────────────────── mIoU ──────────────────────────


def build_near_mask(
    pc_range: tuple,
    voxel_size: tuple,
    grid_size: tuple,
    near_radius_m: float,
) -> np.ndarray:
    """构造 (X, Y, Z) bool 近景 mask：voxel 中心的 (x, y) 世界坐标
    满足 sqrt(x² + y²) ≤ near_radius_m。z 不限制。
    pc_range 在 ego frame，自车在 (0, 0)。所有 sample 复用一次构造。"""
    X, Y, Z = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    x_min, y_min = float(pc_range[0]), float(pc_range[1])
    vx, vy = float(voxel_size[0]), float(voxel_size[1])
    ii, jj = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    xx = x_min + (ii + 0.5) * vx  # (X, Y)
    yy = y_min + (jj + 0.5) * vy
    near_xy = (xx * xx + yy * yy) <= float(near_radius_m) ** 2
    return np.broadcast_to(near_xy[:, :, None], (X, Y, Z)).copy()


def per_sample_miou(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    free_index: int,
    extra_mask: np.ndarray | None = None,
    class_subset: tuple[int, ...] | None = None,
) -> float:
    """单帧 per-sample mIoU；(mask>0) ∧ extra_mask 区域内对每个目标类算
    IoU 后平均，跳过 union==0 的类。所有目标类都没有时返回 nan。

    class_subset:
      - None: 评估所有非 free 类（默认 mIoU = full mIoU）
      - tuple[int]: 只评估这些类（如 OCC3D_DYNAMIC_OBJECT_IDX → mIoU_D）
    """
    valid = mask > 0
    if extra_mask is not None:
        valid = np.logical_and(valid, extra_mask)
    if not valid.any():
        return float("nan")
    p = pred[valid].astype(np.int64).ravel()
    g = gt[valid].astype(np.int64).ravel()

    if class_subset is None:
        target_classes: tuple[int, ...] = tuple(
            c for c in range(num_classes) if c != free_index
        )
    else:
        target_classes = tuple(int(c) for c in class_subset if int(c) != free_index)

    ious: list[float] = []
    for c in target_classes:
        pc = p == c
        gc = g == c
        union = int(np.logical_or(pc, gc).sum())
        if union == 0:
            continue
        inter = int(np.logical_and(pc, gc).sum())
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else float("nan")


def _load_gt(
    gt_root: str, scene: str, token: str, mask_key: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = os.path.join(gt_root, scene, token, "labels.npz")
    if not os.path.exists(path):
        return None, None
    with np.load(path, allow_pickle=False) as g:
        gt = g["semantics"]
        mask = (
            g[mask_key] if mask_key in g.files
            else np.ones_like(gt, dtype=np.uint8)
        )
    return gt, mask


# ────────────────────────── main ──────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--info-path", required=True,
                    help="evolve_infos pkl（schema=online_ncde_evolve_infos_v1）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker 数（数据加载并行）")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="bs=1 时按 num_real_frames 截断输入，避免 pad 段无意义 ODE 推演；"
                         "bs>1 时各样本 num_real 不同，模型跑满 num_frames，结果一致但慢一点")
    ap.add_argument("--solver", choices=["heun", "euler"], default="euler")
    ap.add_argument("--sort-by",
                    choices=["aligned_minus_fast", "abs", "fast_minus_aligned"],
                    default="aligned_minus_fast",
                    help="默认 aligned-fast 大→小（找 aligner 比 fast 强的）；"
                         "abs=|diff| 两端极端；fast-aligned 找 fast 反超的 case")
    ap.add_argument("--limit", type=int, default=-1,
                    help="只评估前 N 个样本（debug 用），<=0 表示全部")
    ap.add_argument("--near-radius-m", type=float, default=20.0,
                    help="近景半径 (m)，仅评估 sqrt(x²+y²) ≤ 此值的 voxel；"
                         "<=0 时退化为不限制（全 pc_range）")
    ap.add_argument("--metric", choices=["miou_d", "miou"], default="miou_d",
                    help="评估指标：miou_d=只统计 OCC3D 动态物体类（默认，"
                         f"id={list(OCC3D_DYNAMIC_OBJECT_IDX)}）；"
                         "miou=全部非 free 类")
    ap.add_argument("--log-interval", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config_with_base(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    root_path = cfg["root_path"]
    num_classes = int(data_cfg["num_classes"])
    free_index = int(data_cfg["free_index"])
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    gt_mask_key = str(data_cfg.get("gt_mask_key", "mask_camera"))
    pc_range = tuple(data_cfg["pc_range"])
    voxel_size = tuple(data_cfg["voxel_size"])
    grid_size = tuple(data_cfg["grid_size"])

    # 构造近景 mask：所有 sample 共享同一个 (X,Y,Z) bool tensor
    if args.near_radius_m > 0:
        near_mask = build_near_mask(pc_range, voxel_size, grid_size, args.near_radius_m)
        n_near = int(near_mask.sum())
        n_total = int(np.prod(grid_size))
        print(f"[near] radius={args.near_radius_m:.1f}m  "
              f"near_voxels={n_near}/{n_total} ({100*n_near/n_total:.1f}%)")
    else:
        near_mask = None
        print("[near] disabled (--near-radius-m <= 0)")

    # 选择评估类
    if args.metric == "miou_d":
        class_subset: tuple[int, ...] | None = tuple(OCC3D_DYNAMIC_OBJECT_IDX)
        print(f"[metric] miou_d  classes={list(class_subset)} (OCC3D 动态物体)")
    else:
        class_subset = None
        print(f"[metric] miou  classes=all non-free")

    # schema 校验
    info_abs = resolve_path(root_path, args.info_path)
    with open(info_abs, "rb") as f:
        payload = pickle.load(f)
    md = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    schema = str(md.get("schema_version", ""))
    if schema != EXPECTED_SCHEMA:
        raise ValueError(
            f"schema={schema!r}，期望 {EXPECTED_SCHEMA}（evolve_infos）；"
            f"该脚本依赖 evolve schema 的 num_real_frames / max_evolve 字段。"
        )
    del payload, md

    # dataset
    logits_loader = build_logits_loader(data_cfg, root_path)
    ds_full = Occ3DOnlineNcdeDataset(
        info_path=args.info_path,
        root_path=root_path,
        gt_root=data_cfg["gt_root"],
        num_classes=num_classes,
        free_index=free_index,
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=gt_mask_key,
        logits_loader=logits_loader,
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
        min_history_completeness=0,
        eval_only_mode=True,
    )
    if args.limit > 0:
        keep = min(args.limit, len(ds_full))
        ds: Any = Subset(ds_full, list(range(keep)))
    else:
        ds = ds_full
    base_indices: list[int] = (
        list(ds.indices) if isinstance(ds, Subset) else list(range(len(ds_full)))
    )
    print(f"[info] dataset size={len(ds)} (full={len(ds_full)})  workers={args.num_workers}  bs={args.batch_size}")

    kwargs: dict[str, Any] = dict(
        batch_size=max(int(args.batch_size), 1),
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=online_ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(ds, **kwargs)

    # 推断 dt 网格
    info0 = ds_full.infos[0]
    history_keyframes = int(info0.get("history_keyframes", 4))
    stride = int(data_cfg.get("fast_frame_stride", 1))
    num_output = int(
        info0.get(
            "num_output_frames",
            history_keyframes * info0.get("steps_per_interval", 1) + 1,
        )
    )
    if (num_output - 1) % stride != 0:
        raise ValueError(f"num_output_frames-1={num_output-1} 与 fast_frame_stride={stride} 不整除")
    num_frames = (num_output - 1) // stride + 1
    if (num_frames - 1) % history_keyframes != 0:
        raise ValueError(
            f"抽帧后 num_frames-1={num_frames-1} 与 history_keyframes={history_keyframes} 不整除"
        )
    steps_per_kf = (num_frames - 1) // history_keyframes
    print(f"[eval] history_keyframes={history_keyframes} num_frames={num_frames} "
          f"steps_per_kf={steps_per_kf}")

    # model
    device = torch.device(eval_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = OnlineNcdeAligner(
        num_classes=num_classes,
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=free_index,
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=model_cfg.get("decoder_init_scale", 1.0e-3),
        use_fast_residual=bool(model_cfg.get("use_fast_residual", True)),
        func_g_inner_dim=model_cfg.get("func_g_inner_dim", 32),
        func_g_body_dilations=tuple(model_cfg.get("func_g_body_dilations", [1, 2, 3])),
        func_g_gn_groups=int(model_cfg.get("func_g_gn_groups", 8)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
        solver_variant=args.solver,
    ).to(device)
    load_checkpoint_for_eval(args.checkpoint, model=model, strict=False)
    model.eval()
    print(f"[model] device={device} solver={args.solver}")

    # 推理 + 收集 per-sample mIoU
    results: list[tuple] = []
    err_counter: dict[str, int] = {}
    sample_pos = 0
    t0 = time.time()

    with torch.inference_mode():
        for batch_idx, sample in enumerate(loader):
            sample_d = move_to_device(sample, device)
            B = sample_d["fast_logits"].shape[0]
            fast_in = sample_d["fast_logits"]
            slow_in = sample_d["slow_logits"]
            ego_in = sample_d["frame_ego2global"]
            ts_in = sample_d.get("frame_timestamps", None)
            dt_in = sample_d.get("frame_dt", None)

            # bs=1 时按 num_real 截断，避免 pad 段无意义 ODE
            n_real_b0 = None
            if B == 1:
                meta_b0 = sample["meta"][0]
                n_real_b0 = int(meta_b0.get("num_real_frames", fast_in.shape[1]))
                if 0 < n_real_b0 < fast_in.shape[1]:
                    fast_in = fast_in[:, :n_real_b0]
                    ego_in = ego_in[:, :n_real_b0]
                    if ts_in is not None:
                        ts_in = ts_in[:, :n_real_b0]
                    if dt_in is not None:
                        dt_in = dt_in[:, :n_real_b0]

            outputs = model.forward_stepwise_eval(
                fast_logits=fast_in,
                slow_logits=slow_in,
                frame_ego2global=ego_in,
                frame_timestamps=ts_in,
                frame_dt=dt_in,
                rollout_start_step=sample_d.get("rollout_start_step", None),
            )
            step_logits = outputs["step_logits"]                    # (B, S, C, X, Y, Z)
            step_indices_arr = outputs["step_indices"].detach().cpu().numpy()
            step_idx_to_local = {int(v): i for i, v in enumerate(step_indices_arr)}
            step_preds = step_logits.argmax(dim=2).to(torch.uint8).cpu().numpy()  # (B, S, X, Y, Z)
            fast_preds = fast_in.argmax(dim=2).to(torch.uint8).cpu().numpy()      # (B, T, X, Y, Z)

            for b in range(B):
                global_idx = base_indices[sample_pos]
                sample_pos += 1
                meta = sample["meta"][b]
                scene = str(meta.get("scene_name", ""))
                ek_tokens = list(meta.get("evolve_keyframe_sample_tokens", []))
                ek_gt_exists = list(meta.get("evolve_keyframe_gt_exists", []))
                max_evolve = int(meta.get("max_evolve_keyframes", -1))
                num_real = int(meta.get("num_real_frames", num_frames))

                if max_evolve <= 0 or max_evolve >= len(ek_tokens):
                    err_counter["max_evolve_invalid"] = err_counter.get("max_evolve_invalid", 0) + 1
                    continue
                if max_evolve < len(ek_gt_exists) and not int(ek_gt_exists[max_evolve]):
                    err_counter["no_end_gt"] = err_counter.get("no_end_gt", 0) + 1
                    continue
                end_token = str(ek_tokens[max_evolve])

                gt, mask = _load_gt(gt_root, scene, end_token, gt_mask_key)
                if gt is None:
                    err_counter["gt_missing"] = err_counter.get("gt_missing", 0) + 1
                    continue

                # aligner 输出在 step = max_evolve * steps_per_kf
                step_d = max_evolve * steps_per_kf
                if step_d not in step_idx_to_local:
                    err_counter["step_not_in_indices"] = err_counter.get("step_not_in_indices", 0) + 1
                    continue
                aligned_pred = step_preds[b, step_idx_to_local[step_d]]

                # fast(curr) = end keyframe 对应的 fast 帧 = 索引 num_real - 1
                # bs=1 已截断到 n_real，[-1] 即 num_real-1；bs>1 时 fast_in 长度 = num_frames，
                # 用 num_real-1 取真实 end frame（pad 部分不会用到）
                fast_idx = num_real - 1
                if fast_idx < 0 or fast_idx >= fast_preds.shape[1]:
                    err_counter["bad_fast_idx"] = err_counter.get("bad_fast_idx", 0) + 1
                    continue
                fast_pred = fast_preds[b, fast_idx]

                am = per_sample_miou(
                    aligned_pred, gt, mask, num_classes, free_index,
                    extra_mask=near_mask, class_subset=class_subset,
                )
                fm = per_sample_miou(
                    fast_pred, gt, mask, num_classes, free_index,
                    extra_mask=near_mask, class_subset=class_subset,
                )
                if np.isnan(am) or np.isnan(fm):
                    err_counter["nan_miou"] = err_counter.get("nan_miou", 0) + 1
                    continue
                results.append((global_idx, fm, am, am - fm, scene, end_token))

            if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == len(loader):
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / max(elapsed, 1e-9)
                print(f"[progress] batch {batch_idx+1}/{len(loader)} "
                      f"({rate:.2f} batch/s, elapsed {elapsed:.0f}s)",
                      flush=True)

    if err_counter:
        print(f"[warn] errors/skips by reason: {err_counter}")
    print(f"[done] valid={len(results)} / total_processed={sample_pos}")

    # 排序
    if args.sort_by == "abs":
        results.sort(key=lambda r: -abs(r[3]))
    elif args.sort_by == "aligned_minus_fast":
        results.sort(key=lambda r: -r[3])
    else:  # fast_minus_aligned
        results.sort(key=lambda r: r[3])

    top = results[: args.top_k]

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# checkpoint: {args.checkpoint}\n")
        f.write(f"# config: {args.config}\n")
        f.write(f"# info_path: {args.info_path}\n")
        f.write(f"# schema: {schema}\n")
        f.write(f"# solver: {args.solver}\n")
        f.write(f"# near_radius_m: {args.near_radius_m}    "
                f"(mIoU 仅在 sqrt(x²+y²) ≤ near_radius ∧ mask_camera 区域内统计)\n")
        if args.metric == "miou_d":
            f.write(f"# metric: miou_d  (dynamic_object_classes={list(OCC3D_DYNAMIC_OBJECT_IDX)})\n")
        else:
            f.write(f"# metric: miou  (all non-free classes)\n")
        f.write(f"# sort_by: {args.sort_by}    diff = aligned_{args.metric} - fast_{args.metric}\n")
        f.write(f"# top_k: {len(top)} / valid={len(results)}\n")
        f.write(f"# columns: idx  fast_{args.metric}  aligned_{args.metric}  diff  scene  end_token\n")
        for idx, fm, am, diff, scene, tok in top:
            f.write(f"{idx}\t{fm:.4f}\t{am:.4f}\t{diff:+.4f}\t{scene}\t{tok}\n")
    print(f"[saved] {out_path}  (top-{len(top)})")


if __name__ == "__main__":
    main()
