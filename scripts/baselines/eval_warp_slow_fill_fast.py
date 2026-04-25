#!/usr/bin/env python3
"""Baseline 评估：warp_slow_fill_fast（无参数纯几何）。

流程（对齐 scripts/eval_online_ncde.py）：
  1. 推理阶段：逐 batch 跑 baseline.predict_sample，累加 mIoU，并把
     每个样本的 dense pred/gt/token 收集到内存。
  2. RayIoU 阶段：推理结束后，按 token 查 lidar origin，一次性调
     ray_metrics.main 统一批量计算 RayIoU@1/2/4。
  - 另外统计连续 coverage 平均值、退化样本数。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde.baselines import WarpSlowFillFastBaseline  # noqa: E402
from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.labels_io import load_labels_npz  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402
from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl  # noqa: E402
from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou  # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402

try:
    import progressbar
except Exception:
    progressbar = None


class _LastFrameOnlyFastLogitsLoader:
    """Wrap 现有 logits_loader，让 load_fast_logits 只返回末帧 (1, C, X, Y, Z)。

    原理：dataset worker 调用 load_fast_logits 前，把 info.frame_rel_paths
    裁剪到只剩末帧路径，inner loader 只解码 1 帧并 stack 出 shape (1, ...)。
    baseline 下游只用 fast_logits[-1]，且 num_frames 从 frame_ego2global
    读取，不依赖 fast_logits.shape[0]。

    相比前一版（前 T-1 填空串走占位分支），这一版同时省：
      - 磁盘 I/O 与 sparse→dense 解码（T 次 → 1 次）
      - worker 侧 dense 张量分配（某些 loader 如 AloccDenseTopkLoader 的
        _empty_frame 仍分配 full dense，占位≠零成本）
      - multiprocessing IPC 传输（worker → 主进程，T 帧 → 1 帧）

    5s pkl (T=31) 下 IPC 传输量从 ~700MB/样本 降到 ~23MB/样本。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def load_fast_logits(self, info: dict, device: torch.device) -> torch.Tensor:
        paths = info.get("frame_rel_paths", None)
        if not paths:
            return self._inner.load_fast_logits(info, device)
        info_view = dict(info)
        info_view["frame_rel_paths"] = [paths[-1]]
        return self._inner.load_fast_logits(info_view, device)

    def load_slow_logits(self, info: dict, device: torch.device) -> torch.Tensor:
        return self._inner.load_slow_logits(info, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径（沿用 online_ncde config）")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 个样本，0 表示全量")
    parser.add_argument("--batch-size", type=int, default=0, help="覆盖 eval.batch_size")
    parser.add_argument(
        "--sweep-info-path",
        default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
        help="sweep pkl（用于 RayIoU lidar origin 查询）",
    )
    parser.add_argument("--dump-json", default="", help="可选：统计结果 json")
    parser.add_argument(
        "--exclude-short-history",
        action="store_true",
        help="只评估满足 config.min_history_completeness（通常 4）的完整历史样本；"
             "默认包含全部短历史样本（min_history_completeness=0，h=0 退化为 slow 直出）。",
    )
    parser.add_argument(
        "--val-info-path",
        default="",
        help="覆盖 config 的 data.val_info_path（空则沿用 config）。",
    )
    parser.add_argument("--no-rayiou", action="store_true", help="跳过 RayIoU 评估")
    return parser.parse_args()


def _to_json_number(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    root_path = cfg["root_path"]

    # baseline 只用 fast_logits[-1]，用 wrapper 屏蔽前 T-1 帧的真实解码
    logits_loader = _LastFrameOnlyFastLogitsLoader(build_logits_loader(data_cfg, root_path))
    print("[io] fast_logits worker 只返回末帧 (1,C,X,Y,Z)，省 I/O + 内存分配 + IPC 传输")

    min_hc = int(data_cfg.get("min_history_completeness", 4)) if args.exclude_short_history else 0
    print(f"[eval-baseline] min_history_completeness={min_hc}"
          + (f"  (--exclude-short-history 使用 config 阈值 {min_hc})" if args.exclude_short_history else ""))

    info_path = args.val_info_path if args.val_info_path else data_cfg.get("val_info_path", data_cfg["info_path"])
    if args.val_info_path:
        print(f"[eval-baseline] --val-info-path 覆盖 -> {info_path}")

    dataset = Occ3DOnlineNcdeDataset(
        info_path=info_path,
        root_path=root_path,
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        logits_loader=logits_loader,
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
        min_history_completeness=min_hc,
    )
    if args.limit > 0:
        keep = min(args.limit, len(dataset))
        dataset = Subset(dataset, list(range(keep)))

    num_workers = int(eval_cfg.get("num_workers", 4))
    batch_size = int(args.batch_size) if args.batch_size > 0 else int(eval_cfg.get("batch_size", 1))
    loader_kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=online_ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(dataset, **loader_kwargs)

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")

    baseline = WarpSlowFillFastBaseline(
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        free_index=int(data_cfg["free_index"]),
    )
    print(f"[baseline] {baseline.name} (logits-level blend)")

    num_classes = int(data_cfg["num_classes"])
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    gt_mask_key = data_cfg.get("gt_mask_key", "mask_camera")

    metric_miou = MetricMiouOcc3D(
        num_classes=num_classes,
        use_image_mask=True,
        use_lidar_mask=False,
    )
    class_names = metric_miou.class_names

    enable_rayiou = not args.no_rayiou
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)

    # 推理阶段累积预测；RayIoU 阶段统一批量计算（不做流式 per-sample raycast）
    collected: list[dict[str, Any]] = []
    coverage_sum = 0.0
    coverage_count = 0
    degenerate_count = 0  # scene 首帧退化（slow 直出）
    missing_gt_count = 0
    processed = 0

    total_batches = len(loader)
    iterator = (
        progressbar.progressbar(loader, max_value=total_batches, prefix="[baseline] ")
        if progressbar is not None
        else loader
    )
    log_interval = int(eval_cfg.get("log_interval", 100))

    with torch.inference_mode():
        for batch_idx, sample in enumerate(iterator, start=1):
            # worker 侧 wrapper 已将 fast_logits 裁到末帧 (1, C, X, Y, Z)，
            # 主进程无需再切片；num_frames 的真值由 frame_ego2global 提供。
            sample = move_to_device(sample, device)
            fast_logits = cast(torch.Tensor, sample["fast_logits"])   # (B, 1, C, X, Y, Z)
            slow_logits = cast(torch.Tensor, sample["slow_logits"])   # (B, C, X, Y, Z)
            frame_ego2global = cast(torch.Tensor, sample["frame_ego2global"])  # (B, T, 4, 4)
            rollout_start_step = sample.get("rollout_start_step", None)
            meta_list = cast(list[dict[str, Any]], sample["meta"])

            B = fast_logits.shape[0]
            num_frames = int(frame_ego2global.shape[1])
            for b in range(B):
                rss_b = int(rollout_start_step[b].item()) if rollout_start_step is not None else 0
                out = baseline.predict_sample(
                    fast_logits=fast_logits[b],
                    slow_logits=slow_logits[b],
                    frame_ego2global=frame_ego2global[b],
                    rollout_start_step=rss_b,
                )
                pred = cast(torch.Tensor, out["pred"])
                coverage = cast(torch.Tensor, out["coverage"])

                if rss_b >= num_frames - 1:
                    degenerate_count += 1
                else:
                    coverage_sum += float(coverage.mean().item())
                    coverage_count += 1

                meta = meta_list[b]
                scene_name = str(meta.get("scene_name", ""))
                token = str(meta.get("token", ""))
                if not scene_name or not token:
                    continue

                gt_path = os.path.join(gt_root, scene_name, token, "labels.npz")
                if not os.path.exists(gt_path):
                    missing_gt_count += 1
                    continue

                gt_npz = load_labels_npz(gt_path)
                gt_semantics = gt_npz["semantics"]
                gt_mask = gt_npz.get(gt_mask_key, np.ones(gt_semantics.shape, dtype=np.float32))
                pred_np = pred.detach().cpu().numpy()

                metric_miou.add_batch(
                    semantics_pred=pred_np,
                    semantics_gt=gt_semantics,
                    mask_lidar=None,
                    mask_camera=gt_mask,
                )

                if enable_rayiou:
                    # 只存 RayIoU 所需字段，用 uint8 省内存（~640KB/样本）
                    collected.append({
                        "pred": pred_np.astype(np.uint8),
                        "gt": gt_semantics.astype(np.uint8),
                        "token": token,
                    })

                processed += 1

            if progressbar is None and (batch_idx % log_interval == 0 or batch_idx == total_batches):
                print(f"[baseline] batch={batch_idx}/{total_batches} processed={processed}")

    print(f"[baseline] processed={processed} degenerate={degenerate_count} "
          f"coverage_mean={coverage_sum / max(coverage_count, 1):.4f}")

    if metric_miou.cnt > 0:
        miou = float(metric_miou.count_miou(verbose=False))
        miou_d = float(metric_miou.count_miou_d(verbose=False))
        per_class = np.nan_to_num(metric_miou.get_per_class_iou(), nan=0.0).tolist()
        print(f"[miou] num={metric_miou.cnt} miou={miou:.2f} miou_d={miou_d:.2f}")
        for name, value in zip(class_names, per_class):
            print(f"  {name}: {float(value):.2f}")
    else:
        miou = float("nan")
        miou_d = float("nan")
        per_class = []
        print("[miou] no samples")

    rayiou_result: dict[str, Any] | None = None
    missing_origin_count = 0
    rayiou_num_samples = 0
    if enable_rayiou:
        print(f"\n[rayiou] 加载 lidar origins: {sweep_info_path}")
        origins_by_token = load_origins_from_sweep_pkl(sweep_info_path)
        print(f"[rayiou] 共 {len(origins_by_token)} 个 token 的 origin")

        sem_pred_list: list[np.ndarray] = []
        sem_gt_list: list[np.ndarray] = []
        lidar_origin_list: list[Any] = []
        for item in collected:
            origin = origins_by_token.get(item["token"], None)
            if origin is None:
                missing_origin_count += 1
                continue
            sem_pred_list.append(item["pred"])
            sem_gt_list.append(item["gt"])
            lidar_origin_list.append(origin)
        if missing_origin_count:
            print(f"[rayiou] 跳过 {missing_origin_count} 个样本（无对应 lidar origin）")
        rayiou_num_samples = len(sem_pred_list)
        print(f"[rayiou] {rayiou_num_samples} 个样本参与计算")

        if rayiou_num_samples > 0:
            rayiou_result = calc_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list)
            print(
                f"[rayiou] RayIoU={rayiou_result['RayIoU']:.4f} "
                f"@1={rayiou_result['RayIoU@1']:.4f} "
                f"@2={rayiou_result['RayIoU@2']:.4f} "
                f"@4={rayiou_result['RayIoU@4']:.4f}"
            )
        else:
            print("[rayiou] no samples")

    print(f"[meta] missing_gt={missing_gt_count} missing_origin={missing_origin_count}")

    if args.dump_json:
        payload = {
            "baseline": baseline.name,
            "fuse_strategy": "logits_blend",
            "num_samples": int(metric_miou.cnt),
            "miou": _to_json_number(miou),
            "miou_d": _to_json_number(miou_d),
            "per_class_iou": [float(v) for v in per_class],
            "class_names": class_names,
            "rayiou": rayiou_result,
            "rayiou_num_samples": int(rayiou_num_samples),
            "coverage_mean": _to_json_number(coverage_sum / max(coverage_count, 1)),
            "degenerate_count": int(degenerate_count),
            "missing_gt_count": int(missing_gt_count),
            "missing_origin_count": int(missing_origin_count),
            "config": str(args.config),
            "val_info_path": str(info_path),
        }
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
