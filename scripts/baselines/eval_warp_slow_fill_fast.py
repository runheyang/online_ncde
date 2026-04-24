#!/usr/bin/env python3
"""Baseline 评估：warp_slow_fill_fast（无参数纯几何）。

流程：
  - 复用 Occ3DOnlineNcdeDataset 吃 canonical val pkl。
  - 对每个样本用 WarpSlowFillFastBaseline 预测当前 keyframe 的 dense 语义。
  - 累加 mIoU（MetricMiouOcc3D）+ RayIoU（RayIouAccumulator）。
  - 另外统计 valid_mask 占比、退化样本数。
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
from online_ncde.ops.dvr.ray_metrics import RayIouAccumulator  # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402

try:
    import progressbar
except Exception:
    progressbar = None


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
        "--include-short-history",
        action="store_true",
        help="强制 min_history_completeness=0，评估覆盖短历史样本。",
    )
    parser.add_argument(
        "--val-info-path",
        default="",
        help="覆盖 config 的 data.val_info_path（空则沿用 config）。",
    )
    parser.add_argument("--no-rayiou", action="store_true", help="跳过 RayIoU 评估")
    parser.add_argument(
        "--mask-thresh",
        type=float,
        default=0.5,
        help="warp ones 后的 valid mask 阈值（默认 0.5）",
    )
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

    logits_loader = build_logits_loader(data_cfg, root_path)

    min_hc = 0 if args.include_short_history else int(data_cfg.get("min_history_completeness", 4))
    print(f"[eval-baseline] min_history_completeness={min_hc}"
          + ("  (--include-short-history 强制为 0)" if args.include_short_history else ""))

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
        mask_thresh=float(args.mask_thresh),
    )
    print(f"[baseline] {baseline.name} mask_thresh={baseline.mask_thresh}")

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
    ray_acc: RayIouAccumulator | None = None
    origins_by_token: dict[str, Any] = {}
    missing_origin_count = 0
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)
    if enable_rayiou:
        print(f"[rayiou] 加载 lidar origins: {sweep_info_path}")
        origins_by_token = load_origins_from_sweep_pkl(sweep_info_path)
        print(f"[rayiou] 共 {len(origins_by_token)} 个 token 的 origin")
        ray_acc = RayIouAccumulator()

    valid_ratio_sum = 0.0
    valid_ratio_count = 0
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
            sample = move_to_device(sample, device)
            fast_logits = cast(torch.Tensor, sample["fast_logits"])   # (B, T, C, X, Y, Z)
            slow_logits = cast(torch.Tensor, sample["slow_logits"])   # (B, C, X, Y, Z)
            frame_ego2global = cast(torch.Tensor, sample["frame_ego2global"])  # (B, T, 4, 4)
            rollout_start_step = sample.get("rollout_start_step", None)
            meta_list = cast(list[dict[str, Any]], sample["meta"])

            B = fast_logits.shape[0]
            for b in range(B):
                rss_b = int(rollout_start_step[b].item()) if rollout_start_step is not None else 0
                out = baseline.predict_sample(
                    fast_logits=fast_logits[b],
                    slow_logits=slow_logits[b],
                    frame_ego2global=frame_ego2global[b],
                    rollout_start_step=rss_b,
                )
                pred = cast(torch.Tensor, out["pred"])
                valid_mask = cast(torch.Tensor, out["valid_mask"])

                num_frames = int(fast_logits.shape[1])
                if rss_b >= num_frames - 1:
                    degenerate_count += 1
                else:
                    valid_ratio_sum += float(valid_mask.float().mean().item())
                    valid_ratio_count += 1

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

                if enable_rayiou and ray_acc is not None:
                    origin = origins_by_token.get(token, None)
                    if origin is None:
                        missing_origin_count += 1
                    else:
                        ray_acc.add_sample(pred_np, gt_semantics, origin)

                processed += 1

            if progressbar is None and (batch_idx % log_interval == 0 or batch_idx == total_batches):
                print(f"[baseline] batch={batch_idx}/{total_batches} processed={processed}")

    print(f"[baseline] processed={processed} degenerate={degenerate_count} "
          f"valid_ratio_mean={valid_ratio_sum / max(valid_ratio_count, 1):.4f}")

    if metric_miou.cnt > 0:
        miou = float(metric_miou.count_miou(verbose=False))
        per_class = np.nan_to_num(metric_miou.get_per_class_iou(), nan=0.0).tolist()
        print(f"[miou] num={metric_miou.cnt} miou={miou:.2f}")
        for name, value in zip(class_names, per_class):
            print(f"  {name}: {float(value):.2f}")
    else:
        miou = float("nan")
        per_class = []
        print("[miou] no samples")

    rayiou_result: dict[str, Any] | None = None
    if enable_rayiou and ray_acc is not None:
        if ray_acc.num_samples > 0:
            rayiou_result = ray_acc.finalize(print_table=True)
            print(
                f"[rayiou] num={rayiou_result['num_samples']} "
                f"RayIoU={rayiou_result['RayIoU']:.4f} "
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
            "mask_thresh": baseline.mask_thresh,
            "num_samples": int(metric_miou.cnt),
            "miou": _to_json_number(miou),
            "per_class_iou": [float(v) for v in per_class],
            "class_names": class_names,
            "rayiou": rayiou_result,
            "valid_mask_ratio_mean": _to_json_number(valid_ratio_sum / max(valid_ratio_count, 1)),
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
