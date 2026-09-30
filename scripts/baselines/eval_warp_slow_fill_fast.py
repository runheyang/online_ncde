#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from evoocc.baselines import WarpSlowFillFastBaseline  # noqa: E402
from evoocc.config import load_config_with_base, resolve_path  # noqa: E402
from evoocc.data.build_dataset import build_evoocc_dataset  # noqa: E402
from evoocc.data.build_logits_loader import build_logits_loader  # noqa: E402
from evoocc.metrics import build_miou_metric  # noqa: E402
from evoocc.trainer import move_to_device, evoocc_collate  # noqa: E402

try:
    import progressbar
except Exception:
    progressbar = None


class _LastFrameOnlyFastLogitsLoader:
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
    parser.add_argument("--config", required=True, help="配置文件路径（沿用 evoocc config）")
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
    dataset_variant = str(data_cfg.get("dataset_variant", "occ3d")).strip().lower()
    metric_variant = str(data_cfg.get("metric_variant", dataset_variant)).strip().lower()

    raw_logits_loader = build_logits_loader(data_cfg, root_path)
    if dataset_variant == "surroundocc":
        logits_loader = raw_logits_loader
        print("[io] SurroundOcc 使用完整 fast_logits 序列，保留 LIDAR_TOP pose 对齐")
    else:
        logits_loader = _LastFrameOnlyFastLogitsLoader(raw_logits_loader)
        print("[io] fast_logits worker 只返回末帧 (1,C,X,Y,Z)，省 I/O + 内存分配 + IPC 传输")

    min_hc = int(data_cfg.get("min_history_completeness", 4)) if args.exclude_short_history else 0
    print(f"[eval-baseline] min_history_completeness={min_hc}"
          + (f"  (--exclude-short-history 使用 config 阈值 {min_hc})" if args.exclude_short_history else ""))

    info_path = args.val_info_path if args.val_info_path else data_cfg.get("val_info_path", data_cfg["info_path"])
    if args.val_info_path:
        print(f"[eval-baseline] --val-info-path 覆盖 -> {info_path}")

    dataset = build_evoocc_dataset(
        data_cfg,
        info_path=info_path,
        root_path=root_path,
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
        collate_fn=evoocc_collate,
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

    metric_miou = build_miou_metric(
        num_classes=num_classes,
        use_image_mask=True,
        use_lidar_mask=False,
        variant=metric_variant,
    )
    class_names = metric_miou.class_names

    enable_rayiou = not args.no_rayiou
    if enable_rayiou and dataset_variant != "occ3d":
        print("[rayiou] 当前 RayIoU 实现固定 Occ3D label space / pc_range，SurroundOcc 分支自动跳过")
        enable_rayiou = False
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)

    collected: list[dict[str, Any]] = []
    coverage_sum = 0.0
    coverage_count = 0
    degenerate_count = 0  # scene first frame, slow output used directly
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
            fast_logits = cast(torch.Tensor, sample["fast_logits"])
            slow_logits = cast(torch.Tensor, sample["slow_logits"])
            frame_ego2global = cast(torch.Tensor, sample["frame_ego2global"])
            gt_labels = cast(torch.Tensor, sample["gt_labels"])
            gt_mask = cast(torch.Tensor, sample["gt_mask"])
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

                pred_np = pred.detach().cpu().numpy()
                gt_semantics = gt_labels[b].detach().cpu().numpy()
                gt_mask_np = gt_mask[b].detach().cpu().numpy()

                metric_miou.add_batch(
                    semantics_pred=pred_np,
                    semantics_gt=gt_semantics,
                    mask_lidar=None,
                    mask_camera=gt_mask_np,
                )

                meta = meta_list[b]
                token = str(meta.get("token", ""))
                if enable_rayiou:
                    if token:
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
        occupied_iou = (
            float(metric_miou.count_occupied_iou(verbose=False))
            if hasattr(metric_miou, "count_occupied_iou")
            else None
        )
        occupied_text = "" if occupied_iou is None else f" occupied_iou={occupied_iou:.2f}"
        print(f"[miou] num={metric_miou.cnt} miou={miou:.2f} miou_d={miou_d:.2f}{occupied_text}")
        for name, value in zip(class_names, per_class):
            print(f"  {name}: {float(value):.2f}")
    else:
        miou = float("nan")
        miou_d = float("nan")
        occupied_iou = None
        per_class = []
        print("[miou] no samples")

    rayiou_result: dict[str, Any] | None = None
    missing_origin_count = 0
    rayiou_num_samples = 0
    if enable_rayiou:
        from evoocc.ops.dvr.ego_pose import load_origins_from_sweep_pkl
        from evoocc.ops.dvr.ray_metrics import main as calc_rayiou

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

    print(f"[meta] missing_origin={missing_origin_count}")

    if args.dump_json:
        payload = {
            "baseline": baseline.name,
            "fuse_strategy": "logits_blend",
            "num_samples": int(metric_miou.cnt),
            "miou": _to_json_number(miou),
            "miou_d": _to_json_number(miou_d),
            "occupied_iou": None if occupied_iou is None else _to_json_number(occupied_iou),
            "per_class_iou": [float(v) for v in per_class],
            "class_names": class_names,
            "rayiou": rayiou_result,
            "rayiou_num_samples": int(rayiou_num_samples),
            "coverage_mean": _to_json_number(coverage_sum / max(coverage_count, 1)),
            "degenerate_count": int(degenerate_count),
            "missing_origin_count": int(missing_origin_count),
            "dataset_variant": dataset_variant,
            "metric_variant": metric_variant,
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
