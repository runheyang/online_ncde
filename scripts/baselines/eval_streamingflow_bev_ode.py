#!/usr/bin/env python3
"""StreamingFlow-style BEV GRU-ODE baseline 逐步评估脚本。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
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

from online_ncde.baselines import StreamingFlowBEVOdeAligner  # noqa: E402
from online_ncde.config import load_config, load_config_with_base, merge_dict, resolve_path  # noqa: E402
from online_ncde.data.build_dataset import build_online_ncde_dataset  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.keyframe_mapping import NuScenesKeyFrameResolver  # noqa: E402
from online_ncde.data.labels_io import load_labels_npz  # noqa: E402
from online_ncde.metrics import build_miou_metric  # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


STREAMINGFLOW_OVERLAY = (
    ROOT / "src" / "online_ncde" / "baselines" / "streamingflow" / "occ3d_config.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--nusc-dataroot", default="data/nuscenes")
    parser.add_argument("--nusc-version", default="v1.0-trainval")
    parser.add_argument("--sweep-info-path", default="data/nuscenes/nuscenes_infos_val_sweep.pkl")
    parser.add_argument("--dump-json", default="")
    parser.add_argument("--val-info-path", default="")
    parser.add_argument("--exclude-short-history", action="store_true")
    parser.add_argument("--no-rayiou", action="store_true")
    return parser.parse_args()


def _assert_occ3d(data_cfg: dict) -> None:
    if int(data_cfg["num_classes"]) != 18:
        raise ValueError("StreamingFlow baseline 只支持 Occ3D num_classes=18")
    if tuple(data_cfg["grid_size"]) != (200, 200, 16):
        raise ValueError("StreamingFlow baseline 只支持 Occ3D grid_size=(200,200,16)")


def _load_config_with_streamingflow_overlay(config_path: str) -> dict:
    cfg = load_config_with_base(config_path)
    overlay = load_config(str(STREAMINGFLOW_OVERLAY))
    return merge_dict(cfg, overlay)


def _safe_avg(value_sum: float, count: int) -> float:
    return value_sum / max(int(count), 1)


def _to_json_number(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def _build_model(data_cfg: dict, model_cfg: dict) -> StreamingFlowBEVOdeAligner:
    return StreamingFlowBEVOdeAligner(
        num_classes=int(data_cfg["num_classes"]),
        feat_dim=int(model_cfg.get("feat_dim", 192)),
        hidden_dim=int(model_cfg.get("hidden_dim", 192)),
        encoder_in_channels=int(model_cfg.get("encoder_in_channels", 18)),
        free_index=int(data_cfg["free_index"]),
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=model_cfg.get("decoder_init_scale", None),
        timestamp_scale=float(data_cfg.get("timestamp_scale", 1.0e-6)),
        streamingflow_cfg=dict(model_cfg.get("streamingflow", {}) or {}),
    )


def main() -> None:
    args = parse_args()
    cfg = _load_config_with_streamingflow_overlay(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    root_path = cfg["root_path"]
    _assert_occ3d(data_cfg)

    logits_loader = build_logits_loader(data_cfg, root_path)
    min_hc = int(data_cfg.get("min_history_completeness", 4)) if args.exclude_short_history else 0
    info_path = args.val_info_path or data_cfg.get("val_info_path", data_cfg["info_path"])
    dataset = build_online_ncde_dataset(
        data_cfg,
        info_path=info_path,
        root_path=root_path,
        logits_loader=logits_loader,
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
        min_history_completeness=min_hc,
    )
    if args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))

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
    model = _build_model(data_cfg, model_cfg).to(device)
    load_checkpoint_for_eval(args.checkpoint, model=model, strict=False)
    model.eval()

    nusc_dataroot = resolve_path(root_path, args.nusc_dataroot)
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)
    keyframe_resolver = NuScenesKeyFrameResolver(
        dataroot=nusc_dataroot,
        version=args.nusc_version,
        sweep_info_path=sweep_info_path,
    )

    num_classes = int(data_cfg["num_classes"])
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    gt_mask_key = data_cfg.get("gt_mask_key", "mask_camera")
    class_names = build_miou_metric(num_classes=num_classes).class_names

    metric_all = build_miou_metric(num_classes=num_classes, use_image_mask=True, use_lidar_mask=False)
    per_step_metrics = {}

    enable_rayiou = not args.no_rayiou
    per_step_ray: dict[int, Any] = {}
    ray_acc_all: Any | None = None
    origins_by_token = {}
    RayIouAccumulator = None
    if enable_rayiou:
        from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
        from online_ncde.ops.dvr.ray_metrics import RayIouAccumulator as _RayIouAccumulator

        RayIouAccumulator = _RayIouAccumulator
        ray_acc_all = RayIouAccumulator()
        origins_by_token = load_origins_from_sweep_pkl(sweep_info_path)
    missing_origin_count = 0

    step_time_sum = defaultdict(float)
    step_warp_sum = defaultdict(float)
    step_solver_sum = defaultdict(float)
    step_decode_sum = defaultdict(float)
    step_time_count = defaultdict(int)
    missing_gt_count = 0

    iterator = (
        progressbar.progressbar(loader, max_value=len(loader), prefix="[eval sf] ")
        if progressbar is not None
        else loader
    )
    with torch.inference_mode():
        for sample in iterator:
            sample = move_to_device(sample, device)
            outputs = model.forward_stepwise_eval(
                fast_logits=sample["fast_logits"],
                slow_logits=sample["slow_logits"],
                frame_ego2global=sample["frame_ego2global"],
                frame_timestamps=sample.get("frame_timestamps", None),
                frame_dt=sample.get("frame_dt", None),
                rollout_start_step=sample.get("rollout_start_step", None),
            )
            step_logits = cast(torch.Tensor, outputs["step_logits"])
            step_time_ms = cast(torch.Tensor, outputs["step_time_ms"])
            step_warp_ms = cast(torch.Tensor, outputs["step_warp_ms"])
            step_solver_ms = cast(torch.Tensor, outputs["step_solver_ms"])
            step_decode_ms = cast(torch.Tensor, outputs["step_decode_ms"])
            step_indices = [int(v) for v in cast(torch.Tensor, outputs["step_indices"]).cpu().tolist()]

            meta_list = cast(list[dict[str, Any]], sample["meta"])
            for b, meta in enumerate(meta_list):
                frame_tokens = [str(tok) for tok in meta.get("frame_tokens", [])]
                scene_name = str(meta.get("scene_name", ""))
                if not frame_tokens or not scene_name:
                    continue
                keyframe_steps = keyframe_resolver.resolve_keyframe_steps(frame_tokens)

                for local_idx, step_idx in enumerate(step_indices):
                    t_ms = float(step_time_ms[b, local_idx].detach().cpu().item())
                    w_ms = float(step_warp_ms[b, local_idx].detach().cpu().item())
                    s_ms = float(step_solver_ms[b, local_idx].detach().cpu().item())
                    d_ms = float(step_decode_ms[b, local_idx].detach().cpu().item())
                    step_time_sum[step_idx] += t_ms
                    step_warp_sum[step_idx] += w_ms
                    step_solver_sum[step_idx] += s_ms
                    step_decode_sum[step_idx] += d_ms
                    step_time_count[step_idx] += 1

                    gt_token = keyframe_steps.get(step_idx, None)
                    if gt_token is None:
                        continue
                    gt_path = os.path.join(gt_root, scene_name, gt_token, "labels.npz")
                    if not os.path.exists(gt_path):
                        missing_gt_count += 1
                        continue

                    gt_npz = load_labels_npz(gt_path)
                    gt_semantics = gt_npz["semantics"]
                    gt_mask = gt_npz.get(gt_mask_key, np.ones(gt_semantics.shape, dtype=np.float32))
                    preds = step_logits[b, local_idx].argmax(dim=0).detach().cpu().numpy()

                    metric = per_step_metrics.setdefault(
                        step_idx,
                        build_miou_metric(num_classes=num_classes, use_image_mask=True, use_lidar_mask=False),
                    )
                    metric.add_batch(preds, gt_semantics, mask_lidar=None, mask_camera=gt_mask)
                    metric_all.add_batch(preds, gt_semantics, mask_lidar=None, mask_camera=gt_mask)

                    if enable_rayiou:
                        origin = origins_by_token.get(gt_token, None)
                        if origin is None:
                            missing_origin_count += 1
                        else:
                            assert RayIouAccumulator is not None
                            ray_acc = per_step_ray.setdefault(step_idx, RayIouAccumulator())
                            ray_acc.add_sample(preds, gt_semantics, origin)
                            assert ray_acc_all is not None
                            ray_acc_all.add_sample(preds, gt_semantics, origin)

    print("[timing]")
    per_step_results: dict[str, Any] = {}
    for step_idx in sorted(step_time_count.keys()):
        avg_ms = _safe_avg(step_time_sum[step_idx], step_time_count[step_idx])
        avg_warp = _safe_avg(step_warp_sum[step_idx], step_time_count[step_idx])
        avg_solver = _safe_avg(step_solver_sum[step_idx], step_time_count[step_idx])
        avg_decode = _safe_avg(step_decode_sum[step_idx], step_time_count[step_idx])
        print(
            f"  step={step_idx} avg_ms={avg_ms:.4f} warp={avg_warp:.4f} "
            f"solver={avg_solver:.4f} decode={avg_decode:.4f} count={step_time_count[step_idx]}"
        )

        metric = per_step_metrics.get(step_idx)
        if metric is None or metric.cnt == 0:
            continue
        miou = float(metric.count_miou(verbose=False))
        miou_d = float(metric.count_miou_d(verbose=False))
        per_class = np.nan_to_num(metric.get_per_class_iou(), nan=0.0).tolist()
        print(f"[keyframe][step={step_idx}] num={metric.cnt} miou={miou:.2f} miou_d={miou_d:.2f}")
        ray_result = None
        if enable_rayiou and step_idx in per_step_ray and per_step_ray[step_idx].num_samples > 0:
            ray_result = per_step_ray[step_idx].finalize(print_table=False)
            print(f"[rayiou][step={step_idx}] RayIoU={ray_result['RayIoU']:.4f}")
        per_step_results[str(step_idx)] = {
            "num_keyframes": int(metric.cnt),
            "miou": miou,
            "miou_d": _to_json_number(miou_d),
            "per_class_iou": [float(v) for v in per_class],
            "class_names": class_names,
            "avg_time_ms": avg_ms,
            "avg_warp_ms": avg_warp,
            "avg_solver_ms": avg_solver,
            "avg_decode_ms": avg_decode,
            "rayiou": ray_result,
        }

    if metric_all.cnt > 0:
        all_miou = float(metric_all.count_miou(verbose=False))
        all_miou_d = float(metric_all.count_miou_d(verbose=False))
        all_per_class = np.nan_to_num(metric_all.get_per_class_iou(), nan=0.0).tolist()
        print(f"[keyframe][all] num={metric_all.cnt} miou={all_miou:.2f} miou_d={all_miou_d:.2f}")
    else:
        all_miou = float("nan")
        all_miou_d = float("nan")
        all_per_class = []
        print("[keyframe][all] no samples")

    all_ray = None
    if enable_rayiou and ray_acc_all is not None and ray_acc_all.num_samples > 0:
        all_ray = ray_acc_all.finalize(print_table=True)
        print(f"[rayiou][all] RayIoU={all_ray['RayIoU']:.4f}")

    if missing_gt_count:
        print(f"[warn] missing_gt_count={missing_gt_count}")
    if missing_origin_count:
        print(f"[warn] missing_origin_count={missing_origin_count}")

    if args.dump_json:
        payload = {
            "model_kind": "streamingflow-bev-ode",
            "all": {
                "num_keyframes": int(metric_all.cnt),
                "miou": _to_json_number(all_miou),
                "miou_d": _to_json_number(all_miou_d),
                "per_class_iou": [float(v) for v in all_per_class],
                "class_names": class_names,
                "rayiou": all_ray,
            },
            "per_step": per_step_results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_json)), exist_ok=True)
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[json] saved: {args.dump_json}")


if __name__ == "__main__":
    main()
