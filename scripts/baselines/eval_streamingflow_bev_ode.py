#!/usr/bin/env python3
"""StreamingFlow-style BEV GRU-ODE baseline 评估脚本。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Subset

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from evoocc.baselines import StreamingFlowBEVOdeAligner  # noqa: E402
from evoocc.config import load_config, load_config_with_base, merge_dict, resolve_path  # noqa: E402
from evoocc.data.build_dataset import build_evoocc_dataset  # noqa: E402
from evoocc.data.build_logits_loader import build_logits_loader  # noqa: E402
from evoocc.data.keyframe_mapping import NuScenesKeyFrameResolver  # noqa: E402
from evoocc.evaluation import attach_occ3d_targets, evaluate_dense_occ, make_dense_occ_prediction  # noqa: E402
from evoocc.trainer import move_to_device, evoocc_collate  # noqa: E402
from evoocc.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


STREAMINGFLOW_OVERLAY = (
    ROOT / "src" / "evoocc" / "baselines" / "streamingflow" / "occ3d_config.yaml"
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
    parser.add_argument("--stepwise", action="store_true", help="评估所有 keyframe step；默认只评估当前时刻")
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


def _empty_timing_stats() -> dict[str, Any]:
    return {
        "time_sum": defaultdict(float),
        "warp_sum": defaultdict(float),
        "solver_sum": defaultdict(float),
        "decode_sum": defaultdict(float),
        "count": defaultdict(int),
    }


def _add_timing(stats: dict[str, Any], step_idx: int, t_ms: float, w_ms: float, s_ms: float, d_ms: float) -> None:
    stats["time_sum"][step_idx] += t_ms
    stats["warp_sum"][step_idx] += w_ms
    stats["solver_sum"][step_idx] += s_ms
    stats["decode_sum"][step_idx] += d_ms
    stats["count"][step_idx] += 1


def _print_timing(stats: dict[str, Any], *, final_only: bool = False) -> dict[str, dict[str, float | int]]:
    timing_results: dict[str, dict[str, float | int]] = {}
    print("[timing]")
    for step_idx in sorted(stats["count"].keys()):
        count = int(stats["count"][step_idx])
        avg_ms = _safe_avg(float(stats["time_sum"][step_idx]), count)
        avg_warp = _safe_avg(float(stats["warp_sum"][step_idx]), count)
        avg_solver = _safe_avg(float(stats["solver_sum"][step_idx]), count)
        avg_decode = _safe_avg(float(stats["decode_sum"][step_idx]), count)
        timing_key = "current" if final_only and int(step_idx) == 0 else str(step_idx)
        timing_label = "current" if timing_key == "current" else f"step={step_idx}"
        if timing_key == "current":
            print(f"  {timing_label} avg_ms={avg_ms:.4f} count={count}")
            timing_results[timing_key] = {
                "avg_time_ms": avg_ms,
                "num_predictions": count,
            }
            continue
        print(
            f"  {timing_label} avg_ms={avg_ms:.4f} warp={avg_warp:.4f} "
            f"solver={avg_solver:.4f} decode={avg_decode:.4f} count={count}"
        )
        timing_results[timing_key] = {
            "avg_time_ms": avg_ms,
            "avg_warp_ms": avg_warp,
            "avg_solver_ms": avg_solver,
            "avg_decode_ms": avg_decode,
            "num_step_preds": count,
        }
    return timing_results


def _build_eval_loader_and_model(
    *,
    args: argparse.Namespace,
    data_cfg: dict,
    model_cfg: dict,
    eval_cfg: dict,
    loader_cfg: dict,
    root_path: str,
) -> tuple[DataLoader, StreamingFlowBEVOdeAligner, torch.device]:
    logits_loader = build_logits_loader(data_cfg, root_path)
    min_hc = int(data_cfg.get("min_history_completeness", 4)) if args.exclude_short_history else 0
    info_path = args.val_info_path or data_cfg.get("val_info_path", data_cfg["info_path"])
    dataset = build_evoocc_dataset(
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
        collate_fn=evoocc_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = False
    loader = DataLoader(dataset, **loader_kwargs)

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = _build_model(data_cfg, model_cfg).to(device)
    load_checkpoint_for_eval(args.checkpoint, model=model, strict=False)
    model.eval()
    return loader, model, device


def _measure_forward_ms(device: torch.device, fn) -> tuple[Any, float]:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize(device=device)
        return result, float(start.elapsed_time(end))
    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000.0


def _collect_streamingflow_final_predictions(
    *,
    args: argparse.Namespace,
    data_cfg: dict,
    model_cfg: dict,
    eval_cfg: dict,
    loader_cfg: dict,
    root_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    loader, model, device = _build_eval_loader_and_model(
        args=args,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        eval_cfg=eval_cfg,
        loader_cfg=loader_cfg,
        root_path=root_path,
    )

    grid_size = tuple(int(v) for v in data_cfg["grid_size"])
    timing_stats = _empty_timing_stats()
    predictions: list[dict[str, Any]] = []
    missing_meta_count = 0

    iterator = (
        progressbar.progressbar(loader, max_value=len(loader), prefix="[infer sf-final] ")
        if progressbar is not None
        else loader
    )
    with torch.inference_mode():
        for sample in iterator:
            sample = move_to_device(sample, device)
            outputs, forward_ms = _measure_forward_ms(
                device,
                lambda: model(
                    fast_logits=sample["fast_logits"],
                    slow_logits=sample["slow_logits"],
                    frame_ego2global=sample["frame_ego2global"],
                    frame_timestamps=sample.get("frame_timestamps", None),
                    frame_dt=sample.get("frame_dt", None),
                    rollout_start_step=sample.get("rollout_start_step", None),
                ),
            )
            aligned = cast(torch.Tensor, outputs["aligned"])
            preds = aligned.argmax(dim=1).to(torch.uint8).cpu().numpy()
            if tuple(preds.shape[-3:]) != grid_size:
                raise ValueError(f"preds shape={preds.shape[-3:]} 与 grid_size={grid_size} 不一致")

            batch_size = int(preds.shape[0])
            avg_ms = float(forward_ms) / max(batch_size, 1)
            for _ in range(batch_size):
                _add_timing(timing_stats, 0, avg_ms, 0.0, avg_ms, 0.0)

            meta_list = cast("list[dict[str, Any]]", sample["meta"])
            for b, meta in enumerate(meta_list):
                scene_name = str(meta.get("scene_name", ""))
                token = str(meta.get("token", ""))
                if not scene_name or not token:
                    missing_meta_count += 1
                    continue
                predictions.append(
                    make_dense_occ_prediction(
                        pred=preds[b],
                        scene_name=scene_name,
                        token=token,
                        step_idx=None,
                    )
                )

    print(f"[pred] collected {len(predictions)} current-frame uint8 predictions in memory")
    return predictions, timing_stats, missing_meta_count


def _collect_streamingflow_stepwise_predictions(
    *,
    args: argparse.Namespace,
    data_cfg: dict,
    model_cfg: dict,
    eval_cfg: dict,
    loader_cfg: dict,
    root_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    loader, model, device = _build_eval_loader_and_model(
        args=args,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        eval_cfg=eval_cfg,
        loader_cfg=loader_cfg,
        root_path=root_path,
    )

    nusc_dataroot = resolve_path(root_path, args.nusc_dataroot)
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)
    keyframe_resolver = NuScenesKeyFrameResolver(
        dataroot=nusc_dataroot,
        version=args.nusc_version,
        sweep_info_path=sweep_info_path,
    )

    grid_size = tuple(int(v) for v in data_cfg["grid_size"])
    timing_stats = _empty_timing_stats()
    predictions: list[dict[str, Any]] = []
    missing_keyframe_count = 0

    iterator = (
        progressbar.progressbar(loader, max_value=len(loader), prefix="[infer sf] ")
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
            step_preds = step_logits.argmax(dim=2).to(torch.uint8).cpu().numpy()
            if tuple(step_preds.shape[-3:]) != grid_size:
                raise ValueError(f"step_preds shape={step_preds.shape[-3:]} 与 grid_size={grid_size} 不一致")

            meta_list = cast("list[dict[str, Any]]", sample["meta"])
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
                    _add_timing(timing_stats, step_idx, t_ms, w_ms, s_ms, d_ms)

                    gt_token = keyframe_steps.get(step_idx, None)
                    if gt_token is None:
                        missing_keyframe_count += 1
                        continue
                    predictions.append(
                        make_dense_occ_prediction(
                            pred=step_preds[b, local_idx],
                            scene_name=scene_name,
                            token=str(gt_token),
                            step_idx=step_idx,
                        )
                    )

    print(f"[pred] collected {len(predictions)} uint8 predictions in memory")
    return predictions, timing_stats, missing_keyframe_count


def main() -> None:
    args = parse_args()
    cfg = _load_config_with_streamingflow_overlay(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    root_path = cfg["root_path"]
    _assert_occ3d(data_cfg)

    eval_mode = "stepwise" if args.stepwise else "final"
    collect_fn = (
        _collect_streamingflow_stepwise_predictions
        if args.stepwise
        else _collect_streamingflow_final_predictions
    )
    predictions, timing_stats, missing_pred_meta_count = collect_fn(
        args=args,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        eval_cfg=eval_cfg,
        loader_cfg=loader_cfg,
        root_path=root_path,
    )
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    predictions_with_gt, missing_gt_count = attach_occ3d_targets(
        predictions,
        gt_root=gt_root,
        gt_mask_key=data_cfg.get("gt_mask_key", "mask_camera"),
        grid_size=tuple(int(v) for v in data_cfg["grid_size"]),
    )
    print(f"[target] attached_gt={len(predictions_with_gt)} missing_gt_count={missing_gt_count}")

    result = evaluate_dense_occ(
        predictions_with_gt,
        num_classes=int(data_cfg["num_classes"]),
        enable_rayiou=not args.no_rayiou,
        sweep_pkl=sweep_info_path if not args.no_rayiou else None,
        print_rayiou_table=True,
    )
    num_predictions = int(len(predictions))
    num_evaluated_predictions = int(len(predictions_with_gt))
    del predictions, predictions_with_gt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    timing_results = _print_timing(timing_stats, final_only=not args.stepwise)
    per_step_results = dict(result["per_step"])
    class_names = result["all"].get("class_names", [])
    if args.stepwise:
        for step_idx, timing in timing_results.items():
            payload = per_step_results.setdefault(
                step_idx,
                {
                    "num_keyframes": 0,
                    "miou": None,
                    "miou_d": None,
                    "per_class_iou": [],
                    "class_names": class_names,
                    "rayiou": None,
                },
            )
            payload.update(timing)

    for step_idx in sorted(per_step_results.keys(), key=lambda x: int(x)):
        step_payload = per_step_results[step_idx]
        if int(step_payload.get("num_keyframes", 0)) > 0:
            print(
                f"[keyframe][step={step_idx}] num={step_payload['num_keyframes']} "
                f"miou={float(step_payload['miou']):.2f} "
                f"miou_d={float(step_payload['miou_d']):.2f}"
            )
        ray_result = step_payload.get("rayiou", None)
        if ray_result is not None:
            print(
                f"[rayiou][step={step_idx}] num={ray_result['num_samples']} "
                f"RayIoU={ray_result['RayIoU']:.4f}"
            )

    all_result = result["all"]
    if int(all_result.get("num_keyframes", 0)) > 0:
        print(
            f"[keyframe][all] num={all_result['num_keyframes']} "
            f"miou={float(all_result['miou']):.2f} "
            f"miou_d={float(all_result['miou_d']):.2f}"
        )
    else:
        print("[keyframe][all] no samples")
    if all_result.get("rayiou", None) is not None:
        all_ray = all_result["rayiou"]
        print(
            f"[rayiou][all] num={all_ray['num_samples']} "
            f"RayIoU={all_ray['RayIoU']:.4f} "
            f"@1={all_ray['RayIoU@1']:.4f} "
            f"@2={all_ray['RayIoU@2']:.4f} "
            f"@4={all_ray['RayIoU@4']:.4f}"
        )

    rayiou_meta = result.get("rayiou_meta", None) or {}
    missing_origin_count = int(rayiou_meta.get("missing_origin_count", 0))
    if missing_pred_meta_count:
        warn_key = "missing_keyframe_count" if args.stepwise else "missing_meta_count"
        print(f"[warn] {warn_key}={missing_pred_meta_count}")
    if missing_gt_count:
        print(f"[warn] missing_gt_count={missing_gt_count}")
    if missing_origin_count:
        print(f"[warn] missing_origin_count={missing_origin_count}")

    if args.dump_json:
        payload = {
            "model_kind": "streamingflow-bev-ode",
            "eval_mode": eval_mode,
            "all": all_result,
            "per_step": per_step_results,
            "timing": timing_results,
            "num_predictions": num_predictions,
            "num_evaluated_predictions": num_evaluated_predictions,
            "missing_prediction_meta_count": int(missing_pred_meta_count),
            "missing_gt_count": int(missing_gt_count),
            "missing_origin_count": int(missing_origin_count),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_json)), exist_ok=True)
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[json] saved: {args.dump_json}")


if __name__ == "__main__":
    main()
