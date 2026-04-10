#!/usr/bin/env python3
"""Unified stepwise evaluation for online NCDE with configurable frame sampling.

Two ways to specify which frames to evolve:

  1. --frame-indices "0,3,6,9,12"   (directly specify frame indices)
  2. --random                        (seeded random gap path)

Score modes:
  all_steps    — compute per-step and aggregated metrics for every evolved step
  final_only   — only score the final step

Examples:
  # explicit frame indices, score all steps
  python eval_online_ncde_stepwise.py --checkpoint ckpt.pt --frame-indices "0,3,6,9,12"

  # explicit frame indices, score final step only
  python eval_online_ncde_stepwise.py --checkpoint ckpt.pt --frame-indices "0,1,4,7,12" --score-mode final_only

  # random 6/3/2Hz path, score final step only
  python eval_online_ncde_stepwise.py --checkpoint ckpt.pt --random --gap-choices "1,2,3" --target-last-step 12 --score-mode final_only

  # random with custom seed
  python eval_online_ncde_stepwise.py --checkpoint ckpt.pt --random --gap-choices "1,2,3" --target-last-step 12 --seed 42
"""

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
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.keyframe_mapping import NuScenesKeyFrameResolver  # noqa: E402
from online_ncde.data.labels_io import load_labels_npz  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint  # noqa: E402

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


# ---------------------------------------------------------------------------
# Frame index builders
# ---------------------------------------------------------------------------

def _build_random_frame_indices(
    target_last_step: int,
    gap_choices: list[int],
    seed: int,
) -> tuple[list[int], list[int]]:
    if target_last_step < 1:
        raise ValueError(f"target_last_step must be >= 1, got {target_last_step}")
    rng = np.random.RandomState(seed)
    indices = [0]
    applied_gaps: list[int] = []
    curr = 0
    while curr < target_last_step:
        chosen_gap = int(rng.choice(gap_choices))
        next_step = min(curr + chosen_gap, target_last_step)
        indices.append(next_step)
        applied_gaps.append(next_step - curr)
        curr = next_step
    return indices, applied_gaps


def _parse_int_list(spec: str) -> list[int]:
    values = [item.strip() for item in str(spec).split(",")]
    return [int(item) for item in values if item]


# ---------------------------------------------------------------------------
# Unified dataset wrapper
# ---------------------------------------------------------------------------

def _infer_num_frames(info: dict[str, Any]) -> int:
    for key in ("frame_ego2global", "frame_timestamps", "frame_dt", "frame_tokens"):
        value = info.get(key, None)
        if value is None:
            continue
        if hasattr(value, "shape"):
            shape = getattr(value, "shape")
            if len(shape) > 0:
                return int(shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


class SubsampledStepwiseEvalDataset(Dataset):
    """Wrap the base dataset with arbitrary frame subsampling."""

    def __init__(
        self,
        base_dataset: Occ3DOnlineNcdeDataset,
        sampled_frame_indices: list[int],
    ) -> None:
        if len(sampled_frame_indices) < 2:
            raise ValueError(f"Need at least 2 frame indices, got {sampled_frame_indices}")
        self.base_dataset = base_dataset
        self.sampled_frame_indices = list(sampled_frame_indices)
        self.orig_step_indices = self.sampled_frame_indices[1:]
        self.required_num_frames = max(self.sampled_frame_indices) + 1

        self.valid_indices: list[int] = []
        self.filter_stats = defaultdict(int)
        for idx, info in enumerate(self.base_dataset.infos):
            if not str(info.get("slow_logit_path", "")):
                self.filter_stats["missing_slow_logit_path"] += 1
                continue
            if _infer_num_frames(info) < self.required_num_frames:
                self.filter_stats["short_sequence"] += 1
                continue
            self.valid_indices.append(idx)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _slice_optional(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value[self.sampled_frame_indices]

    def _subsample_frame_dt(
        self,
        frame_dt: torch.Tensor | None,
        total_num_frames: int,
    ) -> torch.Tensor | None:
        if frame_dt is None:
            return None
        if frame_dt.numel() == total_num_frames:
            return frame_dt[self.sampled_frame_indices]
        if frame_dt.numel() == total_num_frames - 1:
            merged_dt = []
            for start_idx, end_idx in zip(self.sampled_frame_indices[:-1], self.sampled_frame_indices[1:]):
                merged_dt.append(frame_dt[start_idx:end_idx].sum())
            return torch.stack(merged_dt, dim=0) if merged_dt else frame_dt.new_zeros((0,))
        return frame_dt.reshape(-1)[: len(self.sampled_frame_indices)]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_idx = self.valid_indices[idx]
        sample = self.base_dataset[base_idx]
        sampled = dict(sample)

        fast_logits = cast(torch.Tensor, sample["fast_logits"])
        frame_ego2global = cast(torch.Tensor, sample["frame_ego2global"])
        if fast_logits.shape[0] < self.required_num_frames:
            raise IndexError(
                f"sample {base_idx} only has {fast_logits.shape[0]} frames, "
                f"need at least {self.required_num_frames}"
            )

        sampled["fast_logits"] = fast_logits[self.sampled_frame_indices]
        sampled["frame_ego2global"] = frame_ego2global[self.sampled_frame_indices]
        frame_timestamps = self._slice_optional(cast(torch.Tensor | None, sample.get("frame_timestamps", None)))
        sampled["frame_timestamps"] = frame_timestamps
        if frame_timestamps is not None:
            sampled["frame_dt"] = None
        else:
            sampled["frame_dt"] = self._subsample_frame_dt(
                cast(torch.Tensor | None, sample.get("frame_dt", None)),
                total_num_frames=int(fast_logits.shape[0]),
            )

        meta = dict(cast(dict[str, Any], sample.get("meta", {})))
        orig_frame_tokens = [str(tok) for tok in meta.get("frame_tokens", [])]
        sampled_frame_tokens = [
            orig_frame_tokens[orig_idx] if orig_idx < len(orig_frame_tokens) else ""
            for orig_idx in self.sampled_frame_indices
        ]
        meta["orig_frame_tokens"] = orig_frame_tokens
        meta["frame_tokens"] = sampled_frame_tokens
        meta["sampled_frame_indices"] = list(self.sampled_frame_indices)
        meta["orig_step_indices"] = list(self.orig_step_indices)
        sampled["meta"] = meta
        return sampled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_avg(value_sum: float, count: int) -> float:
    return value_sum / max(count, 1)


def _to_json_number(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified stepwise evaluation for online NCDE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(ROOT / "configs/online_ncde/fast_opusv1t__slow_opusv2l/eval.yaml"), help="config path")
    parser.add_argument("--checkpoint", required=True, help="checkpoint path")
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N valid samples, 0 for all")
    parser.add_argument("--batch-size", type=int, default=0, help="override eval.batch_size, 0 keeps config")

    # --- frame selection (mutually exclusive) ---
    parser.add_argument(
        "--frame-indices",
        default="",
        help="comma-separated frame indices to evolve, e.g. '0,3,6,9,12'; must start with 0",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="use seeded random gap path instead of explicit --frame-indices",
    )
    # random mode params
    parser.add_argument("--gap-choices", default="1,2,3", help="[random] comma-separated gap candidates")
    parser.add_argument("--seed", type=int, default=0, help="[random] RNG seed for reproducibility")
    parser.add_argument("--target-last-step", type=int, default=12, help="[random] final step to reach")

    # --- score mode ---
    parser.add_argument(
        "--score-mode",
        choices=["all_steps", "final_only"],
        default="all_steps",
        help="all_steps: per-step + aggregated metrics; final_only: only score the last step",
    )

    # --- nuscenes ---
    parser.add_argument("--nusc-dataroot", default="data/nuscenes", help="NuScenes dataroot")
    parser.add_argument("--nusc-version", default="v1.0-trainval", help="NuScenes version")
    parser.add_argument("--sweep-info-path", default="data/nuscenes/nuscenes_infos_val_sweep.pkl", help="sweep pkl")
    parser.add_argument("--dump-json", default="", help="optional json output path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    # --- build sampled_frame_indices ---
    sampling_meta: dict[str, Any] = {}
    applied_gaps: list[int] | None = None

    if args.random:
        gap_choices = _parse_int_list(args.gap_choices)
        if not gap_choices or any(g <= 0 for g in gap_choices):
            raise ValueError(f"gap_choices must be all positive, got {gap_choices}")
        sampled_frame_indices, applied_gaps = _build_random_frame_indices(
            args.target_last_step, gap_choices, args.seed,
        )
        sampling_meta["mode"] = "random"
        sampling_meta["gap_choices"] = gap_choices
        sampling_meta["applied_gaps"] = applied_gaps
        sampling_meta["seed"] = args.seed
        sampling_meta["target_last_step"] = args.target_last_step
    elif args.frame_indices:
        sampled_frame_indices = _parse_int_list(args.frame_indices)
        sampled_frame_indices = sorted(set(sampled_frame_indices))
        if not sampled_frame_indices or sampled_frame_indices[0] != 0:
            raise ValueError(f"frame_indices must start with 0, got {sampled_frame_indices}")
        sampling_meta["mode"] = "explicit"
    else:
        raise ValueError("must specify either --frame-indices or --random")

    sampling_meta["sampled_frame_indices"] = sampled_frame_indices
    sampling_meta["orig_step_indices"] = sampled_frame_indices[1:]

    mode_str = "random" if args.random else "explicit"
    print(f"[config] mode={mode_str} score_mode={args.score_mode}")
    print(f"[config] sampled_frame_indices={sampled_frame_indices} ({len(sampled_frame_indices)} frames, {len(sampled_frame_indices)-1} steps)")
    if applied_gaps is not None:
        print(f"[config] applied_gaps={applied_gaps}")

    # --- dataset ---
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    root_path = cfg["root_path"]

    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root and data.slow_logit_root are required")

    base_dataset = Occ3DOnlineNcdeDataset(
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
        root_path=root_path,
        fast_logits_root=fast_logits_root,
        slow_logit_root=slow_logit_root,
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        slow_noise_std=0.0,
        topk_other_fill_value=data_cfg.get("topk_other_fill_value", -5.0),
        topk_free_fill_value=data_cfg.get("topk_free_fill_value", 5.0),
    )
    dataset = SubsampledStepwiseEvalDataset(
        base_dataset=base_dataset,
        sampled_frame_indices=sampled_frame_indices,
    )
    if args.limit > 0:
        keep = min(args.limit, len(dataset))
        dataset = Subset(dataset, list(range(keep)))

    num_workers = int(eval_cfg.get("num_workers", 4))
    batch_size = int(args.batch_size) if args.batch_size > 0 else int(eval_cfg.get("batch_size", 1))
    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=online_ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(dataset, **kwargs)

    # --- model ---
    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = OnlineNcdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        fast_occ_thresh=data_cfg.get("fast_occ_thresh", 0.25),
        decoder_init_scale=model_cfg.get("decoder_init_scale", 1.0e-3),
        use_fast_residual=bool(model_cfg.get("use_fast_residual", True)),
        func_g_inner_dim=model_cfg.get("func_g_inner_dim", 32),
        func_g_body_dilations=tuple(model_cfg.get("func_g_body_dilations", [1, 2, 3])),
        func_g_gn_groups=int(model_cfg.get("func_g_gn_groups", 8)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
        amp_fp16=bool(eval_cfg.get("amp_fp16", False)),
    ).to(device)
    load_checkpoint(args.checkpoint, model=model, strict=False)
    model.eval()

    # --- keyframe resolver ---
    nusc_dataroot = resolve_path(root_path, args.nusc_dataroot)
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)
    keyframe_resolver = NuScenesKeyFrameResolver(
        dataroot=nusc_dataroot,
        version=args.nusc_version,
        sweep_info_path=sweep_info_path,
    )

    # --- metric setup ---
    num_classes = int(data_cfg["num_classes"])
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    gt_mask_key = data_cfg.get("gt_mask_key", "mask_camera")
    class_names = MetricMiouOcc3D(num_classes=num_classes).class_names

    dataset_ref = dataset.dataset if isinstance(dataset, Subset) else dataset
    orig_step_indices = dataset_ref.orig_step_indices
    local_to_orig_step = {local_step: orig_step for local_step, orig_step in enumerate(orig_step_indices, start=1)}

    score_final_only = args.score_mode == "final_only"
    final_orig_step_idx = sampled_frame_indices[-1]
    final_local_step = len(sampled_frame_indices) - 1

    per_step_metrics: dict[int, MetricMiouOcc3D] = {}
    metric_all = MetricMiouOcc3D(num_classes=num_classes, use_image_mask=True, use_lidar_mask=False)

    step_time_sum = defaultdict(float)
    step_time_count = defaultdict(int)
    keyframe_time_sum = 0.0
    keyframe_time_count = 0
    all_time_sum = 0.0
    all_time_count = 0
    missing_gt_count = 0
    no_scene_count = 0
    no_frame_tokens_count = 0
    missing_final_step_count = 0
    missing_final_keyframe_count = 0

    # --- eval loop ---
    total_steps = len(loader)
    desc = f"[eval stepwise {mode_str}]"
    iterator = progressbar.progressbar(loader, max_value=total_steps, prefix=desc + " ") if progressbar is not None else loader
    log_interval = int(eval_cfg.get("log_interval", 20))

    with torch.inference_mode():
        for batch_idx, sample in enumerate(iterator, start=1):
            sample = move_to_device(sample, device)
            outputs = model.forward_stepwise_eval(
                fast_logits=sample["fast_logits"],
                slow_logits=sample["slow_logits"],
                frame_ego2global=sample["frame_ego2global"],
                frame_timestamps=sample.get("frame_timestamps", None),
                frame_dt=sample.get("frame_dt", None),
            )
            step_logits = cast(torch.Tensor, outputs["step_logits"])
            step_time_ms = cast(torch.Tensor, outputs["step_time_ms"])
            step_indices = cast(torch.Tensor, outputs["step_indices"])
            step_indices_list = [int(v) for v in step_indices.detach().cpu().tolist()]

            meta_list = cast(list[dict[str, Any]], sample["meta"])
            for b, meta in enumerate(meta_list):
                step_times_list = [float(v) for v in step_time_ms[b].detach().cpu().tolist()]
                # accumulate per-step timing
                for local_step_idx, local_step in enumerate(step_indices_list):
                    orig_step_idx = local_to_orig_step.get(local_step, local_step)
                    t_ms = step_times_list[local_step_idx]
                    step_time_sum[orig_step_idx] += t_ms
                    step_time_count[orig_step_idx] += 1
                    all_time_sum += t_ms
                    all_time_count += 1

                scene_name = str(meta.get("scene_name", ""))
                frame_tokens_raw = meta.get("orig_frame_tokens", meta.get("frame_tokens", []))
                frame_tokens = [str(tok) for tok in frame_tokens_raw] if frame_tokens_raw else []
                if not scene_name:
                    no_scene_count += 1
                    continue
                if not frame_tokens or not any(frame_tokens):
                    no_frame_tokens_count += 1
                    continue

                keyframe_steps = keyframe_resolver.resolve_keyframe_steps(frame_tokens)

                if score_final_only:
                    # --- final_only scoring ---
                    if final_local_step not in step_indices_list:
                        missing_final_step_count += 1
                        continue
                    final_local_step_idx = step_indices_list.index(final_local_step)

                    gt_token = keyframe_steps.get(final_orig_step_idx, None)
                    if gt_token is None:
                        missing_final_keyframe_count += 1
                        continue

                    gt_path = os.path.join(gt_root, scene_name, gt_token, "labels.npz")
                    if not os.path.exists(gt_path):
                        missing_gt_count += 1
                        continue

                    gt_npz = load_labels_npz(gt_path)
                    gt_semantics = gt_npz["semantics"]
                    gt_mask = gt_npz.get(gt_mask_key, np.ones(gt_semantics.shape, dtype=np.float32))
                    preds = step_logits[b, final_local_step_idx].argmax(dim=0).detach().cpu().numpy()

                    metric_all.add_batch(
                        semantics_pred=preds,
                        semantics_gt=gt_semantics,
                        mask_lidar=None,
                        mask_camera=gt_mask,
                    )
                    keyframe_time_sum += step_times_list[final_local_step_idx]
                    keyframe_time_count += 1
                else:
                    # --- all_steps scoring ---
                    for local_step_idx, local_step in enumerate(step_indices_list):
                        orig_step_idx = local_to_orig_step.get(local_step, local_step)
                        gt_token = keyframe_steps.get(orig_step_idx, None)
                        if gt_token is None:
                            continue

                        gt_path = os.path.join(gt_root, scene_name, gt_token, "labels.npz")
                        if not os.path.exists(gt_path):
                            missing_gt_count += 1
                            continue

                        gt_npz = load_labels_npz(gt_path)
                        gt_semantics = gt_npz["semantics"]
                        gt_mask = gt_npz.get(gt_mask_key, np.ones(gt_semantics.shape, dtype=np.float32))
                        preds = step_logits[b, local_step_idx].argmax(dim=0).detach().cpu().numpy()

                        metric = per_step_metrics.setdefault(
                            orig_step_idx,
                            MetricMiouOcc3D(num_classes=num_classes, use_image_mask=True, use_lidar_mask=False),
                        )
                        metric.add_batch(
                            semantics_pred=preds,
                            semantics_gt=gt_semantics,
                            mask_lidar=None,
                            mask_camera=gt_mask,
                        )
                        metric_all.add_batch(
                            semantics_pred=preds,
                            semantics_gt=gt_semantics,
                            mask_lidar=None,
                            mask_camera=gt_mask,
                        )
                        t_ms = step_times_list[local_step_idx]
                        keyframe_time_sum += t_ms
                        keyframe_time_count += 1

            if progressbar is None and (batch_idx % log_interval == 0 or batch_idx == total_steps):
                print(f"{desc} batch={batch_idx}/{total_steps}")

    # --- report timing ---
    step_time_avg = {
        step_idx: _safe_avg(step_time_sum[step_idx], step_time_count[step_idx])
        for step_idx in sorted(step_time_count.keys())
    }
    avg_all_step_time_ms = _safe_avg(all_time_sum, all_time_count)
    avg_keyframe_step_time_ms = _safe_avg(keyframe_time_sum, keyframe_time_count)

    print(f"[meta] mode={mode_str} sampled_frame_indices={sampled_frame_indices}")
    print(f"[timing] avg_all_steps_ms={avg_all_step_time_ms:.4f} steps={all_time_count}")
    print(f"[timing] avg_keyframe_steps_ms={avg_keyframe_step_time_ms:.4f} key_steps={keyframe_time_count}")
    for orig_step_idx in sorted(step_time_avg.keys()):
        print(
            f"[timing] orig_step={orig_step_idx} "
            f"avg_ms={step_time_avg[orig_step_idx]:.4f} count={step_time_count[orig_step_idx]}"
        )

    # --- report metrics ---
    per_step_results: dict[str, Any] = {}

    if score_final_only:
        if metric_all.cnt > 0:
            final_miou = float(metric_all.count_miou(verbose=False))
            final_per_class = np.nan_to_num(metric_all.get_per_class_iou(), nan=0.0).tolist()
            print(
                f"[final][orig_step={final_orig_step_idx}] "
                f"num={metric_all.cnt} miou={final_miou:.2f}"
            )
            for name, value in zip(class_names, final_per_class):
                print(f"  {name}: {float(value):.2f}")
        else:
            final_miou = float("nan")
            final_per_class = []
            print(f"[final][orig_step={final_orig_step_idx}] no samples")
    else:
        for orig_step_idx in sorted(per_step_metrics.keys()):
            metric = per_step_metrics[orig_step_idx]
            if metric.cnt == 0:
                print(f"[keyframe][orig_step={orig_step_idx}] no samples")
                per_step_results[str(orig_step_idx)] = {
                    "orig_step_idx": int(orig_step_idx),
                    "num_keyframes": 0,
                    "miou": None,
                    "per_class_iou": [],
                    "class_names": class_names,
                    "avg_time_ms": float(step_time_avg.get(orig_step_idx, 0.0)),
                    "num_step_preds": int(step_time_count.get(orig_step_idx, 0)),
                }
                continue

            step_miou = float(metric.count_miou(verbose=False))
            step_per_class = np.nan_to_num(metric.get_per_class_iou(), nan=0.0).tolist()
            print(
                f"[keyframe][orig_step={orig_step_idx}] "
                f"num={metric.cnt} miou={step_miou:.2f}"
            )
            for name, value in zip(class_names, step_per_class):
                print(f"  {name}: {float(value):.2f}")

            per_step_results[str(orig_step_idx)] = {
                "orig_step_idx": int(orig_step_idx),
                "num_keyframes": int(metric.cnt),
                "miou": float(step_miou),
                "per_class_iou": [float(v) for v in step_per_class],
                "class_names": class_names,
                "avg_time_ms": float(step_time_avg.get(orig_step_idx, 0.0)),
                "num_step_preds": int(step_time_count.get(orig_step_idx, 0)),
            }

        # aggregated all-step metric
        if metric_all.cnt > 0:
            all_miou = float(metric_all.count_miou(verbose=False))
            all_per_class = np.nan_to_num(metric_all.get_per_class_iou(), nan=0.0).tolist()
            print(f"[keyframe][all] num={metric_all.cnt} miou={all_miou:.2f}")
            for name, value in zip(class_names, all_per_class):
                print(f"  {name}: {float(value):.2f}")
        else:
            all_miou = float("nan")
            all_per_class = []
            print("[keyframe][all] no samples")

    # --- filter stats ---
    filter_stats = dict(dataset_ref.filter_stats)
    print(
        f"[meta] missing_gt={missing_gt_count} "
        f"no_scene={no_scene_count} no_frame_tokens={no_frame_tokens_count} "
        f"filtered_missing_slow={filter_stats.get('missing_slow_logit_path', 0)} "
        f"filtered_short_sequence={filter_stats.get('short_sequence', 0)}"
    )
    if score_final_only:
        print(
            f"[meta] missing_final_step={missing_final_step_count} "
            f"missing_final_keyframe={missing_final_keyframe_count}"
        )

    # --- JSON output ---
    if args.dump_json:
        payload: dict[str, Any] = {
            "sampling": sampling_meta,
            "score_mode": args.score_mode,
            "timing": {
                "avg_all_steps_ms": _to_json_number(avg_all_step_time_ms),
                "avg_keyframe_steps_ms": _to_json_number(avg_keyframe_step_time_ms),
                "per_step_avg_ms": {str(k): _to_json_number(v) for k, v in step_time_avg.items()},
                "per_step_count": {str(k): int(v) for k, v in step_time_count.items()},
            },
        }

        if score_final_only:
            payload["final_keyframe"] = {
                "local_step": int(final_local_step),
                "orig_step_idx": int(final_orig_step_idx),
                "num_keyframes": int(metric_all.cnt),
                "miou": _to_json_number(final_miou),
                "per_class_iou": [float(v) for v in final_per_class],
                "class_names": class_names,
            }
        else:
            payload["keyframe_per_step"] = per_step_results
            payload["keyframe_all"] = {
                "num_keyframes": int(metric_all.cnt),
                "miou": _to_json_number(all_miou),
                "per_class_iou": [float(v) for v in all_per_class],
                "class_names": class_names,
            }

        payload["meta"] = {
            "missing_gt_count": int(missing_gt_count),
            "no_scene_count": int(no_scene_count),
            "no_frame_tokens_count": int(no_frame_tokens_count),
            "filtered_missing_slow_logit_path_count": int(filter_stats.get("missing_slow_logit_path", 0)),
            "filtered_short_sequence_count": int(filter_stats.get("short_sequence", 0)),
            "num_batches": int(total_steps),
            "num_valid_samples": int(len(dataset_ref)),
        }
        if score_final_only:
            payload["meta"]["missing_final_step_count"] = int(missing_final_step_count)
            payload["meta"]["missing_final_keyframe_count"] = int(missing_final_keyframe_count)

        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
