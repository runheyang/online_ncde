#!/usr/bin/env python3
"""Evaluate the decoder output from the step-0 hidden state before any rollout."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config path")
    parser.add_argument("--checkpoint", required=True, help="checkpoint path")
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N valid samples, 0 for all")
    parser.add_argument("--batch-size", type=int, default=0, help="override eval.batch_size, 0 keeps config")
    parser.add_argument(
        "--nusc-dataroot",
        default="data/nuscenes",
        help="NuScenes dataroot for frame_token -> keyframe lookup",
    )
    parser.add_argument(
        "--nusc-version",
        default="v1.0-trainval",
        help="NuScenes version",
    )
    parser.add_argument(
        "--sweep-info-path",
        default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
        help="sweep pkl used to bound valid val sample tokens",
    )
    parser.add_argument("--dump-json", default="", help="optional json output path")
    return parser.parse_args()


def _safe_avg(value_sum: float, count: int) -> float:
    return value_sum / max(count, 1)


def _to_json_number(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


class FilteredOnlineNcdeDataset(Dataset):
    """Filter out entries that cannot be evaluated because slow logits are missing."""

    def __init__(self, base_dataset: Occ3DOnlineNcdeDataset) -> None:
        self.base_dataset = base_dataset
        self.valid_indices: list[int] = []
        self.filter_stats: dict[str, int] = {"missing_slow_logit_path": 0}
        for idx, info in enumerate(self.base_dataset.infos):
            if not str(info.get("slow_logit_path", "")):
                self.filter_stats["missing_slow_logit_path"] += 1
                continue
            self.valid_indices.append(idx)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.base_dataset[self.valid_indices[idx]]


def forward_step0_eval(
    model: OnlineNcdeAligner,
    fast_logits: torch.Tensor,
    slow_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the initial hidden state and add the step-0 fast logits residual."""
    if fast_logits.dim() == 5:
        fast_logits = fast_logits.unsqueeze(0)
        slow_logits = slow_logits.unsqueeze(0)

    logits_list: list[torch.Tensor] = []
    time_ms_values: list[float] = []
    step0_time_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    use_cuda_timing = fast_logits.is_cuda

    for b in range(fast_logits.shape[0]):
        fast_seq = fast_logits[b].float()
        slow_now = slow_logits[b].float()

        if use_cuda_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        fast_feat = model._encode_fast(fast_seq)
        slow_feat = model._encode_slow(slow_now)
        if model.use_fast_residual:
            z0_dense = (slow_feat - fast_feat[0]).float()
        else:
            z0_dense = slow_feat.float()

        logits_delta = model._decode_dense_state(z0_dense)
        if model.use_fast_residual:
            logits0 = logits_delta.float() + fast_seq[0].float()
        else:
            logits0 = logits_delta.float()
        logits_list.append(logits0.float())

        if use_cuda_timing:
            end_event.record()
            step0_time_events.append((start_event, end_event))
        else:
            time_ms_values.append((time.perf_counter() - start_time) * 1000.0)

    if use_cuda_timing:
        torch.cuda.synchronize(device=fast_logits.device)
        time_ms_values = [s.elapsed_time(e) for s, e in step0_time_events]

    logits = torch.stack(logits_list, dim=0)
    step0_time_ms = torch.tensor(time_ms_values, device=fast_logits.device, dtype=torch.float32)
    return logits, step0_time_ms


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    root_path = cfg["root_path"]

    logits_loader = build_logits_loader(data_cfg, root_path)

    base_dataset = Occ3DOnlineNcdeDataset(
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
        root_path=root_path,
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        logits_loader=logits_loader,
    )
    dataset = FilteredOnlineNcdeDataset(base_dataset=base_dataset)
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

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = OnlineNcdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
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
    metric_step0 = MetricMiouOcc3D(
        num_classes=num_classes,
        use_image_mask=True,
        use_lidar_mask=False,
    )
    class_names = metric_step0.class_names

    step0_time_sum = 0.0
    step0_time_count = 0
    missing_gt_count = 0
    no_scene_count = 0
    no_frame_tokens_count = 0
    no_step0_keyframe_count = 0

    total_steps = len(loader)
    iterator = progressbar.progressbar(loader, max_value=total_steps, prefix="[eval step0 init] ") if progressbar is not None else loader
    log_interval = int(eval_cfg.get("log_interval", 20))

    with torch.inference_mode():
        for batch_idx, sample in enumerate(iterator, start=1):
            sample = move_to_device(sample, device)
            logits0, step0_time_ms = forward_step0_eval(
                model=model,
                fast_logits=cast(torch.Tensor, sample["fast_logits"]),
                slow_logits=cast(torch.Tensor, sample["slow_logits"]),
            )

            meta_list = cast(list[dict[str, Any]], sample["meta"])
            for b, meta in enumerate(meta_list):
                t_ms = float(step0_time_ms[b].detach().cpu().item())
                step0_time_sum += t_ms
                step0_time_count += 1

                scene_name = str(meta.get("scene_name", ""))
                frame_tokens_raw = meta.get("frame_tokens", [])
                frame_tokens = [str(tok) for tok in frame_tokens_raw] if frame_tokens_raw else []
                if not scene_name:
                    no_scene_count += 1
                    continue
                if not frame_tokens or not any(frame_tokens):
                    no_frame_tokens_count += 1
                    continue

                keyframe_steps = keyframe_resolver.resolve_keyframe_steps(frame_tokens)
                gt_token = keyframe_steps.get(0, None)
                if gt_token is None:
                    no_step0_keyframe_count += 1
                    continue

                gt_path = os.path.join(gt_root, scene_name, gt_token, "labels.npz")
                if not os.path.exists(gt_path):
                    missing_gt_count += 1
                    continue

                gt_npz = load_labels_npz(gt_path)
                gt_semantics = gt_npz["semantics"]
                gt_mask = gt_npz.get(gt_mask_key, np.ones(gt_semantics.shape, dtype=np.float32))
                preds = logits0[b].argmax(dim=0).detach().cpu().numpy()
                metric_step0.add_batch(
                    semantics_pred=preds,
                    semantics_gt=gt_semantics,
                    mask_lidar=None,
                    mask_camera=gt_mask,
                )

            if progressbar is None and (batch_idx % log_interval == 0 or batch_idx == total_steps):
                print(f"[eval step0 init] batch={batch_idx}/{total_steps}")

    avg_step0_time_ms = _safe_avg(step0_time_sum, step0_time_count)
    print(f"[timing] avg_step0_ms={avg_step0_time_ms:.4f} samples={step0_time_count}")

    if metric_step0.cnt > 0:
        step0_miou = float(metric_step0.count_miou(verbose=False))
        step0_per_class = np.nan_to_num(metric_step0.get_per_class_iou(), nan=0.0).tolist()
        print(f"[keyframe][step=0] num={metric_step0.cnt} miou={step0_miou:.2f}")
        for name, value in zip(class_names, step0_per_class):
            print(f"  {name}: {float(value):.2f}")
    else:
        step0_miou = float("nan")
        step0_per_class = []
        print("[keyframe][step=0] no samples")

    dataset_ref = dataset.dataset if isinstance(dataset, Subset) else dataset
    filter_stats = dict(dataset_ref.filter_stats)
    print(
        f"[meta] missing_gt={missing_gt_count} "
        f"no_scene={no_scene_count} no_frame_tokens={no_frame_tokens_count} "
        f"no_step0_keyframe={no_step0_keyframe_count} "
        f"filtered_missing_slow={filter_stats.get('missing_slow_logit_path', 0)}"
    )

    if args.dump_json:
        payload = {
            "timing": {
                "avg_step0_ms": _to_json_number(avg_step0_time_ms),
                "num_timed_samples": int(step0_time_count),
            },
            "keyframe_step0": {
                "num_keyframes": int(metric_step0.cnt),
                "miou": _to_json_number(step0_miou),
                "per_class_iou": [float(v) for v in step0_per_class],
                "class_names": class_names,
            },
            "meta": {
                "missing_gt_count": int(missing_gt_count),
                "no_scene_count": int(no_scene_count),
                "no_frame_tokens_count": int(no_frame_tokens_count),
                "no_step0_keyframe_count": int(no_step0_keyframe_count),
                "filtered_missing_slow_logit_path_count": int(filter_stats.get("missing_slow_logit_path", 0)),
                "num_batches": int(total_steps),
                "num_valid_samples": int(len(dataset_ref)),
            },
        }
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
