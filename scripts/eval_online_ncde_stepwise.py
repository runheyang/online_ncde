#!/usr/bin/env python3
"""Online NCDE 逐步评估：每步解码 + key_frame IoU + 推理时延。"""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.keyframe_mapping import NuScenesKeyFrameResolver  # noqa: E402
from online_ncde.data.labels_io import load_labels_npz  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde_200x200x16 import OnlineNcdeAligner200              # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/online_ncde/fast_opusv1t__slow_opusv2l/eval.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--200x200x16", dest="use_200", action="store_true",
                        help="使用 200x200x16 分辨率模型结构")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 个样本，0 表示全量")
    parser.add_argument("--batch-size", type=int, default=0, help="覆盖配置中的 eval.batch_size，0 表示不覆盖")
    parser.add_argument(
        "--nusc-dataroot",
        default="data/nuscenes",
        help="NuScenes dataroot（用于 frame_token -> key_frame 解析）",
    )
    parser.add_argument(
        "--nusc-version",
        default="v1.0-trainval",
        help="NuScenes 版本（默认 v1.0-trainval）",
    )
    parser.add_argument(
        "--sweep-info-path",
        default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
        help="sweep pkl 路径（用于限定 val sample token 范围）",
    )
    parser.add_argument("--dump-json", default="", help="可选：将统计结果写入 json")
    return parser.parse_args()


def _safe_avg(value_sum: float, count: int) -> float:
    return value_sum / max(count, 1)


def _to_json_number(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    root_path = cfg["root_path"]

    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root 和 data.slow_logit_root 为必填项。")

    # 按 data.logits_format 构造 LogitsLoader（alocc/composite/opus_sparse_full 等走新路径）
    logits_loader = build_logits_loader(data_cfg, root_path)

    dataset = Occ3DOnlineNcdeDataset(
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
        root_path=root_path,
        fast_logits_root=fast_logits_root,
        slow_logit_root=slow_logit_root,
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        topk_other_fill_value=data_cfg.get("topk_other_fill_value", -5.0),
        topk_free_fill_value=data_cfg.get("topk_free_fill_value", 5.0),
        fast_logits_variant=data_cfg.get("fast_logits_variant", "topk"),
        slow_logit_variant=data_cfg.get("slow_logit_variant", "topk"),
        full_logits_clamp_min=data_cfg.get("full_logits_clamp_min", None),
        full_topk_k=data_cfg.get("full_topk_k", 3),
        logits_loader=logits_loader,
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
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

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    ModelClass = OnlineNcdeAligner200 if args.use_200 else OnlineNcdeAligner
    model = ModelClass(
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
    class_names = MetricMiouOcc3D(num_classes=num_classes).class_names

    per_step_metrics: dict[int, MetricMiouOcc3D] = {}
    metric_all = MetricMiouOcc3D(
        num_classes=num_classes,
        use_image_mask=True,
        use_lidar_mask=False,
    )

    step_time_sum = defaultdict(float)
    step_time_count = defaultdict(int)
    # 分段计时：warp / solver / decode
    step_warp_sum = defaultdict(float)
    step_solver_sum = defaultdict(float)
    step_decode_sum = defaultdict(float)
    keyframe_time_sum = 0.0
    keyframe_time_count = 0
    keyframe_warp_sum = 0.0
    keyframe_solver_sum = 0.0
    keyframe_decode_sum = 0.0
    all_time_sum = 0.0
    all_time_count = 0
    all_warp_sum = 0.0
    all_solver_sum = 0.0
    all_decode_sum = 0.0
    missing_gt_count = 0
    no_scene_count = 0
    no_frame_tokens_count = 0

    total_steps = len(loader)
    iterator = progressbar.progressbar(loader, max_value=total_steps, prefix="[eval stepwise] ") if progressbar is not None else loader
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
            step_logits = cast(torch.Tensor, outputs["step_logits"])  # (B, S, C, X, Y, Z)
            step_time_ms = cast(torch.Tensor, outputs["step_time_ms"])  # (B, S)
            step_warp_ms = cast(torch.Tensor, outputs["step_warp_ms"])  # (B, S)
            step_solver_ms = cast(torch.Tensor, outputs["step_solver_ms"])  # (B, S)
            step_decode_ms = cast(torch.Tensor, outputs["step_decode_ms"])  # (B, S)
            step_indices = cast(torch.Tensor, outputs["step_indices"])  # (S,)
            step_indices_list = [int(v) for v in step_indices.detach().cpu().tolist()]

            meta_list = cast(list[dict[str, Any]], sample["meta"])
            for b, meta in enumerate(meta_list):
                step_times_list = [float(v) for v in step_time_ms[b].detach().cpu().tolist()]
                step_warp_list = [float(v) for v in step_warp_ms[b].detach().cpu().tolist()]
                step_solver_list = [float(v) for v in step_solver_ms[b].detach().cpu().tolist()]
                step_decode_list = [float(v) for v in step_decode_ms[b].detach().cpu().tolist()]
                for local_step_idx, step_idx in enumerate(step_indices_list):
                    t_ms = step_times_list[local_step_idx]
                    w_ms = step_warp_list[local_step_idx]
                    s_ms = step_solver_list[local_step_idx]
                    d_ms = step_decode_list[local_step_idx]
                    step_time_sum[step_idx] += t_ms
                    step_warp_sum[step_idx] += w_ms
                    step_solver_sum[step_idx] += s_ms
                    step_decode_sum[step_idx] += d_ms
                    step_time_count[step_idx] += 1
                    all_time_sum += t_ms
                    all_warp_sum += w_ms
                    all_solver_sum += s_ms
                    all_decode_sum += d_ms
                    all_time_count += 1

                scene_name = str(meta.get("scene_name", ""))
                frame_tokens_raw = meta.get("frame_tokens", [])
                frame_tokens = [str(tok) for tok in frame_tokens_raw] if frame_tokens_raw else []
                if not scene_name:
                    no_scene_count += 1
                    continue
                if not frame_tokens:
                    no_frame_tokens_count += 1
                    continue

                # key_frame 判定不依赖当前对齐 pkl 的扩展字段，直接用 NuScenes sample_data 元数据。
                keyframe_steps = keyframe_resolver.resolve_keyframe_steps(frame_tokens)

                for local_step_idx, step_idx in enumerate(step_indices_list):
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
                    preds = step_logits[b, local_step_idx].argmax(dim=0).detach().cpu().numpy()

                    metric = per_step_metrics.setdefault(
                        step_idx,
                        MetricMiouOcc3D(
                            num_classes=num_classes,
                            use_image_mask=True,
                            use_lidar_mask=False,
                        ),
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
                    keyframe_warp_sum += step_warp_list[local_step_idx]
                    keyframe_solver_sum += step_solver_list[local_step_idx]
                    keyframe_decode_sum += step_decode_list[local_step_idx]
                    keyframe_time_count += 1

            if progressbar is None and (batch_idx % log_interval == 0 or batch_idx == total_steps):
                print(f"[eval stepwise] batch={batch_idx}/{total_steps}")

    step_time_avg = {
        step_idx: _safe_avg(step_time_sum[step_idx], step_time_count[step_idx])
        for step_idx in sorted(step_time_count.keys())
    }
    step_warp_avg = {
        step_idx: _safe_avg(step_warp_sum[step_idx], step_time_count[step_idx])
        for step_idx in sorted(step_time_count.keys())
    }
    step_solver_avg = {
        step_idx: _safe_avg(step_solver_sum[step_idx], step_time_count[step_idx])
        for step_idx in sorted(step_time_count.keys())
    }
    step_decode_avg = {
        step_idx: _safe_avg(step_decode_sum[step_idx], step_time_count[step_idx])
        for step_idx in sorted(step_time_count.keys())
    }
    avg_all_step_time_ms = _safe_avg(all_time_sum, all_time_count)
    avg_all_warp_ms = _safe_avg(all_warp_sum, all_time_count)
    avg_all_solver_ms = _safe_avg(all_solver_sum, all_time_count)
    avg_all_decode_ms = _safe_avg(all_decode_sum, all_time_count)
    avg_keyframe_step_time_ms = _safe_avg(keyframe_time_sum, keyframe_time_count)
    avg_keyframe_warp_ms = _safe_avg(keyframe_warp_sum, keyframe_time_count)
    avg_keyframe_solver_ms = _safe_avg(keyframe_solver_sum, keyframe_time_count)
    avg_keyframe_decode_ms = _safe_avg(keyframe_decode_sum, keyframe_time_count)

    print(f"[timing] avg_all_steps_ms={avg_all_step_time_ms:.4f} steps={all_time_count}")
    print(
        f"[timing]   warp={avg_all_warp_ms:.4f} "
        f"solver={avg_all_solver_ms:.4f} decode={avg_all_decode_ms:.4f}"
    )
    print(f"[timing] avg_keyframe_steps_ms={avg_keyframe_step_time_ms:.4f} key_steps={keyframe_time_count}")
    print(
        f"[timing]   warp={avg_keyframe_warp_ms:.4f} "
        f"solver={avg_keyframe_solver_ms:.4f} decode={avg_keyframe_decode_ms:.4f}"
    )
    for step_idx in sorted(step_time_avg.keys()):
        print(
            f"[timing] step={step_idx} avg_ms={step_time_avg[step_idx]:.4f} "
            f"warp={step_warp_avg[step_idx]:.4f} solver={step_solver_avg[step_idx]:.4f} "
            f"decode={step_decode_avg[step_idx]:.4f} count={step_time_count[step_idx]}"
        )

    per_step_results: dict[str, Any] = {}
    for step_idx in sorted(step_time_avg.keys()):
        metric = per_step_metrics.get(step_idx, None)
        if metric is None or metric.cnt == 0:
            print(f"[keyframe][step={step_idx}] no samples")
            per_step_results[str(step_idx)] = {
                "num_keyframes": 0,
                "miou": None,
                "per_class_iou": [],
                "class_names": class_names,
                "avg_time_ms": float(step_time_avg[step_idx]),
                "avg_warp_ms": float(step_warp_avg[step_idx]),
                "avg_solver_ms": float(step_solver_avg[step_idx]),
                "avg_decode_ms": float(step_decode_avg[step_idx]),
                "num_step_preds": int(step_time_count[step_idx]),
            }
            continue

        step_miou = float(metric.count_miou(verbose=False))
        step_per_class = np.nan_to_num(metric.get_per_class_iou(), nan=0.0).tolist()
        print(
            f"[keyframe][step={step_idx}] "
            f"num={metric.cnt} miou={step_miou:.2f}"
        )
        for name, value in zip(class_names, step_per_class):
            print(f"  {name}: {float(value):.2f}")

        per_step_results[str(step_idx)] = {
            "num_keyframes": int(metric.cnt),
            "miou": float(step_miou),
            "per_class_iou": [float(v) for v in step_per_class],
            "class_names": class_names,
            "avg_time_ms": float(step_time_avg[step_idx]),
            "avg_warp_ms": float(step_warp_avg[step_idx]),
            "avg_solver_ms": float(step_solver_avg[step_idx]),
            "avg_decode_ms": float(step_decode_avg[step_idx]),
            "num_step_preds": int(step_time_count[step_idx]),
        }

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

    print(
        f"[meta] missing_gt={missing_gt_count} "
        f"no_scene={no_scene_count} no_frame_tokens={no_frame_tokens_count}"
    )

    if args.dump_json:
        payload = {
            "timing": {
                "avg_all_steps_ms": _to_json_number(avg_all_step_time_ms),
                "avg_all_warp_ms": _to_json_number(avg_all_warp_ms),
                "avg_all_solver_ms": _to_json_number(avg_all_solver_ms),
                "avg_all_decode_ms": _to_json_number(avg_all_decode_ms),
                "avg_keyframe_steps_ms": _to_json_number(avg_keyframe_step_time_ms),
                "avg_keyframe_warp_ms": _to_json_number(avg_keyframe_warp_ms),
                "avg_keyframe_solver_ms": _to_json_number(avg_keyframe_solver_ms),
                "avg_keyframe_decode_ms": _to_json_number(avg_keyframe_decode_ms),
                "per_step_avg_ms": {k: _to_json_number(v) for k, v in step_time_avg.items()},
                "per_step_warp_ms": {k: _to_json_number(v) for k, v in step_warp_avg.items()},
                "per_step_solver_ms": {k: _to_json_number(v) for k, v in step_solver_avg.items()},
                "per_step_decode_ms": {k: _to_json_number(v) for k, v in step_decode_avg.items()},
                "per_step_count": {str(k): int(v) for k, v in step_time_count.items()},
            },
            "keyframe_per_step": per_step_results,
            "keyframe_all": {
                "num_keyframes": int(metric_all.cnt),
                "miou": _to_json_number(all_miou),
                "per_class_iou": [float(v) for v in all_per_class],
                "class_names": class_names,
            },
            "meta": {
                "missing_gt_count": int(missing_gt_count),
                "no_scene_count": int(no_scene_count),
                "no_frame_tokens_count": int(no_frame_tokens_count),
                "num_batches": int(total_steps),
            },
        }
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
