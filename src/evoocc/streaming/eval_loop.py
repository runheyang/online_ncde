"""Streaming eval 公共流程.

入口脚本只负责构建 fast backend 和 scenes_meta；这里统一处理普通/延迟 slow 注入、
mIoU、RayIoU 和 JSON 汇总。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from evoocc.evaluation import attach_dense_occ_targets, compute_dense_miou
from evoocc.streaming.slow_cache import SlowLogitsGPUCache
from evoocc.streaming.slow_schedule import schedule_slow_steps
from evoocc.streaming.stream_aligner import StreamAligner
from evoocc.streaming.streaming_loader import make_streaming_loader, scatter_to_device


PredictionList = List[Dict[str, Any]]
GtCache = Dict[Tuple[str, str], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]


def cache_one_scene_fast(fast, kf_list, loader_iter) -> List[torch.Tensor]:
    """跑 1 个 scene 的 fast forward，缓存 dense logits fp16."""
    fast.reset_history()
    cache = []
    for _ in kf_list:
        raw = next(loader_iter)
        batch = scatter_to_device(raw, 0)
        fast_logits = fast.forward_keyframe(batch)
        cache.append(fast_logits.to(torch.float16))
    return cache


def _prediction_record(meta, step: int, pred: np.ndarray) -> Dict[str, Any]:
    """构造 dense occupancy 评估记录。"""
    return {
        "token": meta.sample_token,
        "scene_name": meta.scene_name,
        "step_idx": int(step),
        "pred": pred,
    }


def stream_scene_one_interval(
    kf_list,
    scene_fast_cache: List[torch.Tensor],
    slow_cache: Optional[SlowLogitsGPUCache],
    slow_decoder_fn: Callable[[str], torch.Tensor],
    stream_aligner: StreamAligner,
    slow_interval: float,
    device: torch.device,
    pred_list_out: PredictionList,
) -> Tuple[int, int]:
    """普通 streaming：slow 到达即 reset，其余 keyframe evolve."""
    n_reset, n_evolve = 0, 0
    ts = [m.timestamp_us for _, m in kf_list]
    slow_steps = schedule_slow_steps(ts, slow_interval)
    stream_aligner.reset_scene()

    for step, ((_idx, meta), fl_fp16) in enumerate(zip(kf_list, scene_fast_cache)):
        fast_logits = fl_fp16.float()
        ego_t = torch.from_numpy(meta.ego2global).to(device=device, dtype=torch.float32)
        slow_available = (
            step in slow_steps
            and (
                slow_cache.has(meta.slow_logit_path)
                if slow_cache is not None
                else os.path.exists(meta.slow_logit_path)
            )
        )
        if slow_available:
            slow_logits = (
                slow_cache.get(meta.slow_logit_path)
                if slow_cache is not None
                else slow_decoder_fn(meta.slow_logit_path)
            )
            aligned = stream_aligner.reset_with_slow(
                fast_logits, slow_logits, ego_t, meta.timestamp_us
            )
            n_reset += 1
        else:
            aligned = stream_aligner.evolve(fast_logits, ego_t, meta.timestamp_us)
            n_evolve += 1

        pred_uint8 = aligned.argmax(0).to(torch.uint8).cpu().numpy()
        pred_list_out.append(_prediction_record(meta, step, pred_uint8))

    return n_reset, n_evolve


def schedule_delayed_slow_steps(
    timestamps_us: Iterable[int],
    interval_sec: float,
    delay_kf: int,
) -> set:
    """delayed 场景的 slow 到达调度，首份 slow 在 step=delay_kf 到达."""
    ts = list(timestamps_us)
    used = set()
    if len(ts) <= delay_kf or interval_sec < 0:
        return used
    used.add(delay_kf)
    last_t = ts[delay_kf] / 1e6
    for i in range(delay_kf + 1, len(ts)):
        t = ts[i] / 1e6
        if t - last_t + 1e-3 >= interval_sec:
            used.add(i)
            last_t = t
    return used


@torch.no_grad()
def delayed_reset_and_evolve(
    stream_aligner: StreamAligner,
    *,
    kf_list,
    scene_fast_cache: List[torch.Tensor],
    prev_step: int,
    curr_step: int,
    slow_logits_prev: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """slow 对应 prev_step，逐 2Hz evolve 到 curr_step 后输出当前 logits."""
    _, meta_prev = kf_list[prev_step]
    fast_logits_prev = scene_fast_cache[prev_step].float()
    ego_prev = torch.from_numpy(meta_prev.ego2global).to(device=device, dtype=torch.float32)
    stream_aligner.reset_with_slow(
        fast_logits_prev, slow_logits_prev, ego_prev, meta_prev.timestamp_us
    )

    aligned = None
    for k in range(prev_step + 1, curr_step + 1):
        _, meta_k = kf_list[k]
        fast_logits_k = scene_fast_cache[k].float()
        ego_k = torch.from_numpy(meta_k.ego2global).to(device=device, dtype=torch.float32)
        aligned = stream_aligner.evolve(fast_logits_k, ego_k, meta_k.timestamp_us)
    assert aligned is not None, "curr_step must be > prev_step"
    return aligned


def stream_scene_one_interval_delayed(
    kf_list,
    scene_fast_cache: List[torch.Tensor],
    slow_cache: Optional[SlowLogitsGPUCache],
    slow_decoder_fn: Callable[[str], torch.Tensor],
    stream_aligner: StreamAligner,
    slow_interval: float,
    slow_delay_kf: int,
    device: torch.device,
    pred_list_out: PredictionList,
) -> Tuple[int, int, int]:
    """delayed slow 注入版本；slow 到达前输出 fast."""
    n_reset, n_evolve, n_fast_only = 0, 0, 0
    ts = [m.timestamp_us for _, m in kf_list]
    slow_steps = schedule_delayed_slow_steps(ts, slow_interval, slow_delay_kf)
    stream_aligner.reset_scene()

    for step, ((_idx, meta), fl_fp16) in enumerate(zip(kf_list, scene_fast_cache)):
        fast_logits_curr = fl_fp16.float()

        if step < slow_delay_kf:
            aligned = fast_logits_curr
            n_fast_only += 1
        else:
            slow_arrived = step in slow_steps
            if slow_arrived:
                prev_step = step - slow_delay_kf
                _, meta_prev = kf_list[prev_step]
                slow_logit_path = meta_prev.slow_logit_path
                slow_present = (
                    slow_cache.has(slow_logit_path)
                    if slow_cache is not None
                    else os.path.exists(slow_logit_path)
                )
            else:
                slow_present = False

            if slow_arrived and slow_present:
                prev_step = step - slow_delay_kf
                _, meta_prev = kf_list[prev_step]
                slow_logits_prev = (
                    slow_cache.get(meta_prev.slow_logit_path)
                    if slow_cache is not None
                    else slow_decoder_fn(meta_prev.slow_logit_path)
                )
                aligned = delayed_reset_and_evolve(
                    stream_aligner,
                    kf_list=kf_list,
                    scene_fast_cache=scene_fast_cache,
                    prev_step=prev_step,
                    curr_step=step,
                    slow_logits_prev=slow_logits_prev,
                    device=device,
                )
                n_reset += 1
            elif stream_aligner.hidden is not None:
                ego_curr = torch.from_numpy(meta.ego2global).to(
                    device=device, dtype=torch.float32
                )
                aligned = stream_aligner.evolve(
                    fast_logits_curr, ego_curr, meta.timestamp_us
                )
                n_evolve += 1
            else:
                aligned = fast_logits_curr
                n_fast_only += 1

        pred_uint8 = aligned.argmax(0).to(torch.uint8).cpu().numpy()
        pred_list_out.append(_prediction_record(meta, step, pred_uint8))

    return n_reset, n_evolve, n_fast_only


def _cfg_path(data_cfg: dict, key: str) -> Optional[str]:
    value = data_cfg.get(key, None)
    if not value:
        return None
    return value if os.path.isabs(str(value)) else os.path.abspath(str(value))


def _dataset_variant(data_cfg: dict) -> str:
    return str(data_cfg.get("dataset_variant", "occ3d")).strip().lower()


def _metric_variant(data_cfg: dict) -> str:
    return str(data_cfg.get("metric_variant", _dataset_variant(data_cfg))).strip().lower()


def _attach_targets(pred_list: PredictionList, data_cfg: dict, gt_cache: GtCache) -> List[Dict[str, Any]]:
    """按 data_cfg 的数据集类型补齐 GT/mask。"""
    gt_root = _cfg_path(data_cfg, "gt_root")
    if not gt_root:
        raise ValueError("streaming eval 需要 data.gt_root")
    attached, missing = attach_dense_occ_targets(
        pred_list,
        dataset_variant=_dataset_variant(data_cfg),
        gt_root=gt_root,
        gt_mask_key=str(data_cfg.get("gt_mask_key", "mask_camera")),
        grid_size=tuple(data_cfg.get("grid_size", (200, 200, 16))),
        nuscenes_root=_cfg_path(data_cfg, "nuscenes_root"),
        nuscenes_version=str(data_cfg.get("nuscenes_version", "v1.0-trainval")),
        gt_cache=gt_cache,
    )
    if missing:
        print(f"  [metric] 跳过 {missing} 个缺少 GT 的预测")
    return attached


def eval_miou(pred_list: PredictionList, data_cfg: dict, gt_cache: GtCache):
    """返回 mIoU、mIoU_D、逐类 IoU、类别名、payload 和附 GT 记录。"""
    attached = _attach_targets(pred_list, data_cfg, gt_cache)
    metric_payload = compute_dense_miou(
        attached,
        num_classes=int(data_cfg["num_classes"]),
        use_lidar_mask=False,
        use_image_mask=True,
        metric_variant=_metric_variant(data_cfg),
    )["all"]
    miou = metric_payload.get("miou", None)
    miou_d = metric_payload.get("miou_d", None)
    per_cls = np.asarray(metric_payload.get("per_class_iou", []), dtype=np.float64)
    class_names = list(metric_payload.get("class_names", []))
    return (
        float(miou) if miou is not None else float("nan"),
        float(miou_d) if miou_d is not None else float("nan"),
        per_cls,
        class_names,
        metric_payload,
        attached,
    )


def print_per_class_iou(per_cls, class_names, miou, miou_d, label: str = "") -> None:
    """逐类 IoU 紧凑打印."""
    if label:
        print(f"  {label}")
    if len(class_names) == 0 or len(per_cls) == 0:
        print(f"    no valid GT samples    ->  mIoU = {miou:.2f}   mIoU_D = {miou_d:.2f}")
        return
    sem_count = min(len(class_names), len(per_cls)) - 1
    cells = [f"{class_names[i]}={per_cls[i]:5.2f}" for i in range(sem_count)]
    line, lines, max_per_line = "", [], 4
    for j, cell in enumerate(cells, 1):
        line += cell + "  "
        if j % max_per_line == 0:
            lines.append(line.rstrip())
            line = ""
    if line:
        lines.append(line.rstrip())
    for ln in lines:
        print(f"    {ln}")
    free_iou = float(per_cls[min(len(per_cls), len(class_names)) - 1])
    print(f"    free={free_iou:.2f}    ->  mIoU = {miou:.2f}   mIoU_D = {miou_d:.2f}")


def eval_rayiou(records: PredictionList, origins_by_token: dict):
    from evoocc.ops.dvr.ray_metrics import main as calc_rayiou

    sem_pred_list, sem_gt_list, lidar_origin_list = [], [], []
    skipped = 0
    for item in records:
        token = str(item.get("token", ""))
        if token not in origins_by_token:
            skipped += 1
            continue
        gt = item.get("gt", None)
        if gt is None:
            skipped += 1
            continue
        sem_pred_list.append(np.asarray(item["pred"]))
        sem_gt_list.append(gt)
        lidar_origin_list.append(origins_by_token[token])
    if skipped:
        print(f"  [rayiou] 跳过 {skipped} 个无 origin 的样本")
    print(f"  [rayiou] 参与计算: {len(sem_pred_list)}")
    return calc_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list)


def _interval_key(interval: float) -> str:
    return str(float(interval))


def run_streaming_eval(
    *,
    fast,
    scenes_meta,
    data_cfg: dict,
    stream_aligner: StreamAligner,
    slow_decoder_fn: Callable[[str], torch.Tensor],
    slow_intervals: List[float],
    device: torch.device,
    num_workers: int,
    prefetch_factor: int,
    preload_slow: bool,
    no_rayiou: bool,
    sweep_pkl: str,
    out_json: Optional[str],
    fast_backend: str,
    aligner_label: str = "EvoOcc",
    delayed: bool = False,
    slow_delay_keyframes: int = 0,
    extra_out: Optional[dict] = None,
) -> dict:
    """运行完整 streaming eval，返回汇总 dict."""
    flat_indices = [idx for _, kf_list in scenes_meta for (idx, _) in kf_list]
    total_kf = len(flat_indices)
    print(f"  scenes count={len(scenes_meta)}, total kf={total_kf}")

    print(f"[4] DataLoader (num_workers={num_workers}, prefetch={prefetch_factor}) ...")
    loader = make_streaming_loader(
        fast.dataset,
        flat_indices,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    loader_iter = iter(loader)

    predictions = {it: [] for it in slow_intervals}
    gt_cache: GtCache = {}
    reset_cnt = {it: 0 for it in slow_intervals}
    evolve_cnt = {it: 0 for it in slow_intervals}
    fast_only_cnt = {it: 0 for it in slow_intervals}

    mode_label = "delayed slow" if delayed else "stream"
    print(f"\n[5] Phase 1: per-scene fast + {aligner_label} {mode_label} ...")
    t_overall = time.time()
    for s_i, (scene_name, kf_list) in enumerate(scenes_meta):
        scene_fast_cache = cache_one_scene_fast(fast, kf_list, loader_iter)

        scene_slow_cache = None
        if preload_slow:
            scene_slow_cache = SlowLogitsGPUCache(device=device, decoder_fn=slow_decoder_fn)
            scene_slow_cache.preload(
                [m.slow_logit_path for _, m in kf_list],
                skip_missing=True,
                verbose=False,
            )

        for interval in slow_intervals:
            if delayed:
                n_r, n_e, n_f = stream_scene_one_interval_delayed(
                    kf_list,
                    scene_fast_cache,
                    scene_slow_cache,
                    slow_decoder_fn,
                    stream_aligner,
                    interval,
                    slow_delay_keyframes,
                    device,
                    pred_list_out=predictions[interval],
                )
                fast_only_cnt[interval] += n_f
            else:
                n_r, n_e = stream_scene_one_interval(
                    kf_list,
                    scene_fast_cache,
                    scene_slow_cache,
                    slow_decoder_fn,
                    stream_aligner,
                    interval,
                    device,
                    pred_list_out=predictions[interval],
                )
            reset_cnt[interval] += n_r
            evolve_cnt[interval] += n_e

        del scene_fast_cache, scene_slow_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t_overall
        eta = elapsed / (s_i + 1) * (len(scenes_meta) - s_i - 1)
        if (s_i + 1) % 10 == 0 or s_i + 1 == len(scenes_meta) or s_i < 2:
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            print(
                f"  [{s_i+1:3d}/{len(scenes_meta)}] {scene_name} kf={len(kf_list)} "
                f"elapsed={elapsed:6.1f}s eta={eta:6.1f}s gpu={gpu_mb:.0f}MB"
            )

    inference_wall = time.time() - t_overall
    print(f"\n[6] Phase 1 完成: wall={inference_wall:.1f}s, predictions={total_kf} per interval")

    print("\n[7] Phase 2a: mIoU ...")
    miou_results, miou_d_results, per_cls_results = {}, {}, {}
    class_names_results, metric_payloads, attached_predictions = {}, {}, {}
    for interval in slow_intervals:
        miou, miou_d, per_cls, class_names, payload, attached = eval_miou(
            predictions[interval], data_cfg, gt_cache
        )
        miou_results[interval] = miou
        miou_d_results[interval] = miou_d
        per_cls_results[interval] = per_cls
        class_names_results[interval] = class_names
        metric_payloads[interval] = payload
        attached_predictions[interval] = attached
        label = f"[interval={interval}s"
        if delayed:
            label += f", delay={slow_delay_keyframes}kf"
        label += "]"
        print_per_class_iou(per_cls, class_names, miou, miou_d, label=label)
        if "occupied_iou" in payload:
            print(f"    occupied_iou = {payload['occupied_iou']:.2f}")

    rayiou_results = {}
    metric_variant = _metric_variant(data_cfg)
    if not no_rayiou and metric_variant == "occ3d":
        print("\n[8] Phase 2b: RayIoU ...")
        from evoocc.ops.dvr.ego_pose import load_origins_from_sweep_pkl

        origins_by_token = load_origins_from_sweep_pkl(sweep_pkl)
        print(f"  loaded {len(origins_by_token)} origins from {sweep_pkl}")
        for interval in slow_intervals:
            print(f"\n  --- interval={interval} ---")
            r = eval_rayiou(attached_predictions[interval], origins_by_token)
            rayiou_results[interval] = r
            print(
                f"  RayIoU={r['RayIoU']:.4f}  @1={r['RayIoU@1']:.4f}  "
                f"@2={r['RayIoU@2']:.4f}  @4={r['RayIoU@4']:.4f}"
            )
    else:
        reason = "--no-rayiou" if no_rayiou else f"metric_variant={metric_variant}"
        print(f"\n[8] Phase 2b: RayIoU skipped ({reason})")

    suffix = f", slow_delay={slow_delay_keyframes}kf" if delayed else ""
    print(f"\n=== 最终汇总 ({total_kf} keyframes, {len(scenes_meta)} scenes{suffix}) ===")
    print(f"  inference wall: {inference_wall:.1f}s")
    show_rayiou_cols = metric_variant == "occ3d"
    show_occupied_iou = any("occupied_iou" in metric_payloads[it] for it in slow_intervals)
    header = f"  {'interval':>10s}  {'mIoU':>7s}  {'mIoU_D':>7s}"
    if show_occupied_iou:
        header += f"  {'OccIoU':>7s}"
    if show_rayiou_cols:
        header += f"  {'RayIoU':>7s}  {'@1':>6s}  {'@2':>6s}  {'@4':>6s}"
    header += f"  {'reset':>6s}  {'evolve':>7s}"
    if delayed:
        header += f"  {'fastonly':>8s}"
    print(header)

    out = {
        "fast_backend": fast_backend,
        "dataset_variant": _dataset_variant(data_cfg),
        "metric_variant": metric_variant,
        "scenes": len(scenes_meta),
        "total_kf": total_kf,
        "inference_wall_s": float(inference_wall),
        "intervals": {},
    }
    if delayed:
        out["slow_delay_keyframes"] = int(slow_delay_keyframes)
    if extra_out:
        out.update(extra_out)

    for interval in slow_intervals:
        class_names = class_names_results[interval]
        r = rayiou_results.get(interval, {})
        ri = r.get("RayIoU", float("nan"))
        ri1 = r.get("RayIoU@1", float("nan"))
        ri2 = r.get("RayIoU@2", float("nan"))
        ri4 = r.get("RayIoU@4", float("nan"))
        row = f"  {interval:>10.2f}  {miou_results[interval]:>7.2f}  {miou_d_results[interval]:>7.2f}"
        if show_occupied_iou:
            occ_iou = metric_payloads[interval].get("occupied_iou", float("nan"))
            row += f"  {occ_iou:>7.2f}"
        if show_rayiou_cols:
            row += f"  {ri:>7.4f}  {ri1:>6.4f}  {ri2:>6.4f}  {ri4:>6.4f}"
        row += f"  {reset_cnt[interval]:>6d}  {evolve_cnt[interval]:>7d}"
        if delayed:
            row += f"  {fast_only_cnt[interval]:>8d}"
        print(row)

        n_cls = min(len(class_names), len(per_cls_results[interval]), int(data_cfg["num_classes"]))
        per_cls_dict = {class_names[i]: float(per_cls_results[interval][i]) for i in range(n_cls)}
        item = {
            "miou": float(miou_results[interval]),
            "miou_d": float(miou_d_results[interval]),
            "per_class_iou": per_cls_dict,
            "num_keyframes": int(metric_payloads[interval].get("num_keyframes", 0)),
            "rayiou": float(ri) if not np.isnan(ri) else None,
            "rayiou_at_1": float(ri1) if not np.isnan(ri1) else None,
            "rayiou_at_2": float(ri2) if not np.isnan(ri2) else None,
            "rayiou_at_4": float(ri4) if not np.isnan(ri4) else None,
            "n_reset": int(reset_cnt[interval]),
            "n_evolve": int(evolve_cnt[interval]),
        }
        if "occupied_iou" in metric_payloads[interval]:
            item["occupied_iou"] = metric_payloads[interval]["occupied_iou"]
        if delayed:
            item["n_fast_only"] = int(fast_only_cnt[interval])
        out["intervals"][_interval_key(interval)] = item

    if out_json:
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {out_json}")
    return out
