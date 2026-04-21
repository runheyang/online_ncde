#!/usr/bin/env python3
"""真实快系统 + Online NCDE 端到端 benchmark。"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402

from realtime_eval_utils import dense_logits_to_occ_result, evaluate_final_predictions  # noqa: E402
from realtime_ncde_runtime import RealtimeNcdeRuntime  # noqa: E402
from realtime_opus_step_runtime import OpusRealtimeStepRunner  # noqa: E402

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


@dataclass
class ModeStats:
    """聚合单个模式的耗时与评估统计。"""

    step_times_ms: list[float] = field(default_factory=list)
    step0_times_ms: list[float] = field(default_factory=list)
    rollout_times_ms: list[float] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    skip_examples: dict[str, str] = field(default_factory=dict)
    num_samples: int = 0
    num_total_steps: int = 0
    num_eval_results: int = 0

    def add_step_time(self, step_idx: int, time_ms: float) -> None:
        self.step_times_ms.append(float(time_ms))
        self.num_total_steps += 1
        if step_idx == 0:
            self.step0_times_ms.append(float(time_ms))
        else:
            self.rollout_times_ms.append(float(time_ms))

    def add_skip(self, reason: str) -> None:
        self.skipped[reason] += 1

    def add_skip_example(self, reason: str, exc: Exception) -> None:
        if reason not in self.skip_examples:
            self.skip_examples[reason] = repr(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="both",
        choices=["fast_only", "fast_plus_ncde", "both"],
        help="benchmark 模式：仅快系统、快系统+NCDE，或顺序跑两者",
    )
    parser.add_argument("--opus-config", required=True, help="OPUS 配置路径")
    parser.add_argument("--opus-weights", required=True, help="OPUS 权重路径")
    parser.add_argument("--ncde-config", required=True, help="online_ncde 配置路径")
    parser.add_argument(
        "--ncde-checkpoint",
        default="",
        help="online_ncde 权重路径；fast_plus_ncde / both 时必填",
    )
    parser.add_argument(
        "--info-path",
        default="",
        help="可选：覆盖 benchmark 使用的 info pkl，默认取 ncde-config 的 val_info_path",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 个样本，0 表示全量")
    parser.add_argument("--warmup-samples", type=int, default=1, help="warmup 样本数，不计入最终统计")
    parser.add_argument("--device", default="cuda:0", help="推理设备，默认 cuda:0")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="OPUS 预取 worker 数；负数表示沿用 OPUS 配置里的 workers_per_gpu",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor")
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="开启 DataLoader pin_memory，加速 host->device 搬运",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="关闭 DataLoader pin_memory",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        help="开启 DataLoader persistent_workers，减少 worker 重建开销",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="关闭 DataLoader persistent_workers",
    )
    parser.add_argument(
        "--disable-fast-logit-topk-truncation",
        action="store_true",
        help="关闭 fast logits 的 top-3 截断，直接把 OPUS 原始语义 logits 稠密重建后送入后续推理",
    )
    parser.add_argument(
        "--nusc-version",
        default="v1.0-trainval",
        help="NuScenes 版本，用于构造 OPUS 历史帧 ego pose 查表",
    )
    parser.add_argument("--dump-json", default="", help="可选：将结果写入 json")
    parser.set_defaults(pin_memory=True, persistent_workers=True)
    return parser.parse_args()


def _safe_avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _to_json_number(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _load_infos(config_path: str, override_info_path: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    cfg = load_config_with_base(config_path)
    data_cfg = cfg["data"]
    root_path = cfg["root_path"]
    info_path = override_info_path or data_cfg.get("val_info_path", data_cfg["info_path"])
    info_abs = resolve_path(root_path, info_path)
    with open(info_abs, "rb") as f:
        payload = pickle.load(f)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    valid_infos = [dict(info) for info in infos if info.get("valid", True)]
    return cfg, valid_infos, info_abs


def _select_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["fast_only", "fast_plus_ncde"]
    return [mode]


def _normalize_step_count(info: dict[str, Any]) -> int:
    for key in ("num_output_frames", "frame_ego2global", "frame_timestamps", "frame_dt", "frame_tokens"):
        value = info.get(key, None)
        if value is None:
            continue
        if key == "num_output_frames" and int(value) > 0:
            return int(value)
        if hasattr(value, "shape") and len(getattr(value, "shape")) > 0:
            return int(getattr(value, "shape")[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("无法从 info 中推断 step 数。")


def _run_single_mode(
    mode: str,
    infos: list[dict[str, Any]],
    opus_runner: OpusRealtimeStepRunner,
    ncde_cfg: dict[str, Any],
    ncde_runtime: RealtimeNcdeRuntime | None,
    *,
    num_workers: int | None,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    fast_logit_topk: int | None,
    collect_results: bool,
    desc: str,
) -> dict[str, Any]:
    data_cfg = ncde_cfg["data"]
    free_index = int(data_cfg["free_index"])
    num_classes = int(data_cfg["num_classes"])
    grid_size = tuple(int(v) for v in data_cfg["grid_size"])
    other_fill_value = float(data_cfg.get("topk_other_fill_value", -5.0))
    free_fill_value = float(data_cfg.get("topk_free_fill_value", 5.0))

    stats = ModeStats()
    results_by_token: dict[str, dict[str, np.ndarray]] = {}
    prepared_infos: list[dict[str, Any]] = []
    prepared_samples = []
    for info in infos:
        token = str(info.get("token", ""))
        if not token:
            stats.add_skip("missing_token")
            continue
        if opus_runner.get_dataset_index(token) is None:
            stats.add_skip("token_not_in_opus_dataset")
            continue
        if mode == "fast_plus_ncde" and not str(info.get("slow_logit_path", "")):
            stats.add_skip("missing_slow_logit_path")
            continue
        try:
            expected_steps = _normalize_step_count(info)
        except Exception as exc:
            stats.add_skip("invalid_info_step_count")
            stats.add_skip_example("invalid_info_step_count", exc)
            continue
        try:
            prepared = opus_runner.prepare_sample(info)
        except Exception as exc:
            stats.add_skip("prepare_sample_failed")
            stats.add_skip_example("prepare_sample_failed", exc)
            continue
        if prepared.num_steps != expected_steps:
            stats.add_skip("step_count_mismatch")
            continue
        prepared_infos.append(info)
        prepared_samples.append(prepared)

    dataloader = opus_runner.build_prefetch_dataloader(
        prepared_samples,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    progress = progressbar.ProgressBar(max_value=len(prepared_infos), prefix=desc + " ").start() if progressbar is not None else None

    current_sample_idx: int | None = None
    current_info: dict[str, Any] | None = None
    current_token = ""
    current_sample_failed = False
    final_dense_logits: torch.Tensor | None = None
    _progress_count = 0  # 已完成样本数（progressbar2 需要绝对值）

    def finalize_current_sample() -> None:
        nonlocal current_sample_idx, current_info, current_token, current_sample_failed, final_dense_logits, _progress_count
        if current_sample_idx is None:
            return
        if collect_results and (not current_sample_failed) and final_dense_logits is not None:
            stats.num_samples += 1
            results_by_token[current_token] = dense_logits_to_occ_result(
                dense_logits=final_dense_logits,
                free_index=free_index,
            )
            stats.num_eval_results = len(results_by_token)
        if progress is not None:
            _progress_count += 1
            progress.update(_progress_count)
        current_sample_idx = None
        current_info = None
        current_token = ""
        current_sample_failed = False
        final_dense_logits = None

    # 设计文档明确要求 no_grad；这里不能用 inference_mode，
    # 否则 NCDE 的状态张量更新会在部分算子路径下直接报错。
    with torch.no_grad():
        for step_batch in dataloader:
            sample_idx = int(step_batch["sample_idx"])
            step_idx = int(step_batch["step_idx"])
            if current_sample_idx is None or sample_idx != current_sample_idx:
                finalize_current_sample()
                current_sample_idx = sample_idx
                current_info = prepared_infos[sample_idx]
                current_token = str(current_info.get("token", ""))
                if mode == "fast_plus_ncde":
                    if ncde_runtime is None:
                        raise RuntimeError("fast_plus_ncde 模式要求 ncde_runtime 已初始化。")
                    try:
                        sample_num_frames = ncde_runtime.begin_sample(current_info)
                    except Exception as exc:
                        current_sample_failed = True
                        stats.add_skip("ncde_begin_failed")
                        stats.add_skip_example("ncde_begin_failed", exc)
                    else:
                        if sample_num_frames != int(step_batch["num_steps"]):
                            current_sample_failed = True
                            stats.add_skip("ncde_step_count_mismatch")

            if current_sample_failed:
                continue

            if str(step_batch.get("status", "")) != "ok":
                current_sample_failed = True
                error_reason = str(step_batch.get("error_reason", "worker_pipeline_failed"))
                stats.add_skip(error_reason)
                error_message = str(step_batch.get("error_message", ""))
                if error_message and error_reason not in stats.skip_examples:
                    stats.skip_examples[error_reason] = error_message
                continue

            start_time = time.perf_counter()
            try:
                opus_result = opus_runner.run_processed_step_result(cast(dict[str, Any], step_batch["processed"]))
                fast_dense_logits = opus_runner.opus_result_to_fast_logits(
                    opus_result,
                    num_classes=num_classes,
                    free_index=free_index,
                    grid_size=grid_size,
                    other_fill_value=other_fill_value,
                    free_fill_value=free_fill_value,
                    topk=fast_logit_topk,
                )
                if mode == "fast_only":
                    step_dense_logits = fast_dense_logits
                else:
                    step_dense_logits = cast(RealtimeNcdeRuntime, ncde_runtime).step(
                        step_idx=step_idx,
                        fast_logits_dense=fast_dense_logits,
                        info=cast(dict[str, Any], current_info),
                    )
            except Exception as exc:
                current_sample_failed = True
                stats.add_skip("step_runtime_failed")
                stats.add_skip_example("step_runtime_failed", exc)
                continue

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if collect_results:
                stats.add_step_time(step_idx=step_idx, time_ms=elapsed_ms)
            final_dense_logits = step_dense_logits

    finalize_current_sample()
    if progress is not None:
        progress.finish()

    metrics = {}
    if collect_results and results_by_token:
        metrics = evaluate_final_predictions(
            base_dataset=opus_runner.dataset,
            token_to_dataset_idx=opus_runner.token_to_dataset_idx,
            results_by_token=results_by_token,
        )

    result = {
        "mode": mode,
        "num_samples": int(stats.num_samples),
        "num_total_steps": int(stats.num_total_steps),
        "num_eval_results": int(stats.num_eval_results),
        "avg_ms_all_steps": _to_json_number(_safe_avg(stats.step_times_ms)),
        "avg_ms_step0": _to_json_number(_safe_avg(stats.step0_times_ms)),
        "avg_ms_rollout_steps": _to_json_number(_safe_avg(stats.rollout_times_ms)),
        "p50_ms_all_steps": _to_json_number(_percentile(stats.step_times_ms, 50)),
        "p90_ms_all_steps": _to_json_number(_percentile(stats.step_times_ms, 90)),
        "p95_ms_all_steps": _to_json_number(_percentile(stats.step_times_ms, 95)),
        "metrics": {
            key: _to_json_number(float(value))
            if isinstance(value, (int, float, np.integer, np.floating))
            else value
            for key, value in metrics.items()
        },
        "skipped": {key: int(value) for key, value in sorted(stats.skipped.items())},
        "skip_examples": dict(stats.skip_examples),
    }
    return result


def main() -> None:
    args = parse_args()
    if args.mode in {"fast_plus_ncde", "both"} and not args.ncde_checkpoint:
        raise ValueError("fast_plus_ncde / both 模式必须提供 --ncde-checkpoint。")

    ncde_cfg, all_infos, info_abs = _load_infos(args.ncde_config, args.info_path)
    warmup_count = max(int(args.warmup_samples), 0)
    warmup_infos = all_infos[:warmup_count]
    benchmark_infos = all_infos[warmup_count:]
    if args.limit > 0:
        benchmark_infos = benchmark_infos[: int(args.limit)]

    modes = _select_modes(args.mode)
    opus_runner = OpusRealtimeStepRunner(
        config_path=args.opus_config,
        weights_path=args.opus_weights,
        device=args.device,
        amp_enabled=True,
        nusc_version=args.nusc_version,
    )
    ncde_runtime = None
    if "fast_plus_ncde" in modes:
        ncde_runtime = RealtimeNcdeRuntime(
            config_path=args.ncde_config,
            checkpoint_path=args.ncde_checkpoint,
            device=args.device,
        )

    if warmup_infos:
        for mode in modes:
            _run_single_mode(
                mode=mode,
                infos=warmup_infos,
                opus_runner=opus_runner,
                ncde_cfg=ncde_cfg,
                ncde_runtime=ncde_runtime if mode == "fast_plus_ncde" else None,
                num_workers=None if args.num_workers < 0 else int(args.num_workers),
                prefetch_factor=max(int(args.prefetch_factor), 1),
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                fast_logit_topk=None if args.disable_fast_logit_topk_truncation else 3,
                collect_results=False,
                desc=f"[warmup {mode}]",
            )

    outputs: dict[str, Any] = {
        "info_path": info_abs,
        "warmup_samples": int(warmup_count),
        "benchmark_num_infos": int(len(benchmark_infos)),
        "opus_force_offline_count": int(opus_runner.force_offline_count),
        "prefetch_loader": {
            "num_workers": int(opus_runner.loader_num_workers if args.num_workers < 0 else args.num_workers),
            "prefetch_factor": max(int(args.prefetch_factor), 1),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
        },
        "fast_logit_rebuild": {
            "topk_truncation_enabled": not bool(args.disable_fast_logit_topk_truncation),
            "topk": None if args.disable_fast_logit_topk_truncation else 3,
        },
        "modes": {},
    }

    for mode in modes:
        outputs["modes"][mode] = _run_single_mode(
            mode=mode,
            infos=benchmark_infos,
            opus_runner=opus_runner,
            ncde_cfg=ncde_cfg,
            ncde_runtime=ncde_runtime if mode == "fast_plus_ncde" else None,
            num_workers=None if args.num_workers < 0 else int(args.num_workers),
            prefetch_factor=max(int(args.prefetch_factor), 1),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            fast_logit_topk=None if args.disable_fast_logit_topk_truncation else 3,
            collect_results=True,
            desc=f"[benchmark {mode}]",
        )

    if "fast_only" in outputs["modes"] and "fast_plus_ncde" in outputs["modes"]:
        fast_only_avg = outputs["modes"]["fast_only"].get("avg_ms_all_steps", None)
        fast_plus_avg = outputs["modes"]["fast_plus_ncde"].get("avg_ms_all_steps", None)
        delta_ms = None
        ratio = None
        if fast_only_avg is not None and fast_plus_avg is not None:
            delta_ms = float(fast_plus_avg - fast_only_avg)
            ratio = float(fast_plus_avg / fast_only_avg) if float(fast_only_avg) > 0 else None
        outputs["comparison"] = {
            "delta_ms_all_steps": _to_json_number(delta_ms),
            "ratio_ms_all_steps": _to_json_number(ratio),
        }

    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    if args.dump_json:
        dump_path = os.path.abspath(args.dump_json)
        dump_dir = os.path.dirname(dump_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
