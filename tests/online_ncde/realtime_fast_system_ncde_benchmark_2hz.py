#!/usr/bin/env python3
"""2Hz realtime benchmark: align OPUS step gap with online-cache history interval."""

from __future__ import annotations

import argparse
import json
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

from realtime_eval_utils import dense_logits_to_occ_result, evaluate_final_predictions  # noqa: E402
from realtime_fast_system_ncde_benchmark import (  # noqa: E402
    _load_infos,
    _percentile,
    _safe_avg,
    _select_modes,
    _to_json_number,
)
from realtime_ncde_runtime import RealtimeNcdeRuntime  # noqa: E402
from realtime_opus_step_runtime import OpusRealtimeStepRunner  # noqa: E402

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


@dataclass
class ModeStats2Hz:
    total_times_ms: list[float] = field(default_factory=list)
    opus_times_ms: list[float] = field(default_factory=list)
    dense_times_ms: list[float] = field(default_factory=list)
    ncde_times_ms: list[float] = field(default_factory=list)
    step0_total_times_ms: list[float] = field(default_factory=list)
    rollout_total_times_ms: list[float] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    skip_examples: dict[str, str] = field(default_factory=dict)
    num_samples: int = 0
    num_total_steps: int = 0
    num_eval_results: int = 0
    num_bootstrap_steps: int = 0

    def add_step_timing(
        self,
        *,
        step_idx: int,
        total_ms: float,
        opus_ms: float,
        dense_ms: float,
        ncde_ms: float | None,
    ) -> None:
        self.total_times_ms.append(float(total_ms))
        self.opus_times_ms.append(float(opus_ms))
        self.dense_times_ms.append(float(dense_ms))
        if ncde_ms is not None:
            self.ncde_times_ms.append(float(ncde_ms))
        self.num_total_steps += 1
        if step_idx == 0:
            self.step0_total_times_ms.append(float(total_ms))
        else:
            self.rollout_total_times_ms.append(float(total_ms))

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
    parser.add_argument(
        "--ncde-config",
        default=str(ROOT / "configs/online_ncde/fast_opusv1t__slow_opusv2l/eval.yaml"),
        help="online_ncde 配置路径",
    )
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
        help="开启 DataLoader pin_memory",
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
        help="开启 DataLoader persistent_workers",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="关闭 DataLoader persistent_workers",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="逻辑 fast-logit 序列的采样 stride；3 对应约 2Hz",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="保留的采样帧数；默认 5，其中第 1 帧做 offline bootstrap，后 4 帧在线计时",
    )
    parser.add_argument(
        "--allow-cache-misaligned",
        action="store_true",
        help="允许 output_stride * frame_stride 与 history_interval 不一致；默认跳过这类样本",
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
    return int(info.get("num_output_frames", 0))


def _slice_value(value: Any, indices: list[int]) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value[indices]
    if isinstance(value, np.ndarray):
        return value[indices]
    if isinstance(value, (list, tuple)):
        return [value[idx] for idx in indices]
    return value


def _subsample_frame_dt(
    frame_dt: Any,
    *,
    total_num_frames: int,
    sampled_frame_indices: list[int],
) -> np.ndarray | None:
    if frame_dt is None:
        return None
    arr = np.asarray(frame_dt)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.shape[0] == total_num_frames:
        return arr[sampled_frame_indices]
    if arr.shape[0] == total_num_frames - 1:
        merged = [
            arr[start_idx:end_idx].sum()
            for start_idx, end_idx in zip(sampled_frame_indices[:-1], sampled_frame_indices[1:])
        ]
        return np.asarray(merged, dtype=arr.dtype)
    return arr.reshape(-1)[: len(sampled_frame_indices)]


def _build_sampled_info(
    info: dict[str, Any],
    *,
    frame_stride: int,
    max_frames: int,
    require_cache_aligned: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be > 0, got {frame_stride}")
    if max_frames < 2:
        raise ValueError(f"max_frames must be >= 2, got {max_frames}")

    sampled_frame_indices = [i * int(frame_stride) for i in range(int(max_frames))]
    required_num_frames = sampled_frame_indices[-1] + 1
    total_num_frames = _infer_num_frames(info)
    if total_num_frames < required_num_frames:
        return None, "short_sequence"

    output_stride = int(info.get("output_stride", 2))
    history_interval = int(info.get("history_interval", 6))
    sensor_frame_gap = output_stride * int(frame_stride)
    if require_cache_aligned and sensor_frame_gap != history_interval:
        return None, "cache_stride_mismatch"

    sampled = dict(info)
    sampled["num_output_frames"] = int(max_frames)
    sampled["output_stride"] = int(sensor_frame_gap)
    sampled["sampled_frame_indices"] = list(sampled_frame_indices)
    sampled["frame_stride"] = int(frame_stride)
    sampled["num_sampled_frames"] = int(max_frames)

    sampled["frame_ego2global"] = _slice_value(info.get("frame_ego2global", None), sampled_frame_indices)
    sampled["frame_timestamps"] = _slice_value(info.get("frame_timestamps", None), sampled_frame_indices)
    if sampled.get("frame_timestamps", None) is not None:
        sampled["frame_dt"] = None
    else:
        sampled["frame_dt"] = _subsample_frame_dt(
            info.get("frame_dt", None),
            total_num_frames=total_num_frames,
            sampled_frame_indices=sampled_frame_indices,
        )
    frame_tokens = info.get("frame_tokens", None)
    if frame_tokens is not None:
        sampled["frame_tokens"] = [frame_tokens[idx] for idx in sampled_frame_indices]
    return sampled, None

def _time_call_ms(fn: Any, *, device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def _run_single_mode_2hz(
    mode: str,
    infos: list[dict[str, Any]],
    opus_runner: OpusRealtimeStepRunner,
    ncde_cfg: dict[str, Any],
    ncde_runtime: RealtimeNcdeRuntime | None,
    *,
    frame_stride: int,
    max_frames: int,
    require_cache_aligned: bool,
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

    stats = ModeStats2Hz()
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

        sampled_info, skip_reason = _build_sampled_info(
            info,
            frame_stride=frame_stride,
            max_frames=max_frames,
            require_cache_aligned=require_cache_aligned,
        )
        if sampled_info is None:
            stats.add_skip(cast(str, skip_reason))
            continue

        try:
            prepared = opus_runner.prepare_sample(sampled_info)
        except Exception as exc:
            stats.add_skip("prepare_sample_failed")
            stats.add_skip_example("prepare_sample_failed", exc)
            continue

        prepared_infos.append(sampled_info)
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

    with torch.no_grad():
        for step_batch in dataloader:
            sample_idx = int(step_batch["sample_idx"])
            step_idx = int(step_batch["step_idx"])
            if current_sample_idx is None or sample_idx != current_sample_idx:
                finalize_current_sample()
                current_sample_idx = sample_idx
                current_info = prepared_infos[sample_idx]
                current_token = str(current_info.get("token", ""))
                opus_runner.reset_model_cache()
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

                bootstrap_source_idx = int(prepared_samples[sample_idx].step_source_indices[0])
                try:
                    bootstrap_result = opus_runner.run_source_index_result(
                        prepared_samples[sample_idx],
                        bootstrap_source_idx,
                        force_offline_override=True,
                    )
                    bootstrap_fast_dense_logits = opus_runner.opus_result_to_fast_logits(
                        bootstrap_result,
                        num_classes=num_classes,
                        free_index=free_index,
                        grid_size=grid_size,
                        other_fill_value=other_fill_value,
                        free_fill_value=free_fill_value,
                        topk=fast_logit_topk,
                    )
                    stats.num_bootstrap_steps += 1
                except Exception as exc:
                    current_sample_failed = True
                    stats.add_skip("offline_bootstrap_failed")
                    stats.add_skip_example("offline_bootstrap_failed", exc)
                    continue

                if mode == "fast_plus_ncde":
                    try:
                        cast(RealtimeNcdeRuntime, ncde_runtime).step(
                            step_idx=0,
                            fast_logits_dense=bootstrap_fast_dense_logits,
                            info=cast(dict[str, Any], current_info),
                        )
                    except Exception as exc:
                        current_sample_failed = True
                        stats.add_skip("ncde_bootstrap_failed")
                        stats.add_skip_example("ncde_bootstrap_failed", exc)
                        continue

            if current_sample_failed:
                continue

            if step_idx == 0:
                # step0 已经通过 offline bootstrap 执行过，这里跳过 online 版本且不计时。
                continue

            if str(step_batch.get("status", "")) != "ok":
                current_sample_failed = True
                error_reason = str(step_batch.get("error_reason", "worker_pipeline_failed"))
                stats.add_skip(error_reason)
                error_message = str(step_batch.get("error_message", ""))
                if error_message and error_reason not in stats.skip_examples:
                    stats.skip_examples[error_reason] = error_message
                continue

            try:
                opus_result, opus_ms = _time_call_ms(
                    lambda: opus_runner.run_processed_step_result(cast(dict[str, Any], step_batch["processed"])),
                    device=opus_runner.device,
                )
                fast_dense_logits, dense_ms = _time_call_ms(
                    lambda: opus_runner.opus_result_to_fast_logits(
                        opus_result,
                        num_classes=num_classes,
                        free_index=free_index,
                        grid_size=grid_size,
                        other_fill_value=other_fill_value,
                        free_fill_value=free_fill_value,
                        topk=fast_logit_topk,
                    ),
                    device=opus_runner.device,
                )
                if mode == "fast_only":
                    step_dense_logits = fast_dense_logits
                    ncde_ms = None
                else:
                    step_dense_logits, ncde_ms = _time_call_ms(
                        lambda: cast(RealtimeNcdeRuntime, ncde_runtime).step(
                            step_idx=step_idx,
                            fast_logits_dense=fast_dense_logits,
                            info=cast(dict[str, Any], current_info),
                        ),
                        device=opus_runner.device,
                    )
            except Exception as exc:
                current_sample_failed = True
                stats.add_skip("step_runtime_failed")
                stats.add_skip_example("step_runtime_failed", exc)
                continue

            total_ms = float(opus_ms + dense_ms + (ncde_ms or 0.0))
            if collect_results:
                stats.add_step_timing(
                    step_idx=step_idx - 1,
                    total_ms=total_ms,
                    opus_ms=opus_ms,
                    dense_ms=dense_ms,
                    ncde_ms=ncde_ms,
                )
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
        "num_bootstrap_steps": int(stats.num_bootstrap_steps),
        "avg_ms_all_steps": _to_json_number(_safe_avg(stats.total_times_ms)),
        "avg_ms_step0": _to_json_number(_safe_avg(stats.step0_total_times_ms)),
        "avg_ms_rollout_steps": _to_json_number(_safe_avg(stats.rollout_total_times_ms)),
        "avg_ms_opus_runtime": _to_json_number(_safe_avg(stats.opus_times_ms)),
        "avg_ms_dense_rebuild": _to_json_number(_safe_avg(stats.dense_times_ms)),
        "avg_ms_ncde_runtime": _to_json_number(_safe_avg(stats.ncde_times_ms)),
        "p50_ms_all_steps": _to_json_number(_percentile(stats.total_times_ms, 50)),
        "p90_ms_all_steps": _to_json_number(_percentile(stats.total_times_ms, 90)),
        "p95_ms_all_steps": _to_json_number(_percentile(stats.total_times_ms, 95)),
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

    torch.backends.cudnn.benchmark = True

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
        force_offline_sweeps=False,
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
            _run_single_mode_2hz(
                mode=mode,
                infos=warmup_infos,
                opus_runner=opus_runner,
                ncde_cfg=ncde_cfg,
                ncde_runtime=ncde_runtime if mode == "fast_plus_ncde" else None,
                frame_stride=int(args.frame_stride),
                max_frames=int(args.max_frames),
                require_cache_aligned=not bool(args.allow_cache_misaligned),
                num_workers=None if args.num_workers < 0 else int(args.num_workers),
                prefetch_factor=max(int(args.prefetch_factor), 1),
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                fast_logit_topk=None if args.disable_fast_logit_topk_truncation else 3,
                collect_results=False,
                desc=f"[warmup 2hz {mode}]",
            )

    sampled_frame_indices = [i * int(args.frame_stride) for i in range(int(args.max_frames))]
    outputs: dict[str, Any] = {
        "info_path": info_abs,
        "warmup_samples": int(warmup_count),
        "benchmark_num_infos": int(len(benchmark_infos)),
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
        "sampled_frame_indices": list(sampled_frame_indices),
        "cache_alignment_required": not bool(args.allow_cache_misaligned),
        "opus_force_offline_count": int(opus_runner.force_offline_count),
        "timing_notes": {
            "bootstrap": "each sample runs step0 once with offline pipeline and does not time it",
            "avg_ms_step0": "the first timed online step after bootstrap",
            "avg_ms_rollout_steps": "timed online steps after the first timed online step",
        },
        "fast_logit_rebuild": {
            "topk_truncation_enabled": not bool(args.disable_fast_logit_topk_truncation),
            "topk": None if args.disable_fast_logit_topk_truncation else 3,
        },
        "opus_cache": {
            "online_pipeline": True,
            "reset_per_sample": True,
            "bootstrap_strategy": "offline_step0_not_timed_then_online_steps_1_to_end",
            "num_input_frames": int(opus_runner.num_input_frames),
        },
        "prefetch_loader": {
            "num_workers": int(opus_runner.loader_num_workers if args.num_workers < 0 else args.num_workers),
            "prefetch_factor": max(int(args.prefetch_factor), 1),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
        },
        "modes": {},
    }

    for mode in modes:
        outputs["modes"][mode] = _run_single_mode_2hz(
            mode=mode,
            infos=benchmark_infos,
            opus_runner=opus_runner,
            ncde_cfg=ncde_cfg,
            ncde_runtime=ncde_runtime if mode == "fast_plus_ncde" else None,
            frame_stride=int(args.frame_stride),
            max_frames=int(args.max_frames),
            require_cache_aligned=not bool(args.allow_cache_misaligned),
            num_workers=None if args.num_workers < 0 else int(args.num_workers),
            prefetch_factor=max(int(args.prefetch_factor), 1),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            fast_logit_topk=None if args.disable_fast_logit_topk_truncation else 3,
            collect_results=True,
            desc=f"[benchmark 2hz {mode}]",
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
        dump_path = Path(args.dump_json).expanduser().resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
