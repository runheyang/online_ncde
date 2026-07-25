"""ALOcc streaming 系统级 benchmark 公共执行循环。"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from evoocc.streaming.streaming_loader import scatter_to_device


@dataclass(frozen=True)
class SystemFrame:
    """fast/slow dataset 对齐后的单个流式帧。"""

    fast_index: int
    slow_index: int
    meta: Any
    is_scene_start: bool
    is_slow: bool


@dataclass
class FrameTiming:
    """单帧端到端和 CUDA 分阶段计时。"""

    frame_idx: int
    sample_token: str
    scene_name: str
    is_slow: bool
    frame_wall_ms: float
    input_wait_ms: float
    h2d_ms: float
    fast_ms: float
    slow_ms: float
    align_ms: float
    post_ms: float


def index_scene_frames_by_token(scenes_meta) -> Dict[str, Tuple[int, Any]]:
    """把 iter_scenes 输出转换成 sample token 索引。"""
    out: Dict[str, Tuple[int, Any]] = {}
    for _scene_name, kf_list in scenes_meta:
        for dataset_index, meta in kf_list:
            token = str(meta.sample_token)
            if token in out:
                raise ValueError(f"slow dataset 出现重复 sample token: {token}")
            out[token] = (int(dataset_index), meta)
    return out


def build_dual_system_schedule(
    fast_frames: Sequence[Tuple[int, Any]],
    slow_by_token: Mapping[str, Tuple[int, Any]],
    slow_interval_sec: float,
) -> List[SystemFrame]:
    """按 scene/timestamp 对齐 fast、slow dataset，并生成 slow 调度。"""
    schedule: List[SystemFrame] = []
    cur_scene: Optional[str] = None
    last_slow_t_sec = -1e9

    for fast_index, fast_meta in fast_frames:
        token = str(fast_meta.sample_token)
        if token not in slow_by_token:
            raise KeyError(f"slow dataset 缺少 sample token: {token}")
        slow_index, slow_meta = slow_by_token[token]
        if str(slow_meta.scene_name) != str(fast_meta.scene_name):
            raise ValueError(
                f"fast/slow scene 不一致: token={token}, "
                f"fast={fast_meta.scene_name}, slow={slow_meta.scene_name}"
            )
        if int(slow_meta.timestamp_us) != int(fast_meta.timestamp_us):
            raise ValueError(
                f"fast/slow timestamp 不一致: token={token}, "
                f"fast={fast_meta.timestamp_us}, slow={slow_meta.timestamp_us}"
            )

        scene_name = str(fast_meta.scene_name)
        is_scene_start = scene_name != cur_scene
        if is_scene_start:
            cur_scene = scene_name
            last_slow_t_sec = -1e9

        t_sec = int(fast_meta.timestamp_us) / 1e6
        is_slow = is_scene_start or (
            slow_interval_sec >= 0
            and t_sec - last_slow_t_sec + 1e-3 >= slow_interval_sec
        )
        if is_slow:
            last_slow_t_sec = t_sec

        schedule.append(
            SystemFrame(
                fast_index=int(fast_index),
                slow_index=int(slow_index),
                meta=fast_meta,
                is_scene_start=is_scene_start,
                is_slow=is_slow,
            )
        )
    return schedule


def _latency_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def summarize_system_benchmark(
    *,
    name: str,
    timings: Sequence[FrameTiming],
    total_wall_s: float,
    warmup: int,
    peak_memory_mb: float,
) -> Dict[str, Any]:
    """汇总 E2E、CUDA stage 和 regular/slow tick 指标。"""
    if not timings:
        raise ValueError("system benchmark 没有 measured timing")
    if total_wall_s <= 0:
        raise ValueError(f"total_wall_s 必须为正数，得到 {total_wall_s}")

    frame_values = [x.frame_wall_ms for x in timings]
    regular_values = [x.frame_wall_ms for x in timings if not x.is_slow]
    slow_values = [x.frame_wall_ms for x in timings if x.is_slow]
    has_align = any(abs(x.align_ms) > 0.0 for x in timings)
    evolve_values = (
        [x.align_ms for x in timings if not x.is_slow] if has_align else []
    )
    reset_values = (
        [x.align_ms for x in timings if x.is_slow] if has_align else []
    )
    stage_names = ("input_wait_ms", "h2d_ms", "fast_ms", "slow_ms", "align_ms", "post_ms")
    stage_mean = {
        key: float(np.mean([getattr(x, key) for x in timings]))
        for key in stage_names
    }
    total_model_ms = sum(x.fast_ms + x.slow_ms + x.align_ms for x in timings)
    total_gpu_pipeline_ms = sum(
        x.h2d_ms + x.fast_ms + x.slow_ms + x.align_ms + x.post_ms
        for x in timings
    )
    n_measured = len(timings)

    return {
        "name": name,
        "n_measured": int(n_measured),
        "n_warmup": int(warmup),
        "n_slow": int(len(slow_values)),
        "n_regular": int(len(regular_values)),
        "total_wall_s": float(total_wall_s),
        "fps_e2e": float(n_measured / total_wall_s),
        "latency_ms": _latency_stats(frame_values),
        "regular_latency_ms": _latency_stats(regular_values),
        "slow_tick_latency_ms": _latency_stats(slow_values),
        "evolve_latency_ms": _latency_stats(evolve_values),
        "reset_latency_ms": _latency_stats(reset_values),
        "stage_ms_mean": stage_mean,
        "model_latency_ms_mean": float(total_model_ms / n_measured),
        "fps_model_amortized": float(1000.0 * n_measured / total_model_ms),
        "gpu_pipeline_latency_ms_mean": float(total_gpu_pipeline_ms / n_measured),
        "fps_gpu_pipeline": float(1000.0 * n_measured / total_gpu_pipeline_ms),
        "peak_memory_mb": float(peak_memory_mb),
        "trace": [asdict(x) for x in timings],
    }


def attach_system_flops(
    result: Dict[str, Any],
    *,
    mode: str,
    fast_gflops: Optional[float] = None,
    slow_gflops: Optional[float] = None,
    evolve_gflops: Optional[float] = None,
    reset_gflops: Optional[float] = None,
) -> Dict[str, Any]:
    """按实际调度次数汇总预先测得的组件 GFLOPs。"""
    supported_modes = ("fast-only", "slow-only", "fast-ours")
    if mode not in supported_modes:
        raise ValueError(f"mode 必须是 {supported_modes}，得到 {mode!r}")

    components = {
        "fast_gflops_per_call": fast_gflops,
        "slow_gflops_per_call": slow_gflops,
        "evolve_gflops_per_call": evolve_gflops,
        "reset_gflops_per_call": reset_gflops,
    }
    for name, value in components.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} 不能为负数，得到 {value}")

    n_measured = int(result["n_measured"])
    if mode == "fast-only":
        calls = {"fast": n_measured, "slow": 0, "evolve": 0, "reset": 0}
        required = ("fast_gflops_per_call",)
    elif mode == "slow-only":
        calls = {"fast": 0, "slow": n_measured, "evolve": 0, "reset": 0}
        required = ("slow_gflops_per_call",)
    else:
        calls = {
            "fast": n_measured,
            "slow": int(result["n_slow"]),
            "evolve": int(result["n_regular"]),
            "reset": int(result["n_slow"]),
        }
        required = tuple(components)

    missing = [name for name in required if components[name] is None]
    total_gflops = None
    amortized_gflops = None
    if not missing:
        total_gflops = (
            calls["fast"] * float(fast_gflops or 0.0)
            + calls["slow"] * float(slow_gflops or 0.0)
            + calls["evolve"] * float(evolve_gflops or 0.0)
            + calls["reset"] * float(reset_gflops or 0.0)
        )
        amortized_gflops = total_gflops / n_measured

    result["flops"] = {
        "source": "preprofiled_component_gflops",
        "components": components,
        "calls": calls,
        "total_gflops": total_gflops,
        "amortized_gflops_per_output": amortized_gflops,
        "missing_components": missing,
    }
    return result


def _new_cuda_events(stage_names: Iterable[str]):
    return {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in stage_names
    }


def _record_start(events, name: str) -> None:
    events[name][0].record()


def _record_end(events, name: str) -> None:
    events[name][1].record()


def _elapsed_ms(events, name: str) -> float:
    return float(events[name][0].elapsed_time(events[name][1]))


def _prepare_measurement(i: int, warmup: int, device: torch.device):
    if i != warmup:
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _finish_result(
    *,
    name: str,
    timings: List[FrameTiming],
    wall_start: Optional[float],
    warmup: int,
    device: torch.device,
) -> Dict[str, Any]:
    if wall_start is None:
        raise RuntimeError("system benchmark 未进入 measured 阶段")
    total_wall_s = time.perf_counter() - wall_start
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return summarize_system_benchmark(
        name=name,
        timings=timings,
        total_wall_s=total_wall_s,
        warmup=warmup,
        peak_memory_mb=peak_memory_mb,
    )


def run_alocc_only_system(
    *,
    name: str,
    runner,
    raw_batches: Iterable,
    metas_list: Sequence[Any],
    warmup: int,
    samples: int,
    device: torch.device,
    role: str,
) -> Dict[str, Any]:
    """运行 pipeline-matched fast-only 或 slow-only 系统。"""
    if role not in ("fast", "slow"):
        raise ValueError(f"role 仅支持 fast/slow，得到 {role!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("system benchmark 需要 CUDA")

    events = _new_cuda_events(("h2d", role, "post"))
    timings: List[FrameTiming] = []
    wall_start: Optional[float] = None
    cur_scene: Optional[str] = None
    batch_iter = iter(raw_batches)

    with torch.inference_mode():
        for i, meta in enumerate(metas_list):
            if i >= warmup + samples:
                break
            measured_start = _prepare_measurement(i, warmup, device)
            if measured_start is not None:
                wall_start = measured_start
            frame_start = time.perf_counter()

            if str(meta.scene_name) != cur_scene:
                runner.reset_history()
                cur_scene = str(meta.scene_name)

            input_start = time.perf_counter()
            raw = next(batch_iter)
            input_wait_ms = (time.perf_counter() - input_start) * 1000.0

            _record_start(events, "h2d")
            batch = scatter_to_device(raw, 0)
            _record_end(events, "h2d")

            _record_start(events, role)
            logits = runner.forward_keyframe(batch)
            _record_end(events, role)

            _record_start(events, "post")
            logits.argmax(0).to(torch.uint8).cpu().numpy()
            _record_end(events, "post")
            torch.cuda.synchronize(device)

            if i >= warmup:
                frame_wall_ms = (time.perf_counter() - frame_start) * 1000.0
                timings.append(
                    FrameTiming(
                        frame_idx=i - warmup,
                        sample_token=str(meta.sample_token),
                        scene_name=str(meta.scene_name),
                        is_slow=role == "slow",
                        frame_wall_ms=frame_wall_ms,
                        input_wait_ms=input_wait_ms,
                        h2d_ms=_elapsed_ms(events, "h2d"),
                        fast_ms=_elapsed_ms(events, "fast") if role == "fast" else 0.0,
                        slow_ms=_elapsed_ms(events, "slow") if role == "slow" else 0.0,
                        align_ms=0.0,
                        post_ms=_elapsed_ms(events, "post"),
                    )
                )
            del logits, batch, raw

    return _finish_result(
        name=name,
        timings=timings,
        wall_start=wall_start,
        warmup=warmup,
        device=device,
    )


def run_fast_ours_system(
    *,
    fast,
    slow,
    stream_aligner,
    fast_batches: Iterable,
    slow_batches: Iterable,
    schedule: Sequence[SystemFrame],
    warmup: int,
    samples: int,
    device: torch.device,
) -> Dict[str, Any]:
    """运行 fast@2Hz + sparse slow + EvoOcc 的系统级 benchmark。"""
    if not torch.cuda.is_available():
        raise RuntimeError("system benchmark 需要 CUDA")

    events = _new_cuda_events(("h2d_fast", "h2d_slow", "fast", "slow", "align", "post"))
    timings: List[FrameTiming] = []
    wall_start: Optional[float] = None
    fast_iter = iter(fast_batches)
    slow_iter = iter(slow_batches)

    fast.reset_history()
    slow.reset_history()
    stream_aligner.reset_scene()

    with torch.inference_mode():
        for i, frame in enumerate(schedule):
            if i >= warmup + samples:
                break
            measured_start = _prepare_measurement(i, warmup, device)
            if measured_start is not None:
                wall_start = measured_start
            frame_start = time.perf_counter()

            if frame.is_scene_start:
                fast.reset_history()
                slow.reset_history()
                stream_aligner.reset_scene()

            input_start = time.perf_counter()
            raw_fast = next(fast_iter)
            raw_slow = next(slow_iter) if frame.is_slow else None
            input_wait_ms = (time.perf_counter() - input_start) * 1000.0

            _record_start(events, "h2d_fast")
            fast_batch = scatter_to_device(raw_fast, 0)
            _record_end(events, "h2d_fast")

            _record_start(events, "fast")
            fast_logits = fast.forward_keyframe(fast_batch)
            _record_end(events, "fast")

            ego_t = torch.from_numpy(frame.meta.ego2global).to(
                device=device, dtype=torch.float32
            )
            h2d_slow_ms = 0.0
            slow_ms = 0.0
            if frame.is_slow:
                slow.reset_history()
                _record_start(events, "h2d_slow")
                slow_batch = scatter_to_device(raw_slow, 0)
                _record_end(events, "h2d_slow")

                _record_start(events, "slow")
                slow_logits = slow.forward_keyframe(slow_batch)
                _record_end(events, "slow")

                _record_start(events, "align")
                aligned = stream_aligner.reset_with_slow(
                    fast_logits, slow_logits, ego_t, frame.meta.timestamp_us
                )
                _record_end(events, "align")
            else:
                _record_start(events, "align")
                aligned = stream_aligner.evolve(
                    fast_logits, ego_t, frame.meta.timestamp_us
                )
                _record_end(events, "align")

            _record_start(events, "post")
            aligned.argmax(0).to(torch.uint8).cpu().numpy()
            _record_end(events, "post")
            torch.cuda.synchronize(device)

            if frame.is_slow:
                h2d_slow_ms = _elapsed_ms(events, "h2d_slow")
                slow_ms = _elapsed_ms(events, "slow")
                # sparse slow 不跨调用保留内部历史，权重仍常驻 GPU。
                slow.reset_history()

            if i >= warmup:
                frame_wall_ms = (time.perf_counter() - frame_start) * 1000.0
                timings.append(
                    FrameTiming(
                        frame_idx=i - warmup,
                        sample_token=str(frame.meta.sample_token),
                        scene_name=str(frame.meta.scene_name),
                        is_slow=bool(frame.is_slow),
                        frame_wall_ms=frame_wall_ms,
                        input_wait_ms=input_wait_ms,
                        h2d_ms=_elapsed_ms(events, "h2d_fast") + h2d_slow_ms,
                        fast_ms=_elapsed_ms(events, "fast"),
                        slow_ms=slow_ms,
                        align_ms=_elapsed_ms(events, "align"),
                        post_ms=_elapsed_ms(events, "post"),
                    )
                )
            if frame.is_slow:
                del slow_logits, slow_batch, raw_slow
            del aligned, fast_logits, fast_batch, raw_fast, ego_t

    return _finish_result(
        name="fast+ours-system",
        timings=timings,
        wall_start=wall_start,
        warmup=warmup,
        device=device,
    )


def print_system_summary(result: Mapping[str, Any]) -> None:
    """打印紧凑的系统级测速汇总。"""
    latency = result["latency_ms"]
    regular = result["regular_latency_ms"]
    slow_tick = result["slow_tick_latency_ms"]
    print(
        f"  ★ {result['name']}: E2E={result['fps_e2e']:.2f} FPS, "
        f"mean={latency['mean']:.2f}ms, p95={latency['p95']:.2f}ms"
    )
    if regular["mean"] is not None:
        print(
            f"    regular: mean={regular['mean']:.2f}ms, "
            f"p95={regular['p95']:.2f}ms, n={result['n_regular']}"
        )
    if slow_tick["mean"] is not None:
        print(
            f"    slow tick: mean={slow_tick['mean']:.2f}ms, "
            f"p95={slow_tick['p95']:.2f}ms, n={result['n_slow']}"
        )
    print(
        f"    model amortized={result['model_latency_ms_mean']:.2f}ms / "
        f"{result['fps_model_amortized']:.2f} FPS, "
        f"peak VRAM={result['peak_memory_mb']:.0f}MB"
    )
    evolve = result["evolve_latency_ms"]
    reset = result["reset_latency_ms"]
    if evolve["mean"] is not None:
        print(
            f"    EvoOcc evolve: mean={evolve['mean']:.2f}ms, "
            f"p95={evolve['p95']:.2f}ms"
        )
    if reset["mean"] is not None:
        print(
            f"    EvoOcc reset:  mean={reset['mean']:.2f}ms, "
            f"p95={reset['p95']:.2f}ms"
        )
    flops = result.get("flops")
    if flops is not None:
        amortized = flops["amortized_gflops_per_output"]
        if amortized is None:
            missing = ", ".join(flops["missing_components"])
            print(f"    amortized GFLOPs/output=N/A (missing: {missing})")
        else:
            print(f"    amortized GFLOPs/output={amortized:.3f}")
