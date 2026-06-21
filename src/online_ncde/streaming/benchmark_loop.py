"""Streaming benchmark 公共计时骨架."""
from __future__ import annotations

import time
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch

from online_ncde.streaming.slow_cache import SlowLogitsGPUCache, build_slow_decoder_fn
from online_ncde.streaming.streaming_loader import make_streaming_loader, scatter_to_device


def select_benchmark_frames(scenes_meta, warmup: int, samples: int):
    """按 scene 顺序取 warmup+samples 个连续 keyframe."""
    need = int(warmup) + int(samples)
    flat = []
    for _scene_name, kf_list in scenes_meta:
        flat.extend(kf_list)
        if len(flat) >= need:
            break
    if len(flat) < need:
        raise ValueError(f"需要 {need} keyframe, 只找到 {len(flat)}")
    flat = flat[:need]
    flat_indices = [idx for idx, _ in flat]
    flat_metas = [m for _, m in flat]
    print(f"  benchmark sample 总数: {len(flat)} (warmup={warmup} + measured={samples})")
    print(f"  跨 {len(set(m.scene_name for m in flat_metas))} 个 scene")
    return flat, flat_indices, flat_metas


def preload_slow_cache(data_cfg: dict, device: torch.device, flat_metas) -> SlowLogitsGPUCache:
    """预加载 slow logits，避免解压时间污染 benchmark."""
    slow_decoder_fn = build_slow_decoder_fn(data_cfg, device)
    slow_cache = SlowLogitsGPUCache(device=device, decoder_fn=slow_decoder_fn)
    slow_cache.preload([m.slow_logit_path for m in flat_metas], skip_missing=True, verbose=False)
    print(f"  cached {len(slow_cache)} slow paths")
    return slow_cache


def summarize_latency(name: str, per_iter_ms: List[float], pure_inf_time: float, warmup: int, samples: int):
    """输出并返回 latency/FPS 统计."""
    fps = samples / pure_inf_time
    arr = np.asarray(per_iter_ms, dtype=np.float64)
    print(
        f"  ★ {name}: latency mean={1000/fps:.2f}ms "
        f"median={np.median(arr):.2f}ms p95={np.percentile(arr, 95):.2f}ms / FPS={fps:.2f}"
    )
    return {
        "latency_ms_mean": float(1000 / fps),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "fps": float(fps),
        "n_measured": int(samples),
        "n_warmup": int(warmup),
    }


def benchmark_callable(
    name: str,
    raw_batches: Iterable,
    warmup: int,
    samples: int,
    step_fn: Callable,
    log_interval: int = 50,
):
    """对单个 step_fn(raw) 做 mmdet 风格 warmup/measured 计时."""
    pure_inf_time = 0.0
    per_iter = []
    for i, raw in enumerate(raw_batches):
        if i >= warmup + samples:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            step_fn(raw)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i >= warmup:
            pure_inf_time += elapsed
            per_iter.append(elapsed * 1000)
            done = i + 1 - warmup
            if done % log_interval == 0 or done == samples:
                fps = done / pure_inf_time
                print(f"    [{done:>3}/{samples}] FPS: {fps:6.2f}  latency: {1000/fps:6.2f} ms")
    return summarize_latency(name, per_iter, pure_inf_time, warmup, samples)


def benchmark_stream_aligned(
    *,
    name: str,
    fast,
    stream_aligner,
    slow_cache: SlowLogitsGPUCache,
    raw_batches: Iterable,
    metas_list,
    warmup: int,
    samples: int,
    slow_interval: float,
    device: torch.device,
    fallback_fast_before_first_slow: bool = False,
    log_interval: int = 50,
):
    """fast.forward_keyframe -> stream aligner -> argmax/cpu 的通用计时."""
    pure_inf_time = 0.0
    per_iter = []
    n_reset, n_evolve = 0, 0
    cur_scene = None
    last_slow_t_sec = -1e9

    fast.reset_history()
    stream_aligner.reset_scene()

    for i, (raw, meta) in enumerate(zip(raw_batches, metas_list)):
        if i >= warmup + samples:
            break
        if meta.scene_name != cur_scene:
            stream_aligner.reset_scene()
            cur_scene = meta.scene_name
            last_slow_t_sec = -1e9

        t_sec = meta.timestamp_us / 1e6
        is_slow = (last_slow_t_sec < 0) or (t_sec - last_slow_t_sec + 1e-3 >= slow_interval)
        ego_t = torch.from_numpy(meta.ego2global).to(device=device, dtype=torch.float32)
        slow_logits = (
            slow_cache.get(meta.slow_logit_path)
            if is_slow and slow_cache.has(meta.slow_logit_path)
            else None
        )

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            batch = scatter_to_device(raw, 0)
            fast_logits = fast.forward_keyframe(batch)
            if slow_logits is not None:
                aligned = stream_aligner.reset_with_slow(
                    fast_logits, slow_logits, ego_t, meta.timestamp_us
                )
            elif fallback_fast_before_first_slow and stream_aligner.hidden is None:
                aligned = fast_logits
            else:
                aligned = stream_aligner.evolve(fast_logits, ego_t, meta.timestamp_us)
            aligned.argmax(0).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i >= warmup:
            pure_inf_time += elapsed
            per_iter.append(elapsed * 1000)
            if slow_logits is not None:
                n_reset += 1
            else:
                n_evolve += 1
            done = i + 1 - warmup
            if done % log_interval == 0 or done == samples:
                fps = done / pure_inf_time
                print(
                    f"    [{done:>3}/{samples}] FPS: {fps:6.2f}  latency: {1000/fps:6.2f} ms  "
                    f"(reset={n_reset} evolve={n_evolve})"
                )
        if slow_logits is not None:
            last_slow_t_sec = t_sec

    out = summarize_latency(name, per_iter, pure_inf_time, warmup, samples)
    out.update({
        "n_reset": int(n_reset),
        "n_evolve": int(n_evolve),
        "slow_interval_sec": float(slow_interval),
    })
    print(f"     reset/evolve in measured: {n_reset}/{n_evolve}")
    return out


def make_loader_iter(fast, flat_indices, num_workers: int, prefetch_factor: int):
    loader = make_streaming_loader(
        fast.dataset,
        flat_indices,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    return iter(loader)


def wants_mode(mode: str, name: str, both_names: Tuple[str, ...]) -> bool:
    if mode == "all":
        return True
    if mode == "both" and name in both_names:
        return True
    return mode == name
