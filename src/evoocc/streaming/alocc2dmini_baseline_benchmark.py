"""新增 baseline 共用的 ALOcc2DMini streaming benchmark 驱动。"""
from __future__ import annotations

import argparse
import json
import os

import torch

from evoocc.streaming.aligner_factory import (
    resolve_repo_path,
    resolve_slow_root,
)
from evoocc.streaming.alocc2dmini_runtime import (
    DEFAULT_BDV2_PKL,
    DEFAULT_OCCSTUDIO_ROOT,
    build_alocc2dmini_fast_runner,
    resolve_cfg_path,
)
from evoocc.streaming.baseline_factory import (
    SUPPORTED_BASELINES,
    build_streaming_baseline,
    load_config_with_baseline_overlay,
)
from evoocc.streaming.baseline_stream_aligner import (
    build_baseline_stream_aligner,
)
from evoocc.streaming.benchmark_loop import (
    benchmark_stream_aligned,
    make_loader_iter,
    preload_slow_cache,
    select_benchmark_frames,
)
from evoocc.streaming.benchmark_modes import benchmark_alocc_fast_only
from evoocc.streaming.benchmark_runtime import (
    configure_torch_benchmark_runtime,
)
from evoocc.streaming.scene_iterator import (
    build_sample_meta_index,
    iter_scenes,
)


configure_torch_benchmark_runtime(torch)


def parse_args(baseline_name: str):
    if baseline_name not in SUPPORTED_BASELINES:
        raise ValueError(
            f"未知 baseline: {baseline_name!r}，可选 {SUPPORTED_BASELINES}"
        )
    parser = argparse.ArgumentParser(
        description=f"ALOcc2DMini + {baseline_name} streaming benchmark"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--occ-root", default=DEFAULT_OCCSTUDIO_ROOT)
    parser.add_argument("--occ-config", default=None)
    parser.add_argument("--occ-ckpt", default=None)
    parser.add_argument("--bevdetv2-pkl", default=DEFAULT_BDV2_PKL)
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--slow-interval", type=float, default=5.0)
    parser.add_argument(
        "--mode",
        choices=["fast-only", "fast-baseline", "both"],
        default="both",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    if args.mode in ("fast-baseline", "both") and not args.checkpoint:
        parser.error(f"--mode {args.mode} 需要 --checkpoint")
    return args


def _select_frames(
    fast,
    args,
    data_cfg,
    repo_root: str,
):
    slow_root = resolve_slow_root(data_cfg, repo_root)
    gt_root = resolve_cfg_path(
        data_cfg,
        "gt_root",
        repo_root,
        args.gt_root,
    )
    nuscenes_root = resolve_cfg_path(
        data_cfg,
        "nuscenes_root",
        repo_root,
    )
    slow_format = data_cfg.get(
        "slow_logit_format",
        data_cfg.get("logits_format", "alocc_dense_topk"),
    )
    print(
        f"[3] sample meta index "
        f"(slow_format={slow_format}, slow_root={slow_root}) ..."
    )
    sample_to_meta = build_sample_meta_index(
        args.bevdetv2_pkl,
        slow_root,
        gt_root,
        dataset_variant=data_cfg.get("dataset_variant", "occ3d"),
        nuscenes_root=nuscenes_root,
        nuscenes_version=data_cfg.get(
            "nuscenes_version",
            "v1.0-trainval",
        ),
    )
    scenes_meta = list(
        iter_scenes(fast.dataset, sample_to_meta, limit_scenes=None)
    )
    _flat, flat_indices, flat_metas = select_benchmark_frames(
        scenes_meta,
        args.warmup,
        args.samples,
    )
    return flat_indices, flat_metas


def _print_results(
    baseline_name: str,
    results: dict,
    warmup: int,
    samples: int,
) -> None:
    display_name = baseline_name.replace("_", "-")
    print(f"\n{'=' * 72}")
    print(
        f"Final {display_name} benchmark "
        f"(warmup={warmup}, measured={samples})"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 72}")
    if "fast_only" in results:
        fast_only = results["fast_only"]
        print(
            f"  fast-only baseline       | "
            f"{fast_only['latency_ms_mean']:6.2f} ms | "
            f"{fast_only['fps']:6.2f} FPS"
        )
    result_key = f"fast_{baseline_name}"
    if result_key in results:
        baseline = results[result_key]
        print(
            f"  fast + {display_name:<18}| "
            f"{baseline['latency_ms_mean']:6.2f} ms | "
            f"{baseline['fps']:6.2f} FPS  "
            f"(reset={baseline['n_reset']}/"
            f"evolve={baseline['n_evolve']})"
        )
    if "fast_only" in results and result_key in results:
        fast_only = results["fast_only"]
        baseline = results[result_key]
        delta_ms = (
            baseline["latency_ms_mean"]
            - fast_only["latency_ms_mean"]
        )
        delta_pct = delta_ms / fast_only["latency_ms_mean"] * 100
        print(
            f"  baseline overhead        | "
            f"{delta_ms:+6.2f} ms | {delta_pct:+6.1f}%"
        )


def main(baseline_name: str, repo_root: str) -> None:
    args = parse_args(baseline_name)
    os.chdir(repo_root)
    args.config = resolve_repo_path(args.config, repo_root)
    args.checkpoint = resolve_repo_path(args.checkpoint, repo_root)
    args.occ_root = resolve_repo_path(args.occ_root, repo_root)
    args.bevdetv2_pkl = resolve_repo_path(
        args.bevdetv2_pkl,
        repo_root,
    )
    args.out_json = resolve_repo_path(args.out_json, repo_root)
    device = torch.device("cuda:0")

    cfg = load_config_with_baseline_overlay(
        args.config,
        baseline_name,
    )
    data_cfg = cfg["data"]
    stream_aligner = None
    if args.mode in ("fast-baseline", "both"):
        print(f"[1] {baseline_name} build & load ckpt ...")
        model, model_data_cfg = build_streaming_baseline(
            args.config,
            args.checkpoint,
            baseline_name,
            device,
        )
        if model_data_cfg != data_cfg:
            raise ValueError(
                "baseline 构建返回的 data_cfg 与配置加载结果不一致"
            )
        stream_aligner = build_baseline_stream_aligner(
            baseline_name,
            model,
        )

    print("[2] ALOcc2DMini fast runner build ...")
    fast = build_alocc2dmini_fast_runner(
        data_cfg,
        occstudio_root=args.occ_root,
        occ_config=args.occ_config,
        occ_ckpt=args.occ_ckpt,
        device="cuda:0",
    )
    flat_indices, flat_metas = _select_frames(
        fast,
        args,
        data_cfg,
        repo_root,
    )

    results = {}
    if args.mode in ("fast-only", "both"):
        print("\n=== Mode A: fast-only baseline ===")
        results["fast_only"] = benchmark_alocc_fast_only(
            fast,
            make_loader_iter(
                fast,
                flat_indices,
                args.num_workers,
                args.prefetch_factor,
            ),
            args.warmup,
            args.samples,
        )

    if args.mode in ("fast-baseline", "both"):
        print("[4] preload slow logits ...")
        slow_cache = preload_slow_cache(
            data_cfg,
            device,
            flat_metas,
        )
        display_name = baseline_name.replace("_", "-")
        print(
            f"\n=== Mode B: fast + {display_name} "
            f"(slow_interval={args.slow_interval}s) ==="
        )
        result_key = f"fast_{baseline_name}"
        results[result_key] = benchmark_stream_aligned(
            name=f"fast+{display_name}",
            fast=fast,
            stream_aligner=stream_aligner,
            slow_cache=slow_cache,
            raw_batches=make_loader_iter(
                fast,
                flat_indices,
                args.num_workers,
                args.prefetch_factor,
            ),
            metas_list=flat_metas,
            warmup=args.warmup,
            samples=args.samples,
            slow_interval=args.slow_interval,
            device=device,
        )

    _print_results(
        baseline_name,
        results,
        args.warmup,
        args.samples,
    )

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as file:
            json.dump(
                {
                    "fast_backend": "alocc2dmini",
                    "aligner": baseline_name,
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "mode": args.mode,
                    "warmup": args.warmup,
                    "measured": args.samples,
                    "slow_interval_sec": args.slow_interval,
                    "gpu": torch.cuda.get_device_name(0),
                    "results": results,
                },
                file,
                indent=2,
            )
        print(f"\n[saved] {args.out_json}")
