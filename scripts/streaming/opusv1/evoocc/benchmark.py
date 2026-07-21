"""OPUSv1-T streaming benchmark."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from evoocc.streaming.benchmark_runtime import configure_benchmark_env

configure_benchmark_env()

import torch

from evoocc.config import load_config_with_base
from evoocc.streaming.benchmark_runtime import configure_torch_benchmark_runtime
from evoocc.streaming.aligner_factory import (
    build_evoocc_aligner,
    resolve_repo_path,
    resolve_slow_root,
)
from evoocc.streaming.benchmark_loop import (
    benchmark_stream_aligned,
    make_loader_iter,
    preload_slow_cache,
    select_benchmark_frames,
)
from evoocc.streaming.benchmark_modes import (
    benchmark_opus_fast_only,
    benchmark_opus_native_only,
    benchmark_opus_only,
)
from evoocc.streaming.opus_runtime import (
    DEFAULT_GT_ROOT,
    DEFAULT_META_PKL,
    DEFAULT_OPUS_ROOT,
    DEFAULT_OPUSV1_CKPT,
    DEFAULT_OPUSV1_CONFIG,
    DEFAULT_OPUSV2_CKPT,
    DEFAULT_OPUSV2_CONFIG,
    build_opus_runner,
    resolve_opus_path,
)
from evoocc.streaming.scene_iterator import build_opus_sample_meta_index, iter_scenes
from evoocc.streaming.stream_aligner import StreamAligner

configure_torch_benchmark_runtime(torch)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--opus-root", default=DEFAULT_OPUS_ROOT)
    p.add_argument("--opus-config", default=DEFAULT_OPUSV1_CONFIG)
    p.add_argument("--opus-ckpt", default=DEFAULT_OPUSV1_CKPT)
    p.add_argument("--slow-opus-config", default=DEFAULT_OPUSV2_CONFIG)
    p.add_argument("--slow-opus-ckpt", default=DEFAULT_OPUSV2_CKPT)
    p.add_argument("--meta-pkl", default=DEFAULT_META_PKL)
    p.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--slow-interval", type=float, default=5.0)
    p.add_argument(
        "--mode",
        choices=["native-only", "fast-only", "slow-only", "fast-ours", "both", "all"],
        default="all",
        help="默认 all，运行全部模式；both 仅运行 fast-only + fast-ours",
    )
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()
    if args.mode in ("fast-ours", "both", "all") and not args.checkpoint:
        p.error(f"--mode {args.mode} 需要 --checkpoint")
    return args


def build_fast_runner(args, data_cfg):
    return build_opus_runner(
        data_cfg,
        opus_root=args.opus_root,
        config_path=args.opus_config,
        ckpt_path=args.opus_ckpt,
        role="fast",
        repo_root=REPO_ROOT,
        device="cuda:0",
    )


def build_slow_runner(args, data_cfg):
    return build_opus_runner(
        data_cfg,
        opus_root=args.opus_root,
        config_path=args.slow_opus_config,
        ckpt_path=args.slow_opus_ckpt,
        role="slow",
        repo_root=REPO_ROOT,
        device="cuda:0",
    )


def mode_enabled(mode: str, name: str) -> bool:
    if mode == "all":
        return True
    if mode == "both":
        return name in ("fast-only", "fast-ours")
    return mode == name


def select_frames(runner, sample_meta_index, args):
    scenes_meta = list(iter_scenes(runner.dataset, sample_meta_index, limit_scenes=None))
    _flat, flat_indices, flat_metas = select_benchmark_frames(
        scenes_meta, args.warmup, args.samples
    )
    return flat_indices, flat_metas


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    args.config = resolve_repo_path(args.config, REPO_ROOT)
    args.checkpoint = resolve_repo_path(args.checkpoint, REPO_ROOT)
    args.opus_root = resolve_repo_path(args.opus_root, REPO_ROOT)
    args.opus_config = resolve_opus_path(args.opus_config, args.opus_root, REPO_ROOT)
    args.opus_ckpt = resolve_opus_path(args.opus_ckpt, args.opus_root, REPO_ROOT)
    args.slow_opus_config = resolve_opus_path(
        args.slow_opus_config, args.opus_root, REPO_ROOT
    )
    args.slow_opus_ckpt = resolve_opus_path(
        args.slow_opus_ckpt, args.opus_root, REPO_ROOT
    )
    args.meta_pkl = resolve_repo_path(args.meta_pkl, REPO_ROOT)
    args.gt_root = resolve_repo_path(args.gt_root, REPO_ROOT)
    args.out_json = resolve_repo_path(args.out_json, REPO_ROOT)
    device = torch.device("cuda:0")
    data_cfg = load_config_with_base(args.config)["data"]
    slow_root = resolve_slow_root(data_cfg, REPO_ROOT)
    slow_format = data_cfg.get(
        "slow_logit_format", data_cfg.get("logits_format", "opus_sparse_full")
    )
    print(f"[meta] slow_format={slow_format}, slow_root={slow_root}")
    sample_meta_index = build_opus_sample_meta_index(
        args.meta_pkl, slow_root, args.gt_root
    )

    results = {}

    if mode_enabled(args.mode, "slow-only"):
        print("[slow] OPUSv2-L slow runner build ...")
        slow = build_slow_runner(args, data_cfg)
        slow_indices, _slow_metas = select_frames(slow, sample_meta_index, args)
        print("\n=== Mode S: OPUSv2-L slow-only raw-top3 ===")
        results["slow_only"] = benchmark_opus_only(
            slow,
            make_loader_iter(slow, slow_indices, args.num_workers, args.prefetch_factor),
            args.warmup,
            args.samples,
            name="slow-only(raw-top3)",
        )
        del slow
        torch.cuda.empty_cache()

    needs_fast = any(
        mode_enabled(args.mode, name)
        for name in ("native-only", "fast-only", "fast-ours")
    )
    if needs_fast:
        stream_aligner = None
        if mode_enabled(args.mode, "fast-ours"):
            print("[aligner] build & load ckpt ...")
            aligner, aligner_data_cfg = build_evoocc_aligner(
                args.config, args.checkpoint, device, solver=args.solver
            )
            if aligner_data_cfg != data_cfg:
                raise ValueError("aligner 构建返回的 data_cfg 与配置加载结果不一致")
            stream_aligner = StreamAligner(aligner)

        print("[fast] OPUSv1-T fast runner build ...")
        fast = build_fast_runner(args, data_cfg)
        flat_indices, flat_metas = select_frames(fast, sample_meta_index, args)

    if mode_enabled(args.mode, "native-only"):
        print("\n=== Mode A0: OPUS native simple_test + sparse->dense uint8 ===")
        results["native_only"] = benchmark_opus_native_only(
            fast,
            make_loader_iter(fast, flat_indices, args.num_workers, args.prefetch_factor),
            args.warmup,
            args.samples,
            data_cfg,
        )

    if mode_enabled(args.mode, "fast-only"):
        print("\n=== Mode A: OPUS raw-top3 fast-only ===")
        results["fast_only"] = benchmark_opus_fast_only(
            fast,
            make_loader_iter(fast, flat_indices, args.num_workers, args.prefetch_factor),
            args.warmup,
            args.samples,
        )

    if mode_enabled(args.mode, "fast-ours"):
        print("[slow-cache] preload OPUSv2-L logits ...")
        slow_cache = preload_slow_cache(data_cfg, device, flat_metas)
        print(f"\n=== Mode B: OPUS raw-top3 fast + EvoOcc (slow_interval={args.slow_interval}s) ===")
        results["fast_ours"] = benchmark_stream_aligned(
            name="fast+ours",
            fast=fast,
            stream_aligner=stream_aligner,
            slow_cache=slow_cache,
            raw_batches=make_loader_iter(fast, flat_indices, args.num_workers, args.prefetch_factor),
            metas_list=flat_metas,
            warmup=args.warmup,
            samples=args.samples,
            slow_interval=args.slow_interval,
            device=device,
            fallback_fast_before_first_slow=True,
        )

    print(f"\n{'=' * 72}")
    print(f"Final OPUS benchmark (warmup={args.warmup}, measured={args.samples})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 72}")
    if "native_only" in results:
        a0 = results["native_only"]
        print(f"  OPUS native          | {a0['latency_ms_mean']:6.2f} ms | {a0['fps']:6.2f} FPS")
    if "slow_only" in results:
        s = results["slow_only"]
        print(f"  OPUSv2-L slow-only   | {s['latency_ms_mean']:6.2f} ms | {s['fps']:6.2f} FPS")
    if "fast_only" in results:
        a = results["fast_only"]
        print(f"  raw-top3 fast-only   | {a['latency_ms_mean']:6.2f} ms | {a['fps']:6.2f} FPS")
    if "fast_ours" in results:
        b = results["fast_ours"]
        print(
            f"  raw-top3 fast + ours | {b['latency_ms_mean']:6.2f} ms | {b['fps']:6.2f} FPS  "
            f"(reset={b['n_reset']}/evolve={b['n_evolve']})"
        )
    if "fast_only" in results and "fast_ours" in results:
        a, b = results["fast_only"], results["fast_ours"]
        d_ms = b["latency_ms_mean"] - a["latency_ms_mean"]
        d_pct = d_ms / a["latency_ms_mean"] * 100
        print(f"  aligner overhead     | +{d_ms:5.2f} ms | {d_pct:+6.1f}%")

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "fast_backend": "opusv1t_raw_top3",
                "slow_backend": "opusv2l_raw_top3",
                "fast_config": args.opus_config,
                "fast_checkpoint": args.opus_ckpt,
                "slow_config": args.slow_opus_config,
                "slow_checkpoint": args.slow_opus_ckpt,
                "dataset_variant": data_cfg.get("dataset_variant", "occ3d"),
                "mode": args.mode,
                "warmup": args.warmup,
                "measured": args.samples,
                "slow_interval_sec": args.slow_interval,
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
            }, f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
