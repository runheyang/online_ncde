"""OPUSv1-T fast + StreamingFlow streaming benchmark。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from evoocc.streaming.benchmark_runtime import configure_benchmark_env

configure_benchmark_env()

import torch

from evoocc.streaming.aligner_factory import resolve_repo_path, resolve_slow_root
from evoocc.streaming.benchmark_loop import (
    benchmark_stream_aligned,
    make_loader_iter,
    preload_slow_cache,
    select_benchmark_frames,
)
from evoocc.streaming.benchmark_modes import benchmark_opus_fast_only
from evoocc.streaming.benchmark_runtime import configure_torch_benchmark_runtime
from evoocc.streaming.opus_runtime import (
    DEFAULT_GT_ROOT,
    DEFAULT_META_PKL,
    DEFAULT_OPUS_ROOT,
    DEFAULT_OPUSV1_CKPT,
    DEFAULT_OPUSV1_CONFIG,
    build_opus_runner,
    resolve_opus_path,
)
from evoocc.streaming.scene_iterator import build_opus_sample_meta_index, iter_scenes
from evoocc.streaming.streamingflow_aligner import (
    StreamingFlowStreamAligner,
    build_streamingflow_model,
)

configure_torch_benchmark_runtime(torch)


STREAMINGFLOW_OVERLAY = (
    Path(REPO_ROOT) / "src" / "evoocc" / "baselines" / "streamingflow" / "occ3d_config.yaml"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opus-root", default=DEFAULT_OPUS_ROOT)
    p.add_argument("--opus-config", default=DEFAULT_OPUSV1_CONFIG)
    p.add_argument("--opus-ckpt", default=DEFAULT_OPUSV1_CKPT)
    p.add_argument("--meta-pkl", default=DEFAULT_META_PKL)
    p.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--slow-interval", type=float, default=5.0)
    p.add_argument(
        "--mode",
        choices=["fast-only", "fast-streamingflow", "both"],
        default="both",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


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


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    args.config = resolve_repo_path(args.config, REPO_ROOT)
    args.checkpoint = resolve_repo_path(args.checkpoint, REPO_ROOT)
    args.opus_root = resolve_repo_path(args.opus_root, REPO_ROOT)
    args.opus_config = resolve_opus_path(args.opus_config, args.opus_root, REPO_ROOT)
    args.opus_ckpt = resolve_opus_path(args.opus_ckpt, args.opus_root, REPO_ROOT)
    args.meta_pkl = resolve_repo_path(args.meta_pkl, REPO_ROOT)
    args.gt_root = resolve_repo_path(args.gt_root, REPO_ROOT)
    args.out_json = resolve_repo_path(args.out_json, REPO_ROOT)
    device = torch.device("cuda:0")

    print("[1] StreamingFlow build & load ckpt ...")
    model, data_cfg = build_streamingflow_model(
        args.config,
        args.checkpoint,
        str(STREAMINGFLOW_OVERLAY),
        device,
    )
    stream_aligner = StreamingFlowStreamAligner(model)

    print("[2] OPUSv1-T fast runner build ...")
    fast = build_fast_runner(args, data_cfg)

    slow_root = resolve_slow_root(data_cfg, REPO_ROOT)
    slow_format = data_cfg.get(
        "slow_logit_format", data_cfg.get("logits_format", "opus_sparse_full")
    )
    print(f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}) ...")
    s2m = build_opus_sample_meta_index(args.meta_pkl, slow_root, args.gt_root)
    scenes_meta = list(iter_scenes(fast.dataset, s2m, limit_scenes=None))
    _flat, flat_indices, flat_metas = select_benchmark_frames(
        scenes_meta, args.warmup, args.samples
    )

    print("[4] preload OPUSv2-L slow logits ...")
    slow_cache = preload_slow_cache(data_cfg, device, flat_metas)

    results = {}
    if args.mode in ("fast-only", "both"):
        print("\n=== Mode A: OPUS raw-top3 fast-only ===")
        results["fast_only"] = benchmark_opus_fast_only(
            fast,
            make_loader_iter(fast, flat_indices, args.num_workers, args.prefetch_factor),
            args.warmup,
            args.samples,
        )

    if args.mode in ("fast-streamingflow", "both"):
        print(
            f"\n=== Mode B: OPUS raw-top3 fast + StreamingFlow "
            f"(slow_interval={args.slow_interval}s) ==="
        )
        results["fast_streamingflow"] = benchmark_stream_aligned(
            name="fast+streamingflow",
            fast=fast,
            stream_aligner=stream_aligner,
            slow_cache=slow_cache,
            raw_batches=make_loader_iter(
                fast, flat_indices, args.num_workers, args.prefetch_factor
            ),
            metas_list=flat_metas,
            warmup=args.warmup,
            samples=args.samples,
            slow_interval=args.slow_interval,
            device=device,
            fallback_fast_before_first_slow=True,
        )

    print(f"\n{'=' * 72}")
    print(f"Final OPUS StreamingFlow benchmark (warmup={args.warmup}, measured={args.samples})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 72}")
    if "fast_only" in results:
        a = results["fast_only"]
        print(f"  raw-top3 fast-only       | {a['latency_ms_mean']:6.2f} ms | {a['fps']:6.2f} FPS")
    if "fast_streamingflow" in results:
        b = results["fast_streamingflow"]
        print(
            f"  fast + StreamingFlow     | {b['latency_ms_mean']:6.2f} ms | {b['fps']:6.2f} FPS  "
            f"(reset={b['n_reset']}/evolve={b['n_evolve']})"
        )
    if "fast_only" in results and "fast_streamingflow" in results:
        a, b = results["fast_only"], results["fast_streamingflow"]
        d_ms = b["latency_ms_mean"] - a["latency_ms_mean"]
        d_pct = d_ms / a["latency_ms_mean"] * 100
        print(f"  StreamingFlow overhead   | {d_ms:+6.2f} ms | {d_pct:+6.1f}%")

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fast_backend": "opusv1t_raw_top3",
                    "slow_backend": "opusv2l_sparse_full",
                    "aligner": "streamingflow_bev_ode",
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "opus_config": args.opus_config,
                    "opus_checkpoint": args.opus_ckpt,
                    "warmup": args.warmup,
                    "measured": args.samples,
                    "slow_interval_sec": args.slow_interval,
                    "gpu": torch.cuda.get_device_name(0),
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
