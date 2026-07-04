"""ALOcc2DMini fast + StreamingFlow streaming benchmark."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = "/root/autodl-tmp/online_ncde"
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "src")))

from online_ncde.streaming.benchmark_runtime import configure_benchmark_env

configure_benchmark_env()

import torch

from online_ncde.streaming.benchmark_runtime import configure_torch_benchmark_runtime
from online_ncde.streaming.aligner_factory import resolve_repo_path, resolve_slow_root
from online_ncde.streaming.benchmark_loop import (
    benchmark_stream_aligned,
    make_loader_iter,
    preload_slow_cache,
    select_benchmark_frames,
)
from online_ncde.streaming.benchmark_modes import benchmark_alocc_fast_only
from online_ncde.streaming.alocc2dmini_runtime import (
    DEFAULT_BDV2_PKL,
    DEFAULT_OCCSTUDIO_ROOT,
    build_alocc2dmini_fast_runner,
    resolve_cfg_path,
)
from online_ncde.streaming.scene_iterator import build_sample_meta_index, iter_scenes
from online_ncde.streaming.streamingflow_aligner import (
    StreamingFlowStreamAligner,
    build_streamingflow_model,
)

configure_torch_benchmark_runtime(torch)


STREAMINGFLOW_OVERLAY = (
    Path(REPO_ROOT) / "src" / "online_ncde" / "baselines" / "streamingflow" / "occ3d_config.yaml"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--occ-root", default=DEFAULT_OCCSTUDIO_ROOT)
    p.add_argument("--occ-config", default=None)
    p.add_argument("--occ-ckpt", default=None)
    p.add_argument("--bevdetv2-pkl", default=DEFAULT_BDV2_PKL)
    p.add_argument("--gt-root", default=None)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--slow-interval", type=float, default=4.0)
    p.add_argument("--mode", choices=["fast-only", "fast-streamingflow", "both"], default="both")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def build_fast_runner(args, data_cfg):
    return build_alocc2dmini_fast_runner(
        data_cfg,
        occstudio_root=args.occ_root,
        occ_config=args.occ_config,
        occ_ckpt=args.occ_ckpt,
        device="cuda:0",
    )


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    args.config = resolve_repo_path(args.config, REPO_ROOT)
    args.checkpoint = resolve_repo_path(args.checkpoint, REPO_ROOT)
    args.bevdetv2_pkl = resolve_repo_path(args.bevdetv2_pkl, REPO_ROOT)
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

    print("[2] ALOcc2DMini fast runner build ...")
    fast = build_fast_runner(args, data_cfg)

    slow_root = resolve_slow_root(data_cfg, REPO_ROOT)
    gt_root = resolve_cfg_path(data_cfg, "gt_root", REPO_ROOT, args.gt_root)
    nuscenes_root = resolve_cfg_path(data_cfg, "nuscenes_root", REPO_ROOT)
    slow_format = data_cfg.get("slow_logit_format", data_cfg.get("logits_format", "alocc_dense_topk"))
    print(f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}) ...")
    s2m = build_sample_meta_index(
        args.bevdetv2_pkl,
        slow_root,
        gt_root,
        dataset_variant=data_cfg.get("dataset_variant", "occ3d"),
        nuscenes_root=nuscenes_root,
        nuscenes_version=data_cfg.get("nuscenes_version", "v1.0-trainval"),
    )
    scenes_meta = list(iter_scenes(fast.dataset, s2m, limit_scenes=None))
    _flat, flat_indices, flat_metas = select_benchmark_frames(
        scenes_meta, args.warmup, args.samples
    )

    print("[4] preload slow logits ...")
    slow_cache = preload_slow_cache(data_cfg, device, flat_metas)

    results = {}
    if args.mode in ("fast-only", "both"):
        print("\n=== Mode A: fast-only baseline ===")
        results["fast_only"] = benchmark_alocc_fast_only(
            fast,
            make_loader_iter(fast, flat_indices, args.num_workers, args.prefetch_factor),
            args.warmup,
            args.samples,
        )

    if args.mode in ("fast-streamingflow", "both"):
        print(f"\n=== Mode B: fast + StreamingFlow (slow_interval={args.slow_interval}s) ===")
        results["fast_streamingflow"] = benchmark_stream_aligned(
            name="fast+streamingflow",
            fast=fast,
            stream_aligner=stream_aligner,
            slow_cache=slow_cache,
            raw_batches=make_loader_iter(fast, flat_indices, args.num_workers, args.prefetch_factor),
            metas_list=flat_metas,
            warmup=args.warmup,
            samples=args.samples,
            slow_interval=args.slow_interval,
            device=device,
        )

    print(f"\n{'=' * 72}")
    print(f"Final StreamingFlow benchmark (warmup={args.warmup}, measured={args.samples})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 72}")
    if "fast_only" in results:
        a = results["fast_only"]
        print(f"  fast-only baseline       | {a['latency_ms_mean']:6.2f} ms | {a['fps']:6.2f} FPS")
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
        print(f"  StreamingFlow overhead   | +{d_ms:5.2f} ms | {d_pct:+6.1f}%")

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "fast_backend": "alocc2dmini",
                "aligner": "streamingflow_bev_ode",
                "warmup": args.warmup,
                "measured": args.samples,
                "slow_interval_sec": args.slow_interval,
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
            }, f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
