"""ALOcc2DMini no-warp attention streaming benchmark."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

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
from online_ncde.streaming.fast_runner import FastRunner
from online_ncde.streaming.no_warp_attn import NoWarpAttnStreamAligner, build_no_warp_aligner
from online_ncde.streaming.scene_iterator import build_sample_meta_index, iter_scenes

configure_torch_benchmark_runtime(torch)


OCC_ROOT = "/root/autodl-tmp/online_ncde/third_party/OccStudio"
OCC_CONFIG = "configs/alocc/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.py"
OCC_CKPT = "ckpts/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.pth"
BDV2_PKL = "/root/autodl-tmp/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl"
GT_ROOT = "/root/autodl-tmp/data/nuscenes/gts"
DEFAULT_ALIGNER_CFG = "configs/online_ncde/fast_alocc2dmini__slow_alocc3d/base.yaml"
DEFAULT_ALIGNER_CKPT = (
    "ckpts/fast_alocc2dmini__slow_alocc3d/"
    "no_warp_attn_20260503_112331/epoch_9.pth"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aligner-cfg", default=DEFAULT_ALIGNER_CFG)
    p.add_argument("--aligner-ckpt", default=DEFAULT_ALIGNER_CKPT)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--slow-interval", type=float, default=4.0)
    p.add_argument("--mode", choices=["fast-only", "fast-nowarp", "both"], default="both")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--out-json", default=None)
    p.add_argument("--use-fast-residual", dest="use_fast_residual", action="store_true", default=False)
    p.add_argument("--no-use-fast-residual", dest="use_fast_residual", action="store_false")
    return p.parse_args()


def build_fast_runner(data_cfg):
    fast = FastRunner(
        occstudio_root=OCC_ROOT,
        config_path=OCC_CONFIG,
        ckpt_path=OCC_CKPT,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        topk_k=int(data_cfg.get("alocc_topk_k", 3)),
        clamp_min=float(data_cfg.get("alocc_clamp_min", -5.0)),
        fill_value=float(data_cfg.get("alocc_fill_value", -5.0)),
        max_centering=bool(data_cfg.get("alocc_max_centering", False)),
        device="cuda:0",
    )
    fast.build()
    return fast


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    args.aligner_cfg = resolve_repo_path(args.aligner_cfg, REPO_ROOT)
    args.aligner_ckpt = resolve_repo_path(args.aligner_ckpt, REPO_ROOT)
    args.out_json = resolve_repo_path(args.out_json, REPO_ROOT)
    device = torch.device("cuda:0")

    print("[1] no-warp attn aligner build & load ckpt ...")
    aligner, data_cfg = build_no_warp_aligner(
        args.aligner_cfg,
        args.aligner_ckpt,
        device,
        use_fast_residual=args.use_fast_residual,
    )
    stream_aligner = NoWarpAttnStreamAligner(aligner)
    print(f"  ckpt={args.aligner_ckpt}")
    print(f"  fast_residual={args.use_fast_residual}")

    print("[2] ALOcc2DMini fast runner build ...")
    fast = build_fast_runner(data_cfg)

    slow_root = resolve_slow_root(data_cfg, REPO_ROOT)
    slow_format = data_cfg.get("slow_logit_format", data_cfg.get("logits_format", "alocc_dense_topk"))
    print(f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}) ...")
    s2m = build_sample_meta_index(BDV2_PKL, slow_root, GT_ROOT)
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

    if args.mode in ("fast-nowarp", "both"):
        print(f"\n=== Mode B: fast + no-warp-attn (slow_interval={args.slow_interval}s) ===")
        results["fast_no_warp_attn"] = benchmark_stream_aligned(
            name="fast+no-warp-attn",
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
    print(f"Final no-warp-attn benchmark (warmup={args.warmup}, measured={args.samples})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 72}")
    if "fast_only" in results:
        a = results["fast_only"]
        print(f"  fast-only baseline       | {a['latency_ms_mean']:6.2f} ms | {a['fps']:6.2f} FPS")
    if "fast_no_warp_attn" in results:
        b = results["fast_no_warp_attn"]
        print(
            f"  fast + no-warp-attn      | {b['latency_ms_mean']:6.2f} ms | {b['fps']:6.2f} FPS  "
            f"(reset={b['n_reset']}/evolve={b['n_evolve']})"
        )
    if "fast_only" in results and "fast_no_warp_attn" in results:
        a, b = results["fast_only"], results["fast_no_warp_attn"]
        d_ms = b["latency_ms_mean"] - a["latency_ms_mean"]
        d_pct = d_ms / a["latency_ms_mean"] * 100
        print(f"  no-warp aligner overhead | +{d_ms:5.2f} ms | {d_pct:+6.1f}%")

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "fast_backend": "alocc2dmini",
                "aligner": "no_warp_attn",
                "aligner_cfg": args.aligner_cfg,
                "aligner_ckpt": args.aligner_ckpt,
                "use_fast_residual": args.use_fast_residual,
                "warmup": args.warmup,
                "measured": args.samples,
                "slow_interval_sec": args.slow_interval,
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
            }, f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
