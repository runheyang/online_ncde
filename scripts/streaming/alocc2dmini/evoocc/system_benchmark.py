"""ALOcc2DMini + sparse ALOcc3D + EvoOcc 系统级吞吐 benchmark。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from evoocc.streaming.benchmark_runtime import configure_benchmark_env

configure_benchmark_env()

import torch

from evoocc.config import load_config_with_base
from evoocc.streaming.aligner_factory import (
    build_evoocc_aligner,
    resolve_repo_path,
    resolve_slow_root,
)
from evoocc.streaming.alocc2dmini_runtime import (
    DEFAULT_BDV2_PKL,
    DEFAULT_OCCSTUDIO_ROOT,
    build_alocc2dmini_fast_runner,
    build_alocc3d_slow_runner,
    resolve_cfg_path,
)
from evoocc.streaming.benchmark_loop import (
    make_loader_iter,
    select_benchmark_frames,
)
from evoocc.streaming.benchmark_runtime import configure_torch_benchmark_runtime
from evoocc.streaming.scene_iterator import build_sample_meta_index, iter_scenes
from evoocc.streaming.stream_aligner import StreamAligner
from evoocc.streaming.system_benchmark_loop import (
    build_dual_system_schedule,
    index_scene_frames_by_token,
    print_system_summary,
    run_alocc_only_system,
    run_fast_ours_system,
)

configure_torch_benchmark_runtime(torch)


def parse_args():
    p = argparse.ArgumentParser(
        description="ALOcc streaming 系统级最大吞吐 benchmark（不做真实时间 sleep）"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--occ-root", default=DEFAULT_OCCSTUDIO_ROOT)
    p.add_argument("--occ-config", default=None)
    p.add_argument("--occ-ckpt", default=None)
    p.add_argument("--slow-occ-config", default=None)
    p.add_argument("--slow-occ-ckpt", default=None)
    p.add_argument("--bevdetv2-pkl", default=DEFAULT_BDV2_PKL)
    p.add_argument("--gt-root", default=None)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--slow-interval", type=float, default=5.0)
    p.add_argument(
        "--mode",
        choices=["fast-only", "slow-only", "fast-ours"],
        default="fast-ours",
    )
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()
    if args.mode == "fast-ours" and not args.checkpoint:
        p.error("--mode fast-ours 需要 --checkpoint")
    if args.samples <= 0:
        p.error("--samples 必须大于 0")
    if args.warmup < 0:
        p.error("--warmup 不能小于 0")
    return args


def build_fast_runner(args, data_cfg):
    return build_alocc2dmini_fast_runner(
        data_cfg,
        occstudio_root=args.occ_root,
        occ_config=args.occ_config,
        occ_ckpt=args.occ_ckpt,
        device="cuda:0",
    )


def build_slow_runner(args, data_cfg):
    return build_alocc3d_slow_runner(
        data_cfg,
        occstudio_root=args.occ_root,
        occ_config=args.slow_occ_config,
        occ_ckpt=args.slow_occ_ckpt,
        device="cuda:0",
    )


def build_scenes(runner, sample_meta_index):
    return list(iter_scenes(runner.dataset, sample_meta_index, limit_scenes=None))


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    args.config = resolve_repo_path(args.config, REPO_ROOT)
    args.checkpoint = resolve_repo_path(args.checkpoint, REPO_ROOT)
    args.occ_root = resolve_repo_path(args.occ_root, REPO_ROOT)
    args.bevdetv2_pkl = resolve_repo_path(args.bevdetv2_pkl, REPO_ROOT)
    args.out_json = resolve_repo_path(args.out_json, REPO_ROOT)

    if not torch.cuda.is_available():
        raise RuntimeError("system benchmark 需要 CUDA")
    device = torch.device("cuda:0")
    data_cfg = load_config_with_base(args.config)["data"]
    slow_root = resolve_slow_root(data_cfg, REPO_ROOT)
    gt_root = resolve_cfg_path(data_cfg, "gt_root", REPO_ROOT, args.gt_root)
    nuscenes_root = resolve_cfg_path(data_cfg, "nuscenes_root", REPO_ROOT)
    slow_format = data_cfg.get(
        "slow_logit_format", data_cfg.get("logits_format", "alocc_dense_topk")
    )
    print(
        f"[meta] dataset_variant={data_cfg.get('dataset_variant', 'occ3d')}, "
        f"slow_format={slow_format}"
    )
    sample_meta_index = build_sample_meta_index(
        args.bevdetv2_pkl,
        slow_root,
        gt_root,
        dataset_variant=data_cfg.get("dataset_variant", "occ3d"),
        nuscenes_root=nuscenes_root,
        nuscenes_version=data_cfg.get("nuscenes_version", "v1.0-trainval"),
    )

    startup_start = time.perf_counter()
    schedule_info = {}

    if args.mode == "fast-only":
        print("[build] ALOcc2DMini fast runner ...")
        fast = build_fast_runner(args, data_cfg)
        scenes_meta = build_scenes(fast, sample_meta_index)
        flat, flat_indices, flat_metas = select_benchmark_frames(
            scenes_meta, args.warmup, args.samples
        )
        startup_s = time.perf_counter() - startup_start
        resident_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        result = run_alocc_only_system(
            name="fast-only-system",
            runner=fast,
            raw_batches=make_loader_iter(
                fast, flat_indices, args.num_workers, args.prefetch_factor
            ),
            metas_list=flat_metas,
            warmup=args.warmup,
            samples=args.samples,
            device=device,
            role="fast",
        )
        schedule_info["total_frames"] = len(flat)

    elif args.mode == "slow-only":
        print("[build] ALOcc3D slow runner ...")
        slow = build_slow_runner(args, data_cfg)
        scenes_meta = build_scenes(slow, sample_meta_index)
        flat, flat_indices, flat_metas = select_benchmark_frames(
            scenes_meta, args.warmup, args.samples
        )
        startup_s = time.perf_counter() - startup_start
        resident_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        result = run_alocc_only_system(
            name="slow-only-system",
            runner=slow,
            raw_batches=make_loader_iter(
                slow, flat_indices, args.num_workers, args.prefetch_factor
            ),
            metas_list=flat_metas,
            warmup=args.warmup,
            samples=args.samples,
            device=device,
            role="slow",
        )
        schedule_info["total_frames"] = len(flat)

    else:
        print("[build] EvoOcc aligner ...")
        aligner, aligner_data_cfg = build_evoocc_aligner(
            args.config, args.checkpoint, device, solver=args.solver
        )
        if aligner_data_cfg != data_cfg:
            raise ValueError("aligner 构建返回的 data_cfg 与配置加载结果不一致")
        stream_aligner = StreamAligner(aligner)

        print("[build] ALOcc2DMini fast runner ...")
        fast = build_fast_runner(args, data_cfg)
        print("[build] ALOcc3D sparse slow runner ...")
        slow = build_slow_runner(args, data_cfg)

        fast_scenes = build_scenes(fast, sample_meta_index)
        slow_scenes = build_scenes(slow, sample_meta_index)
        fast_flat, fast_indices, _fast_metas = select_benchmark_frames(
            fast_scenes, args.warmup, args.samples
        )
        slow_by_token = index_scene_frames_by_token(slow_scenes)
        schedule = build_dual_system_schedule(
            fast_flat,
            slow_by_token,
            slow_interval_sec=args.slow_interval,
        )
        slow_indices = [frame.slow_index for frame in schedule if frame.is_slow]
        startup_s = time.perf_counter() - startup_start
        resident_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        schedule_info = {
            "total_frames": len(schedule),
            "scheduled_slow_total": len(slow_indices),
            "scheduled_slow_measured": sum(
                1 for frame in schedule[args.warmup:] if frame.is_slow
            ),
        }
        print(
            f"[schedule] fast={len(fast_indices)}, slow={len(slow_indices)}, "
            f"slow measured={schedule_info['scheduled_slow_measured']}"
        )
        result = run_fast_ours_system(
            fast=fast,
            slow=slow,
            stream_aligner=stream_aligner,
            fast_batches=make_loader_iter(
                fast, fast_indices, args.num_workers, args.prefetch_factor
            ),
            slow_batches=make_loader_iter(
                slow, slow_indices, args.num_workers, args.prefetch_factor
            ),
            schedule=schedule,
            warmup=args.warmup,
            samples=args.samples,
            device=device,
        )

    print("\n=== System benchmark summary ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  mode={args.mode}, startup={startup_s:.2f}s, resident={resident_memory_mb:.0f}MB")
    print_system_summary(result)

    payload = {
        "benchmark_type": "streaming_system_max_throughput",
        "mode": args.mode,
        "fast_backend": "alocc2dmini",
        "dataset_variant": data_cfg.get("dataset_variant", "occ3d"),
        "warmup": args.warmup,
        "measured": args.samples,
        "slow_interval_sec": args.slow_interval,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "solver": args.solver,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "startup_s": float(startup_s),
        "resident_memory_mb": float(resident_memory_mb),
        "schedule": schedule_info,
        "result": result,
    }
    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
