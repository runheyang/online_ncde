"""OPUSv1-T + OPUSv2-L + EvoOcc 系统级吞吐 benchmark。"""
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
from evoocc.streaming.benchmark_loop import make_loader_iter, select_benchmark_frames
from evoocc.streaming.benchmark_runtime import configure_torch_benchmark_runtime
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
from evoocc.streaming.system_benchmark_loop import (
    attach_system_flops,
    build_dual_system_schedule,
    index_scene_frames_by_token,
    print_system_summary,
    run_alocc_only_system as run_opus_only_system,
    run_fast_ours_system,
)

configure_torch_benchmark_runtime(torch)


def parse_args():
    p = argparse.ArgumentParser(
        description="OPUS streaming 系统级最大吞吐 benchmark（不做真实时间 sleep）"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--opus-root", default=DEFAULT_OPUS_ROOT)
    p.add_argument("--opus-config", default=DEFAULT_OPUSV1_CONFIG)
    p.add_argument("--opus-ckpt", default=DEFAULT_OPUSV1_CKPT)
    p.add_argument("--slow-opus-config", default=DEFAULT_OPUSV2_CONFIG)
    p.add_argument("--slow-opus-ckpt", default=DEFAULT_OPUSV2_CKPT)
    p.add_argument("--meta-pkl", default=DEFAULT_META_PKL)
    p.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
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
    p.add_argument(
        "--fast-gflops",
        type=float,
        default=None,
        help="预先测得的单次 fast forward GFLOPs",
    )
    p.add_argument(
        "--slow-gflops",
        type=float,
        default=None,
        help="预先测得的单次 slow forward GFLOPs",
    )
    p.add_argument(
        "--evolve-gflops",
        type=float,
        default=None,
        help="预先测得的单次 EvoOcc evolve GFLOPs",
    )
    p.add_argument(
        "--reset-gflops",
        type=float,
        default=None,
        help="预先测得的单次 EvoOcc reset GFLOPs",
    )
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


def build_scenes(runner, sample_meta_index):
    return list(iter_scenes(runner.dataset, sample_meta_index, limit_scenes=None))


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

    if not torch.cuda.is_available():
        raise RuntimeError("system benchmark 需要 CUDA")
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

    startup_start = time.perf_counter()
    schedule_info = {}

    if args.mode == "fast-only":
        print("[build] OPUSv1-T fast runner ...")
        fast = build_fast_runner(args, data_cfg)
        scenes_meta = build_scenes(fast, sample_meta_index)
        flat, flat_indices, flat_metas = select_benchmark_frames(
            scenes_meta, args.warmup, args.samples
        )
        startup_s = time.perf_counter() - startup_start
        resident_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        result = run_opus_only_system(
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
        print("[build] OPUSv2-L slow runner ...")
        slow = build_slow_runner(args, data_cfg)
        scenes_meta = build_scenes(slow, sample_meta_index)
        flat, flat_indices, flat_metas = select_benchmark_frames(
            scenes_meta, args.warmup, args.samples
        )
        startup_s = time.perf_counter() - startup_start
        resident_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        result = run_opus_only_system(
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

        print("[build] OPUSv1-T fast runner ...")
        fast = build_fast_runner(args, data_cfg)
        print("[build] OPUSv2-L slow runner ...")
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

    attach_system_flops(
        result,
        mode=args.mode,
        fast_gflops=args.fast_gflops,
        slow_gflops=args.slow_gflops,
        evolve_gflops=args.evolve_gflops,
        reset_gflops=args.reset_gflops,
    )
    if args.mode == "fast-only":
        schedule_label = "Fast@2Hz"
    elif args.mode == "slow-only":
        schedule_label = "Slow@2Hz"
    elif args.slow_interval > 0:
        schedule_label = f"Fast@2Hz + Slow@{1.0 / args.slow_interval:g}Hz"
    elif args.slow_interval == 0:
        schedule_label = "Fast@2Hz + Slow@2Hz"
    else:
        schedule_label = "Fast@2Hz + Slow@scene-start"

    print("\n=== System benchmark summary ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"  mode={args.mode}, schedule={schedule_label}, startup={startup_s:.2f}s, "
        f"resident={resident_memory_mb:.0f}MB"
    )
    print_system_summary(result)

    payload = {
        "benchmark_type": "streaming_system_max_throughput",
        "mode": args.mode,
        "schedule_label": schedule_label,
        "fast_backend": "opusv1t_raw_top3",
        "slow_backend": "opusv2l_raw_top3",
        "fast_config": args.opus_config,
        "fast_checkpoint": args.opus_ckpt,
        "slow_config": args.slow_opus_config,
        "slow_checkpoint": args.slow_opus_ckpt,
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
