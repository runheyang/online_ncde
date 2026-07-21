"""OPUSv1-T fast + EvoOcc delayed streaming eval."""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch

from evoocc.streaming.aligner_factory import (
    build_evoocc_aligner,
    resolve_repo_path,
    resolve_slow_root,
)
from evoocc.streaming.eval_loop import run_streaming_eval
from evoocc.streaming.opus_runtime import (
    DEFAULT_GT_ROOT,
    DEFAULT_META_PKL,
    DEFAULT_OPUS_ROOT,
    DEFAULT_OPUSV1_CKPT,
    DEFAULT_OPUSV1_CONFIG,
    DEFAULT_SWEEP_PKL,
    build_opus_runner,
    resolve_opus_path,
)
from evoocc.streaming.scene_iterator import build_opus_sample_meta_index, iter_scenes
from evoocc.streaming.slow_cache import build_slow_decoder_fn
from evoocc.streaming.stream_aligner import StreamAligner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aligner-cfg", required=True)
    p.add_argument("--aligner-ckpt", required=True)
    p.add_argument("--slow-intervals", type=float, nargs="+", required=True)
    p.add_argument("--slow-delay-keyframes", type=int, default=2)
    p.add_argument("--limit-scenes", type=int, default=None)
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--preload-slow", action="store_true")
    p.add_argument("--opus-root", default=DEFAULT_OPUS_ROOT)
    p.add_argument("--opus-config", default=DEFAULT_OPUSV1_CONFIG)
    p.add_argument("--opus-ckpt", default=DEFAULT_OPUSV1_CKPT)
    p.add_argument("--meta-pkl", default=DEFAULT_META_PKL)
    p.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    p.add_argument("--sweep-pkl", default=DEFAULT_SWEEP_PKL)
    p.add_argument("--no-rayiou", action="store_true")
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
    args.aligner_cfg = resolve_repo_path(args.aligner_cfg, REPO_ROOT)
    args.aligner_ckpt = resolve_repo_path(args.aligner_ckpt, REPO_ROOT)
    args.opus_root = resolve_repo_path(args.opus_root, REPO_ROOT)
    args.opus_config = resolve_opus_path(args.opus_config, args.opus_root, REPO_ROOT)
    args.opus_ckpt = resolve_opus_path(args.opus_ckpt, args.opus_root, REPO_ROOT)
    args.meta_pkl = resolve_repo_path(args.meta_pkl, REPO_ROOT)
    args.gt_root = resolve_repo_path(args.gt_root, REPO_ROOT)
    args.sweep_pkl = resolve_repo_path(args.sweep_pkl, REPO_ROOT)
    args.out_json = resolve_repo_path(args.out_json, REPO_ROOT)
    device = torch.device("cuda:0")

    print("[1] aligner build & load ckpt ...")
    aligner, data_cfg = build_evoocc_aligner(
        args.aligner_cfg, args.aligner_ckpt, device, solver=args.solver
    )
    stream_aligner = StreamAligner(aligner)

    print("[2] OPUSv1-T fast runner build ...")
    fast = build_fast_runner(args, data_cfg)

    slow_root = resolve_slow_root(data_cfg, REPO_ROOT)
    slow_format = data_cfg.get(
        "slow_logit_format", data_cfg.get("logits_format", "opus_sparse_full")
    )
    print(
        f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}, "
        f"slow_delay_kf={args.slow_delay_keyframes}) ..."
    )
    s2m = build_opus_sample_meta_index(args.meta_pkl, slow_root, args.gt_root)
    scenes_meta = list(iter_scenes(fast.dataset, s2m, limit_scenes=args.limit_scenes))
    slow_decoder_fn = build_slow_decoder_fn(data_cfg, device)

    run_streaming_eval(
        fast=fast,
        scenes_meta=scenes_meta,
        data_cfg=data_cfg,
        stream_aligner=stream_aligner,
        slow_decoder_fn=slow_decoder_fn,
        slow_intervals=args.slow_intervals,
        device=device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preload_slow=args.preload_slow,
        no_rayiou=args.no_rayiou,
        sweep_pkl=args.sweep_pkl,
        out_json=args.out_json,
        fast_backend="opusv1t_raw_top3",
        delayed=True,
        slow_delay_keyframes=args.slow_delay_keyframes,
        extra_out={"opus_config": args.opus_config},
    )


if __name__ == "__main__":
    main()
