"""ALOcc2DMini fast + EvoOcc delayed streaming eval."""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = "/root/autodl-tmp/online_ncde"
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "src")))

import torch

from evoocc.streaming.aligner_factory import (
    build_evoocc_aligner,
    resolve_repo_path,
    resolve_slow_root,
)
from evoocc.streaming.eval_loop import run_streaming_eval
from evoocc.streaming.alocc2dmini_runtime import (
    DEFAULT_BDV2_PKL,
    DEFAULT_OCCSTUDIO_ROOT,
    build_alocc2dmini_fast_runner,
    resolve_cfg_path,
)
from evoocc.streaming.scene_iterator import build_sample_meta_index, iter_scenes
from evoocc.streaming.slow_cache import build_slow_decoder_fn
from evoocc.streaming.stream_aligner import StreamAligner


DEFAULT_SWEEP_PKL = "/root/autodl-tmp/data/nuscenes/nuscenes_infos_val_sweep.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aligner-cfg", required=True)
    p.add_argument("--aligner-ckpt", required=True)
    p.add_argument("--occ-root", default=DEFAULT_OCCSTUDIO_ROOT)
    p.add_argument("--occ-config", default=None)
    p.add_argument("--occ-ckpt", default=None)
    p.add_argument("--bevdetv2-pkl", default=DEFAULT_BDV2_PKL)
    p.add_argument("--gt-root", default=None)
    p.add_argument("--slow-intervals", type=float, nargs="+", required=True)
    p.add_argument("--slow-delay-keyframes", type=int, default=2)
    p.add_argument("--limit-scenes", type=int, default=None)
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--preload-slow", action="store_true")
    p.add_argument("--sweep-pkl", default=DEFAULT_SWEEP_PKL)
    p.add_argument("--no-rayiou", action="store_true")
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
    args.bevdetv2_pkl = resolve_repo_path(args.bevdetv2_pkl, REPO_ROOT)
    args.sweep_pkl = resolve_repo_path(args.sweep_pkl, REPO_ROOT)
    args.out_json = resolve_repo_path(args.out_json, REPO_ROOT)
    device = torch.device("cuda:0")

    print("[1] aligner build & load ckpt ...")
    aligner, data_cfg = build_evoocc_aligner(
        args.aligner_cfg, args.aligner_ckpt, device, solver=args.solver
    )
    stream_aligner = StreamAligner(aligner)

    print("[2] ALOcc2DMini fast runner build ...")
    fast = build_fast_runner(args, data_cfg)

    slow_root = resolve_slow_root(data_cfg, REPO_ROOT)
    gt_root = resolve_cfg_path(data_cfg, "gt_root", REPO_ROOT, args.gt_root)
    nuscenes_root = resolve_cfg_path(data_cfg, "nuscenes_root", REPO_ROOT)
    slow_format = data_cfg.get("slow_logit_format", data_cfg.get("logits_format", "alocc_dense_topk"))
    print(
        f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}, "
        f"slow_delay_kf={args.slow_delay_keyframes}) ..."
    )
    s2m = build_sample_meta_index(
        args.bevdetv2_pkl,
        slow_root,
        gt_root,
        dataset_variant=data_cfg.get("dataset_variant", "occ3d"),
        nuscenes_root=nuscenes_root,
        nuscenes_version=data_cfg.get("nuscenes_version", "v1.0-trainval"),
    )
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
        fast_backend="alocc2dmini",
        delayed=True,
        slow_delay_keyframes=args.slow_delay_keyframes,
    )


if __name__ == "__main__":
    main()
