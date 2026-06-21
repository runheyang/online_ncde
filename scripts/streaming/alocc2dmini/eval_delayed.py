"""ALOcc2DMini fast + OnlineNCDE delayed streaming eval."""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = "/root/autodl-tmp/online_ncde"
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "src")))

import torch

from online_ncde.streaming.aligner_factory import (
    build_online_ncde_aligner,
    resolve_repo_path,
    resolve_slow_root,
)
from online_ncde.streaming.eval_loop import run_streaming_eval
from online_ncde.streaming.fast_runner import FastRunner
from online_ncde.streaming.scene_iterator import build_sample_meta_index, iter_scenes
from online_ncde.streaming.slow_cache import build_slow_decoder_fn
from online_ncde.streaming.stream_aligner import StreamAligner


OCC_ROOT = "/root/autodl-tmp/online_ncde/third_party/OccStudio"
OCC_CONFIG = "configs/alocc/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.py"
OCC_CKPT = "ckpts/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.pth"
BDV2_PKL = "/root/autodl-tmp/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl"
GT_ROOT = "/root/autodl-tmp/data/nuscenes/gts"
DEFAULT_SWEEP_PKL = "/root/autodl-tmp/data/nuscenes/nuscenes_infos_val_sweep.pkl"


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
    p.add_argument("--sweep-pkl", default=DEFAULT_SWEEP_PKL)
    p.add_argument("--no-rayiou", action="store_true")
    p.add_argument("--out-json", default=None)
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
    args.out_json = resolve_repo_path(args.out_json, REPO_ROOT)
    device = torch.device("cuda:0")

    print("[1] aligner build & load ckpt ...")
    aligner, data_cfg = build_online_ncde_aligner(
        args.aligner_cfg, args.aligner_ckpt, device, solver=args.solver
    )
    stream_aligner = StreamAligner(aligner)

    print("[2] ALOcc2DMini fast runner build ...")
    fast = build_fast_runner(data_cfg)

    slow_root = resolve_slow_root(data_cfg, REPO_ROOT)
    slow_format = data_cfg.get("slow_logit_format", data_cfg.get("logits_format", "alocc_dense_topk"))
    print(
        f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}, "
        f"slow_delay_kf={args.slow_delay_keyframes}) ..."
    )
    s2m = build_sample_meta_index(BDV2_PKL, slow_root, GT_ROOT)
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
