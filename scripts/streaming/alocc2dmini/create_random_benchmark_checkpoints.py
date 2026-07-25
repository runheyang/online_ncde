#!/usr/bin/env python3
"""为四组 streaming benchmark 生成可复现的随机初始化权重。"""
from __future__ import annotations

import argparse
import gc
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch

from evoocc.config import load_config_with_base
from evoocc.streaming.aligner_factory import (
    build_evoocc_model_from_config,
)
from evoocc.streaming.baseline_factory import (
    SUPPORTED_BASELINES,
    build_baseline_model_from_config,
    load_config_with_baseline_overlay,
)
from evoocc.utils.checkpoints import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def save_random_model(
    name: str,
    model: torch.nn.Module,
    out_dir: str,
) -> str:
    path = os.path.join(out_dir, f"{name}_random.pth")
    save_checkpoint(
        path,
        model=model,
        extra={"random_init": True, "baseline": name},
    )
    print(f"[saved] {name}: {path}")
    return path


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    evoocc_cfg = load_config_with_base(args.config)
    evoocc = build_evoocc_model_from_config(
        evoocc_cfg,
        solver="euler",
    )
    save_random_model("evoocc", evoocc, args.out_dir)
    del evoocc
    gc.collect()

    for baseline_name in SUPPORTED_BASELINES:
        # 每个模型从同一随机种子开始，便于重复测速。
        torch.manual_seed(args.seed)
        cfg = load_config_with_baseline_overlay(
            args.config,
            baseline_name,
        )
        model = build_baseline_model_from_config(cfg, baseline_name)
        save_random_model(baseline_name, model, args.out_dir)
        del model
        gc.collect()

    with open(
        os.path.join(args.out_dir, "seed.txt"),
        "w",
    ) as file:
        file.write(f"{args.seed}\n")


if __name__ == "__main__":
    main()
