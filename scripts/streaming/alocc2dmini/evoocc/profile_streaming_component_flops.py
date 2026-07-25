"""统计 streaming system benchmark 四个组件的单次计算量。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from evoocc.streaming.benchmark_runtime import configure_benchmark_env


configure_benchmark_env()

import torch
import torch.nn as nn

from evoocc.config import load_config_with_base
from evoocc.streaming.aligner_factory import build_evoocc_aligner, resolve_repo_path
from evoocc.streaming.alocc2dmini_runtime import (
    DEFAULT_OCCSTUDIO_ROOT,
    build_alocc2dmini_fast_runner,
    build_alocc3d_slow_runner,
)
from evoocc.streaming.benchmark_runtime import configure_torch_benchmark_runtime
from evoocc.streaming.stream_aligner import StreamAligner


configure_torch_benchmark_runtime(torch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用真实 forward 统计 ALOcc2DMini、ALOcc3D 和 EvoOcc 单次 FLOPs"
    )
    parser.add_argument(
        "--config",
        default="configs/evoocc/fast_alocc2dmini__slow_alocc3d.yaml",
    )
    parser.add_argument("--checkpoint", default="ckpts/epoch_9.pth")
    parser.add_argument("--occ-root", default=DEFAULT_OCCSTUDIO_ROOT)
    parser.add_argument("--occ-config", default=None)
    parser.add_argument("--occ-ckpt", default=None)
    parser.add_argument("--slow-occ-config", default=None)
    parser.add_argument("--slow-occ-ckpt", default=None)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument(
        "--component",
        choices=["all", "alocc2dmini", "alocc3d", "evoocc"],
        default="all",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def _start_counter(model: nn.Module) -> None:
    from mmcv.cnn.utils.flops_counter import add_flops_counting_methods

    add_flops_counting_methods(model)
    model.reset_flops_count()
    model.start_flops_count()


def _collect_counter(model: nn.Module) -> Dict[str, Any]:
    from mmcv.cnn.utils.flops_counter import is_supported_instance

    by_type: Dict[str, float] = {}
    total = 0.0
    for module in model.modules():
        if not is_supported_instance(module):
            continue
        value = float(getattr(module, "__flops__", 0.0))
        total += value
        name = type(module).__name__
        by_type[name] = by_type.get(name, 0.0) + value
    model.stop_flops_count()
    return {
        "counter": "mmcv_module_hooks",
        "convention": "one_multiply_accumulate_is_one_flop",
        "flops": total,
        "gflops": total / 1.0e9,
        "by_module_type_gflops": {
            key: value / 1.0e9
            for key, value in sorted(by_type.items())
            if value > 0
        },
    }


def _profile_runner(runner, sample_index: int) -> Dict[str, Any]:
    if runner.model is None:
        raise RuntimeError("runner 尚未 build")
    runner.reset_history()
    raw = runner.dataset[sample_index]
    batch = runner.prepare_batch(raw)
    _start_counter(runner.model)
    with torch.inference_mode():
        output = runner.forward_keyframe(batch)
        torch.cuda.synchronize()
    result = _collect_counter(runner.model)
    result["output_shape"] = list(output.shape)
    del output, batch, raw
    return result


def _profile_evoocc(
    config_path: str,
    checkpoint_path: str,
    solver: str,
    device: torch.device,
) -> Dict[str, Any]:
    aligner, data_cfg = build_evoocc_aligner(
        config_path,
        checkpoint_path,
        device,
        solver=solver,
    )
    stream = StreamAligner(aligner)
    classes = int(data_cfg["num_classes"])
    grid = tuple(int(v) for v in data_cfg["grid_size"])
    fast_logits = torch.zeros((classes, *grid), device=device)
    slow_logits = torch.zeros_like(fast_logits)
    pose = torch.eye(4, device=device)

    stream.reset_scene()
    _start_counter(aligner)
    with torch.inference_mode():
        stream.reset_with_slow(fast_logits, slow_logits, pose, 0)
        torch.cuda.synchronize()
    reset = _collect_counter(aligner)

    _start_counter(aligner)
    with torch.inference_mode():
        output = stream.evolve(fast_logits, pose, 500_000)
        torch.cuda.synchronize()
    evolve = _collect_counter(aligner)

    # 两次三线性 warp 均为函数式 grid_sample，不会被模块 hook 统计。
    warped_values = 2 * int(aligner.hidden_dim) * int(grid[0] * grid[1] * grid[2])
    evolve["untracked_warp"] = {
        "num_trilinear_outputs": warped_values,
        "mac_convention_gflops_8_taps": warped_values * 8 / 1.0e9,
        "arithmetic_gflops_8_mul_7_add": warped_values * 15 / 1.0e9,
        "note": "不含 sampling-grid 构造；不并入 module-hook GFLOPs",
    }
    reset["output_shape"] = list(slow_logits.shape)
    evolve["output_shape"] = list(output.shape)
    return {"reset": reset, "evolve": evolve}


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    args.config = resolve_repo_path(args.config, REPO_ROOT)
    args.checkpoint = resolve_repo_path(args.checkpoint, REPO_ROOT)
    args.occ_root = resolve_repo_path(args.occ_root, REPO_ROOT)
    args.out_json = resolve_repo_path(args.out_json, REPO_ROOT)

    if not torch.cuda.is_available():
        raise RuntimeError("FLOPs profiling 需要 CUDA 执行真实 ALOcc forward")
    device = torch.device("cuda:0")
    data_cfg = load_config_with_base(args.config)["data"]
    results: Dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "solver": args.solver,
    }

    if args.component in ("all", "alocc2dmini"):
        print("[profile] ALOcc2DMini fast")
        fast = build_alocc2dmini_fast_runner(
            data_cfg,
            occstudio_root=args.occ_root,
            occ_config=args.occ_config,
            occ_ckpt=args.occ_ckpt,
            device="cuda:0",
        )
        results["alocc2dmini_fast"] = _profile_runner(fast, args.sample_index)
        del fast
        torch.cuda.empty_cache()
        os.chdir(REPO_ROOT)

    if args.component in ("all", "alocc3d"):
        print("[profile] ALOcc3D slow")
        slow = build_alocc3d_slow_runner(
            data_cfg,
            occstudio_root=args.occ_root,
            occ_config=args.slow_occ_config,
            occ_ckpt=args.slow_occ_ckpt,
            device="cuda:0",
        )
        results["alocc3d_slow"] = _profile_runner(slow, args.sample_index)
        del slow
        torch.cuda.empty_cache()
        os.chdir(REPO_ROOT)

    if args.component in ("all", "evoocc"):
        print("[profile] EvoOcc reset/evolve")
        results["evoocc"] = _profile_evoocc(
            args.config,
            args.checkpoint,
            args.solver,
            device,
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
