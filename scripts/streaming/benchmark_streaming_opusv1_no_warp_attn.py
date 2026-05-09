"""Streaming benchmark for OPUSv1-T fast system + no-warp attention baseline.

基于 scripts/streaming/benchmark_streaming_opusv1.py，计时协议保持一致：
  - warmup K iter 不计时
  - measured N iter 计时
  - 计时区间含 scatter_to_device + OPUS forward_keyframe + stream align + argmax/cpu

默认权重：
  ckpts/fast_opusv1t__slow_opusv2l/no_warp_attn_20260504_100601/epoch_6.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT_LOCAL = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT_LOCAL, "src"))
sys.path.insert(0, SCRIPT_DIR)

import benchmark_streaming_opusv1 as upstream  # noqa: E402
from online_ncde.baselines import NoWarpMotionBiasAttnAligner  # noqa: E402
from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.data.ego_warp_list import compute_transform_prev_to_curr  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402

DEFAULT_ALIGNER_CFG = "configs/online_ncde/fast_opusv1t__slow_opusv2l/base.yaml"
DEFAULT_ALIGNER_CKPT = (
    "ckpts/fast_opusv1t__slow_opusv2l/"
    "no_warp_attn_20260504_100601/epoch_6.pth"
)


class NoWarpAttnStreamAligner:
    """把 NoWarpMotionBiasAttnAligner 拆成 OPUSv1T 逐 keyframe 调用接口。"""

    def __init__(self, base_aligner: NoWarpMotionBiasAttnAligner) -> None:
        self.m = base_aligner
        self.m.eval()
        self.hidden: Optional[torch.Tensor] = None
        self.prev_ego: Optional[torch.Tensor] = None
        self.prev_t_us: Optional[int] = None
        self._spatial_shape = None
        self._ts_scale = float(self.m.timestamp_scale)

    def reset_scene(self) -> None:
        self.hidden = None
        self.prev_ego = None
        self.prev_t_us = None
        self._spatial_shape = None

    @torch.no_grad()
    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        return self.m._encode_fast(fast_logits.unsqueeze(0))[0]

    @torch.no_grad()
    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        return self.m._encode_slow(slow_logits)

    @torch.no_grad()
    def reset_with_slow(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        fast_feat = self._encode_fast(fast_logits)
        self.hidden = self._encode_slow(slow_logits)
        self.prev_ego = ego2global
        self.prev_t_us = int(t_us)
        self._spatial_shape = (
            int(fast_feat.shape[1]),
            int(fast_feat.shape[2]),
            int(fast_feat.shape[3]),
        )
        return slow_logits

    @torch.no_grad()
    def evolve(
        self,
        fast_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        if self.hidden is None or self.prev_ego is None or self.prev_t_us is None:
            raise RuntimeError("evolve called before reset_with_slow; first keyframe must be a slow injection")

        fast_feat = self._encode_fast(fast_logits)
        if self._spatial_shape is None:
            self._spatial_shape = (
                int(fast_feat.shape[1]),
                int(fast_feat.shape[2]),
                int(fast_feat.shape[3]),
            )

        transform = compute_transform_prev_to_curr(
            pose_prev_ego2global=self.prev_ego,
            pose_curr_ego2global=ego2global,
        )
        dt = torch.tensor(
            float(t_us - self.prev_t_us) * self._ts_scale,
            device=fast_feat.device,
            dtype=fast_feat.dtype,
        )
        h_new = self.m._advance_no_warp(
            h_dense=self.hidden,
            fast_curr=fast_feat,
            dt_value=dt,
            transform_prev_to_curr=transform,
            spatial_shape_xyz=self._spatial_shape,
        )
        logits_delta = self.m._decode_dense_state(h_new)
        aligned = logits_delta + fast_logits if self.m.use_fast_residual else logits_delta

        self.hidden = h_new
        self.prev_ego = ego2global
        self.prev_t_us = int(t_us)
        return aligned


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aligner-cfg", default=DEFAULT_ALIGNER_CFG)
    p.add_argument("--aligner-ckpt", default=DEFAULT_ALIGNER_CKPT)
    p.add_argument("--samples", type=int, default=200, help="measured iterations")
    p.add_argument("--warmup", type=int, default=20, help="warmup iterations")
    p.add_argument("--slow-interval", type=float, default=4.0)
    p.add_argument(
        "--mode",
        choices=["native-only", "fast-only", "fast-nowarp", "both", "all"],
        default="both",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--opus-root", default=upstream.OPUS_ROOT)
    p.add_argument("--opus-config", default=upstream.OPUS_CONFIG)
    p.add_argument("--opus-ckpt", default=upstream.OPUS_CKPT)
    p.add_argument("--meta-pkl", default=upstream.META_PKL)
    p.add_argument("--gt-root", default=upstream.GT_ROOT)
    p.add_argument("--out-json", default=None)
    p.add_argument(
        "--use-fast-residual",
        dest="use_fast_residual",
        action="store_true",
        default=False,
        help="开启 fast residual；默认关闭，与 no_warp_attn 训练入口对齐",
    )
    p.add_argument(
        "--no-use-fast-residual",
        dest="use_fast_residual",
        action="store_false",
        help="关闭 fast residual；默认值",
    )
    return p.parse_args()


def build_no_warp_aligner(args, device):
    cfg = load_config_with_base(args.aligner_cfg)
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    inner_dim = int(model_cfg.get("no_warp_inner_dim", model_cfg.get("func_g_inner_dim", 24)))
    num_heads = int(model_cfg.get("no_warp_attn_num_heads", 3))
    if inner_dim % num_heads != 0:
        raise ValueError(
            f"no-warp attention inner_dim={inner_dim} 必须能被 num_heads={num_heads} 整除"
        )

    decoder_init_scale = model_cfg.get("decoder_init_scale", 1.0e-3) if args.use_fast_residual else None
    aligner = NoWarpMotionBiasAttnAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=decoder_init_scale,
        use_fast_residual=args.use_fast_residual,
        fusion_inner_dim=inner_dim,
        fusion_attn_num_heads=num_heads,
        fusion_attn_window_size=tuple(model_cfg.get("no_warp_attn_window_size", [8, 8, 4])),
        fusion_attn_head_dilations=tuple(model_cfg.get("no_warp_attn_head_dilations", [1, 2])),
        fusion_gn_groups=int(model_cfg.get("no_warp_gn_groups", model_cfg.get("func_g_gn_groups", 8))),
        fusion_attn_mlp_ratio=float(model_cfg.get("no_warp_attn_mlp_ratio", 2.0)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
    ).to(device)
    load_checkpoint_for_eval(args.aligner_ckpt, model=aligner, strict=False)
    aligner.eval()
    return aligner, data_cfg


def benchmark_fast_no_warp(
    fast,
    stream_aligner,
    slow_cache,
    raw_batches,
    metas_list,
    K,
    N,
    slow_interval,
    device,
    log_interval=50,
):
    """Mode B: OPUS raw-top3 fast logits -> no-warp attn -> argmax+cpu."""
    pure_inf_time = 0.0
    per_iter = []
    n_reset, n_evolve = 0, 0
    cur_scene = None
    last_slow_t_sec = -1e9

    fast.reset_history()
    stream_aligner.reset_scene()

    for i, (raw, meta) in enumerate(zip(raw_batches, metas_list)):
        if i >= K + N:
            break

        if meta.scene_name != cur_scene:
            stream_aligner.reset_scene()
            cur_scene = meta.scene_name
            last_slow_t_sec = -1e9

        t_sec = meta.timestamp_us / 1e6
        is_slow = (last_slow_t_sec < 0) or (t_sec - last_slow_t_sec + 1e-3 >= slow_interval)
        ego_t = torch.from_numpy(meta.ego2global).to(device=device, dtype=torch.float32)
        slow_logits = slow_cache.get(meta.slow_logit_path) if (
            is_slow and slow_cache.has(meta.slow_logit_path)
        ) else None

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            batch = upstream.scatter_to_device(raw, 0)
            fast_logits = fast.forward_keyframe(batch)
            if slow_logits is not None:
                aligned = stream_aligner.reset_with_slow(
                    fast_logits, slow_logits, ego_t, meta.timestamp_us
                )
            elif stream_aligner.hidden is None:
                aligned = fast_logits
            else:
                aligned = stream_aligner.evolve(fast_logits, ego_t, meta.timestamp_us)
            pred_uint8 = aligned.argmax(0).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i >= K:
            pure_inf_time += elapsed
            per_iter.append(elapsed * 1000)
            if slow_logits is not None:
                n_reset += 1
            else:
                n_evolve += 1
            done = i + 1 - K
            if done % log_interval == 0 or done == N:
                fps = done / pure_inf_time
                print(
                    f"    [{done:>3}/{N}] FPS: {fps:6.2f}  latency: {1000/fps:6.2f} ms  "
                    f"(reset={n_reset} evolve={n_evolve})"
                )

        if slow_logits is not None:
            last_slow_t_sec = t_sec

    out = upstream._summarize_latency("fast+no-warp-attn", per_iter, pure_inf_time, K, N)
    out.update({
        "n_reset": int(n_reset),
        "n_evolve": int(n_evolve),
        "slow_interval_sec": float(slow_interval),
    })
    print(f"     reset/evolve in measured: {n_reset}/{n_evolve}")
    return out


def _wants(mode: str, name: str) -> bool:
    if mode == "all":
        return True
    if mode == "both" and name in {"fast-only", "fast-nowarp"}:
        return True
    return mode == name


def main():
    args = parse_args()
    os.chdir(upstream.REPO_ROOT)
    args.aligner_cfg = upstream._resolve_repo_path(args.aligner_cfg)
    args.aligner_ckpt = upstream._resolve_repo_path(args.aligner_ckpt)
    args.out_json = upstream._resolve_repo_path(args.out_json)
    device = torch.device("cuda:0")

    print("[1] no-warp attn aligner build & load ckpt ...")
    aligner, data_cfg = build_no_warp_aligner(args, device)
    stream_aligner = NoWarpAttnStreamAligner(aligner)
    print(f"  ckpt={args.aligner_ckpt}")
    print(f"  fast_residual={args.use_fast_residual}")

    print("[2] OPUSv1-T fast runner build ...")
    fast = upstream.OpusV1FastRunner(
        opus_root=args.opus_root,
        config_path=args.opus_config,
        ckpt_path=args.opus_ckpt,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        other_fill_value=float(data_cfg.get("opus_other_fill_value", -5.0)),
        free_fill_value=float(data_cfg.get("opus_free_fill_value", 5.0)),
        topk_k=int(data_cfg.get("opus_full_topk_k", 3)),
        clamp_min=float(data_cfg.get("opus_clamp_min", -5.0)),
        device="cuda:0",
    )
    fast.build()

    slow_root = upstream.resolve_slow_root(data_cfg)
    slow_format = data_cfg.get("slow_logit_format", data_cfg.get("logits_format", "opus_sparse_full"))
    print(f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}) ...")
    s2m = upstream.build_opus_sample_meta_index(args.meta_pkl, slow_root, args.gt_root)
    scenes_meta = list(upstream.iter_scenes(fast.dataset, s2m, limit_scenes=None))

    flat = []
    for _scene_name, kf_list in scenes_meta:
        flat.extend(kf_list)
        if len(flat) >= args.warmup + args.samples:
            break
    need = args.warmup + args.samples
    if len(flat) < need:
        raise ValueError(f"需要 {need} keyframe, 只找到 {len(flat)}")
    flat = flat[:need]
    flat_indices = [idx for idx, _ in flat]
    flat_metas = [m for _, m in flat]
    print(f"  benchmark sample 总数: {len(flat)} (warmup={args.warmup} + measured={args.samples})")
    print(f"  跨 {len(set(m.scene_name for m in flat_metas))} 个 scene")

    print("[4] preload slow logits (避免 zlib 解压噪声污染 fast+no-warp 时间测量) ...")
    slow_decoder_fn = upstream.build_slow_decoder_fn(data_cfg, device)
    slow_cache = upstream.SlowLogitsGPUCache(device=device, decoder_fn=slow_decoder_fn)
    slow_cache.preload([m.slow_logit_path for m in flat_metas], skip_missing=True, verbose=False)
    print(f"  cached {len(slow_cache)} slow paths")

    results = {}

    if _wants(args.mode, "native-only"):
        print("\n=== Mode A0: OPUS native simple_test + sparse->dense uint8 ===")
        loader = upstream.make_streaming_loader(
            fast.dataset,
            flat_indices,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        results["native_only"] = upstream.benchmark_native_only(
            fast, iter(loader), args.warmup, args.samples, data_cfg,
        )

    if _wants(args.mode, "fast-only"):
        print("\n=== Mode A: OPUS raw-top3 fast-only (NCDE input path) ===")
        loader = upstream.make_streaming_loader(
            fast.dataset,
            flat_indices,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        results["fast_only"] = upstream.benchmark_fast_only(
            fast, iter(loader), args.warmup, args.samples,
        )

    if _wants(args.mode, "fast-nowarp"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n=== Mode B: OPUS raw-top3 fast + no-warp-attn (slow_interval={args.slow_interval}s) ===")
        loader = upstream.make_streaming_loader(
            fast.dataset,
            flat_indices,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        results["fast_no_warp_attn"] = benchmark_fast_no_warp(
            fast,
            stream_aligner,
            slow_cache,
            iter(loader),
            flat_metas,
            args.warmup,
            args.samples,
            args.slow_interval,
            device,
        )

    print(f"\n{'=' * 72}")
    print(f"Final OPUSv1 no-warp-attn benchmark (warmup={args.warmup}, measured={args.samples})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 72}")
    if "native_only" in results:
        a0 = results["native_only"]
        print(f"  OPUS native             | {a0['latency_ms_mean']:6.2f} ms | {a0['fps']:6.2f} FPS")
    if "fast_only" in results:
        a = results["fast_only"]
        print(f"  raw-top3 fast-only      | {a['latency_ms_mean']:6.2f} ms | {a['fps']:6.2f} FPS")
    if "fast_no_warp_attn" in results:
        b = results["fast_no_warp_attn"]
        print(
            f"  raw-top3 fast + no-warp | {b['latency_ms_mean']:6.2f} ms | {b['fps']:6.2f} FPS  "
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
                "fast_backend": "opusv1t_raw_top3",
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
