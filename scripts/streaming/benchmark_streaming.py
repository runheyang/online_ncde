"""Streaming benchmark: fast-only baseline vs fast + ours 端到端单帧延迟.

测时协议跟 OccStudio tools/analysis_tools/benchmark.py 完全一致:
  - warmup K iter (不计时)
  - measured N iter (计时)
  - 每帧: torch.cuda.synchronize() + time.perf_counter() 卡区间
  - 输出 mean latency (ms) + FPS (img/s)

两种 mode 跑相同的 K + N 连续 keyframe (跨 scene 连续, 跟 OccStudio benchmark 一致):

Mode A (fast-only, OccStudio 标准):
  fast.model.simple_test(pts, metas, img)
  ↑ 含 extract_feat + occupancy_head + permute/flip/rot + softmax + argmax + .cpu().numpy() → uint8

Mode B (fast + ours):
  fast.forward_keyframe(batch)
  ↑ 含 extract_feat + occupancy_head + permute/flip/rot + topk(3) + clamp + scatter → dense logits on GPU
  → stream_aligner.reset_with_slow / evolve
  → aligned.argmax(0).to(uint8).cpu().numpy()
  ↑ 最终 [200,200,16] uint8 on CPU, 跟 baseline 输出对齐

aligner overhead = Mode B latency - Mode A latency
"""
from __future__ import annotations
import os, sys, time, argparse, json
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch

from online_ncde.config import load_config_with_base
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner
from online_ncde.utils.checkpoints import load_checkpoint_for_eval
from online_ncde.streaming.fast_runner import FastRunner
from online_ncde.streaming.scene_iterator import build_sample_meta_index, iter_scenes
from online_ncde.streaming.stream_aligner import StreamAligner
from online_ncde.streaming.streaming_loader import make_streaming_loader, scatter_to_device
from online_ncde.streaming.slow_cache import SlowLogitsGPUCache, build_slow_decoder_fn

REPO_ROOT = "/root/autodl-tmp/online_ncde"
OCC_ROOT = "/root/autodl-tmp/online_ncde/third_party/OccStudio"
OCC_CONFIG = "configs/alocc/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.py"
OCC_CKPT = "ckpts/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.pth"
BDV2_PKL = "/root/autodl-tmp/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl"
GT_ROOT = "/root/autodl-tmp/data/nuscenes/gts"


def resolve_slow_root(data_cfg) -> str:
    rel = data_cfg.get("slow_logit_root")
    if rel is None:
        raise ValueError("data_cfg.slow_logit_root 缺失")
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aligner-cfg", required=True)
    p.add_argument("--aligner-ckpt", required=True)
    p.add_argument("--samples", type=int, default=200, help="measured iterations (mmdet 标准)")
    p.add_argument("--warmup", type=int, default=20,
                   help="warmup iterations (>=alocc history_cat_num=16, 20 安全)")
    p.add_argument("--slow-interval", type=float, default=4.0,
                   help="慢系统注入间隔(秒). 默认 4.0s = 你描述的现实场景")
    p.add_argument("--mode", choices=["fast-only", "fast-ours", "both"], default="both")
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def build_aligner(args, device):
    cfg = load_config_with_base(args.aligner_cfg)
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    aligner = OnlineNcdeAligner(
        num_classes=data_cfg["num_classes"], feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"], encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"], pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=model_cfg.get("decoder_init_scale", 1.0e-3),
        use_fast_residual=bool(model_cfg.get("use_fast_residual", True)),
        func_g_inner_dim=model_cfg.get("func_g_inner_dim", 32),
        func_g_body_dilations=tuple(model_cfg.get("func_g_body_dilations", [1, 2, 3])),
        func_g_gn_groups=int(model_cfg.get("func_g_gn_groups", 8)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
        solver_variant=args.solver,
    ).to(device)
    load_checkpoint_for_eval(args.aligner_ckpt, model=aligner, strict=False)
    aligner.eval()
    return aligner, data_cfg


def _unwrap(x):
    return x.data[0] if hasattr(x, "data") else x


def _unwrap_batch_for_simple_test(batch):
    """把 streaming_loader.scatter_to_device 后的 batch unwrap 成 simple_test 参数."""
    img_inputs = batch["img_inputs"]
    img_metas = batch["img_metas"]
    points = batch.get("points", None)
    if not isinstance(img_inputs, list): img_inputs = [img_inputs]
    if not isinstance(img_metas, list):  img_metas = [img_metas]
    if points is not None and not isinstance(points, list): points = [points]
    img = _unwrap(img_inputs[0])
    metas = _unwrap(img_metas[0])
    pts = [None] if points is None else _unwrap(points[0])
    return pts, metas, img


def benchmark_fast_only(fast: FastRunner, raw_batches, K, N, log_interval=50):
    """Mode A: 调 model.simple_test (含 softmax+argmax+.cpu().numpy() → uint8).
    严格对齐 OccStudio benchmark.py: scatter-to-GPU 也在 timer 内."""
    pure_inf_time = 0.0
    per_iter = []
    for i, raw in enumerate(raw_batches):
        if i >= K + N:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            batch = scatter_to_device(raw, 0)
            pts, metas, img = _unwrap_batch_for_simple_test(batch)
            result = fast.model.simple_test(pts, metas, img)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= K:
            pure_inf_time += elapsed
            per_iter.append(elapsed * 1000)
            done = i + 1 - K
            if done % log_interval == 0 or done == N:
                fps = done / pure_inf_time
                print(f"    [{done:>3}/{N}] FPS: {fps:6.2f}  latency: {1000/fps:6.2f} ms")
    fps = N / pure_inf_time
    arr = np.array(per_iter)
    print(f"  ★ fast-only: latency mean={1000/fps:.2f}ms median={np.median(arr):.2f}ms p95={np.percentile(arr, 95):.2f}ms / FPS={fps:.2f}")
    return {
        "latency_ms_mean": float(1000 / fps),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "fps": float(fps),
        "n_measured": int(N),
        "n_warmup": int(K),
    }


def benchmark_fast_ours(
    fast: FastRunner, stream_aligner: StreamAligner, slow_cache: SlowLogitsGPUCache,
    raw_batches, metas_list, K, N, slow_interval, device, log_interval=50,
):
    """Mode B: fast.forward_keyframe → stream_aligner → argmax+cpu→uint8."""
    pure_inf_time = 0.0
    per_iter = []
    n_reset, n_evolve = 0, 0
    cur_scene = None
    last_slow_t_sec = -1e9

    # 重置 alocc internal history (与 fast-only mode 起点一致)
    fast.reset_history()
    stream_aligner.reset_scene()

    for i, (raw, meta) in enumerate(zip(raw_batches, metas_list)):
        if i >= K + N:
            break
        # scene 切换: stream aligner reset (alocc 内部 history 不手动 reset, 与 fast-only mode 一致)
        if meta.scene_name != cur_scene:
            stream_aligner.reset_scene()
            cur_scene = meta.scene_name
            last_slow_t_sec = -1e9
        # slow 注入决策
        t_sec = meta.timestamp_us / 1e6
        is_slow = (last_slow_t_sec < 0) or (t_sec - last_slow_t_sec + 1e-3 >= slow_interval)
        ego_t = torch.from_numpy(meta.ego2global).to(device=device, dtype=torch.float32)
        slow_logits = slow_cache.get(meta.slow_logit_path) if (is_slow and slow_cache.has(meta.slow_logit_path)) else None

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            batch = scatter_to_device(raw, 0)
            fl = fast.forward_keyframe(batch)
            if slow_logits is not None:
                aligned = stream_aligner.reset_with_slow(fl, slow_logits, ego_t, meta.timestamp_us)
            else:
                aligned = stream_aligner.evolve(fl, ego_t, meta.timestamp_us)
            pred_uint8 = aligned.argmax(0).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i >= K:
            pure_inf_time += elapsed
            per_iter.append(elapsed * 1000)
            if slow_logits is not None: n_reset += 1
            else: n_evolve += 1
            done = i + 1 - K
            if done % log_interval == 0 or done == N:
                fps = done / pure_inf_time
                print(f"    [{done:>3}/{N}] FPS: {fps:6.2f}  latency: {1000/fps:6.2f} ms  "
                      f"(reset={n_reset} evolve={n_evolve})")
        if slow_logits is not None:
            last_slow_t_sec = t_sec

    fps = N / pure_inf_time
    arr = np.array(per_iter)
    print(f"  ★ fast+ours: latency mean={1000/fps:.2f}ms median={np.median(arr):.2f}ms p95={np.percentile(arr, 95):.2f}ms / FPS={fps:.2f}")
    print(f"     reset/evolve in measured: {n_reset}/{n_evolve}")
    return {
        "latency_ms_mean": float(1000 / fps),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "fps": float(fps),
        "n_measured": int(N),
        "n_warmup": int(K),
        "n_reset": int(n_reset),
        "n_evolve": int(n_evolve),
        "slow_interval_sec": float(slow_interval),
    }


def main():
    args = parse_args()
    os.chdir("/root/autodl-tmp/online_ncde")
    device = torch.device("cuda:0")

    print("[1] aligner build & load ckpt ...")
    aligner, data_cfg = build_aligner(args, device)
    stream_aligner = StreamAligner(aligner)

    print("[2] fast runner build (含切目录到 OccStudio) ...")
    fast = FastRunner(
        occstudio_root=OCC_ROOT, config_path=OCC_CONFIG, ckpt_path=OCC_CKPT,
        num_classes=data_cfg["num_classes"], free_index=data_cfg["free_index"],
        topk_k=int(data_cfg.get("alocc_topk_k", 3)),
        clamp_min=float(data_cfg.get("alocc_clamp_min", -5.0)),
        fill_value=float(data_cfg.get("alocc_fill_value", -5.0)),
        max_centering=bool(data_cfg.get("alocc_max_centering", False)),
        device="cuda:0",
    )
    fast.build()

    slow_root = resolve_slow_root(data_cfg)
    slow_format = data_cfg.get("slow_logit_format", data_cfg.get("logits_format", "alocc_dense_topk"))
    print(f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}) ...")
    s2m = build_sample_meta_index(BDV2_PKL, slow_root, GT_ROOT)
    scenes_meta = list(iter_scenes(fast.dataset, s2m, limit_scenes=None))

    # 收集 K+N 个连续 keyframe (跨 scene 连续, 跟 OccStudio benchmark.py 一致)
    flat = []
    for sn, kf_list in scenes_meta:
        flat.extend(kf_list)
        if len(flat) >= args.warmup + args.samples:
            break
    if len(flat) < args.warmup + args.samples:
        raise ValueError(f"需要 {args.warmup+args.samples} keyframe, 只找到 {len(flat)}")
    flat = flat[: args.warmup + args.samples]
    flat_indices = [idx for idx, _ in flat]
    flat_metas = [m for _, m in flat]
    print(f"  benchmark sample 总数: {len(flat)} (warmup={args.warmup} + measured={args.samples})")
    print(f"  跨 {len(set(m.scene_name for m in flat_metas))} 个 scene")

    print(f"[4] preload slow logits (避免 zlib 解压噪声污染时间测量) ...")
    slow_decoder_fn = build_slow_decoder_fn(data_cfg, device)
    slow_cache = SlowLogitsGPUCache(device=device, decoder_fn=slow_decoder_fn)
    slow_cache.preload([m.slow_logit_path for m in flat_metas], skip_missing=True, verbose=False)
    print(f"  cached {len(slow_cache)} slow paths")

    results = {}
    if args.mode in ("fast-only", "both"):
        print(f"\n=== Mode A: fast-only baseline (OccStudio 标准 simple_test, 含 softmax+argmax+CPU) ===")
        loader_a = make_streaming_loader(fast.dataset, flat_indices,
                                         num_workers=args.num_workers, prefetch_factor=2)
        fast.reset_history()
        def gen_a():
            for raw in loader_a:
                yield raw
        results["fast_only"] = benchmark_fast_only(fast, gen_a(), args.warmup, args.samples)

    if args.mode in ("fast-ours", "both"):
        print(f"\n=== Mode B: fast + ours (NCDE aligner, slow_interval={args.slow_interval}s, 输出 → CPU uint8) ===")
        loader_b = make_streaming_loader(fast.dataset, flat_indices,
                                         num_workers=args.num_workers, prefetch_factor=2)
        def gen_b():
            for raw in loader_b:
                yield raw
        results["fast_ours"] = benchmark_fast_ours(
            fast, stream_aligner, slow_cache,
            gen_b(), flat_metas,
            args.warmup, args.samples, args.slow_interval, device,
        )

    # 论文 table 风格汇总
    print(f"\n{'='*72}")
    print(f"Final benchmark (mmdet-style: warmup={args.warmup}, measured={args.samples})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*72}")
    if "fast_only" in results:
        a = results["fast_only"]
        print(f"  fast-only baseline    | {a['latency_ms_mean']:6.2f} ms | {a['fps']:6.2f} FPS")
    if "fast_ours" in results:
        b = results["fast_ours"]
        print(f"  fast + ours (slow={args.slow_interval}s)| {b['latency_ms_mean']:6.2f} ms | {b['fps']:6.2f} FPS  "
              f"(reset={b['n_reset']}/evolve={b['n_evolve']})")
    if "fast_only" in results and "fast_ours" in results:
        a, b = results["fast_only"], results["fast_ours"]
        d_ms = b["latency_ms_mean"] - a["latency_ms_mean"]
        d_pct = d_ms / a["latency_ms_mean"] * 100
        print(f"  aligner overhead      | +{d_ms:5.2f} ms | {d_pct:+6.1f}%")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "warmup": args.warmup, "measured": args.samples,
                "slow_interval_sec": args.slow_interval,
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
            }, f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
