"""Streaming benchmark for OPUSv1-T fast system + NCDE aligner.

计时协议对齐 scripts/streaming/benchmark_streaming.py:
  - warmup K iter 不计时
  - measured N iter 计时
  - 每帧计时区间含 scatter_to_device + GPU forward + 最终 argmax/cpu

Modes:
  native-only:
    OPUS 原生 simple_test + sparse 输出转 dense uint8。仅作 OPUS 原生端到端参考。
  fast-only:
    OPUSv1FastRunner.forward_keyframe，使用 NCDE 实际输入口径：
    raw logits -> sigmoid 阈值/中心距离过滤 -> voxelize raw logits -> top3 -> clamp_min -> dense。
  fast-ours:
    fast-only logits -> StreamAligner reset/evolve -> argmax/cpu。

默认 mode=both，只测 fast-only 与 fast-ours，便于看 NCDE aligner overhead。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
from pyquaternion import Quaternion

from online_ncde.config import load_config_with_base
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner
from online_ncde.streaming.opusv1_fast_runner import OpusV1FastRunner
from online_ncde.streaming.scene_iterator import KeyframeMeta, iter_scenes
from online_ncde.streaming.slow_cache import SlowLogitsGPUCache, build_slow_decoder_fn
from online_ncde.streaming.stream_aligner import StreamAligner
from online_ncde.streaming.streaming_loader import make_streaming_loader, scatter_to_device
from online_ncde.utils.checkpoints import load_checkpoint_for_eval


REPO_ROOT = "/root/autodl-tmp/online_ncde"
OPUS_ROOT = "/root/autodl-tmp/OPUS"
OPUS_CONFIG = "configs/opusv1_nusc-occ3d/opusv1-t_r50_704x256_8f_nusc-occ3d_100e.py"
OPUS_CKPT = "checkpoints/opusv1-t_r50_704x256_8f_nusc-occ3d_100e.pth"
META_PKL = "/root/autodl-tmp/data/nuscenes/nuscenes_infos_val_sweep.pkl"
GT_ROOT = "/root/autodl-tmp/data/nuscenes/gts"


def _resolve_repo_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def resolve_slow_root(data_cfg: dict) -> str:
    rel = data_cfg.get("slow_logit_root")
    if rel is None:
        raise ValueError("data_cfg.slow_logit_root 缺失")
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def _ego2global_matrix(translation, rotation_wxyz) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def build_opus_sample_meta_index(meta_pkl_path: str, slow_logit_root: str, gt_root: str) -> dict:
    """OPUS sweep pkl -> sample_token 到 KeyframeMeta 的索引."""
    import pickle

    with open(meta_pkl_path, "rb") as f:
        data = pickle.load(f)
    out = {}
    for e in data["infos"]:
        sample_token = e["token"]
        cam_front = e.get("cams", {}).get("CAM_FRONT", {})
        frame_token = (
            cam_front.get("sample_data_token")
            or cam_front.get("token")
            or os.path.splitext(os.path.basename(cam_front.get("data_path", "")))[0]
            or sample_token
        )
        scene_name = e["scene_name"]
        ego = _ego2global_matrix(e["ego2global_translation"], e["ego2global_rotation"])
        out[sample_token] = KeyframeMeta(
            sample_token=sample_token,
            frame_token=frame_token,
            scene_name=scene_name,
            scene_token=e["scene_token"],
            timestamp_us=int(e["timestamp"]),
            ego2global=ego,
            slow_logit_path=os.path.join(slow_logit_root, scene_name, sample_token, "logits.npz"),
            gt_label_path=os.path.join(gt_root, scene_name, sample_token, "labels.npz"),
        )
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aligner-cfg", required=True)
    p.add_argument("--aligner-ckpt", required=True)
    p.add_argument("--samples", type=int, default=200, help="measured iterations")
    p.add_argument("--warmup", type=int, default=20, help="warmup iterations")
    p.add_argument("--slow-interval", type=float, default=4.0)
    p.add_argument(
        "--mode",
        choices=["native-only", "fast-only", "fast-ours", "both", "all"],
        default="both",
    )
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--opus-root", default=OPUS_ROOT)
    p.add_argument("--opus-config", default=OPUS_CONFIG)
    p.add_argument("--opus-ckpt", default=OPUS_CKPT)
    p.add_argument("--meta-pkl", default=META_PKL)
    p.add_argument("--gt-root", default=GT_ROOT)
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


def _take_single_aug(x):
    x = _unwrap(x)
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = _unwrap(x[0])
    return x


def _unwrap_opus_batch(batch):
    img = _take_single_aug(batch["img"])
    img_metas = _take_single_aug(batch["img_metas"])
    if isinstance(img_metas, dict):
        img_metas = [img_metas]
    return img_metas, img


def _sparse_occ_to_dense_uint8(result, grid_size=(200, 200, 16), free_index=17):
    """OPUS 原生 sparse sem_pred/occ_loc -> dense uint8，用于 native-only 对齐输出形态."""
    dense = np.full(tuple(grid_size), fill_value=int(free_index), dtype=np.uint8)
    if not result:
        return dense
    item = result[0]
    occ_loc = item["occ_loc"]
    sem_pred = item["sem_pred"]
    if occ_loc.size > 0:
        dense[occ_loc[:, 0], occ_loc[:, 1], occ_loc[:, 2]] = sem_pred.astype(np.uint8)
    return dense


def _summarize_latency(name: str, per_iter_ms: list[float], pure_inf_time: float, K: int, N: int):
    fps = N / pure_inf_time
    arr = np.asarray(per_iter_ms, dtype=np.float64)
    print(
        f"  ★ {name}: latency mean={1000/fps:.2f}ms "
        f"median={np.median(arr):.2f}ms p95={np.percentile(arr, 95):.2f}ms / FPS={fps:.2f}"
    )
    return {
        "latency_ms_mean": float(1000 / fps),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "fps": float(fps),
        "n_measured": int(N),
        "n_warmup": int(K),
    }


def benchmark_native_only(fast: OpusV1FastRunner, raw_batches, K, N, data_cfg, log_interval=50):
    """Mode A0: OPUS 原生 simple_test + sparse 转 dense uint8。"""
    pure_inf_time = 0.0
    per_iter = []
    fast.reset_history()

    for i, raw in enumerate(raw_batches):
        if i >= K + N:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            batch = scatter_to_device(raw, 0)
            img_metas, img = _unwrap_opus_batch(batch)
            result = fast.model.simple_test(img_metas, img)
            pred_uint8 = _sparse_occ_to_dense_uint8(
                result,
                grid_size=tuple(data_cfg["grid_size"]),
                free_index=int(data_cfg["free_index"]),
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i >= K:
            pure_inf_time += elapsed
            per_iter.append(elapsed * 1000)
            done = i + 1 - K
            if done % log_interval == 0 or done == N:
                fps = done / pure_inf_time
                print(f"    [{done:>3}/{N}] FPS: {fps:6.2f}  latency: {1000/fps:6.2f} ms")

    return _summarize_latency("native-only", per_iter, pure_inf_time, K, N)


def benchmark_fast_only(fast: OpusV1FastRunner, raw_batches, K, N, log_interval=50):
    """Mode A: OPUS raw-top3 dense logits -> argmax+cpu."""
    pure_inf_time = 0.0
    per_iter = []
    fast.reset_history()

    for i, raw in enumerate(raw_batches):
        if i >= K + N:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            batch = scatter_to_device(raw, 0)
            fast_logits = fast.forward_keyframe(batch)
            pred_uint8 = fast_logits.argmax(0).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i >= K:
            pure_inf_time += elapsed
            per_iter.append(elapsed * 1000)
            done = i + 1 - K
            if done % log_interval == 0 or done == N:
                fps = done / pure_inf_time
                print(f"    [{done:>3}/{N}] FPS: {fps:6.2f}  latency: {1000/fps:6.2f} ms")

    return _summarize_latency("fast-only(raw-top3)", per_iter, pure_inf_time, K, N)


def benchmark_fast_ours(
    fast: OpusV1FastRunner,
    stream_aligner: StreamAligner,
    slow_cache: SlowLogitsGPUCache,
    raw_batches,
    metas_list,
    K,
    N,
    slow_interval,
    device,
    log_interval=50,
):
    """Mode B: OPUS raw-top3 fast logits -> StreamAligner -> argmax+cpu."""
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
            batch = scatter_to_device(raw, 0)
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

    out = _summarize_latency("fast+ours", per_iter, pure_inf_time, K, N)
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
    if mode == "both" and name in {"fast-only", "fast-ours"}:
        return True
    return mode == name


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    args.aligner_cfg = _resolve_repo_path(args.aligner_cfg)
    args.aligner_ckpt = _resolve_repo_path(args.aligner_ckpt)
    args.out_json = _resolve_repo_path(args.out_json)
    device = torch.device("cuda:0")

    print("[1] aligner build & load ckpt ...")
    aligner, data_cfg = build_aligner(args, device)
    stream_aligner = StreamAligner(aligner)

    print("[2] OPUSv1-T fast runner build ...")
    fast = OpusV1FastRunner(
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

    slow_root = resolve_slow_root(data_cfg)
    slow_format = data_cfg.get("slow_logit_format", data_cfg.get("logits_format", "opus_sparse_full"))
    print(f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}) ...")
    s2m = build_opus_sample_meta_index(args.meta_pkl, slow_root, args.gt_root)
    scenes_meta = list(iter_scenes(fast.dataset, s2m, limit_scenes=None))

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

    print("[4] preload slow logits (避免 zlib 解压噪声污染 fast+ours 时间测量) ...")
    slow_decoder_fn = build_slow_decoder_fn(data_cfg, device)
    slow_cache = SlowLogitsGPUCache(device=device, decoder_fn=slow_decoder_fn)
    slow_cache.preload([m.slow_logit_path for m in flat_metas], skip_missing=True, verbose=False)
    print(f"  cached {len(slow_cache)} slow paths")

    results = {}

    if _wants(args.mode, "native-only"):
        print("\n=== Mode A0: OPUS native simple_test + sparse->dense uint8 ===")
        loader = make_streaming_loader(
            fast.dataset, flat_indices,
            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        )
        results["native_only"] = benchmark_native_only(
            fast, iter(loader), args.warmup, args.samples, data_cfg,
        )

    if _wants(args.mode, "fast-only"):
        print("\n=== Mode A: OPUS raw-top3 fast-only (NCDE input path) ===")
        loader = make_streaming_loader(
            fast.dataset, flat_indices,
            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        )
        results["fast_only"] = benchmark_fast_only(
            fast, iter(loader), args.warmup, args.samples,
        )

    if _wants(args.mode, "fast-ours"):
        print(f"\n=== Mode B: OPUS raw-top3 fast + NCDE (slow_interval={args.slow_interval}s) ===")
        loader = make_streaming_loader(
            fast.dataset, flat_indices,
            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        )
        results["fast_ours"] = benchmark_fast_ours(
            fast, stream_aligner, slow_cache,
            iter(loader), flat_metas,
            args.warmup, args.samples, args.slow_interval, device,
        )

    print(f"\n{'=' * 72}")
    print(f"Final OPUSv1 benchmark (warmup={args.warmup}, measured={args.samples})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 72}")
    if "native_only" in results:
        a0 = results["native_only"]
        print(f"  OPUS native          | {a0['latency_ms_mean']:6.2f} ms | {a0['fps']:6.2f} FPS")
    if "fast_only" in results:
        a = results["fast_only"]
        print(f"  raw-top3 fast-only   | {a['latency_ms_mean']:6.2f} ms | {a['fps']:6.2f} FPS")
    if "fast_ours" in results:
        b = results["fast_ours"]
        print(
            f"  raw-top3 fast + ours | {b['latency_ms_mean']:6.2f} ms | {b['fps']:6.2f} FPS  "
            f"(reset={b['n_reset']}/evolve={b['n_evolve']})"
        )
    if "fast_only" in results and "fast_ours" in results:
        a, b = results["fast_only"], results["fast_ours"]
        d_ms = b["latency_ms_mean"] - a["latency_ms_mean"]
        d_pct = d_ms / a["latency_ms_mean"] * 100
        print(f"  aligner overhead     | +{d_ms:5.2f} ms | {d_pct:+6.1f}%")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "fast_backend": "opusv1t_raw_top3",
                "warmup": args.warmup,
                "measured": args.samples,
                "slow_interval_sec": args.slow_interval,
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
            }, f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
