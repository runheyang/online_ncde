"""现实在线推理评估 (推理 + 评估分离): per-scene 推理 + collect, 后统一 mIoU + RayIoU.

测速职责留给 scripts/streaming/benchmark_streaming.py; 本脚本只关心精度评估.

Phase 1 (推理):
  for scene in scenes:
    cache_fast(scene)            # GPU fp16, 单 scene <= 1 GB
    preload_slow(scene)          # GPU fp16, 单 scene <= 1 GB
    for interval in intervals:
      stream align all kf;
      collect (token, pred_uint8) → per-interval list (CPU memory)
      load GT/mask 一次, 共享 dict[token] -> (gt, mask)
    free fast/slow                # GPU back to ~0.4 GB

Phase 2a: 统一 mIoU
Phase 2b: 统一 RayIoU (调 ray_metrics.main)

内存预算 (全 val ~6000 kf × 4 intervals):
  GPU peak = ~0.4 GB
  CPU pred  = 6000 × 4 × 640 KB ≈ 15 GB
  CPU gt    = 6000 × 1280 KB     ≈ 7.5 GB
  total CPU ≈ 22 GB

用法:
    python scripts/streaming/eval_streaming_realistic.py \
        --aligner-cfg configs/online_ncde/fast_alocc2dmini__slow_alocc3d/base_2hz.yaml \
        --aligner-ckpt ckpts/.../epoch_9.pth \
        --slow-intervals 0.5 1.0 2.0 4.0 \
        [--limit-scenes N]
        --num-workers 4 --preload-slow
        [--no-rayiou]
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
from online_ncde.metrics import MetricMiouOcc3D
from online_ncde.streaming.fast_runner import FastRunner
from online_ncde.streaming.scene_iterator import build_sample_meta_index, iter_scenes
from online_ncde.streaming.slow_schedule import schedule_slow_steps
from online_ncde.streaming.stream_aligner import StreamAligner
from online_ncde.streaming.streaming_loader import make_streaming_loader, scatter_to_device
from online_ncde.streaming._dense_decode import decode_topk_npz_to_dense_gpu
from online_ncde.streaming.slow_cache import SlowLogitsGPUCache

OCC_ROOT = "/root/autodl-tmp/online_ncde/third_party/OccStudio"
OCC_CONFIG = "configs/alocc/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.py"
OCC_CKPT = "ckpts/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.pth"
BDV2_PKL = "/root/autodl-tmp/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl"
SLOW_ROOT = "/root/autodl-tmp/data/alocc3d_wo_mask"
GT_ROOT = "/root/autodl-tmp/data/nuscenes/gts"
DEFAULT_SWEEP_PKL = "/root/autodl-tmp/data/nuscenes/nuscenes_infos_val_sweep.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aligner-cfg", required=True)
    p.add_argument("--aligner-ckpt", required=True)
    p.add_argument("--slow-intervals", type=float, nargs="+", required=True)
    p.add_argument("--limit-scenes", type=int, default=None)
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--preload-slow", action="store_true",
                   help="预加载本 scene 的 slow logits 到 GPU dense fp16; 用时 ~0.1ms")
    p.add_argument("--sweep-pkl", default=DEFAULT_SWEEP_PKL,
                   help="nuscenes_infos_val_sweep.pkl 路径 (RayIoU 用 lidar origins)")
    p.add_argument("--no-rayiou", action="store_true", help="跳过 RayIoU, 仅评 mIoU")
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


def cache_one_scene_fast(fast: FastRunner, kf_list, loader_iter):
    """跑 1 个 scene 的 fast forward, 缓存 (C,X,Y,Z) fp16 GPU. 不做时延打桩."""
    fast.reset_history()
    cache = []
    for _ in kf_list:
        raw = next(loader_iter)
        batch = scatter_to_device(raw, 0)
        fl = fast.forward_keyframe(batch)
        cache.append(fl.to(torch.float16))
    return cache


def stream_scene_one_interval(
    kf_list, scene_fast_cache, slow_cache, stream_aligner, data_cfg,
    slow_interval, device,
    pred_list_out, gt_cache,
):
    """跑一个 (scene, interval), 收集 (token, pred_uint8) → pred_list_out;
    GT/mask 按 token 在 gt_cache 里 cache 一次, 跨 interval 共享.
    不做时延打桩 (测速请用 benchmark_streaming.py).
    """
    n_reset, n_evolve = 0, 0
    clamp_min = float(data_cfg.get("alocc_clamp_min", -5.0))
    fill_value = float(data_cfg.get("alocc_fill_value", -5.0))
    max_centering = bool(data_cfg.get("alocc_max_centering", False))
    num_classes = data_cfg["num_classes"]

    ts = [m.timestamp_us for _, m in kf_list]
    slow_steps = schedule_slow_steps(ts, slow_interval)
    stream_aligner.reset_scene()

    for step, ((idx, meta), fl_fp16) in enumerate(zip(kf_list, scene_fast_cache)):
        fast_logits = fl_fp16.float()
        ego_t = torch.from_numpy(meta.ego2global).to(device=device, dtype=torch.float32)

        slow_available = (
            step in slow_steps
            and (slow_cache.has(meta.slow_logit_path) if slow_cache is not None
                 else os.path.exists(meta.slow_logit_path))
        )
        if slow_available:
            if slow_cache is not None:
                slow_logits = slow_cache.get(meta.slow_logit_path)
            else:
                slow_logits = decode_topk_npz_to_dense_gpu(
                    meta.slow_logit_path, device=device,
                    num_classes=num_classes, clamp_min=clamp_min,
                    fill_value=fill_value, max_centering=max_centering,
                )
            aligned = stream_aligner.reset_with_slow(fast_logits, slow_logits, ego_t, meta.timestamp_us)
            n_reset += 1
        else:
            aligned = stream_aligner.evolve(fast_logits, ego_t, meta.timestamp_us)
            n_evolve += 1

        pred_uint8 = aligned.argmax(0).to(torch.uint8).cpu().numpy()
        pred_list_out.append((meta.sample_token, pred_uint8))

        if meta.sample_token not in gt_cache:
            gt_npz = np.load(meta.gt_label_path)
            gt_cache[meta.sample_token] = (
                gt_npz["semantics"].astype(np.uint8),
                gt_npz["mask_camera"].astype(np.uint8),
            )

    return n_reset, n_evolve


def eval_miou(pred_list, gt_cache, num_classes):
    """返回 (miou, miou_d, per_class_iou (np.array %), metric)."""
    metric = MetricMiouOcc3D(num_classes=num_classes,
                             use_lidar_mask=False, use_image_mask=True)
    for token, pred in pred_list:
        gt, mask = gt_cache[token]
        metric.add_batch(pred, gt, mask_camera=mask)
    miou = metric.count_miou(verbose=False)
    per_cls = metric.per_class_iu(metric.hist) * 100.0
    miou_d = metric.count_miou_d(verbose=False, class_iou=per_cls / 100.0)
    return float(miou), float(miou_d), per_cls, metric


def _print_per_class_iou(per_cls, class_names, miou, miou_d, label=""):
    """逐类 IoU 紧凑打印, 末尾给 mIoU + mIoU_D."""
    if label:
        print(f"  {label}")
    # 跳过 free 类 (最后一个)
    sem_count = len(class_names) - 1
    cells = [f"{class_names[i]}={per_cls[i]:5.2f}" for i in range(sem_count)]
    line, lines, max_per_line = "", [], 4
    for j, c in enumerate(cells, 1):
        line += c + "  "
        if j % max_per_line == 0:
            lines.append(line.rstrip()); line = ""
    if line: lines.append(line.rstrip())
    for ln in lines:
        print(f"    {ln}")
    print(f"    free={per_cls[-1]:.2f}    →  mIoU = {miou:.2f}   mIoU_D = {miou_d:.2f}")


def eval_rayiou(pred_list, gt_cache, origins_by_token):
    from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou
    sem_pred_list, sem_gt_list, lidar_origin_list = [], [], []
    skipped = 0
    for token, pred in pred_list:
        if token not in origins_by_token:
            skipped += 1; continue
        gt, _ = gt_cache[token]
        sem_pred_list.append(pred)
        sem_gt_list.append(gt)
        lidar_origin_list.append(origins_by_token[token])
    if skipped:
        print(f"  [rayiou] 跳过 {skipped} 个无 origin 的样本")
    print(f"  [rayiou] 参与计算: {len(sem_pred_list)}")
    return calc_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list)


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

    print("[3] sample meta index ...")
    s2m = build_sample_meta_index(BDV2_PKL, SLOW_ROOT, GT_ROOT)
    scenes_meta = list(iter_scenes(fast.dataset, s2m, limit_scenes=args.limit_scenes))
    flat_indices = [idx for _, kf_list in scenes_meta for (idx, _) in kf_list]
    total_kf = len(flat_indices)
    print(f"  scenes count={len(scenes_meta)}, total kf={total_kf}")

    print(f"[4] DataLoader (num_workers={args.num_workers}, prefetch={args.prefetch_factor}) ...")
    loader = make_streaming_loader(
        fast.dataset, flat_indices,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
    )
    loader_iter = iter(loader)

    # ---------- Phase 1: 推理 + collect ----------
    predictions = {it: [] for it in args.slow_intervals}
    gt_cache = {}
    reset_cnt = {it: 0 for it in args.slow_intervals}
    evolve_cnt = {it: 0 for it in args.slow_intervals}

    print(f"\n[5] Phase 1: per-scene 推理 + collect ...")
    t_overall = time.time()
    for s_i, (scene_name, kf_list) in enumerate(scenes_meta):
        scene_fast_cache = cache_one_scene_fast(fast, kf_list, loader_iter)

        scene_slow_cache = None
        if args.preload_slow:
            scene_slow_cache = SlowLogitsGPUCache(
                device=device,
                num_classes=data_cfg["num_classes"],
                clamp_min=float(data_cfg.get("alocc_clamp_min", -5.0)),
                fill_value=float(data_cfg.get("alocc_fill_value", -5.0)),
                max_centering=bool(data_cfg.get("alocc_max_centering", False)),
            )
            scene_slow_cache.preload([m.slow_logit_path for _, m in kf_list],
                                     skip_missing=True, verbose=False)

        for it in args.slow_intervals:
            n_r, n_e = stream_scene_one_interval(
                kf_list, scene_fast_cache, scene_slow_cache, stream_aligner,
                data_cfg, it, device,
                pred_list_out=predictions[it], gt_cache=gt_cache,
            )
            reset_cnt[it] += n_r
            evolve_cnt[it] += n_e

        del scene_fast_cache, scene_slow_cache
        torch.cuda.empty_cache()

        elapsed = time.time() - t_overall
        eta = elapsed / (s_i + 1) * (len(scenes_meta) - s_i - 1)
        if (s_i + 1) % 10 == 0 or s_i + 1 == len(scenes_meta) or s_i < 2:
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  [{s_i+1:3d}/{len(scenes_meta)}] {scene_name} kf={len(kf_list)} "
                  f"elapsed={elapsed:6.1f}s eta={eta:6.1f}s gpu={gpu_mb:.0f}MB")

    inference_wall = time.time() - t_overall
    print(f"\n[6] Phase 1 完成: wall={inference_wall:.1f}s, GT/mask cache={len(gt_cache)} samples")

    # ---------- Phase 2a: mIoU ----------
    print(f"\n[7] Phase 2a: 统一 mIoU 评估 ...")
    miou_results = {}
    miou_d_results = {}
    per_cls_results = {}
    for it in args.slow_intervals:
        miou, miou_d, per_cls, metric = eval_miou(
            predictions[it], gt_cache, data_cfg["num_classes"]
        )
        miou_results[it] = miou
        miou_d_results[it] = miou_d
        per_cls_results[it] = per_cls
        _print_per_class_iou(per_cls, metric.class_names, miou, miou_d,
                             label=f"[interval={it}s]")

    # ---------- Phase 2b: RayIoU ----------
    rayiou_results = {}
    if not args.no_rayiou:
        print(f"\n[8] Phase 2b: 加载 lidar origins 并跑 RayIoU ...")
        from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
        origins_by_token = load_origins_from_sweep_pkl(args.sweep_pkl)
        print(f"  loaded {len(origins_by_token)} origins from {args.sweep_pkl}")
        for it in args.slow_intervals:
            print(f"\n  --- interval={it} ---")
            r = eval_rayiou(predictions[it], gt_cache, origins_by_token)
            rayiou_results[it] = r
            print(f"  RayIoU={r['RayIoU']:.4f}  @1={r['RayIoU@1']:.4f}  "
                  f"@2={r['RayIoU@2']:.4f}  @4={r['RayIoU@4']:.4f}")

    # ---------- 汇总 (仅精度, 时延见 benchmark_streaming.py) ----------
    print(f"\n=== 最终汇总 ({total_kf} keyframes, {len(scenes_meta)} scenes) ===")
    print(f"  inference wall: {inference_wall:.1f}s   (此时间含 phase1 全 IO+forward; 单帧时延请用 benchmark_streaming.py 测)")
    header = (f"  {'interval':>10s}  {'mIoU':>7s}  {'mIoU_D':>7s}  {'RayIoU':>7s}  "
              f"{'@1':>6s}  {'@2':>6s}  {'@4':>6s}  {'reset':>6s}  {'evolve':>7s}")
    print(header)
    out = {
        "scenes": len(scenes_meta), "total_kf": total_kf,
        "inference_wall_s": float(inference_wall),
        "intervals": {},
    }
    for it in args.slow_intervals:
        r = rayiou_results.get(it, {})
        ri = r.get("RayIoU", float("nan"))
        ri1 = r.get("RayIoU@1", float("nan"))
        ri2 = r.get("RayIoU@2", float("nan"))
        ri4 = r.get("RayIoU@4", float("nan"))
        print(f"  {it:>10.2f}  {miou_results[it]:>7.2f}  {miou_d_results[it]:>7.2f}  "
              f"{ri:>7.4f}  {ri1:>6.4f}  {ri2:>6.4f}  {ri4:>6.4f}  "
              f"{reset_cnt[it]:>6d}  {evolve_cnt[it]:>7d}")
        per_cls_dict = {
            MetricMiouOcc3D(num_classes=data_cfg["num_classes"]).class_names[i]: float(per_cls_results[it][i])
            for i in range(data_cfg["num_classes"])
        }
        out["intervals"][it] = {
            "miou": float(miou_results[it]),
            "miou_d": float(miou_d_results[it]),
            "per_class_iou": per_cls_dict,
            "rayiou": float(ri) if not np.isnan(ri) else None,
            "rayiou_at_1": float(ri1) if not np.isnan(ri1) else None,
            "rayiou_at_2": float(ri2) if not np.isnan(ri2) else None,
            "rayiou_at_4": float(ri4) if not np.isnan(ri4) else None,
            "n_reset": reset_cnt[it], "n_evolve": evolve_cnt[it],
        }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
