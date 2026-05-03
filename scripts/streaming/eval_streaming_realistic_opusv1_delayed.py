"""OPUSv1-T fast + NCDE aligner 的 delayed slow logits 流式评估.

参照 eval_streaming_realistic_delayed.py，但 fast 系统换成 OPUSv1-T：
  - fast logits 实时，由 OpusV1FastRunner 输出 raw-top3 dense logits；
  - slow logits 到达 step k 时，内容对应 step k - delay；
  - 前 delay 个 keyframe 无可用 slow，直接输出 fast；
  - slow 到达时从历史 slow 所在帧 reset hidden，再逐 2Hz step evolve 到当前帧。

只使用本项目 mIoU/RayIoU，不调用 OPUS 自带 dataset.evaluate / RayIoU。
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
from online_ncde.metrics import MetricMiouOcc3D
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
DEFAULT_SWEEP_PKL = "/root/autodl-tmp/data/nuscenes/nuscenes_infos_val_sweep.pkl"


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


def schedule_delayed_slow_steps(timestamps_us, interval_sec: float, delay_kf: int):
    """delayed 场景的 slow 到达调度.

    slow 在 step=delay_kf 首次到达，内容对应 step=0；之后每 slow_interval 秒到达一次。
    """
    ts = list(timestamps_us)
    used = set()
    if len(ts) <= delay_kf or interval_sec < 0:
        return used
    used.add(delay_kf)
    last_t = ts[delay_kf] / 1e6
    for i in range(delay_kf + 1, len(ts)):
        t = ts[i] / 1e6
        if t - last_t + 1e-3 >= interval_sec:
            used.add(i)
            last_t = t
    return used


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aligner-cfg", required=True)
    p.add_argument("--aligner-ckpt", required=True)
    p.add_argument("--slow-intervals", type=float, nargs="+", required=True)
    p.add_argument("--slow-delay-keyframes", type=int, default=2,
                   help="slow logits 到达延迟，单位 keyframe；默认 2 = 2Hz 下 1s")
    p.add_argument("--limit-scenes", type=int, default=None)
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--preload-slow", action="store_true")
    p.add_argument("--opus-root", default=OPUS_ROOT)
    p.add_argument("--opus-config", default=OPUS_CONFIG)
    p.add_argument("--opus-ckpt", default=OPUS_CKPT)
    p.add_argument("--meta-pkl", default=META_PKL)
    p.add_argument("--gt-root", default=GT_ROOT)
    p.add_argument("--sweep-pkl", default=DEFAULT_SWEEP_PKL)
    p.add_argument("--no-rayiou", action="store_true")
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


def cache_one_scene_fast(fast: OpusV1FastRunner, kf_list, loader_iter):
    """跑 1 个 scene 的 OPUS fast forward，缓存 dense logits fp16."""
    fast.reset_history()
    cache = []
    for _ in kf_list:
        raw = next(loader_iter)
        batch = scatter_to_device(raw, 0)
        fast_logits = fast.forward_keyframe(batch)
        cache.append(fast_logits.to(torch.float16))
    return cache


@torch.no_grad()
def delayed_reset_and_evolve(
    stream_aligner: StreamAligner,
    *,
    kf_list,
    scene_fast_cache,
    prev_step: int,
    curr_step: int,
    slow_logits_prev: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """slow 描述 prev_step 时刻，当前 curr_step 需要输出。"""
    _, meta_prev = kf_list[prev_step]
    fast_logits_prev = scene_fast_cache[prev_step].float()
    ego_prev = torch.from_numpy(meta_prev.ego2global).to(device=device, dtype=torch.float32)

    stream_aligner.reset_with_slow(
        fast_logits_prev, slow_logits_prev, ego_prev, meta_prev.timestamp_us,
    )

    aligned = None
    for k in range(prev_step + 1, curr_step + 1):
        _, meta_k = kf_list[k]
        fast_logits_k = scene_fast_cache[k].float()
        ego_k = torch.from_numpy(meta_k.ego2global).to(device=device, dtype=torch.float32)
        aligned = stream_aligner.evolve(fast_logits_k, ego_k, meta_k.timestamp_us)
    assert aligned is not None, "curr_step must be > prev_step"
    return aligned


def stream_scene_one_interval_delayed(
    kf_list, scene_fast_cache, slow_cache, slow_decoder_fn,
    stream_aligner, data_cfg,
    slow_interval, slow_delay_kf, device,
    pred_list_out, gt_cache,
):
    """delayed slow 注入版本。前 delay 帧或未注入 slow 前输出 fast。"""
    n_reset, n_evolve, n_fast_only = 0, 0, 0

    ts = [m.timestamp_us for _, m in kf_list]
    slow_steps = schedule_delayed_slow_steps(ts, slow_interval, slow_delay_kf)
    stream_aligner.reset_scene()

    for step, ((idx, meta), fl_fp16) in enumerate(zip(kf_list, scene_fast_cache)):
        fast_logits_curr = fl_fp16.float()

        if step < slow_delay_kf:
            aligned = fast_logits_curr
            n_fast_only += 1
        else:
            slow_arrived = step in slow_steps
            if slow_arrived:
                prev_step = step - slow_delay_kf
                _, meta_prev = kf_list[prev_step]
                slow_logit_path = meta_prev.slow_logit_path
                slow_present = (
                    slow_cache.has(slow_logit_path) if slow_cache is not None
                    else os.path.exists(slow_logit_path)
                )
            else:
                slow_present = False

            if slow_arrived and slow_present:
                prev_step = step - slow_delay_kf
                _, meta_prev = kf_list[prev_step]
                if slow_cache is not None:
                    slow_logits_prev = slow_cache.get(meta_prev.slow_logit_path)
                else:
                    slow_logits_prev = slow_decoder_fn(meta_prev.slow_logit_path)
                aligned = delayed_reset_and_evolve(
                    stream_aligner,
                    kf_list=kf_list,
                    scene_fast_cache=scene_fast_cache,
                    prev_step=prev_step,
                    curr_step=step,
                    slow_logits_prev=slow_logits_prev,
                    device=device,
                )
                n_reset += 1
            elif stream_aligner.hidden is not None:
                ego_curr = torch.from_numpy(meta.ego2global).to(device=device, dtype=torch.float32)
                aligned = stream_aligner.evolve(fast_logits_curr, ego_curr, meta.timestamp_us)
                n_evolve += 1
            else:
                aligned = fast_logits_curr
                n_fast_only += 1

        pred_uint8 = aligned.argmax(0).to(torch.uint8).cpu().numpy()
        pred_list_out.append((meta.sample_token, pred_uint8))

        if meta.sample_token not in gt_cache:
            with np.load(meta.gt_label_path, allow_pickle=False) as gt_npz:
                gt_cache[meta.sample_token] = (
                    gt_npz["semantics"].astype(np.uint8),
                    gt_npz["mask_camera"].astype(np.uint8),
                )

    return n_reset, n_evolve, n_fast_only


def eval_miou(pred_list, gt_cache, num_classes):
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
    if label:
        print(f"  {label}")
    sem_count = len(class_names) - 1
    cells = [f"{class_names[i]}={per_cls[i]:5.2f}" for i in range(sem_count)]
    line, lines, max_per_line = "", [], 4
    for j, c in enumerate(cells, 1):
        line += c + "  "
        if j % max_per_line == 0:
            lines.append(line.rstrip())
            line = ""
    if line:
        lines.append(line.rstrip())
    for ln in lines:
        print(f"    {ln}")
    print(f"    free={per_cls[-1]:.2f}    ->  mIoU = {miou:.2f}   mIoU_D = {miou_d:.2f}")


def eval_rayiou(pred_list, gt_cache, origins_by_token):
    from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou
    sem_pred_list, sem_gt_list, lidar_origin_list = [], [], []
    skipped = 0
    for token, pred in pred_list:
        if token not in origins_by_token:
            skipped += 1
            continue
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
    print(f"[3] sample meta index (slow_format={slow_format}, slow_root={slow_root}, "
          f"slow_delay_kf={args.slow_delay_keyframes}) ...")
    s2m = build_opus_sample_meta_index(args.meta_pkl, slow_root, args.gt_root)
    scenes_meta = list(iter_scenes(fast.dataset, s2m, limit_scenes=args.limit_scenes))
    flat_indices = [idx for _, kf_list in scenes_meta for (idx, _) in kf_list]
    total_kf = len(flat_indices)
    print(f"  scenes count={len(scenes_meta)}, total kf={total_kf}")
    slow_decoder_fn = build_slow_decoder_fn(data_cfg, device)

    print(f"[4] DataLoader (num_workers={args.num_workers}, prefetch={args.prefetch_factor}) ...")
    loader = make_streaming_loader(
        fast.dataset, flat_indices,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    loader_iter = iter(loader)

    predictions = {it: [] for it in args.slow_intervals}
    gt_cache = {}
    reset_cnt = {it: 0 for it in args.slow_intervals}
    evolve_cnt = {it: 0 for it in args.slow_intervals}
    fast_only_cnt = {it: 0 for it in args.slow_intervals}

    print("\n[5] Phase 1: per-scene OPUS fast + delayed NCDE stream ...")
    t_overall = time.time()
    for s_i, (scene_name, kf_list) in enumerate(scenes_meta):
        scene_fast_cache = cache_one_scene_fast(fast, kf_list, loader_iter)

        scene_slow_cache = None
        if args.preload_slow:
            scene_slow_cache = SlowLogitsGPUCache(device=device, decoder_fn=slow_decoder_fn)
            scene_slow_cache.preload([m.slow_logit_path for _, m in kf_list],
                                     skip_missing=True, verbose=False)

        for it in args.slow_intervals:
            n_r, n_e, n_f = stream_scene_one_interval_delayed(
                kf_list, scene_fast_cache, scene_slow_cache, slow_decoder_fn,
                stream_aligner, data_cfg,
                it, args.slow_delay_keyframes, device,
                pred_list_out=predictions[it], gt_cache=gt_cache,
            )
            reset_cnt[it] += n_r
            evolve_cnt[it] += n_e
            fast_only_cnt[it] += n_f

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

    print("\n[7] Phase 2a: mIoU ...")
    miou_results, miou_d_results, per_cls_results = {}, {}, {}
    for it in args.slow_intervals:
        miou, miou_d, per_cls, metric = eval_miou(
            predictions[it], gt_cache, data_cfg["num_classes"]
        )
        miou_results[it] = miou
        miou_d_results[it] = miou_d
        per_cls_results[it] = per_cls
        _print_per_class_iou(per_cls, metric.class_names, miou, miou_d,
                             label=f"[interval={it}s, delay={args.slow_delay_keyframes}kf]")

    rayiou_results = {}
    if not args.no_rayiou:
        print("\n[8] Phase 2b: 本项目 RayIoU ...")
        from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
        origins_by_token = load_origins_from_sweep_pkl(args.sweep_pkl)
        for it in args.slow_intervals:
            r = eval_rayiou(predictions[it], gt_cache, origins_by_token)
            rayiou_results[it] = r
            print(f"  interval={it}: RayIoU={r['RayIoU']:.4f}  "
                  f"@1={r['RayIoU@1']:.4f}  @2={r['RayIoU@2']:.4f}  @4={r['RayIoU@4']:.4f}")
    else:
        print("\n[8] Phase 2b: RayIoU skipped (--no-rayiou)")

    print(f"\n=== 最终汇总 ({total_kf} keyframes, {len(scenes_meta)} scenes, "
          f"slow_delay={args.slow_delay_keyframes}kf) ===")
    print(f"  inference wall: {inference_wall:.1f}s")
    header = (f"  {'interval':>10s}  {'mIoU':>7s}  {'mIoU_D':>7s}  {'RayIoU':>7s}  "
              f"{'@1':>6s}  {'@2':>6s}  {'@4':>6s}  {'reset':>6s}  {'evolve':>7s}  {'fastonly':>8s}")
    print(header)
    out = {
        "fast_backend": "opusv1t_raw_top3",
        "opus_config": args.opus_config,
        "scenes": len(scenes_meta),
        "total_kf": total_kf,
        "slow_delay_keyframes": args.slow_delay_keyframes,
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
              f"{reset_cnt[it]:>6d}  {evolve_cnt[it]:>7d}  {fast_only_cnt[it]:>8d}")
        per_cls_dict = {
            MetricMiouOcc3D(num_classes=data_cfg["num_classes"]).class_names[i]: float(per_cls_results[it][i])
            for i in range(data_cfg["num_classes"])
        }
        out["intervals"][str(it)] = {
            "miou": float(miou_results[it]),
            "miou_d": float(miou_d_results[it]),
            "per_class_iou": per_cls_dict,
            "rayiou": float(ri) if not np.isnan(ri) else None,
            "rayiou_at_1": float(ri1) if not np.isnan(ri1) else None,
            "rayiou_at_2": float(ri2) if not np.isnan(ri2) else None,
            "rayiou_at_4": float(ri4) if not np.isnan(ri4) else None,
            "n_reset": reset_cnt[it],
            "n_evolve": evolve_cnt[it],
            "n_fast_only": fast_only_cnt[it],
        }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
