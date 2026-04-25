#!/usr/bin/env python3
"""按演化时长 T (默认 0.5/1.0/1.5/2.0s) 分桶评估 mIoU + RayIoU。

- 每个桶强制覆盖全部样本：短历史样本 fallback 到自身可达的最远演化（abs_step = num_frames-1）。
- 走 collect-then-evaluate（一次性收集所有 (200,200,16) uint8 预测后批量算 RayIoU），
  不依赖逐步累加器；不统计推理速度。
- 输入 pkl 必须由 gen_online_ncde_canonical_infos.py --allow-short-history 生成（含短历史样本）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.keyframe_mapping import NuScenesKeyFrameResolver  # noqa: E402
from online_ncde.data.labels_io import load_labels_npz  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl  # noqa: E402
from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou  # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


DEFAULT_EVOLUTION_TIMES = "0.5,1.0,1.5,2.0"
DEFAULT_KEYFRAME_INTERVAL_SEC = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="配置文件路径")
    p.add_argument("--checkpoint", required=True, help="模型权重路径")
    p.add_argument("--limit", type=int, default=0, help="仅评估前 N 个样本，0 表示全量")
    p.add_argument("--batch-size", type=int, default=1,
                   help="DataLoader batch_size；默认强制为 1（forward_stepwise_eval 要求 "
                        "batch 内 rollout_start_step 一致，--allow-short-history 数据下连续样本"
                        "几乎一定 mix h，bs>1 会直接 raise）。")
    p.add_argument("--solver", choices=["heun", "euler"], default="euler",
                   help="ODE 求解器；默认 euler（next-fast 单次求值）")
    p.add_argument("--evolution-times", default=DEFAULT_EVOLUTION_TIMES,
                   help="逗号分隔，单位秒；默认 '0.5,1.0,1.5,2.0'")
    p.add_argument("--keyframe-interval-sec", type=float, default=DEFAULT_KEYFRAME_INTERVAL_SEC,
                   help="单个 keyframe interval 时长，默认 0.5（NuScenes 2Hz）")
    p.add_argument("--strict-no-fallback", action="store_true",
                   help="关闭 fallback：每桶仅纳入有足够历史的样本，桶内 count 自然分布。"
                        "默认 ON fallback：每桶覆盖全部样本。")
    p.add_argument("--exclude-short-history", action="store_true",
                   help="额外按 config.min_history_completeness 过滤短历史样本（与 stepwise 脚本一致）。"
                        "默认不过滤，所有样本都参评。")
    p.add_argument("--no-rayiou", action="store_true", help="跳过 RayIoU 评估")
    p.add_argument("--val-info-path", default="",
                   help="覆盖 data.val_info_path（空则沿用 config）")
    p.add_argument("--nusc-dataroot", default="data/nuscenes",
                   help="NuScenes dataroot（用于 frame_token -> key_frame 解析）")
    p.add_argument("--nusc-version", default="v1.0-trainval", help="NuScenes 版本")
    p.add_argument("--sweep-info-path", default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
                   help="sweep pkl 路径（用于 lidar origin 与 keyframe sample_token 集合）")
    p.add_argument("--dump-json", default="", help="可选：将结果写入 json")
    return p.parse_args()


def _to_json_number(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def _to_json_safe(obj: Any) -> Any:
    """递归把 numpy 标量/数组转成 Python 原生类型，避免 json.dump 报错。"""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_json_number(float(obj)) if isinstance(obj, np.floating) else obj.item()
    if isinstance(obj, float):
        return _to_json_number(obj)
    return obj


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    root_path = cfg["root_path"]

    evolution_times = [float(s) for s in args.evolution_times.split(",") if s.strip()]
    keyframe_interval = float(args.keyframe_interval_sec)
    print(f"[eval] evolution_times={evolution_times}s "
          f"keyframe_interval={keyframe_interval}s "
          f"fallback={'OFF' if args.strict_no_fallback else 'ON'}")

    # === dataset ===
    logits_loader = build_logits_loader(data_cfg, root_path)
    min_hc = int(data_cfg.get("min_history_completeness", 4)) if args.exclude_short_history else 0
    print(f"[eval] min_history_completeness={min_hc}")
    info_path = args.val_info_path if args.val_info_path else data_cfg.get(
        "val_info_path", data_cfg["info_path"]
    )
    if args.val_info_path:
        print(f"[eval] --val-info-path 覆盖 -> {info_path}")
    dataset_full = Occ3DOnlineNcdeDataset(
        info_path=info_path,
        root_path=root_path,
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        logits_loader=logits_loader,
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
        min_history_completeness=min_hc,
    )
    if args.limit > 0:
        keep = min(args.limit, len(dataset_full))
        dataset: Any = Subset(dataset_full, list(range(keep)))
    else:
        dataset = dataset_full

    num_workers = int(eval_cfg.get("num_workers", 4))
    # 强制 bs=1：batch 内 rollout_start_step 不一致时 forward_stepwise_eval 会 raise
    batch_size = max(int(args.batch_size), 1)
    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=online_ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(dataset, **kwargs)

    # === 推断 dt_per_step（从 pkl 元信息） ===
    info0 = dataset_full.infos[0]
    history_keyframes = int(info0.get("history_keyframes", 4))
    stride = int(data_cfg.get("fast_frame_stride", 1))
    num_output_frames_canonical = int(info0.get("num_output_frames",
                                                history_keyframes * info0.get("steps_per_interval", 1) + 1))
    if (num_output_frames_canonical - 1) % stride != 0:
        raise ValueError(
            f"canonical num_output_frames-1={num_output_frames_canonical-1} 与 fast_frame_stride={stride} 不整除"
        )
    num_frames = (num_output_frames_canonical - 1) // stride + 1
    if (num_frames - 1) % history_keyframes != 0:
        raise ValueError(
            f"抽帧后 num_frames-1={num_frames-1} 与 history_keyframes={history_keyframes} 不整除"
        )
    steps_per_kf = (num_frames - 1) // history_keyframes
    dt_per_step = keyframe_interval / steps_per_kf
    print(f"[eval] history_keyframes={history_keyframes} num_frames={num_frames} "
          f"steps_per_kf={steps_per_kf} dt_per_step={dt_per_step:.4f}s")

    # 校验：所有 T 必须精确落在 step 网格上，且 desired_off 必须是 steps_per_kf 的正整数倍
    # （否则 abs_step 会落到非 keyframe step，没有可用 GT，破坏“每桶覆盖全部样本”的语义）。
    desired_offs: dict[float, int] = {}
    grid_eps = 1e-6
    for T in evolution_times:
        ratio = T / dt_per_step
        d_off = int(round(ratio))
        if abs(ratio - d_off) > grid_eps:
            raise ValueError(
                f"--evolution-times 中的 T={T}s 不在 step 网格上："
                f"T/dt_per_step={ratio} 不是整数（dt_per_step={dt_per_step:.4f}s）。"
            )
        if d_off <= 0 or d_off % steps_per_kf != 0:
            raise ValueError(
                f"--evolution-times 中的 T={T}s 对应 desired_off={d_off}，"
                f"必须是 steps_per_kf={steps_per_kf} 的正整数倍（dt_per_step={dt_per_step:.4f}s）。"
            )
        desired_offs[T] = d_off

    # === model ===
    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = OnlineNcdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=model_cfg.get("decoder_init_scale", 1.0e-3),
        use_fast_residual=bool(model_cfg.get("use_fast_residual", True)),
        func_g_inner_dim=model_cfg.get("func_g_inner_dim", 32),
        func_g_body_dilations=tuple(model_cfg.get("func_g_body_dilations", [1, 2, 3])),
        func_g_gn_groups=int(model_cfg.get("func_g_gn_groups", 8)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
        amp_fp16=bool(eval_cfg.get("amp_fp16", False)),
        solver_variant=args.solver,
    ).to(device)
    if args.solver == "euler":
        print(f"[solver] {args.solver} (next-fast only, 单次 func_g 求值)")
    else:
        print(f"[solver] {args.solver}")
    load_checkpoint_for_eval(args.checkpoint, model=model, strict=False)
    model.eval()

    # === keyframe resolver / sweep origins ===
    nusc_dataroot = resolve_path(root_path, args.nusc_dataroot)
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)
    keyframe_resolver = NuScenesKeyFrameResolver(
        dataroot=nusc_dataroot,
        version=args.nusc_version,
        sweep_info_path=sweep_info_path,
    )
    enable_rayiou = not args.no_rayiou
    origins_by_token: dict[str, Any] = {}
    if enable_rayiou:
        print(f"[rayiou] 加载 lidar origins: {sweep_info_path}")
        origins_by_token = load_origins_from_sweep_pkl(sweep_info_path)
        print(f"[rayiou] 共 {len(origins_by_token)} 个 token 的 origin")

    num_classes = int(data_cfg["num_classes"])
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    gt_mask_key = data_cfg.get("gt_mask_key", "mask_camera")
    class_names = MetricMiouOcc3D(num_classes=num_classes).class_names

    # === 推理 + 收集 buckets ===
    # buckets[T] -> list of dict(pred uint8 (X,Y,Z), token, scene, abs_step, rss, is_fallback)
    buckets: dict[float, list[dict[str, Any]]] = {T: [] for T in evolution_times}
    no_scene_count = 0
    no_frame_tokens_count = 0
    skip_no_keyframe_token: dict[float, int] = defaultdict(int)
    fallback_count: dict[float, int] = defaultdict(int)
    real_count: dict[float, int] = defaultdict(int)

    total_steps = len(loader)
    iterator = (
        progressbar.progressbar(loader, max_value=total_steps, prefix="[eval evol-times] ")
        if progressbar is not None else loader
    )
    log_interval = int(eval_cfg.get("log_interval", 20))

    with torch.inference_mode():
        for batch_idx, sample in enumerate(iterator, start=1):
            sample = move_to_device(sample, device)
            # 前置检查：bs>1 时 batch 内 rss 必须一致，否则 forward_stepwise_eval 会 raise
            rss_tensor = sample.get("rollout_start_step", None)
            if rss_tensor is not None and rss_tensor.numel() > 1:
                rss_unique = torch.unique(rss_tensor).tolist()
                if len(rss_unique) > 1:
                    raise RuntimeError(
                        f"batch 内 rollout_start_step 不一致：{rss_tensor.tolist()}；"
                        "本脚本 batch 不支持混合短/长历史样本，请使用 --batch-size 1。"
                    )
            outputs = model.forward_stepwise_eval(
                fast_logits=sample["fast_logits"],
                slow_logits=sample["slow_logits"],
                frame_ego2global=sample["frame_ego2global"],
                frame_timestamps=sample.get("frame_timestamps", None),
                frame_dt=sample.get("frame_dt", None),
                rollout_start_step=sample.get("rollout_start_step", None),
            )
            step_logits = cast(torch.Tensor, outputs["step_logits"])  # (B, S, C, X, Y, Z)
            # argmax → uint8 → CPU，省内存
            step_preds = step_logits.argmax(dim=2).to(torch.uint8).cpu().numpy()  # (B, S, X, Y, Z)
            step_indices_arr = cast(torch.Tensor, outputs["step_indices"]).detach().cpu().numpy()  # (S,)
            step_idx_to_local = {int(v): i for i, v in enumerate(step_indices_arr)}

            B = step_preds.shape[0]
            if rss_tensor is None:
                rss_batch = np.zeros((B,), dtype=np.int64)
            else:
                rss_batch = rss_tensor.detach().cpu().numpy()
            meta_list = cast(list[dict[str, Any]], sample["meta"])

            for b in range(B):
                meta = meta_list[b]
                rss_b = int(rss_batch[b])
                scene_name = str(meta.get("scene_name", ""))
                frame_tokens_raw = meta.get("frame_tokens", [])
                frame_tokens = [str(tok) for tok in frame_tokens_raw] if frame_tokens_raw else []
                if not scene_name:
                    no_scene_count += 1
                    continue
                if not frame_tokens:
                    no_frame_tokens_count += 1
                    continue

                keyframe_steps = keyframe_resolver.resolve_keyframe_steps(frame_tokens)
                # rss 必须落在 keyframe 边界，否则 abs_step = rss + chosen_off 会脱离 keyframe step
                assert rss_b % steps_per_kf == 0, \
                    f"rollout_start_step={rss_b} 不是 steps_per_kf={steps_per_kf} 的整数倍"
                max_off = (num_frames - 1) - rss_b  # >= 0

                for T in evolution_times:
                    desired_off = desired_offs[T]
                    # strict 模式：desired_off 超出可达范围（含 max_off==0）一律跳过
                    if args.strict_no_fallback and desired_off > max_off:
                        continue
                    if max_off == 0:
                        # h=0 退化分支：forward_stepwise_eval 只返回 1 个 step（slow_logits）
                        abs_step = num_frames - 1
                        assert abs_step in step_idx_to_local, \
                            f"h=0 退化分支应包含 abs_step={abs_step}，实际 step_indices={step_indices_arr}"
                        local_idx = step_idx_to_local[abs_step]
                        is_fallback = True
                    else:
                        chosen_off = min(desired_off, max_off)
                        abs_step = rss_b + chosen_off
                        assert abs_step in step_idx_to_local, \
                            f"abs_step={abs_step} 不在 step_indices={step_indices_arr}（rss={rss_b}）"
                        local_idx = step_idx_to_local[abs_step]
                        is_fallback = (chosen_off != desired_off)

                    sample_token = keyframe_steps.get(abs_step, None)
                    if sample_token is None:
                        skip_no_keyframe_token[T] += 1
                        continue

                    # 显式拷贝避免持有整块 step_preds tensor 的 base view，
                    # 让每条 bucket item 真正只占 ~640KB
                    pred_arr = np.ascontiguousarray(step_preds[b, local_idx])  # (X, Y, Z) uint8
                    buckets[T].append({
                        "pred": pred_arr,
                        "token": sample_token,
                        "scene": scene_name,
                    })
                    if is_fallback:
                        fallback_count[T] += 1
                    else:
                        real_count[T] += 1

            if progressbar is None and (batch_idx % log_interval == 0 or batch_idx == total_steps):
                print(f"[eval evol-times] batch={batch_idx}/{total_steps}")

    print(f"[meta] no_scene={no_scene_count} no_frame_tokens={no_frame_tokens_count}")
    for T in evolution_times:
        print(f"[bucket T={T}s] collected={len(buckets[T])} "
              f"real={real_count[T]} fallback={fallback_count[T]} "
              f"no_keyframe_token={skip_no_keyframe_token[T]}")

    # === 逐桶评估 mIoU + RayIoU ===
    # GT 缓存：(scene, token) → (semantics, mask)，跨桶共享
    gt_cache: dict[tuple[str, str], tuple[np.ndarray | None, np.ndarray | None]] = {}

    def load_gt(scene: str, token: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        key = (scene, token)
        if key not in gt_cache:
            gt_path = os.path.join(gt_root, scene, token, "labels.npz")
            if not os.path.exists(gt_path):
                gt_cache[key] = (None, None)
            else:
                npz = load_labels_npz(gt_path)
                sem = npz["semantics"]
                mask = npz.get(gt_mask_key, np.ones(sem.shape, dtype=np.float32))
                gt_cache[key] = (sem, mask)
        return gt_cache[key]

    summary: dict[str, Any] = {}
    for T in evolution_times:
        items = buckets[T]
        print(f"\n[bucket T={T}s] 收集到 {len(items)} 个样本，开始评估 mIoU + RayIoU")
        metric = MetricMiouOcc3D(
            num_classes=num_classes,
            use_image_mask=True,
            use_lidar_mask=False,
        )
        sem_pred_list: list[np.ndarray] = []
        sem_gt_list: list[np.ndarray] = []
        lidar_origin_list: list[Any] = []
        missing_gt = 0
        missing_origin = 0

        for it in items:
            sem, mask = load_gt(it["scene"], it["token"])
            if sem is None or mask is None:
                missing_gt += 1
                continue
            metric.add_batch(
                semantics_pred=it["pred"],
                semantics_gt=sem,
                mask_lidar=None,
                mask_camera=mask,
            )
            if enable_rayiou:
                origin = origins_by_token.get(it["token"], None)
                if origin is None:
                    missing_origin += 1
                    continue
                sem_pred_list.append(it["pred"])
                sem_gt_list.append(sem)
                lidar_origin_list.append(origin)

        if metric.cnt > 0:
            miou = float(metric.count_miou(verbose=False))
            miou_d = float(metric.count_miou_d(verbose=False))
            per_class = np.nan_to_num(metric.get_per_class_iou(), nan=0.0).tolist()
            print(f"[bucket T={T}s] miou={miou:.2f} miou_d={miou_d:.2f} num={metric.cnt}")
            for name, value in zip(class_names, per_class):
                print(f"  {name}: {float(value):.2f}")
        else:
            miou, miou_d, per_class = float("nan"), float("nan"), []
            print(f"[bucket T={T}s] no samples")

        rayiou_result: dict[str, Any] | None = None
        if enable_rayiou and len(sem_pred_list) > 0:
            rayiou_result = calc_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list)
            print(f"[bucket T={T}s] RayIoU={rayiou_result['RayIoU']:.4f} "
                  f"@1={rayiou_result['RayIoU@1']:.4f} "
                  f"@2={rayiou_result['RayIoU@2']:.4f} "
                  f"@4={rayiou_result['RayIoU@4']:.4f} "
                  f"num={rayiou_result.get('num_samples', len(sem_pred_list))}")
        elif enable_rayiou:
            print(f"[bucket T={T}s] RayIoU no samples")

        summary[str(T)] = {
            "num_collected": int(len(items)),
            "num_real": int(real_count[T]),
            "num_fallback": int(fallback_count[T]),
            "num_miou_samples": int(metric.cnt),
            "miou": _to_json_number(miou),
            "miou_d": _to_json_number(miou_d),
            "per_class_iou": [float(v) for v in per_class],
            "class_names": class_names,
            "rayiou": _to_json_safe(rayiou_result) if rayiou_result is not None else None,
            "missing_gt": int(missing_gt),
            "missing_origin": int(missing_origin),
        }
        # 释放本桶的 preds 引用，下一桶进入前回收
        items.clear()

    if args.dump_json:
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "evolution_times": evolution_times,
            "keyframe_interval_sec": keyframe_interval,
            "dt_per_step": dt_per_step,
            "history_keyframes": history_keyframes,
            "num_frames": num_frames,
            "steps_per_kf": steps_per_kf,
            "strict_no_fallback": bool(args.strict_no_fallback),
            "exclude_short_history": bool(args.exclude_short_history),
            "buckets": summary,
            "meta": {
                "no_scene": int(no_scene_count),
                "no_frame_tokens": int(no_frame_tokens_count),
                "skip_no_keyframe_token": {str(k): int(v) for k, v in skip_no_keyframe_token.items()},
            },
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
