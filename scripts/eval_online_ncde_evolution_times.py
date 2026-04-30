#!/usr/bin/env python3
"""按推演时长 T 分桶评估 mIoU + RayIoU（start-anchored 版本）。

输入 pkl 必须由 gen_online_ncde_evolve_infos.py 生成（schema_version =
online_ncde_evolve_infos_v1）。每个 sample 是「以某个 keyframe K_j 为 slow 锚 +
往后推到 K_{j+max_evolve}」，max_evolve = min(history_keyframes, scene_end - j)。

每个 sample 跑一次完整 forward，按 T 桶分发 step 输出：
  - desired_step = round(T / dt_per_step)
  - 若 desired_step > num_real_frames - 1（该 sample 推不到 T）：跳过该桶（无 fallback）
  - 否则 pred = step_logits[step_indices == desired_step]，GT 取
    evolve_keyframe_sample_tokens[desired_step // steps_per_kf] 对应的 keyframe GT
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

torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.labels_io import load_labels_npz  # noqa: E402
from online_ncde.data.build_dataset import build_online_ncde_dataset  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D, build_miou_metric  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl  # noqa: E402
from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou  # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


DEFAULT_EVOLUTION_TIMES = "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0"
DEFAULT_KEYFRAME_INTERVAL_SEC = 0.5
EXPECTED_SCHEMA = "online_ncde_evolve_infos_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="配置文件路径")
    p.add_argument("--checkpoint", required=True, help="模型权重路径")
    p.add_argument("--limit", type=int, default=0, help="仅评估前 N 个样本，0 表示全量")
    p.add_argument("--batch-size", type=int, default=1,
                   help="DataLoader batch_size（默认 1）；start-anchored pkl 中所有 sample 的"
                        "rss=0 且 num_output_frames 定长，bs>1 也合法但 num_real_frames 不同的"
                        "sample 进同一 batch 时模型只能跑统一长度（pad 段无影响），bs=1 最稳。")
    p.add_argument("--solver", choices=["heun", "euler"], default="euler",
                   help="ODE 求解器；默认 euler（next-fast 单次求值）")
    p.add_argument("--evolution-times", default=DEFAULT_EVOLUTION_TIMES,
                   help="逗号分隔，单位秒；默认 '0.5,1.0,...,5.0'")
    p.add_argument("--keyframe-interval-sec", type=float, default=DEFAULT_KEYFRAME_INTERVAL_SEC,
                   help="单个 keyframe interval 时长，默认 0.5（NuScenes 2Hz）")
    p.add_argument("--no-rayiou", action="store_true", help="跳过 RayIoU 评估")
    p.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True,
                   help="启用 target-anchored fallback（默认 ON）：每个 keyframe K_target 在 T 桶下若没有"
                        "推演时长正好 T 的 sample（即 target_idx<T*2），fallback 到 start=K_0 的"
                        "sample 在 step target_idx*spi 处的输出（target_idx=0 时取 slow_logits）。"
                        "开启后每桶覆盖全部 ~6017 keyframe。传 --no-fallback 关闭：每桶只评"
                        "『真能推 T 秒』的样本。")
    p.add_argument("--val-info-path", default="",
                   help="覆盖 data.val_info_path（应指向 evolve_infos pkl）")
    p.add_argument("--nusc-dataroot", default="data/nuscenes", help="NuScenes dataroot")
    p.add_argument("--nusc-version", default="v1.0-trainval", help="NuScenes 版本")
    p.add_argument("--sweep-info-path", default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
                   help="sweep pkl 路径（用于 lidar origin 与 keyframe 校验）")
    p.add_argument("--dump-json", default="", help="可选：将结果写入 json")
    return p.parse_args()


def _to_json_number(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def _to_json_safe(obj: Any) -> Any:
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
          f"keyframe_interval={keyframe_interval}s")

    # === dataset ===
    logits_loader = build_logits_loader(data_cfg, root_path)
    info_path = args.val_info_path if args.val_info_path else data_cfg.get(
        "val_info_path", data_cfg["info_path"]
    )
    if args.val_info_path:
        print(f"[eval] --val-info-path 覆盖 -> {info_path}")
    # 校验 schema：必须是 evolve_infos
    info_path_abs = resolve_path(root_path, info_path)
    import pickle
    with open(info_path_abs, "rb") as f:
        _payload = pickle.load(f)
    _meta = _payload.get("metadata", {}) if isinstance(_payload, dict) else {}
    schema_ver = str(_meta.get("schema_version", ""))
    if schema_ver != EXPECTED_SCHEMA:
        raise ValueError(
            f"--val-info-path 的 schema_version={schema_ver!r}，"
            f"本脚本只支持 {EXPECTED_SCHEMA}（由 gen_online_ncde_evolve_infos.py 生成）。"
        )
    del _payload, _meta

    dataset_full = build_online_ncde_dataset(
        data_cfg,
        info_path=info_path,
        root_path=root_path,
        logits_loader=logits_loader,
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
        # evolve_infos 不参与 history-based 过滤（每个 sample 自己的 max_evolve 决定能进哪些桶）
        min_history_completeness=0,
        eval_only_mode=True,
    )
    if args.limit > 0:
        keep = min(args.limit, len(dataset_full))
        dataset: Any = Subset(dataset_full, list(range(keep)))
    else:
        dataset = dataset_full

    num_workers = int(eval_cfg.get("num_workers", 4))
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

    # === 推断 dt_per_step（从 pkl 元信息 / 第一条 info） ===
    info0 = dataset_full.infos[0]
    history_keyframes = int(info0.get("history_keyframes", 10))
    stride = int(data_cfg.get("fast_frame_stride", 1))
    num_output_frames_canonical = int(info0.get("num_output_frames",
                                                history_keyframes * info0.get("steps_per_interval", 1) + 1))
    if (num_output_frames_canonical - 1) % stride != 0:
        raise ValueError(
            f"num_output_frames-1={num_output_frames_canonical-1} 与 fast_frame_stride={stride} 不整除"
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

    # 校验：每个 T 必须落在 keyframe 网格（0.5s 倍数），同时换算到 keyframe distance T_x2
    # （= T / keyframe_interval）。所有内部分桶用 T_x2 整数索引。
    T_x2_to_T: dict[int, float] = {}
    grid_eps = 1e-6
    for T in evolution_times:
        T_x2_f = T / keyframe_interval
        T_x2 = int(round(T_x2_f))
        if abs(T_x2_f - T_x2) > grid_eps:
            raise ValueError(
                f"--evolution-times T={T}s 不在 keyframe 网格上："
                f"T/{keyframe_interval}={T_x2_f} 不是整数。"
            )
        if T_x2 <= 0:
            raise ValueError(f"T={T}s 必须 > 0。")
        if T_x2 > history_keyframes:
            raise ValueError(
                f"T={T}s 对应 T_x2={T_x2} 超出 history_keyframes={history_keyframes}（pkl 不支持该长度）。"
            )
        T_x2_to_T[T_x2] = T
    T_x2_set = set(T_x2_to_T.keys())
    max_T_x2 = max(T_x2_set)
    print(f"[eval] T_x2 集合: {sorted(T_x2_set)} (max={max_T_x2})")
    print(f"[eval] fallback={'ON (target-anchored，每桶 ~6017 个评估单元)' if args.fallback else 'OFF (每桶只评真能推 T 秒的 sample)'}")

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
        solver_variant=args.solver,
    ).to(device)
    if args.solver == "euler":
        print(f"[solver] {args.solver} (next-fast only, 单次 func_g 求值)")
    else:
        print(f"[solver] {args.solver}")
    load_checkpoint_for_eval(args.checkpoint, model=model, strict=False)
    model.eval()

    # === sweep origins for RayIoU ===
    nusc_dataroot = resolve_path(root_path, args.nusc_dataroot)  # noqa: F841 (保留以便将来扩展)
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)
    enable_rayiou = not args.no_rayiou
    origins_by_token: dict[str, Any] = {}
    if enable_rayiou:
        print(f"[rayiou] 加载 lidar origins: {sweep_info_path}")
        origins_by_token = load_origins_from_sweep_pkl(sweep_info_path)
        print(f"[rayiou] 共 {len(origins_by_token)} 个 token 的 origin")

    num_classes = int(data_cfg["num_classes"])
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    gt_mask_key = data_cfg.get("gt_mask_key", "mask_camera")
    class_names = build_miou_metric(num_classes=num_classes).class_names

    # === 推理 + 收集 buckets ===
    # buckets[T] -> list of dict(pred uint8 (X,Y,Z), token, scene, source ∈ {'main', 'fallback'})
    buckets: dict[float, list[dict[str, Any]]] = {T: [] for T in evolution_times}
    skip_too_short: dict[float, int] = defaultdict(int)
    skip_no_gt_token: dict[float, int] = defaultdict(int)
    real_count: dict[float, int] = defaultdict(int)
    fallback_count: dict[float, int] = defaultdict(int)

    total_steps = len(loader)
    iterator = (
        progressbar.progressbar(loader, max_value=total_steps, prefix="[eval evol-times] ")
        if progressbar is not None else loader
    )
    log_interval = int(eval_cfg.get("log_interval", 20))

    with torch.inference_mode():
        for batch_idx, sample in enumerate(iterator, start=1):
            sample = move_to_device(sample, device)
            # bs=1 时按 num_real_frames 截断输入，避免模型在 pad 段做无意义的 ODE step
            # （compute_segment_dt 内部 clamp_min(eps)，pad 段并非真零，会推一点点）。
            # bs>1 时各样本 num_real 可能不同，这里走完整 num_frames，pad 段输出会被 meta
            # 的 num_real_frames 过滤掉，但模型仍会跑满，浪费一些算力。
            B_in = sample["fast_logits"].shape[0]
            fast_in = sample["fast_logits"]
            slow_in = sample["slow_logits"]
            ego_in = sample["frame_ego2global"]
            ts_in = sample.get("frame_timestamps", None)
            dt_in = sample.get("frame_dt", None)
            if B_in == 1:
                meta_b0 = cast(list[dict[str, Any]], sample["meta"])[0]
                n_real = int(meta_b0.get("num_real_frames", fast_in.shape[1]))
                if 0 < n_real < fast_in.shape[1]:
                    fast_in = fast_in[:, :n_real]
                    ego_in = ego_in[:, :n_real]
                    if ts_in is not None:
                        ts_in = ts_in[:, :n_real]
                    if dt_in is not None:
                        dt_in = dt_in[:, :n_real]
            outputs = model.forward_stepwise_eval(
                fast_logits=fast_in,
                slow_logits=slow_in,
                frame_ego2global=ego_in,
                frame_timestamps=ts_in,
                frame_dt=dt_in,
                rollout_start_step=sample.get("rollout_start_step", None),
            )
            step_logits = cast(torch.Tensor, outputs["step_logits"])  # (B, S, C, X, Y, Z)
            step_preds = step_logits.argmax(dim=2).to(torch.uint8).cpu().numpy()  # (B, S, X, Y, Z)
            step_indices_arr = cast(torch.Tensor, outputs["step_indices"]).detach().cpu().numpy()
            step_idx_to_local = {int(v): i for i, v in enumerate(step_indices_arr)}
            # slow_logits.argmax 用作 fallback 路径里 distance=0 (target=K_0) 的预测
            slow_preds = slow_in.argmax(dim=1).to(torch.uint8).cpu().numpy()  # (B, X, Y, Z)

            B = step_preds.shape[0]
            meta_list = cast(list[dict[str, Any]], sample["meta"])

            for b in range(B):
                meta = meta_list[b]
                num_real = int(meta.get("num_real_frames", num_frames))
                max_real_step = num_real - 1  # 该 sample 真实段最末 abs_step（抽帧后坐标）
                ek_step_indices = list(meta.get("evolve_keyframe_step_indices", []))
                ek_sample_tokens = list(meta.get("evolve_keyframe_sample_tokens", []))
                ek_gt_exists = list(meta.get("evolve_keyframe_gt_exists", []))
                scene_name = str(meta.get("scene_name", ""))
                start_kf_idx = int(meta.get("start_keyframe_local_idx", -1))
                max_evolve = int(meta.get("max_evolve_keyframes", -1))

                if not ek_step_indices or not ek_sample_tokens:
                    raise RuntimeError(
                        "meta 缺少 evolve_keyframe_step_indices / sample_tokens；"
                        "请确认 pkl 是 evolve_infos_v1 schema。"
                    )
                if max_evolve < 0 or start_kf_idx < 0:
                    raise RuntimeError(
                        "meta 缺少 max_evolve_keyframes / start_keyframe_local_idx。"
                    )

                def _emit(T: float, dist: int, source: str) -> None:
                    """发射一个 (target=K_{start+dist}, T) 评估单元到桶里。"""
                    if dist >= len(ek_sample_tokens):
                        return
                    if dist < len(ek_gt_exists) and not bool(ek_gt_exists[dist]):
                        skip_no_gt_token[T] += 1
                        return
                    target_token = str(ek_sample_tokens[dist])
                    if not target_token:
                        skip_no_gt_token[T] += 1
                        return
                    if dist == 0:
                        pred = slow_preds[b]
                    else:
                        step = ek_step_indices[dist] if dist < len(ek_step_indices) else (dist * steps_per_kf)
                        step_int = int(step)
                        if step_int not in step_idx_to_local:
                            raise RuntimeError(
                                f"step={step_int} 不在模型 step_indices={step_indices_arr}"
                                f" [scene={scene_name} start_idx={start_kf_idx}]"
                            )
                        local_idx = step_idx_to_local[step_int]
                        pred = step_preds[b, local_idx]
                    buckets[T].append({
                        "pred": np.ascontiguousarray(pred),
                        "token": target_token,
                        "scene": scene_name,
                    })
                    if source == "main":
                        real_count[T] += 1
                    else:
                        fallback_count[T] += 1

                # 主路径：每个 d ∈ [1, max_evolve] 贡献到 T_x2=d 桶
                # （target = K_{start+d}, sample 自身能整步推到 d 个 keyframe）
                for d in range(1, max_evolve + 1):
                    if d not in T_x2_set:
                        continue
                    # 确认 d 对应的 step 在 num_real_frames 内（应该总成立，因为 d ≤ max_evolve）
                    step_d = d * steps_per_kf
                    if step_d > max_real_step:
                        skip_too_short[T_x2_to_T[d]] += 1
                        continue
                    _emit(T_x2_to_T[d], d, "main")

                # fallback 路径：仅 start=K_0 的 sample 触发；为 target_idx ∈ [0, max_evolve] 提供
                # T_x2 > target_idx 桶下的预测。
                # （target_idx=0 时取 slow_logits；target_idx=k>0 时取 step k*spi 输出，
                # 即「以 K_0 为 start 推演 k 个 keyframe 到 K_k」。这就是 K_k 在更长 T 桶下的
                # 「能找到的最早 start 推到 K_k」的输出。）
                if args.fallback and start_kf_idx == 0:
                    for d in range(0, max_evolve + 1):
                        # 主路径覆盖 T_x2 = d；fallback 覆盖 T_x2 > d
                        for T_x2 in range(max(d, 0) + 1, max_T_x2 + 1):
                            if T_x2 not in T_x2_set:
                                continue
                            _emit(T_x2_to_T[T_x2], d, "fallback")

            if progressbar is None and (batch_idx % log_interval == 0 or batch_idx == total_steps):
                print(f"[eval evol-times] batch={batch_idx}/{total_steps}")

    for T in evolution_times:
        print(f"[bucket T={T}s] collected={len(buckets[T])} "
              f"main={real_count[T]} fallback={fallback_count[T]} "
              f"too_short={skip_too_short[T]} no_gt_token={skip_no_gt_token[T]}")

    # === 逐桶评估 mIoU + RayIoU ===
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
        metric = build_miou_metric(
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
            "num_main": int(real_count[T]),
            "num_fallback": int(fallback_count[T]),
            "num_too_short": int(skip_too_short[T]),
            "num_no_gt_token": int(skip_no_gt_token[T]),
            "num_miou_samples": int(metric.cnt),
            "miou": _to_json_number(miou),
            "miou_d": _to_json_number(miou_d),
            "per_class_iou": [float(v) for v in per_class],
            "class_names": class_names,
            "rayiou": _to_json_safe(rayiou_result) if rayiou_result is not None else None,
            "missing_gt": int(missing_gt),
            "missing_origin": int(missing_origin),
        }
        # 释放本桶引用
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
            "schema": EXPECTED_SCHEMA,
            "fallback_enabled": bool(args.fallback),
            "buckets": summary,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
