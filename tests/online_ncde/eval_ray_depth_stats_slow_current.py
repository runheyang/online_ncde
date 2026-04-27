#!/usr/bin/env python3
"""按 ray 统计「慢系统当前帧 logits」与 fast baseline 的 first-hit 深度误差和 RayIoU。

与 eval_ray_depth_stats.py 的差别：
  - 不跑 NCDE 对齐器；不需要 checkpoint。
  - slow_logit_path 重定向到 **当前 keyframe** token（即
    `slow_logit_root/{scene}/{current_token}/logits.npz`），
    而不是 canonical pkl 默认指向的 2s 前 keyframe（slow_sample_token）。
  - 评估慢系统在当前帧时刻能输出什么——给 NCDE 演化的「上界」。

用法示例：
    python tests/online_ncde/eval_ray_depth_stats_slow_current.py \
        --config configs/online_ncde/.../base.yaml --limit 50

统计内容（与 eval_ray_depth_stats.py 一致，只是把 aligned 换成 slow_current）：
  1. 每条 ray 的 GT/Pred first-hit depth、abs 深度误差
  2. 预测比 GT 更近/更远的比例
  3. 深度偏差超过阈值的比例
  4. 分 mask 内/外、近距/远距 统计
  5. 分区域 RayIoU 对比（fast(last) vs slow_current）
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.labels_io import load_labels_npz  # noqa: E402
from online_ncde.losses import resize_labels_and_mask_to_logits  # noqa: E402
from online_ncde.metrics import (  # noqa: E402
    OCC3D_DYNAMIC_OBJECT_IDX,
    MetricMiouOcc3D,
    apply_free_threshold,
)

# --- DVR 相关常量（与 ray_metrics.py 保持一致）---
_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
]
FREE_ID = len(occ_class_names) - 1
DYNAMIC_IDS = set(OCC3D_DYNAMIC_OBJECT_IDX)


# ---------------------------------------------------------------------------
# DVR 加载（延迟编译）
# ---------------------------------------------------------------------------

_dvr = None

def get_dvr():
    global _dvr
    if _dvr is None:
        from torch.utils.cpp_extension import load
        _dvr_dir = os.path.join(str(ROOT), "src", "online_ncde", "ops", "dvr")
        _dvr = load(
            "dvr",
            sources=[
                os.path.join(_dvr_dir, "dvr.cpp"),
                os.path.join(_dvr_dir, "dvr.cu"),
            ],
            verbose=True,
            extra_cuda_cflags=['-allow-unsupported-compiler'],
        )
    return _dvr


def generate_lidar_rays() -> np.ndarray:
    """生成 lidar 射线方向（与 ray_metrics.py 一致）。"""
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch in pitch_angles:
        for az_deg in np.arange(0, 360, 1):
            az = np.deg2rad(az_deg)
            lidar_rays.append((
                np.cos(pitch) * np.cos(az),
                np.cos(pitch) * np.sin(az),
                np.sin(pitch),
            ))
    return np.array(lidar_rays, dtype=np.float32)


# ---------------------------------------------------------------------------
# 单样本 ray casting
# ---------------------------------------------------------------------------

def raycast_one_sample(
    sem_pred: np.ndarray,
    lidar_rays: torch.Tensor,
    output_origin: torch.Tensor,
) -> dict[str, np.ndarray]:
    """对一个 volume 做 ray casting，返回每条 ray 的 first-hit 信息。"""
    dvr = get_dvr()
    T = output_origin.shape[1]

    occ = copy.deepcopy(sem_pred)
    occ[sem_pred < FREE_ID] = 1
    occ[sem_pred == FREE_ID] = 0
    occ = torch.from_numpy(occ).permute(2, 1, 0)  # (Z, Y, X)
    occ = occ[None, None, :].contiguous().float()   # (1, 1, Z, Y, X)

    offset = torch.Tensor(_pc_range[:3])[None, None, :]
    scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]
    lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])

    all_class, all_dist, all_coord = [], [], []
    for t in range(T):
        lidar_origin = output_origin[:, t:t + 1, :]
        lidar_endpts = lidar_rays[None] + lidar_origin
        origin_render = ((lidar_origin - offset) / scaler).float()
        points_render = ((lidar_endpts - offset) / scaler).float()

        with torch.no_grad():
            pred_dist, _, coord_index = dvr.render_forward(
                occ.cuda(), origin_render.cuda(), points_render.cuda(),
                lidar_tindex.cuda(), [1, 16, 200, 200], "test",
            )
            pred_dist *= _voxel_size

        coord_index = coord_index[0, :, :].int().cpu()
        labels = torch.from_numpy(
            sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]]
        )
        all_class.append(labels.numpy())
        all_dist.append(pred_dist[0].cpu().numpy())
        all_coord.append(coord_index.numpy())

    return {
        "class": np.concatenate(all_class, axis=0),
        "dist": np.concatenate(all_dist, axis=0),
        "coord": np.concatenate(all_coord, axis=0),
    }


# ---------------------------------------------------------------------------
# 深度误差统计累加器
# ---------------------------------------------------------------------------

class HitNoHitCounter:
    """GT hit/no-hit × Pred hit/no-hit 四格表统计。"""

    def __init__(self) -> None:
        self.gt_hit_pred_hit: int = 0
        self.gt_hit_pred_nohit: int = 0
        self.gt_nohit_pred_hit: int = 0
        self.gt_nohit_pred_nohit: int = 0

    def update(self, gt_hit: np.ndarray, pred_hit: np.ndarray) -> None:
        self.gt_hit_pred_hit += int((gt_hit & pred_hit).sum())
        self.gt_hit_pred_nohit += int((gt_hit & ~pred_hit).sum())
        self.gt_nohit_pred_hit += int((~gt_hit & pred_hit).sum())
        self.gt_nohit_pred_nohit += int((~gt_hit & ~pred_hit).sum())

    @property
    def total(self) -> int:
        return (self.gt_hit_pred_hit + self.gt_hit_pred_nohit
                + self.gt_nohit_pred_hit + self.gt_nohit_pred_nohit)

    @property
    def gt_hit_total(self) -> int:
        return self.gt_hit_pred_hit + self.gt_hit_pred_nohit

    @property
    def gt_nohit_total(self) -> int:
        return self.gt_nohit_pred_hit + self.gt_nohit_pred_nohit

    @property
    def pred_hit_rate(self) -> float:
        return (self.gt_hit_pred_hit + self.gt_nohit_pred_hit) / max(self.total, 1)

    @property
    def miss_rate(self) -> float:
        return self.gt_hit_pred_nohit / max(self.gt_hit_total, 1)

    @property
    def false_hit_rate(self) -> float:
        return self.gt_nohit_pred_hit / max(self.gt_nohit_total, 1)


class RayDepthStats:
    """累计 per-ray 深度误差统计（含 signed error）。"""

    def __init__(self, depth_thresholds: list[float] = [1.0, 2.0, 4.0]) -> None:
        self.depth_thresholds = depth_thresholds
        self.total_rays: int = 0
        self.abs_err_sum: float = 0.0
        self.signed_err_sum: float = 0.0
        self.signed_errs: list[np.ndarray] = []
        self.closer_count: int = 0
        self.farther_count: int = 0
        self.exact_count: int = 0
        self.exceed_count: dict[float, int] = {t: 0 for t in depth_thresholds}
        self.num_classes = len(occ_class_names)
        self.gt_cnt = np.zeros(self.num_classes)
        self.pred_cnt = np.zeros(self.num_classes)
        self.tp_cnt = np.zeros([len(depth_thresholds), self.num_classes])

    def update(
        self,
        pred_class: np.ndarray,
        pred_dist: np.ndarray,
        gt_class: np.ndarray,
        gt_dist: np.ndarray,
    ) -> None:
        n = len(gt_class)
        if n == 0:
            return
        self.total_rays += n

        signed_err = pred_dist - gt_dist
        abs_err = np.abs(signed_err)
        self.abs_err_sum += float(abs_err.sum())
        self.signed_err_sum += float(signed_err.sum())
        self.signed_errs.append(signed_err.astype(np.float32))
        self.closer_count += int((signed_err < 0).sum())
        self.farther_count += int((signed_err > 0).sum())
        self.exact_count += int((signed_err == 0).sum())

        for t in self.depth_thresholds:
            self.exceed_count[t] += int((abs_err >= t).sum())

        self.gt_cnt += np.bincount(gt_class.astype(np.int64), minlength=self.num_classes)
        self.pred_cnt += np.bincount(pred_class.astype(np.int64), minlength=self.num_classes)
        for j, t in enumerate(self.depth_thresholds):
            tp_mask = (pred_class == gt_class) & (abs_err < t)
            tp_bincount = np.bincount(gt_class[tp_mask].astype(np.int64), minlength=self.num_classes)
            self.tp_cnt[j] += tp_bincount

    @property
    def mean_abs_err(self) -> float:
        return self.abs_err_sum / max(self.total_rays, 1)

    @property
    def mean_signed_err(self) -> float:
        return self.signed_err_sum / max(self.total_rays, 1)

    @property
    def median_signed_err(self) -> float:
        if not self.signed_errs:
            return 0.0
        return float(np.median(np.concatenate(self.signed_errs)))

    @property
    def closer_ratio(self) -> float:
        return self.closer_count / max(self.total_rays, 1)

    @property
    def farther_ratio(self) -> float:
        return self.farther_count / max(self.total_rays, 1)

    def exceed_ratio(self, threshold: float) -> float:
        return self.exceed_count[threshold] / max(self.total_rays, 1)

    def per_class_rayiou_at(self, threshold_idx: int) -> np.ndarray:
        iou = self.tp_cnt[threshold_idx] / np.maximum(
            self.gt_cnt + self.pred_cnt - self.tp_cnt[threshold_idx], 1
        )
        return iou[:-1]

    def rayiou_at(self, threshold_idx: int) -> float:
        return float(np.nanmean(self.per_class_rayiou_at(threshold_idx)))

    @property
    def rayiou(self) -> float:
        return float(np.mean([self.rayiou_at(j) for j in range(len(self.depth_thresholds))]))


# ---------------------------------------------------------------------------
# 结果输出（列名：Fast Baseline vs Slow Current）
# ---------------------------------------------------------------------------

def print_hit_nohit_comparison(
    fast_c: HitNoHitCounter,
    slow_c: HitNoHitCounter,
    region_name: str,
) -> None:
    print(f"\n{'=' * 15} Hit/No-Hit: {region_name} {'=' * 15}")
    header = f"{'Metric':<36} {'Fast (last)':>14} {'Slow Current':>14} {'Delta':>14}"
    print(header)
    print("-" * len(header))

    def _row(name: str, v_f: float, v_s: float, fmt: str = ".4f") -> None:
        delta = v_s - v_f
        print(f"{name:<36} {v_f:>14{fmt}} {v_s:>14{fmt}} {delta:>+14{fmt}}")

    def _row_int(name: str, v_f: int, v_s: int) -> None:
        print(f"{name:<36} {v_f:>14,} {v_s:>14,} {v_s - v_f:>+14,}")

    _row_int("Total rays", fast_c.total, slow_c.total)
    _row_int("GT hit & Pred hit", fast_c.gt_hit_pred_hit, slow_c.gt_hit_pred_hit)
    _row_int("GT hit & Pred no-hit", fast_c.gt_hit_pred_nohit, slow_c.gt_hit_pred_nohit)
    _row_int("GT no-hit & Pred hit", fast_c.gt_nohit_pred_hit, slow_c.gt_nohit_pred_hit)
    _row_int("GT no-hit & Pred no-hit", fast_c.gt_nohit_pred_nohit, slow_c.gt_nohit_pred_nohit)
    _row("Pred hit rate", fast_c.pred_hit_rate, slow_c.pred_hit_rate)
    _row("Miss rate (GT-hit)", fast_c.miss_rate, slow_c.miss_rate)
    _row("False-hit rate (GT-empty)", fast_c.false_hit_rate, slow_c.false_hit_rate)


def print_depth_comparison(
    fast_stats: RayDepthStats,
    slow_stats: RayDepthStats,
    region_name: str,
) -> None:
    header = f"{'Metric':<36} {'Fast (last)':>14} {'Slow Current':>14} {'Delta':>14}"
    sep = "-" * len(header)
    print(f"\n{'=' * 20} {region_name} {'=' * 20}")
    print(header)
    print(sep)

    def _row(name: str, v_f: float, v_s: float, fmt: str = ".4f") -> None:
        delta = v_s - v_f
        print(f"{name:<36} {v_f:>14{fmt}} {v_s:>14{fmt}} {delta:>+14{fmt}}")

    def _row_int(name: str, v_f: int, v_s: int) -> None:
        print(f"{name:<36} {v_f:>14,} {v_s:>14,} {v_s - v_f:>+14,}")

    _row_int("Total rays", fast_stats.total_rays, slow_stats.total_rays)
    _row("Mean signed err (pred-gt)", fast_stats.mean_signed_err, slow_stats.mean_signed_err)
    _row("Median signed err (pred-gt)", fast_stats.median_signed_err, slow_stats.median_signed_err)
    _row("Mean |depth_pred - depth_gt|", fast_stats.mean_abs_err, slow_stats.mean_abs_err)
    _row("Pred closer ratio (err<0)", fast_stats.closer_ratio, slow_stats.closer_ratio)
    _row("Pred farther ratio (err>0)", fast_stats.farther_ratio, slow_stats.farther_ratio)
    for t in fast_stats.depth_thresholds:
        _row(f"Exceed ratio (|err|>={t:.0f}m)", fast_stats.exceed_ratio(t), slow_stats.exceed_ratio(t))
    for j, t in enumerate(fast_stats.depth_thresholds):
        _row(f"RayIoU@{t:.0f}", fast_stats.rayiou_at(j), slow_stats.rayiou_at(j))
    _row("RayIoU (mean)", fast_stats.rayiou, slow_stats.rayiou)


# ---------------------------------------------------------------------------
# sweep pkl 解析（与 eval_ray_depth_stats.py 一致）
# ---------------------------------------------------------------------------

def resolve_sweep_pkl(sweep_pkl_arg: str, cfg: dict) -> str:
    if sweep_pkl_arg:
        p = Path(sweep_pkl_arg)
        return str(p if p.is_absolute() else (ROOT / p).resolve())
    info_path = cfg["data"].get("val_info_path", cfg["data"]["info_path"])
    info_abs = Path(info_path) if Path(info_path).is_absolute() else (ROOT / info_path).resolve()
    with open(info_abs, "rb") as f:
        payload = pickle.load(f)
    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    src = meta.get("source_info_path", "")
    if src and Path(src).exists() and "sweep" in Path(src).stem:
        return src
    if src and Path(src).exists():
        with open(src, "rb") as f:
            inner = pickle.load(f)
        inner_meta = inner.get("metadata", {}) if isinstance(inner, dict) else {}
        inner_src = inner_meta.get("source_info_path", "")
        if inner_src and Path(inner_src).exists() and "sweep" in Path(inner_src).stem:
            return inner_src
    raise FileNotFoundError(
        f"无法自动推断 sweep pkl，请用 --sweep-pkl 指定。\n  info_path={info_abs}"
    )


# ---------------------------------------------------------------------------
# 当前帧 slow logits 加载（覆盖 slow_logit_path）
# ---------------------------------------------------------------------------

def _make_logit_rel_path(scene_name: str, token: str) -> str:
    """与 gen_online_ncde_canonical_infos.make_logit_rel_path 一致。"""
    return str(Path(scene_name) / token / "logits.npz")


def _load_fast_last_frame(logits_loader, info: dict) -> torch.Tensor:
    """只解码 frame_rel_paths[-1] 一帧 fast logits（在 CPU 上）。"""
    info_one = dict(info)
    info_one["frame_rel_paths"] = [info["frame_rel_paths"][-1]]
    return logits_loader.load_fast_logits(info_one, torch.device("cpu"))[0]


def _load_slow_at_current(
    logits_loader, info: dict, scene_name: str, current_token: str
) -> torch.Tensor:
    """把 slow_logit_path 重定向到当前 keyframe 后解码（在 CPU 上）。"""
    info_override = dict(info)
    info_override["slow_logit_path"] = _make_logit_rel_path(scene_name, current_token)
    return logits_loader.load_slow_logits(info_override, torch.device("cpu"))


# ---------------------------------------------------------------------------
# 极简 Dataset：每个样本只加载 fast(last) + slow(current) + GT
# ---------------------------------------------------------------------------

class SlowCurrentEvalDataset(Dataset):
    """绕开 Occ3DOnlineNcdeDataset 的多帧/监督/sidecar 加载，只取所需 3 项。

    比原 dataset 快约 17×（13 帧 fast + 4 帧 sup + ray sidecar → 1 帧 fast + 1 帧 slow + 1 帧 GT）。
    """

    def __init__(
        self,
        infos: list[dict],
        logits_loader,
        gt_root_abs: str,
        grid_size: tuple,
        gt_mask_key: str,
    ) -> None:
        self.infos = infos
        self.logits_loader = logits_loader
        self.gt_root_abs = gt_root_abs
        self.grid_size = grid_size
        self.gt_mask_key = gt_mask_key

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, idx: int) -> dict:
        info = self.infos[idx]
        scene_name = str(info.get("scene_name", ""))
        token = str(info.get("token", ""))

        # fast 最后一帧（=当前 keyframe）
        fast_last = _load_fast_last_frame(self.logits_loader, info)

        # slow 当前 keyframe（覆盖 path）。文件不存在时返回 None 占位。
        slow_current = None
        try:
            slow_current = _load_slow_at_current(
                self.logits_loader, info, scene_name, token
            )
        except FileNotFoundError:
            pass

        # GT
        gt_path = os.path.join(self.gt_root_abs, scene_name, token, "labels.npz")
        gt_npz = load_labels_npz(gt_path)
        gt_labels = torch.from_numpy(gt_npz["semantics"].astype("int64"))
        mask_np = gt_npz.get(self.gt_mask_key, None)
        if mask_np is None:
            mask_np = np.ones(self.grid_size, dtype=np.float32)
        gt_mask = torch.from_numpy(mask_np.astype("float32"))

        return {
            "fast_last": fast_last,
            "slow_current": slow_current,
            "gt_labels": gt_labels,
            "gt_mask": gt_mask,
            "token": token,
            "scene_name": scene_name,
        }


def slow_current_collate(batch: list[dict]) -> dict:
    """支持 slow_current=None 的 batch 拼接。"""
    fast = torch.stack([b["fast_last"] for b in batch], dim=0)
    gt = torch.stack([b["gt_labels"] for b in batch], dim=0)
    mask = torch.stack([b["gt_mask"] for b in batch], dim=0)
    # slow 可能 None，保持 list；后面 GPU 端单独搬
    slow = [b["slow_current"] for b in batch]
    return {
        "fast_last": fast,
        "slow_current": slow,
        "gt_labels": gt,
        "gt_mask": mask,
        "tokens": [b["token"] for b in batch],
        "scene_names": [b["scene_name"] for b in batch],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估慢系统当前帧 logits 的 RayIoU 分箱（无需 NCDE 模型）"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--sweep-pkl", default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
                        help="sweep pkl 路径（相对于项目根目录）")
    parser.add_argument(
        "--val-scene-count",
        type=int,
        default=0,
        help="仅评估按 seed=0 打乱后前 N 个 scene 的样本（0=使用全部 scene）",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--exclude-short-history",
        action="store_true",
        help="只评估满足 config.min_history_completeness 的完整历史样本；"
             "默认含全部短历史样本（min_history_completeness=0）。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    free_index = data_cfg["free_index"]
    grid_size = tuple(data_cfg["grid_size"])
    gt_mask_key = data_cfg["gt_mask_key"]

    # --- 加载 logits_loader 和 infos ---
    logits_loader = build_logits_loader(data_cfg, cfg["root_path"])
    info_path = data_cfg.get("val_info_path", data_cfg["info_path"])
    info_abs = resolve_path(cfg["root_path"], info_path)
    with open(info_abs, "rb") as f:
        payload = pickle.load(f)
    raw_infos = payload["infos"] if isinstance(payload, dict) else payload
    infos = [info for info in raw_infos if info.get("valid", True)]

    min_hc = int(data_cfg.get("min_history_completeness", 4)) if args.exclude_short_history else 0
    if min_hc > 0:
        infos = [
            info for info in infos
            if int(info.get("history_completeness", info.get("history_keyframes", 0))) >= min_hc
        ]
    print(f"[eval] min_history_completeness={min_hc}，过滤后样本数={len(infos)}")

    if args.val_scene_count > 0:
        scene_names = [info.get("scene_name", "") for info in infos]
        unique_scenes = sorted({name for name in scene_names if name})
        if not unique_scenes:
            raise ValueError("未找到有效 scene_name，无法按 scene 划分评估集")
        rng = random.Random(0)
        rng.shuffle(unique_scenes)
        val_scene_count = min(args.val_scene_count, len(unique_scenes))
        val_scene_set = set(unique_scenes[:val_scene_count])
        infos = [info for info in infos if info.get("scene_name", "") in val_scene_set]
        if not infos:
            raise ValueError("验证集 scene 为空，请检查 --val-scene-count 或数据")
        print(f"[eval] --val-scene-count={args.val_scene_count}，"
              f"选中 {val_scene_count}/{len(unique_scenes)} 个 scene，"
              f"共 {len(infos)} 个样本")

    if args.limit > 0:
        infos = infos[: args.limit]
        print(f"[eval] --limit={args.limit}，实际评估 {len(infos)} 个样本")

    gt_root_abs = resolve_path(cfg["root_path"], data_cfg["gt_root"])
    dataset = SlowCurrentEvalDataset(
        infos=infos,
        logits_loader=logits_loader,
        gt_root_abs=gt_root_abs,
        grid_size=grid_size,
        gt_mask_key=gt_mask_key,
    )

    num_workers = int(eval_cfg.get("num_workers", 4))
    dl_kwargs: dict = dict(
        batch_size=int(eval_cfg.get("batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=slow_current_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        dl_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(dataset, **dl_kwargs)

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    free_conf_thresh = eval_cfg.get("free_conf_thresh", None)

    # --- 阶段 1：推理收集 predictions + mIoU 统计 ---
    print(f"[phase1] 加载 slow_current logits + fast(last) + GT (num_workers={num_workers})...")
    predictions: list[dict] = []
    total_steps = len(loader)
    t0 = time.time()

    metric_fast = MetricMiouOcc3D(
        num_classes=data_cfg["num_classes"],
        use_image_mask=True,
        use_lidar_mask=False,
    )
    metric_slow = MetricMiouOcc3D(
        num_classes=data_cfg["num_classes"],
        use_image_mask=True,
        use_lidar_mask=False,
    )

    missing_slow_current = 0
    with torch.no_grad():
        for step, sample in enumerate(loader, start=1):
            fast_last = sample["fast_last"].to(device, non_blocking=True)  # (B, C, X, Y, Z)
            gt_labels = sample["gt_labels"].to(device, non_blocking=True)
            gt_mask = sample["gt_mask"].to(device, non_blocking=True)
            slow_list = sample["slow_current"]
            tokens = sample["tokens"]

            gt_labels_r, gt_mask_r = resize_labels_and_mask_to_logits(
                fast_last, gt_labels, gt_mask
            )

            if free_conf_thresh is not None:
                pred_fast = apply_free_threshold(fast_last, free_index, free_conf_thresh)
            else:
                pred_fast = fast_last.argmax(dim=1)

            pred_fast_np = pred_fast.cpu().numpy()
            gt_np = gt_labels_r.cpu().numpy()
            mask_np = gt_mask_r.cpu().numpy() if gt_mask_r is not None else None

            for b in range(pred_fast_np.shape[0]):
                slow_b_cpu = slow_list[b]
                if slow_b_cpu is None:
                    missing_slow_current += 1
                    continue
                slow_b = slow_b_cpu.unsqueeze(0).to(device, non_blocking=True)
                if free_conf_thresh is not None:
                    pred_slow = apply_free_threshold(slow_b, free_index, free_conf_thresh)
                else:
                    pred_slow = slow_b.argmax(dim=1)
                pred_slow_np_b = pred_slow.cpu().numpy()[0].astype(np.uint8)

                mask_b = mask_np[b] if mask_np is not None else None
                metric_fast.add_batch(
                    semantics_pred=pred_fast_np[b],
                    semantics_gt=gt_np[b],
                    mask_lidar=None,
                    mask_camera=mask_b,
                )
                metric_slow.add_batch(
                    semantics_pred=pred_slow_np_b,
                    semantics_gt=gt_np[b],
                    mask_lidar=None,
                    mask_camera=mask_b,
                )
                predictions.append({
                    "pred_fast": pred_fast_np[b].astype(np.uint8),
                    "pred_slow_current": pred_slow_np_b,
                    "gt": gt_np[b].astype(np.uint8),
                    "mask": mask_b,
                    "token": tokens[b],
                })

            if step % 200 == 0 or step == total_steps:
                spd = step / (time.time() - t0) if time.time() > t0 else 0
                print(f"  step={step}/{total_steps}  {spd:.1f} it/s")

    print(f"[phase1] 共收集 {len(predictions)} 个样本"
          + (f"，跳过 {missing_slow_current} 个缺当前帧 slow logits 的样本"
             if missing_slow_current else ""))

    # --- 阶段 1.5：mIoU 对比 ---
    print("\n" + "=" * 20 + " mIoU 对比 " + "=" * 20)
    miou_fast = metric_fast.count_miou(verbose=False)
    miou_d_fast = metric_fast.count_miou_d(verbose=False)
    per_class_fast = np.nan_to_num(metric_fast.get_per_class_iou(), nan=0.0)

    miou_slow = metric_slow.count_miou(verbose=False)
    miou_d_slow = metric_slow.count_miou_d(verbose=False)
    per_class_slow = np.nan_to_num(metric_slow.get_per_class_iou(), nan=0.0)

    header = f"{'Metric':<28} {'Fast (last)':>14} {'Slow Current':>14} {'Delta':>14}"
    print(header)
    print("-" * len(header))
    print(f"{'mIoU':<28} {miou_fast:>14.4f} {miou_slow:>14.4f} {miou_slow - miou_fast:>+14.4f}")
    print(f"{'mIoU_d (dynamic)':<28} {miou_d_fast:>14.4f} {miou_d_slow:>14.4f} {miou_d_slow - miou_d_fast:>+14.4f}")
    print()
    print(f"{'Class':<28} {'Fast (last)':>14} {'Slow Current':>14} {'Delta':>14}")
    print("-" * len(header))
    for name, v_f, v_s in zip(metric_fast.class_names, per_class_fast, per_class_slow):
        print(f"{name:<28} {float(v_f):>14.4f} {float(v_s):>14.4f} {float(v_s) - float(v_f):>+14.4f}")

    # --- 阶段 2：加载 lidar origins ---
    print("[phase2] 加载 lidar origins...")
    sweep_pkl = resolve_sweep_pkl(args.sweep_pkl, cfg)
    from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
    origins_by_token = load_origins_from_sweep_pkl(sweep_pkl)
    print(f"  共 {len(origins_by_token)} 个 token 的 origin")

    # --- 阶段 3：逐样本 ray casting + 统计 ---
    print("[phase3] Ray casting + 深度统计...")
    lidar_rays = torch.from_numpy(generate_lidar_rays())

    dist_bins = [(0, 10), (10, 20), (20, 40), (40, float("inf"))]
    regions = [
        "all", "mask_in", "mask_out",
        "d_0_10", "d_10_20", "d_20_40", "d_40_plus",
        "near_static", "near_dynamic",
    ]
    stats: dict[tuple[str, str], RayDepthStats] = {
        (src, region): RayDepthStats()
        for src in ("fast", "slow_current")
        for region in regions
    }
    dist_bin_keys = ["d_0_10", "d_10_20", "d_20_40", "d_40_plus"]

    hn_regions = ["all", "mask_in", "d_0_10", "d_10_20"]
    hn_stats: dict[tuple[str, str], HitNoHitCounter] = {
        (src, region): HitNoHitCounter()
        for src in ("fast", "slow_current")
        for region in hn_regions
    }

    skipped = 0
    t0 = time.time()
    for i, item in enumerate(predictions):
        token = item["token"]
        if token not in origins_by_token:
            skipped += 1
            continue

        lidar_origins = origins_by_token[token]
        gt_vol = np.reshape(item["gt"], [200, 200, 16])
        fast_vol = np.reshape(item["pred_fast"], [200, 200, 16])
        slow_vol = np.reshape(item["pred_slow_current"], [200, 200, 16])
        mask_vol = item["mask"]

        rc_gt = raycast_one_sample(gt_vol, lidar_rays, lidar_origins)
        rc_fast = raycast_one_sample(fast_vol, lidar_rays, lidar_origins)
        rc_slow = raycast_one_sample(slow_vol, lidar_rays, lidar_origins)

        # --- hit/no-hit 统计 ---
        all_gt_hit = rc_gt["class"] != FREE_ID
        all_gt_coord = rc_gt["coord"]

        for src_name, rc_src in [("fast", rc_fast), ("slow_current", rc_slow)]:
            all_pred_hit = rc_src["class"] != FREE_ID

            hn_stats[(src_name, "all")].update(all_gt_hit, all_pred_hit)

            if mask_vol is not None:
                m = mask_vol[all_gt_coord[:, 0], all_gt_coord[:, 1], all_gt_coord[:, 2]].astype(bool)
                if m.any():
                    hn_stats[(src_name, "mask_in")].update(all_gt_hit[m], all_pred_hit[m])

            gt_dist_all = rc_gt["dist"]
            for lo, hi, key in [(0, 10, "d_0_10"), (10, 20, "d_10_20")]:
                bin_m = (gt_dist_all >= lo) & (gt_dist_all < hi)
                if bin_m.any():
                    hn_stats[(src_name, key)].update(all_gt_hit[bin_m], all_pred_hit[bin_m])

        # --- 深度统计：只分析 GT 命中非 free 的 ray ---
        valid = rc_gt["class"] != FREE_ID
        gt_class = rc_gt["class"][valid]
        gt_dist = rc_gt["dist"][valid]
        gt_coord = rc_gt["coord"][valid]

        gt_is_dynamic = np.isin(gt_class, list(DYNAMIC_IDS))
        gt_is_static = ~gt_is_dynamic

        for src_name, rc_src in [("fast", rc_fast), ("slow_current", rc_slow)]:
            pred_class = rc_src["class"][valid]
            pred_dist = rc_src["dist"][valid]

            stats[(src_name, "all")].update(pred_class, pred_dist, gt_class, gt_dist)

            if mask_vol is not None:
                in_mask = mask_vol[gt_coord[:, 0], gt_coord[:, 1], gt_coord[:, 2]].astype(bool)
                out_mask = ~in_mask
                if in_mask.any():
                    stats[(src_name, "mask_in")].update(
                        pred_class[in_mask], pred_dist[in_mask],
                        gt_class[in_mask], gt_dist[in_mask],
                    )
                if out_mask.any():
                    stats[(src_name, "mask_out")].update(
                        pred_class[out_mask], pred_dist[out_mask],
                        gt_class[out_mask], gt_dist[out_mask],
                    )

            for (lo, hi), key in zip(dist_bins, dist_bin_keys):
                bin_mask = (gt_dist >= lo) & (gt_dist < hi)
                if bin_mask.any():
                    stats[(src_name, key)].update(
                        pred_class[bin_mask], pred_dist[bin_mask],
                        gt_class[bin_mask], gt_dist[bin_mask],
                    )

            near_mask = gt_dist < 20.0
            ns = near_mask & gt_is_static
            nd = near_mask & gt_is_dynamic
            if ns.any():
                stats[(src_name, "near_static")].update(
                    pred_class[ns], pred_dist[ns], gt_class[ns], gt_dist[ns],
                )
            if nd.any():
                stats[(src_name, "near_dynamic")].update(
                    pred_class[nd], pred_dist[nd], gt_class[nd], gt_dist[nd],
                )

        if (i + 1) % 200 == 0 or (i + 1) == len(predictions):
            spd = (i + 1) / (time.time() - t0) if time.time() > t0 else 0
            print(f"  sample={i + 1}/{len(predictions)}  {spd:.1f} it/s")

    if skipped:
        print(f"  跳过 {skipped} 个样本（无对应 lidar origin）")

    # --- 阶段 4：输出对比表 ---
    region_display = {
        "all": "全部 ray",
        "mask_in": "mask 内 ray",
        "mask_out": "mask 外 ray",
        "d_0_10": "距离 0-10m",
        "d_10_20": "距离 10-20m",
        "d_20_40": "距离 20-40m",
        "d_40_plus": "距离 40m+",
        "near_static": "近距(<20m) static",
        "near_dynamic": "近距(<20m) dynamic",
    }
    for region in regions:
        fast_s = stats[("fast", region)]
        slow_s = stats[("slow_current", region)]
        if fast_s.total_rays == 0 and slow_s.total_rays == 0:
            continue
        print_depth_comparison(fast_s, slow_s, region_display[region])

    # --- 各类别 RayIoU 对比 ---
    def _print_per_class_rayiou(stats_obj: RayDepthStats, label: str) -> None:
        table = PrettyTable(['Class Names', 'RayIoU@1', 'RayIoU@2', 'RayIoU@4'])
        table.float_format = '.3'
        cls_names = occ_class_names[:-1]
        iou1 = stats_obj.per_class_rayiou_at(0)
        iou2 = stats_obj.per_class_rayiou_at(1)
        iou4 = stats_obj.per_class_rayiou_at(2)
        for i, name in enumerate(cls_names):
            table.add_row([name, iou1[i], iou2[i], iou4[i]],
                          divider=(i == len(cls_names) - 1))
        table.add_row(['MEAN',
                       float(np.nanmean(iou1)),
                       float(np.nanmean(iou2)),
                       float(np.nanmean(iou4))])
        print(f"\n{'=' * 20} 各类别 RayIoU: {label} {'=' * 20}")
        print(table)

    fast_all = stats[("fast", "all")]
    slow_all = stats[("slow_current", "all")]
    if fast_all.total_rays > 0:
        _print_per_class_rayiou(fast_all, "Fast (last frame)")
    if slow_all.total_rays > 0:
        _print_per_class_rayiou(slow_all, "Slow Current")

    # --- hit/no-hit 四格表 ---
    hn_display = {
        "all": "全部 ray",
        "mask_in": "mask 内 ray",
        "d_0_10": "距离 0-10m",
        "d_10_20": "距离 10-20m",
    }
    for region in hn_regions:
        fast_c = hn_stats[("fast", region)]
        slow_c = hn_stats[("slow_current", region)]
        if fast_c.total == 0 and slow_c.total == 0:
            continue
        print_hit_nohit_comparison(fast_c, slow_c, hn_display[region])


if __name__ == "__main__":
    main()
