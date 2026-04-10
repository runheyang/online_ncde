#!/usr/bin/env python3
"""按 ray 统计 fast baseline 与 aligner output 的 first-hit 深度误差，诊断 RayIoU 下降原因。

用法示例：
    python tests/online_ncde/eval_ray_depth_stats.py \
        --checkpoint <ckpt_path> --limit 50

统计内容：
  1. 每条 ray 的 GT/Pred first-hit depth、abs 深度误差
  2. 预测比 GT 更近/更远的比例
  3. 深度偏差超过阈值的比例
  4. 分 mask 内/外、近距/远距 统计
  5. 分区域 RayIoU 对比
"""

from __future__ import annotations

import argparse
import copy
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "utils"))

from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.losses import resize_labels_and_mask_to_logits  # noqa: E402
from online_ncde.metrics import (  # noqa: E402
    OCC3D_DYNAMIC_OBJECT_IDX,
    MetricMiouOcc3D,
    apply_free_threshold,
)
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde_200x200x16 import OnlineNcdeAligner200              # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint  # noqa: E402

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
        import os
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
# 单样本 ray casting：返回 per-ray (class, distance, coord_index)
# ---------------------------------------------------------------------------

def raycast_one_sample(
    sem_pred: np.ndarray,
    lidar_rays: torch.Tensor,
    output_origin: torch.Tensor,
) -> dict[str, np.ndarray]:
    """对一个 volume 做 ray casting，返回每条 ray 的 first-hit 信息。

    Returns:
        dict with:
          "class": (N_rays,) int — first-hit 类别
          "dist":  (N_rays,) float — first-hit 距离
          "coord": (N_rays, 3) int — first-hit 体素坐标 (x,y,z)
    """
    dvr = get_dvr()
    T = output_origin.shape[1]

    # 构建 binary occ volume
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
        lidar_origin = output_origin[:, t:t + 1, :]        # (1, 1, 3)
        lidar_endpts = lidar_rays[None] + lidar_origin      # (1, N, 3)
        origin_render = ((lidar_origin - offset) / scaler).float()
        points_render = ((lidar_endpts - offset) / scaler).float()

        with torch.no_grad():
            pred_dist, _, coord_index = dvr.render_forward(
                occ.cuda(), origin_render.cuda(), points_render.cuda(),
                lidar_tindex.cuda(), [1, 16, 200, 200], "test",
            )
            pred_dist *= _voxel_size

        coord_index = coord_index[0, :, :].int().cpu()  # (N, 3)
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
        """Pred hit 占全部 ray 的比例。"""
        return (self.gt_hit_pred_hit + self.gt_nohit_pred_hit) / max(self.total, 1)

    @property
    def miss_rate(self) -> float:
        """GT hit 中 pred 未 hit 的比例。"""
        return self.gt_hit_pred_nohit / max(self.gt_hit_total, 1)

    @property
    def false_hit_rate(self) -> float:
        """GT no-hit 中 pred hit 的比例。"""
        return self.gt_nohit_pred_hit / max(self.gt_nohit_total, 1)


class RayDepthStats:
    """累计 per-ray 深度误差统计（含 signed error）。"""

    def __init__(self, depth_thresholds: list[float] = [1.0, 2.0, 4.0]) -> None:
        self.depth_thresholds = depth_thresholds
        self.total_rays: int = 0
        self.abs_err_sum: float = 0.0
        self.signed_err_sum: float = 0.0   # sum(pred - gt)
        self.signed_errs: list[np.ndarray] = []  # 收集所有 signed error 用于中位数
        self.closer_count: int = 0     # pred 比 GT 更近 (signed < 0)
        self.farther_count: int = 0    # pred 比 GT 更远 (signed > 0)
        self.exact_count: int = 0
        self.exceed_count: dict[float, int] = {t: 0 for t in depth_thresholds}
        # RayIoU 分子分母
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

        signed_err = pred_dist - gt_dist  # 正 = 预测更远，负 = 预测更近
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

    # --- 汇总指标 ---

    @property
    def mean_abs_err(self) -> float:
        return self.abs_err_sum / max(self.total_rays, 1)

    @property
    def mean_signed_err(self) -> float:
        """正 = 系统性预测更远，负 = 系统性预测更近。"""
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

    def rayiou_at(self, threshold_idx: int) -> float:
        iou = self.tp_cnt[threshold_idx] / np.maximum(
            self.gt_cnt + self.pred_cnt - self.tp_cnt[threshold_idx], 1
        )
        return float(np.nanmean(iou[:-1]))

    @property
    def rayiou(self) -> float:
        return float(np.mean([self.rayiou_at(j) for j in range(len(self.depth_thresholds))]))


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------

def print_hit_nohit_comparison(
    fast_c: HitNoHitCounter,
    aligned_c: HitNoHitCounter,
    region_name: str,
) -> None:
    """打印某个区域的 fast vs aligned hit/no-hit 四格表。"""
    print(f"\n{'=' * 15} Hit/No-Hit: {region_name} {'=' * 15}")
    header = f"{'Metric':<36} {'Fast Baseline':>14} {'Aligner Output':>14} {'Delta':>14}"
    print(header)
    print("-" * len(header))

    def _row(name: str, v_f: float, v_a: float, fmt: str = ".4f") -> None:
        delta = v_a - v_f
        print(f"{name:<36} {v_f:>14{fmt}} {v_a:>14{fmt}} {delta:>+14{fmt}}")

    def _row_int(name: str, v_f: int, v_a: int) -> None:
        print(f"{name:<36} {v_f:>14,} {v_a:>14,} {v_a - v_f:>+14,}")

    _row_int("Total rays", fast_c.total, aligned_c.total)
    _row_int("GT hit & Pred hit", fast_c.gt_hit_pred_hit, aligned_c.gt_hit_pred_hit)
    _row_int("GT hit & Pred no-hit", fast_c.gt_hit_pred_nohit, aligned_c.gt_hit_pred_nohit)
    _row_int("GT no-hit & Pred hit", fast_c.gt_nohit_pred_hit, aligned_c.gt_nohit_pred_hit)
    _row_int("GT no-hit & Pred no-hit", fast_c.gt_nohit_pred_nohit, aligned_c.gt_nohit_pred_nohit)
    _row("Pred hit rate", fast_c.pred_hit_rate, aligned_c.pred_hit_rate)
    _row("Miss rate (GT-hit)", fast_c.miss_rate, aligned_c.miss_rate)
    _row("False-hit rate (GT-empty)", fast_c.false_hit_rate, aligned_c.false_hit_rate)


def print_depth_comparison(
    fast_stats: RayDepthStats,
    aligned_stats: RayDepthStats,
    region_name: str,
) -> None:
    """打印某个区域的 fast vs aligned 深度对比表。"""
    header = f"{'Metric':<36} {'Fast Baseline':>14} {'Aligner Output':>14} {'Delta':>14}"
    sep = "-" * len(header)
    print(f"\n{'=' * 20} {region_name} {'=' * 20}")
    print(header)
    print(sep)

    def _row(name: str, v_f: float, v_a: float, fmt: str = ".4f") -> None:
        delta = v_a - v_f
        print(f"{name:<36} {v_f:>14{fmt}} {v_a:>14{fmt}} {delta:>+14{fmt}}")

    def _row_int(name: str, v_f: int, v_a: int) -> None:
        print(f"{name:<36} {v_f:>14,} {v_a:>14,} {v_a - v_f:>+14,}")

    _row_int("Total rays", fast_stats.total_rays, aligned_stats.total_rays)
    _row("Mean signed err (pred-gt)", fast_stats.mean_signed_err, aligned_stats.mean_signed_err)
    _row("Median signed err (pred-gt)", fast_stats.median_signed_err, aligned_stats.median_signed_err)
    _row("Mean |depth_pred - depth_gt|", fast_stats.mean_abs_err, aligned_stats.mean_abs_err)
    _row("Pred closer ratio (err<0)", fast_stats.closer_ratio, aligned_stats.closer_ratio)
    _row("Pred farther ratio (err>0)", fast_stats.farther_ratio, aligned_stats.farther_ratio)
    for t in fast_stats.depth_thresholds:
        _row(f"Exceed ratio (|err|>={t:.0f}m)", fast_stats.exceed_ratio(t), aligned_stats.exceed_ratio(t))
    for j, t in enumerate(fast_stats.depth_thresholds):
        _row(f"RayIoU@{t:.0f}", fast_stats.rayiou_at(j), aligned_stats.rayiou_at(j))
    _row("RayIoU (mean)", fast_stats.rayiou, aligned_stats.rayiou)


# ---------------------------------------------------------------------------
# sweep pkl 解析（复用 eval_online_ncde.py 的逻辑）
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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 ray 统计 first-hit 深度误差，诊断 RayIoU 下降")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/online_ncde/fast_opusv1t__slow_opusv2l/eval.yaml"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sweep-pkl", default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
                        help="sweep pkl 路径（相对于项目根目录）")
    parser.add_argument("--200x200x16", dest="use_200", action="store_true",
                        help="使用 200x200x16 分辨率模型结构")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    free_index = data_cfg["free_index"]

    # --- 数据集 ---
    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root 和 data.slow_logit_root 为必填项。")
    logits_loader = build_logits_loader(data_cfg, cfg["root_path"])

    dataset = Occ3DOnlineNcdeDataset(
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
        root_path=cfg["root_path"],
        fast_logits_root=fast_logits_root,
        slow_logit_root=slow_logit_root,
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=free_index,
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        slow_noise_std=0.0,
        topk_other_fill_value=data_cfg.get("topk_other_fill_value", -5.0),
        topk_free_fill_value=data_cfg.get("topk_free_fill_value", 5.0),
        fast_logits_variant=data_cfg.get("fast_logits_variant", "topk"),
        slow_logit_variant=data_cfg.get("slow_logit_variant", "topk"),
        full_logits_clamp_min=data_cfg.get("full_logits_clamp_min", None),
        full_topk_k=data_cfg.get("full_topk_k", 3),
        logits_loader=logits_loader,
    )
    if args.limit > 0:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
        print(f"[eval] --limit={args.limit}，实际评估 {len(dataset)} 个样本")

    num_workers = int(eval_cfg.get("num_workers", 4))
    dl_kwargs: dict = dict(
        batch_size=int(eval_cfg.get("batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=online_ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        dl_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(dataset, **dl_kwargs)

    # --- 模型 ---
    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    ModelClass = OnlineNcdeAligner200 if args.use_200 else OnlineNcdeAligner
    model = ModelClass(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=free_index,
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        fast_occ_thresh=data_cfg.get("fast_occ_thresh", 0.25),
        decoder_init_scale=model_cfg.get("decoder_init_scale", 1.0e-3),
        use_fast_residual=bool(model_cfg.get("use_fast_residual", True)),
        func_g_inner_dim=model_cfg.get("func_g_inner_dim", 32),
        func_g_body_dilations=tuple(model_cfg.get("func_g_body_dilations", [1, 2, 3])),
        func_g_gn_groups=int(model_cfg.get("func_g_gn_groups", 8)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
        amp_fp16=bool(eval_cfg.get("amp_fp16", False)),
    ).to(device)
    load_checkpoint(args.checkpoint, model=model, strict=False)
    model.eval()

    free_conf_thresh = eval_cfg.get("free_conf_thresh", None)

    # --- 阶段 1：推理收集 predictions + mIoU 统计 ---
    print("[phase1] 推理收集 predictions...")
    predictions: list[dict] = []
    total_steps = len(loader)
    t0 = time.time()

    # mIoU 累加器（与 Trainer.evaluate 口径一致：use_image_mask=True, use_lidar_mask=False）
    metric_fast = MetricMiouOcc3D(
        num_classes=data_cfg["num_classes"],
        use_image_mask=True,
        use_lidar_mask=False,
    )
    metric_aligned = MetricMiouOcc3D(
        num_classes=data_cfg["num_classes"],
        use_image_mask=True,
        use_lidar_mask=False,
    )

    with torch.no_grad():
        for step, sample in enumerate(loader, start=1):
            sample = move_to_device(sample, device)
            outputs = model(
                fast_logits=sample["fast_logits"],
                slow_logits=sample["slow_logits"],
                frame_ego2global=sample["frame_ego2global"],
                frame_timestamps=sample.get("frame_timestamps", None),
                frame_dt=sample.get("frame_dt", None),
            )
            aligned_logits = outputs["aligned"]
            fast_logits_last = sample["fast_logits"][:, -1]

            gt_labels, gt_mask = resize_labels_and_mask_to_logits(
                aligned_logits, sample["gt_labels"], sample["gt_mask"]
            )

            if free_conf_thresh is not None:
                pred_aligned = apply_free_threshold(aligned_logits, free_index, free_conf_thresh)
                pred_fast = apply_free_threshold(fast_logits_last, free_index, free_conf_thresh)
            else:
                pred_aligned = aligned_logits.argmax(dim=1)
                pred_fast = fast_logits_last.argmax(dim=1)

            meta_list = sample.get("meta", [])
            if isinstance(meta_list, dict):
                meta_list = [meta_list]

            pred_fast_np = pred_fast.cpu().numpy()
            pred_aligned_np = pred_aligned.cpu().numpy()
            gt_np = gt_labels.cpu().numpy()
            mask_np = gt_mask.cpu().numpy() if gt_mask is not None else None

            for b in range(pred_aligned_np.shape[0]):
                token = meta_list[b].get("token", "") if b < len(meta_list) else ""
                mask_b = mask_np[b] if mask_np is not None else None
                # 累加 mIoU（fast 和 aligned 各一份）
                metric_fast.add_batch(
                    semantics_pred=pred_fast_np[b],
                    semantics_gt=gt_np[b],
                    mask_lidar=None,
                    mask_camera=mask_b,
                )
                metric_aligned.add_batch(
                    semantics_pred=pred_aligned_np[b],
                    semantics_gt=gt_np[b],
                    mask_lidar=None,
                    mask_camera=mask_b,
                )
                predictions.append({
                    "pred_fast": pred_fast_np[b].astype(np.uint8),
                    "pred_aligned": pred_aligned_np[b].astype(np.uint8),
                    "gt": gt_np[b].astype(np.uint8),
                    "mask": mask_b,
                    "token": token,
                })

            if step % 200 == 0 or step == total_steps:
                spd = step / (time.time() - t0) if time.time() > t0 else 0
                print(f"  step={step}/{total_steps}  {spd:.1f} it/s")

    print(f"[phase1] 共收集 {len(predictions)} 个样本")

    # --- 阶段 1.5：打印 mIoU 和 per-class IoU ---
    print("\n" + "=" * 20 + " mIoU 对比 " + "=" * 20)
    miou_fast = metric_fast.count_miou(verbose=False)
    miou_d_fast = metric_fast.count_miou_d(verbose=False)
    per_class_fast = np.nan_to_num(metric_fast.get_per_class_iou(), nan=0.0)

    miou_aligned = metric_aligned.count_miou(verbose=False)
    miou_d_aligned = metric_aligned.count_miou_d(verbose=False)
    per_class_aligned = np.nan_to_num(metric_aligned.get_per_class_iou(), nan=0.0)

    header = f"{'Metric':<28} {'Fast Baseline':>14} {'Aligner Output':>14} {'Delta':>14}"
    print(header)
    print("-" * len(header))
    print(f"{'mIoU':<28} {miou_fast:>14.4f} {miou_aligned:>14.4f} {miou_aligned - miou_fast:>+14.4f}")
    print(f"{'mIoU_d (dynamic)':<28} {miou_d_fast:>14.4f} {miou_d_aligned:>14.4f} {miou_d_aligned - miou_d_fast:>+14.4f}")
    print()
    print(f"{'Class':<28} {'Fast':>14} {'Aligned':>14} {'Delta':>14}")
    print("-" * len(header))
    for name, v_f, v_a in zip(metric_fast.class_names, per_class_fast, per_class_aligned):
        print(f"{name:<28} {float(v_f):>14.4f} {float(v_a):>14.4f} {float(v_a) - float(v_f):>+14.4f}")

    # --- 阶段 2：加载 lidar origins ---
    print("[phase2] 加载 lidar origins...")
    sweep_pkl = resolve_sweep_pkl(args.sweep_pkl, cfg)
    from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
    origins_by_token = load_origins_from_sweep_pkl(sweep_pkl)
    print(f"  共 {len(origins_by_token)} 个 token 的 origin")

    # --- 阶段 3：逐样本 ray casting + 统计 ---
    print("[phase3] Ray casting + 深度统计...")
    lidar_rays = torch.from_numpy(generate_lidar_rays())

    # 统计区域：全局 / mask 内外 / 距离桶 / 近距 × static/dynamic
    dist_bins = [(0, 10), (10, 20), (20, 40), (40, float("inf"))]
    regions = [
        "all", "mask_in", "mask_out",
        "d_0_10", "d_10_20", "d_20_40", "d_40_plus",
        "near_static", "near_dynamic",
    ]
    stats: dict[tuple[str, str], RayDepthStats] = {
        (src, region): RayDepthStats()
        for src in ("fast", "aligned")
        for region in regions
    }
    dist_bin_keys = ["d_0_10", "d_10_20", "d_20_40", "d_40_plus"]

    # hit/no-hit 四格表统计
    hn_regions = ["all", "mask_in", "d_0_10", "d_10_20"]
    hn_stats: dict[tuple[str, str], HitNoHitCounter] = {
        (src, region): HitNoHitCounter()
        for src in ("fast", "aligned")
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
        aligned_vol = np.reshape(item["pred_aligned"], [200, 200, 16])
        mask_vol = item["mask"]  # (200, 200, 16) or None

        # ray cast 三个 volume
        rc_gt = raycast_one_sample(gt_vol, lidar_rays, lidar_origins)
        rc_fast = raycast_one_sample(fast_vol, lidar_rays, lidar_origins)
        rc_aligned = raycast_one_sample(aligned_vol, lidar_rays, lidar_origins)

        # --- hit/no-hit 统计（使用全部 ray）---
        all_gt_hit = rc_gt["class"] != FREE_ID
        all_gt_coord = rc_gt["coord"]  # (N, 3)

        for src_name, rc_src in [("fast", rc_fast), ("aligned", rc_aligned)]:
            all_pred_hit = rc_src["class"] != FREE_ID

            # all
            hn_stats[(src_name, "all")].update(all_gt_hit, all_pred_hit)

            # mask 内（注意：GT no-hit ray 的 coord 是 DVR 最后遍历的栅格点，
            # 大概率在边界 free 区域、不在 mask 内，因此 mask_in 主要反映 GT hit ray）
            if mask_vol is not None:
                m = mask_vol[all_gt_coord[:, 0], all_gt_coord[:, 1], all_gt_coord[:, 2]].astype(bool)
                if m.any():
                    hn_stats[(src_name, "mask_in")].update(all_gt_hit[m], all_pred_hit[m])

            # 距离桶：统一用 gt_dist 分桶，保证 fast/aligned 对比的是同一批 ray
            gt_dist_all = rc_gt["dist"]
            for lo, hi, key in [(0, 10, "d_0_10"), (10, 20, "d_10_20")]:
                bin_m = (gt_dist_all >= lo) & (gt_dist_all < hi)
                if bin_m.any():
                    hn_stats[(src_name, key)].update(all_gt_hit[bin_m], all_pred_hit[bin_m])

        # 只分析 GT 命中非 free 的 ray（原有深度统计）
        valid = rc_gt["class"] != FREE_ID
        gt_class = rc_gt["class"][valid]
        gt_dist = rc_gt["dist"][valid]
        gt_coord = rc_gt["coord"][valid]  # (N, 3) — 用于判断 mask 内/外

        # 预计算 GT ray 属性 mask
        gt_is_dynamic = np.isin(gt_class, list(DYNAMIC_IDS))
        gt_is_static = ~gt_is_dynamic

        for src_name, rc_src in [("fast", rc_fast), ("aligned", rc_aligned)]:
            pred_class = rc_src["class"][valid]
            pred_dist = rc_src["dist"][valid]

            # all
            stats[(src_name, "all")].update(pred_class, pred_dist, gt_class, gt_dist)

            # mask 内/外
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

            # 距离桶: 0-10, 10-20, 20-40, 40+
            for (lo, hi), key in zip(dist_bins, dist_bin_keys):
                bin_mask = (gt_dist >= lo) & (gt_dist < hi)
                if bin_mask.any():
                    stats[(src_name, key)].update(
                        pred_class[bin_mask], pred_dist[bin_mask],
                        gt_class[bin_mask], gt_dist[bin_mask],
                    )

            # 近距 (<20m) × static / dynamic
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
        aligned_s = stats[("aligned", region)]
        if fast_s.total_rays == 0 and aligned_s.total_rays == 0:
            continue
        print_depth_comparison(fast_s, aligned_s, region_display[region])

    # --- hit/no-hit 四格表 ---
    hn_display = {
        "all": "全部 ray",
        "mask_in": "mask 内 ray",
        "d_0_10": "距离 0-10m",
        "d_10_20": "距离 10-20m",
    }
    for region in hn_regions:
        fast_c = hn_stats[("fast", region)]
        aligned_c = hn_stats[("aligned", region)]
        if fast_c.total == 0 and aligned_c.total == 0:
            continue
        print_hit_nohit_comparison(fast_c, aligned_c, hn_display[region])


if __name__ == "__main__":
    main()
