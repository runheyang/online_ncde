#!/usr/bin/env python3
"""对比 fast baseline 与 aligner output 在 camera-visible mask 内的详细 occupancy 统计。

用法示例：
    python tests/online_ncde/eval_occ_detailed_stats.py \
        --checkpoint <ckpt_path> --limit 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.losses import resize_labels_and_mask_to_logits  # noqa: E402
from online_ncde.metrics import OCC3D_DYNAMIC_OBJECT_IDX, apply_free_threshold  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint  # noqa: E402

# 区域显示名称
REGION_NAMES = {
    "overall": "mask内整体",
    "dynamic": "mask内dynamic区域",
    "boundary": "mask内boundary band",
}


# ---------------------------------------------------------------------------
# 统计累加器
# ---------------------------------------------------------------------------

class OccStats:
    """累计 occupied 类的 TP/FP/FN 等统计量。"""

    def __init__(self) -> None:
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.pred_occ_count: int = 0
        self.gt_occ_count: int = 0
        self.gt_free_count: int = 0

    def update(self, pred: np.ndarray, gt: np.ndarray, free_index: int) -> None:
        pred_occ = pred != free_index
        gt_occ = gt != free_index
        gt_free = gt == free_index
        self.tp += int(np.sum(pred_occ & gt_occ))
        self.fp += int(np.sum(pred_occ & gt_free))
        self.fn += int(np.sum(~pred_occ & gt_occ))
        self.pred_occ_count += int(np.sum(pred_occ))
        self.gt_occ_count += int(np.sum(gt_occ))
        self.gt_free_count += int(np.sum(gt_free))

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def iou(self) -> float:
        return self.tp / max(self.tp + self.fp + self.fn, 1)

    @property
    def fp_rate_on_gt_free(self) -> float:
        return self.fp / max(self.gt_free_count, 1)


# ---------------------------------------------------------------------------
# 区域 mask 计算
# ---------------------------------------------------------------------------

def compute_boundary_band(gt: np.ndarray, free_index: int, iterations: int) -> np.ndarray:
    """计算 GT occupied 的边界带：dilate - erode。"""
    occupied = gt != free_index
    struct = scipy.ndimage.generate_binary_structure(3, 1)  # 6-connected
    dilated = scipy.ndimage.binary_dilation(occupied, structure=struct, iterations=iterations)
    eroded = scipy.ndimage.binary_erosion(occupied, structure=struct, iterations=iterations)
    return dilated & ~eroded


def compute_region_masks(
    gt: np.ndarray,
    cam_mask: np.ndarray,
    free_index: int,
    boundary_width: int,
) -> dict[str, np.ndarray]:
    """返回三个区域的 boolean mask，均已与 camera mask 取交。"""
    m = cam_mask.astype(bool)
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    # dynamic 区域：GT 动态物体 + 膨胀一圈（纳入周围 free space 以观测 FP）
    dynamic_core = np.isin(gt, list(OCC3D_DYNAMIC_OBJECT_IDX))
    dynamic_dilated = scipy.ndimage.binary_dilation(dynamic_core, structure=struct, iterations=boundary_width)
    return {
        "overall": m,
        "dynamic": m & dynamic_dilated,
        "boundary": m & compute_boundary_band(gt, free_index, boundary_width),
    }


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------

def _fmt_int(v: int) -> str:
    return f"{v:>14,}"


def _fmt_pct(v: float) -> str:
    return f"{v:>14.4f}"


def _fmt_delta(v: float, is_int: bool = False) -> str:
    if is_int:
        return f"{v:>+14,}" if isinstance(v, int) else f"{int(v):>+14,}"
    return f"{v:>+14.4f}"


def print_results(stats: dict[tuple[str, str], OccStats]) -> None:
    """打印对比表。"""
    header = f"{'Metric':<28} {'Fast Baseline':>14} {'Aligner Output':>14} {'Delta':>14}"
    sep = "-" * len(header)

    for region in ("overall", "dynamic", "boundary"):
        fast = stats[("fast", region)]
        aligned = stats[("aligned", region)]

        print(f"\n{'=' * 20} {REGION_NAMES[region]} {'=' * 20}")
        print(header)
        print(sep)

        rows: list[tuple[str, int | float, int | float, bool]] = [
            ("TP_occ", fast.tp, aligned.tp, True),
            ("FP_occ", fast.fp, aligned.fp, True),
            ("FN_occ", fast.fn, aligned.fn, True),
            ("Precision_occ", fast.precision, aligned.precision, False),
            ("Recall_occ", fast.recall, aligned.recall, False),
            ("IoU_occ", fast.iou, aligned.iou, False),
            ("Pred occupied count", fast.pred_occ_count, aligned.pred_occ_count, True),
            ("GT occupied count", fast.gt_occ_count, aligned.gt_occ_count, True),
        ]
        # 仅 overall 区域额外输出 FP rate on GT-free
        if region == "overall":
            rows.append(("FP rate on GT-free", fast.fp_rate_on_gt_free, aligned.fp_rate_on_gt_free, False))

        for name, v_fast, v_aligned, is_int in rows:
            if is_int:
                delta = int(v_aligned) - int(v_fast)
                print(f"{name:<28} {_fmt_int(int(v_fast))} {_fmt_int(int(v_aligned))} {_fmt_delta(delta, True)}")
            else:
                delta = float(v_aligned) - float(v_fast)
                print(f"{name:<28} {_fmt_pct(float(v_fast))} {_fmt_pct(float(v_aligned))} {_fmt_delta(delta)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 fast baseline 与 aligner 的详细 occ 统计")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 个样本（0=全量）")
    parser.add_argument("--boundary-width", type=int, default=1, help="boundary band 膨胀/腐蚀迭代次数")
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
    model = OnlineNcdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=free_index,
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
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

    # --- 推理并累计统计 ---
    stats: dict[tuple[str, str], OccStats] = {
        (src, region): OccStats()
        for src in ("fast", "aligned")
        for region in ("overall", "dynamic", "boundary")
    }

    total_steps = len(loader)
    t0 = time.time()

    with torch.no_grad():
        for step, sample in enumerate(loader, start=1):
            sample = move_to_device(sample, device)

            # 模型前向
            outputs = model(
                fast_logits=sample["fast_logits"],
                slow_logits=sample["slow_logits"],
                frame_ego2global=sample["frame_ego2global"],
                frame_timestamps=sample.get("frame_timestamps", None),
                frame_dt=sample.get("frame_dt", None),
            )
            aligned_logits = outputs["aligned"]            # (B, C, X, Y, Z)
            fast_logits_last = sample["fast_logits"][:, -1]  # (B, C, X, Y, Z)

            # resize GT/mask 以匹配 logits 空间维度
            gt_labels, gt_mask = resize_labels_and_mask_to_logits(
                aligned_logits, sample["gt_labels"], sample["gt_mask"]
            )

            # argmax 预测
            if free_conf_thresh is not None:
                pred_aligned = apply_free_threshold(aligned_logits, free_index, free_conf_thresh)
                pred_fast = apply_free_threshold(fast_logits_last, free_index, free_conf_thresh)
            else:
                pred_aligned = aligned_logits.argmax(dim=1)
                pred_fast = fast_logits_last.argmax(dim=1)

            # 转 numpy
            pred_aligned_np = pred_aligned.cpu().numpy()
            pred_fast_np = pred_fast.cpu().numpy()
            gt_np = gt_labels.cpu().numpy()
            mask_np = gt_mask.cpu().numpy() if gt_mask is not None else None

            # 逐样本累计
            for b in range(pred_aligned_np.shape[0]):
                mask_b = mask_np[b].astype(bool) if mask_np is not None else np.ones_like(gt_np[b], dtype=bool)
                gt_b = gt_np[b]
                regions = compute_region_masks(gt_b, mask_b, free_index, args.boundary_width)

                for region_name, region_mask in regions.items():
                    if region_mask.sum() == 0:
                        continue
                    gt_region = gt_b[region_mask]
                    for src_name, pred_src in [("fast", pred_fast_np[b]), ("aligned", pred_aligned_np[b])]:
                        pred_region = pred_src[region_mask]
                        stats[(src_name, region_name)].update(pred_region, gt_region, free_index)

            if step % 200 == 0 or step == total_steps:
                spd = step / (time.time() - t0) if time.time() > t0 else 0
                print(f"[eval] step={step}/{total_steps}  {spd:.1f} it/s")

    # --- 输出对比表 ---
    print_results(stats)


if __name__ == "__main__":
    main()
