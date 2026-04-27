#!/usr/bin/env python3
"""随机/显式帧间隔评估脚本：仅对最后一帧（current frame）打分，输出 mIoU + RayIoU。

逻辑对齐 ``scripts/eval_online_ncde.py``，唯一差别是在喂给模型前用
``SubsampledStepwiseEvalDataset`` 对 fast_logits / 时间戳 / pose 抽帧（保留首尾、
中间任意 gap），从而压力测试 NCDE aligner 在非均匀帧率下的表现。

两种取帧方式（互斥）：
  1. ``--frame-indices "0,3,6,9,12"``  显式给定（必须以 0 开头，按升序）
  2. ``--random``                     固定随机种子的 gap 路径

示例：
  # 显式
  python tests/online_ncde/eval_online_ncde_random_interval.py \
      --config configs/xxx.yaml --checkpoint ckpt.pt --frame-indices "0,3,6,9,12"

  # 随机 6/3/2 Hz 路径，end at step=12
  python tests/online_ncde/eval_online_ncde_random_interval.py \
      --config configs/xxx.yaml --checkpoint ckpt.pt --random \
      --gap-choices "1,2,3" --target-last-step 12 --seed 0
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.losses import resize_labels_and_mask_to_logits  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D, apply_free_threshold  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde.trainer import move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402


# ---------------------------------------------------------------------------
# 抽帧索引构造
# ---------------------------------------------------------------------------

def _build_random_frame_indices(
    target_last_step: int,
    gap_choices: list[int],
    seed: int,
) -> tuple[list[int], list[int]]:
    if target_last_step < 1:
        raise ValueError(f"target_last_step must be >= 1, got {target_last_step}")
    rng = np.random.RandomState(seed)
    indices = [0]
    applied_gaps: list[int] = []
    curr = 0
    while curr < target_last_step:
        chosen_gap = int(rng.choice(gap_choices))
        next_step = min(curr + chosen_gap, target_last_step)
        indices.append(next_step)
        applied_gaps.append(next_step - curr)
        curr = next_step
    return indices, applied_gaps


def _parse_int_list(spec: str) -> list[int]:
    values = [item.strip() for item in str(spec).split(",")]
    return [int(item) for item in values if item]


# ---------------------------------------------------------------------------
# Dataset wrapper：按 sampled_frame_indices 抽帧
# ---------------------------------------------------------------------------

def _infer_num_frames(info: dict[str, Any]) -> int:
    for key in ("frame_ego2global", "frame_timestamps", "frame_dt", "frame_tokens"):
        value = info.get(key, None)
        if value is None:
            continue
        if hasattr(value, "shape"):
            shape = getattr(value, "shape")
            if len(shape) > 0:
                return int(shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


class SubsampledStepwiseEvalDataset(Dataset):
    """对 base dataset 的 fast_logits / 时间 / pose 做抽帧。

    短历史样本（rollout_start_step>0）保留——抽帧后会按"原 pad 区间内有多少个
    抽出的帧"重映射成抽帧坐标系下的 rollout_start_step，让 aligner 正确跳过 pad。
    最后一帧的 token 仍是原始末帧（gt_labels / gt_mask / slow_logits 不动），
    所以下游评估直接对应 current frame。
    """

    def __init__(
        self,
        base_dataset: Occ3DOnlineNcdeDataset,
        sampled_frame_indices: list[int],
    ) -> None:
        if len(sampled_frame_indices) < 2:
            raise ValueError(f"Need at least 2 frame indices, got {sampled_frame_indices}")
        self.base_dataset = base_dataset
        self.sampled_frame_indices = list(sampled_frame_indices)
        self.required_num_frames = max(self.sampled_frame_indices) + 1

        self.valid_indices: list[int] = []
        self.filter_stats: dict[str, int] = defaultdict(int)
        # 短历史样本统计：保留但记录
        self.short_history_kept: int = 0
        self.degenerate_kept: int = 0
        for idx, info in enumerate(self.base_dataset.infos):
            # 抽帧需要 fast_logits 帧数 >= 末尾索引；slow_logits 单帧不影响
            if not str(info.get("slow_logit_path", "")):
                self.filter_stats["missing_slow_logit_path"] += 1
                continue
            if _infer_num_frames(info) < self.required_num_frames:
                self.filter_stats["short_sequence"] += 1
                continue
            orig_R = int(info.get("rollout_start_step", 0))
            if orig_R > 0:
                self.short_history_kept += 1
                # 抽帧后映射的 new_R 若 >= len(sampled)-1，整段都在 pad 区间，
                # aligner 会走退化分支返回 slow_logits（仍是有效预测，不剔除）
                new_R = sum(1 for s in self.sampled_frame_indices if s < orig_R)
                if new_R >= len(self.sampled_frame_indices) - 1:
                    self.degenerate_kept += 1
            self.valid_indices.append(idx)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _slice_optional(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value[self.sampled_frame_indices]

    def _subsample_frame_dt(
        self,
        frame_dt: torch.Tensor | None,
        total_num_frames: int,
    ) -> torch.Tensor | None:
        """frame_dt 既可能是逐帧时间戳差（length=T-1）也可能是逐帧偏移（length=T）。"""
        if frame_dt is None:
            return None
        if frame_dt.numel() == total_num_frames:
            return frame_dt[self.sampled_frame_indices]
        if frame_dt.numel() == total_num_frames - 1:
            merged = []
            for s, e in zip(self.sampled_frame_indices[:-1], self.sampled_frame_indices[1:]):
                merged.append(frame_dt[s:e].sum())
            return torch.stack(merged, dim=0) if merged else frame_dt.new_zeros((0,))
        return frame_dt.reshape(-1)[: len(self.sampled_frame_indices)]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_idx = self.valid_indices[idx]
        sample = self.base_dataset[base_idx]
        sampled = dict(sample)

        fast_logits = cast(torch.Tensor, sample["fast_logits"])
        frame_ego2global = cast(torch.Tensor, sample["frame_ego2global"])
        if fast_logits.shape[0] < self.required_num_frames:
            raise IndexError(
                f"sample {base_idx} only has {fast_logits.shape[0]} frames, "
                f"need at least {self.required_num_frames}"
            )

        sampled["fast_logits"] = fast_logits[self.sampled_frame_indices]
        sampled["frame_ego2global"] = frame_ego2global[self.sampled_frame_indices]
        frame_timestamps = self._slice_optional(
            cast(torch.Tensor | None, sample.get("frame_timestamps", None))
        )
        sampled["frame_timestamps"] = frame_timestamps
        if frame_timestamps is not None:
            sampled["frame_dt"] = None
        else:
            sampled["frame_dt"] = self._subsample_frame_dt(
                cast(torch.Tensor | None, sample.get("frame_dt", None)),
                total_num_frames=int(fast_logits.shape[0]),
            )

        # rollout_start_step 重映射到抽帧坐标系：
        # 原坐标下 [0, orig_R) 是 pad 帧，统计抽帧序列里落在该区间的帧数即为新值
        orig_R = int(cast(torch.Tensor, sample["rollout_start_step"]).item())
        new_R = sum(1 for s in self.sampled_frame_indices if s < orig_R)
        sampled["rollout_start_step"] = torch.tensor(new_R, dtype=torch.long)

        meta = dict(cast(dict[str, Any], sample.get("meta", {})))
        meta["sampled_frame_indices"] = list(self.sampled_frame_indices)
        meta["rollout_start_step"] = new_R
        meta["rollout_start_step_orig"] = orig_R
        sampled["meta"] = meta
        return sampled


# ---------------------------------------------------------------------------
# sweep pkl 解析（与 eval_online_ncde.py 一致）
# ---------------------------------------------------------------------------

def resolve_sweep_pkl(args: argparse.Namespace, cfg: dict) -> str:
    if args.sweep_pkl:
        p = Path(args.sweep_pkl)
        return str(p if p.is_absolute() else (ROOT / p).resolve())
    info_path = cfg["data"].get("val_info_path", cfg["data"]["info_path"])
    info_abs = Path(info_path) if Path(info_path).is_absolute() else (ROOT / info_path).resolve()
    with open(info_abs, "rb") as f:
        meta = pickle.load(f).get("metadata", {})
    src = meta.get("source_info_path", "")
    if src and Path(src).exists():
        return src
    raise FileNotFoundError(
        f"无法推断 sweep pkl 路径（canonical metadata.source_info_path={src}），"
        "请通过 --sweep-pkl 指定。"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online NCDE 抽帧评估（仅末帧 mIoU + RayIoU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--limit", type=int, default=0,
                        help="只评估前 N 个有效样本，0 表示全部")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="覆盖 eval.batch_size，0 保持配置")
    parser.add_argument("--solver", choices=["heun", "euler"], default="euler",
                        help="ODE 求解器：euler（默认）或 heun")
    parser.add_argument("--sweep-pkl", default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
                        help="sweep pkl 路径，相对项目根目录")
    parser.add_argument(
        "--exclude-short-history",
        action="store_true",
        help="启用 config.min_history_completeness（通常 4）过滤短历史样本；"
             "默认仅在抽帧 wrapper 中过滤 rollout_start_step>0 的短历史。",
    )

    # 抽帧方式（二选一）
    parser.add_argument(
        "--frame-indices",
        default="",
        help="显式给定要保留的帧索引（必须以 0 开头），如 '0,3,6,9,12'",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="使用固定 seed 的随机 gap 路径（与 --frame-indices 互斥）",
    )
    parser.add_argument("--gap-choices", default="1,2,3",
                        help="[random] 备选 gap，逗号分隔")
    parser.add_argument("--seed", type=int, default=0,
                        help="[random] RNG seed")
    parser.add_argument("--target-last-step", type=int, default=12,
                        help="[random] 末帧索引（也即原始时间轴上的当前帧步数）")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    # --- 抽帧索引 ---
    applied_gaps: list[int] | None = None
    if args.random:
        gap_choices = _parse_int_list(args.gap_choices)
        if not gap_choices or any(g <= 0 for g in gap_choices):
            raise ValueError(f"gap_choices 必须全为正整数，got {gap_choices}")
        sampled_frame_indices, applied_gaps = _build_random_frame_indices(
            args.target_last_step, gap_choices, args.seed,
        )
        mode_str = "random"
    elif args.frame_indices:
        sampled_frame_indices = sorted(set(_parse_int_list(args.frame_indices)))
        if not sampled_frame_indices or sampled_frame_indices[0] != 0:
            raise ValueError(f"frame_indices 必须以 0 开头，got {sampled_frame_indices}")
        mode_str = "explicit"
    else:
        raise ValueError("必须指定 --frame-indices 或 --random")

    print(f"[config] mode={mode_str}")
    print(f"[config] sampled_frame_indices={sampled_frame_indices} "
          f"({len(sampled_frame_indices)} frames, "
          f"{len(sampled_frame_indices) - 1} steps)")
    if applied_gaps is not None:
        print(f"[config] applied_gaps={applied_gaps}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})

    # --- Dataset ---
    logits_loader = build_logits_loader(data_cfg, cfg["root_path"])

    min_hc = int(data_cfg.get("min_history_completeness", 4)) if args.exclude_short_history else 0
    print(f"[eval] min_history_completeness={min_hc}"
          + ("  (--exclude-short-history)" if args.exclude_short_history else ""))

    base_dataset = Occ3DOnlineNcdeDataset(
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
        root_path=cfg["root_path"],
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        logits_loader=logits_loader,
        ray_sidecar_dir=data_cfg.get("ray_sidecar_dir", None),
        ray_sidecar_split="val",
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
        min_history_completeness=min_hc,
    )
    dataset: Dataset = SubsampledStepwiseEvalDataset(
        base_dataset=base_dataset,
        sampled_frame_indices=sampled_frame_indices,
    )
    wrapper_ref = cast(SubsampledStepwiseEvalDataset, dataset)
    fs = wrapper_ref.filter_stats
    print(f"[dataset] valid={len(wrapper_ref)} "
          f"filtered_short_seq={fs.get('short_sequence', 0)} "
          f"filtered_missing_slow={fs.get('missing_slow_logit_path', 0)} "
          f"short_history_kept={wrapper_ref.short_history_kept} "
          f"(degenerate_kept={wrapper_ref.degenerate_kept})")

    if args.limit > 0:
        keep = min(args.limit, len(dataset))
        dataset = Subset(dataset, list(range(keep)))
        print(f"[eval] --limit={args.limit}，实际评估 {len(dataset)} 个样本")

    num_workers = int(eval_cfg.get("num_workers", 4))
    batch_size = int(args.batch_size) if args.batch_size > 0 else int(eval_cfg.get("batch_size", 1))
    dl_kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
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

    free_index = data_cfg["free_index"]
    free_conf_thresh = eval_cfg.get("free_conf_thresh", None)

    # --- 推理：mIoU + 收集 predictions（用于 RayIoU）---
    metric = MetricMiouOcc3D(
        num_classes=data_cfg["num_classes"],
        use_image_mask=True,
        use_lidar_mask=False,
    )
    predictions: list[dict] = []
    total_steps = len(loader)

    with torch.inference_mode():
        for step, sample in enumerate(loader, start=1):
            sample = move_to_device(sample, device)
            outputs = model(
                fast_logits=sample["fast_logits"],
                slow_logits=sample["slow_logits"],
                frame_ego2global=sample["frame_ego2global"],
                frame_timestamps=sample.get("frame_timestamps", None),
                frame_dt=sample.get("frame_dt", None),
                rollout_start_step=sample.get("rollout_start_step", None),
            )
            aligned_logits = cast(torch.Tensor, outputs["aligned"])

            gt_labels, gt_mask = resize_labels_and_mask_to_logits(
                aligned_logits, sample["gt_labels"], sample["gt_mask"]
            )

            if free_conf_thresh is not None:
                preds = apply_free_threshold(aligned_logits, free_index, free_conf_thresh)
            else:
                preds = aligned_logits.argmax(dim=1)

            preds_np = preds.detach().cpu().numpy()
            gt_np = gt_labels.detach().cpu().numpy()
            mask_np = gt_mask.detach().cpu().numpy() if gt_mask is not None else None

            meta_list = sample.get("meta", [])
            if isinstance(meta_list, dict):
                meta_list = [meta_list]

            for b in range(preds_np.shape[0]):
                mask_b = mask_np[b] if mask_np is not None else None
                metric.add_batch(
                    semantics_pred=preds_np[b],
                    semantics_gt=gt_np[b],
                    mask_lidar=None,
                    mask_camera=mask_b,
                )
                token = meta_list[b].get("token", "") if b < len(meta_list) else ""
                predictions.append({
                    "pred": preds_np[b].astype(np.uint8),
                    "gt": gt_np[b].astype(np.uint8),
                    "token": token,
                })

            if step % int(eval_cfg.get("log_interval", 20)) == 0 or step == total_steps:
                print(f"[eval] step={step}/{total_steps}")

    # --- mIoU ---
    miou = metric.count_miou(verbose=False)
    miou_d = metric.count_miou_d(verbose=False)
    per_class = np.nan_to_num(metric.get_per_class_iou(), nan=0.0).tolist()
    print(f"\n[eval] miou={miou:.4f}  miou_d={miou_d:.4f}  num_samples={metric.cnt}")
    for name, value in zip(metric.class_names, per_class):
        print(f"{name}: {float(value):.2f}")

    # --- RayIoU ---
    print("\n[rayiou] 加载 lidar origins...")
    sweep_pkl = resolve_sweep_pkl(args, cfg)
    print(f"[rayiou] sweep pkl: {sweep_pkl}")

    from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
    origins_by_token = load_origins_from_sweep_pkl(sweep_pkl)
    print(f"[rayiou] 共 {len(origins_by_token)} 个 token 的 origin")

    from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou

    sem_pred_list, sem_gt_list, lidar_origin_list = [], [], []
    skipped = 0
    for item in predictions:
        token = item["token"]
        if token not in origins_by_token:
            skipped += 1
            continue
        sem_pred_list.append(item["pred"])
        sem_gt_list.append(item["gt"])
        lidar_origin_list.append(origins_by_token[token])

    if skipped:
        print(f"[rayiou] 跳过 {skipped} 个样本（无对应 lidar origin）")
    print(f"[rayiou] {len(sem_pred_list)} 个样本参与计算")

    rayiou_result = calc_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list)
    print(f"\n[rayiou] RayIoU={rayiou_result['RayIoU']:.4f}")
    print(f"[rayiou] RayIoU@1={rayiou_result['RayIoU@1']:.4f}")
    print(f"[rayiou] RayIoU@2={rayiou_result['RayIoU@2']:.4f}")
    print(f"[rayiou] RayIoU@4={rayiou_result['RayIoU@4']:.4f}")


if __name__ == "__main__":
    main()
