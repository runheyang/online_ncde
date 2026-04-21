#!/usr/bin/env python3
"""Online NCDE 小样本过拟合脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.losses import build_loss, resize_labels_and_mask_to_logits  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde.trainer import Trainer, move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.utils.reproducibility import set_seed  # noqa: E402


def compute_overfit_miou_from_hist(hist: np.ndarray, free_index: int) -> dict[str, float | int]:
    """小样本 overfit 指标：present/active mIoU 与缺失类误报率。"""
    eps = 1.0e-12
    num_classes = hist.shape[0]
    cls_ids = np.arange(num_classes)

    gt_cnt = hist.sum(axis=1)
    pred_cnt = hist.sum(axis=0)
    inter = np.diag(hist)
    union = gt_cnt + pred_cnt - inter

    non_free = cls_ids != int(free_index)
    present = (gt_cnt > 0) & non_free
    active = ((gt_cnt + pred_cnt) > 0) & non_free
    absent = (gt_cnt == 0) & non_free

    if present.any():
        iou_present = inter[present] / np.maximum(union[present], eps)
        miou_present = float(iou_present.mean() * 100.0)
    else:
        miou_present = 0.0

    if active.any():
        iou_active = inter[active] / np.maximum(union[active], eps)
        miou_active = float(iou_active.mean() * 100.0)
    else:
        miou_active = 0.0

    absent_fp = float(pred_cnt[absent].sum())
    total_non_free_pred = float(np.maximum(pred_cnt[non_free].sum(), eps))
    absent_fp_rate = absent_fp / total_non_free_pred

    return {
        "miou_present": miou_present,
        "miou_active": miou_active,
        "absent_fp_rate": float(absent_fp_rate),
        "present_class_count": int(present.sum()),
        "active_class_count": int(active.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/online_ncde/fast_opusv1t__slow_opusv2l/train.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--sample-count", type=int, default=10, help="过拟合样本数")
    parser.add_argument("--epochs", type=int, default=300, help="过拟合轮数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config_with_base(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]
    logits_loader = build_logits_loader(data_cfg, cfg["root_path"])

    dataset = Occ3DOnlineNcdeDataset(
        info_path=data_cfg["info_path"],
        root_path=cfg["root_path"],
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        logits_loader=logits_loader,
        ray_sidecar_dir=data_cfg.get("ray_sidecar_dir", None),
        ray_sidecar_split="train",
    )
    sub = Subset(dataset, list(range(min(args.sample_count, len(dataset)))))
    loader = DataLoader(
        sub,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=online_ncde_collate,
    )

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
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
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    # ray loss 的 pc_range / free_index 由 data 配置注入，与 train 脚本一致
    ray_cfg = loss_cfg.get("ray", None)
    if ray_cfg is not None:
        ray_cfg.setdefault("pc_range", list(data_cfg["pc_range"]))
        ray_cfg.setdefault("free_index", int(data_cfg["free_index"]))
    loss_fn = build_loss(loss_cfg, num_classes=data_cfg["num_classes"]).to(device)
    # 复用 Trainer 内的多帧监督 loss 组合逻辑，确保与正式训练一致。
    eval_cfg = cfg.get("eval", {})
    loss_aggregator = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        free_conf_thresh=train_cfg.get("free_conf_thresh", None),
        log_interval=int(train_cfg.get("log_interval", 10)),
        clip_norm=float(train_cfg.get("clip_norm", 5.0)),
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.15, 0.20, 0.25, 0.40])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=str(train_cfg.get("rollout_mode", "full")),
        primary_supervision_label=str(eval_cfg.get("primary_supervision_label", "t-1.0")),
        stepwise_max_step_index=train_cfg.get("max_step_index", None),
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        metric = MetricMiouOcc3D(
            num_classes=data_cfg["num_classes"],
            use_image_mask=True,
        )
        total_loss = 0.0
        total_focal = 0.0
        total_aux = 0.0
        total_ray = 0.0
        total_ray_hit = 0.0
        total_ray_empty = 0.0
        total_ray_pre_free = 0.0
        total_ray_depth = 0.0
        total_ray_sup_count = 0
        total_sup_loss: dict[str, float] = {}
        total_sup_count: dict[str, int] = {}
        step_count = 0

        for sample in loader:
            sample = move_to_device(sample, device)

            # 直接复用 Trainer 的多帧 rollout + loss helper，确保 rollout_mode /
            # max_step_index / ray 监督口径与正式训练完全一致。
            _, loss_dict, sup_loss_batch, sup_count_batch, logits, _, _ = (
                loss_aggregator._run_stepwise_and_compute_loss(  # noqa: SLF001
                    sample=sample,
                    for_eval=False,
                )
            )
            loss = loss_dict["total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("clip_norm", 5.0))
            optimizer.step()

            total_loss += float(loss.item())
            total_focal += float(loss_dict["focal"].item())
            total_aux += float(loss_dict["aux"].item())
            if "ray_total" in loss_dict:
                sup_count = int(loss_dict.get("ray_sup_count", torch.tensor(0)).item())
                if sup_count > 0:
                    total_ray += float(loss_dict["ray_total"].item())
                    total_ray_hit += float(loss_dict["ray_hit"].item())
                    total_ray_empty += float(loss_dict["ray_empty"].item())
                    total_ray_pre_free += float(loss_dict.get("ray_pre_free", torch.tensor(0.0)).item())
                    total_ray_depth += float(loss_dict["ray_depth"].item())
                    total_ray_sup_count += 1
            for key, value in sup_loss_batch.items():
                cnt = sup_count_batch.get(key, 0)
                if cnt <= 0:
                    continue
                total_sup_loss[key] = total_sup_loss.get(key, 0.0) + float(value) * float(cnt)
                total_sup_count[key] = total_sup_count.get(key, 0) + int(cnt)
            step_count += 1

            gt_labels_rs, gt_mask_rs = resize_labels_and_mask_to_logits(
                logits, sample["gt_labels"], sample["gt_mask"]
            )
            preds = logits.argmax(dim=1)
            preds_np = preds.detach().cpu().numpy()
            gt_np = gt_labels_rs.detach().cpu().numpy()
            mask_np = gt_mask_rs.detach().cpu().numpy() if gt_mask_rs is not None else None
            for b in range(preds_np.shape[0]):
                metric.add_batch(
                    semantics_pred=preds_np[b],
                    semantics_gt=gt_np[b],
                    mask_lidar=None,
                    mask_camera=mask_np[b] if mask_np is not None else None,
                )

        denom = max(step_count, 1)
        miou_stats = compute_overfit_miou_from_hist(
            metric.hist,
            free_index=data_cfg["free_index"],
        )
        train_metrics = {
            "loss": total_loss / denom,
            "focal": total_focal / denom,
            "aux": total_aux / denom,
            "miou_present": float(miou_stats["miou_present"]),
            "miou_active": float(miou_stats["miou_active"]),
            "absent_fp_rate": float(miou_stats["absent_fp_rate"]),
            "present_class_count": int(miou_stats["present_class_count"]),
            "active_class_count": int(miou_stats["active_class_count"]),
        }
        if total_ray_sup_count > 0:
            rdenom = float(total_ray_sup_count)
            train_metrics["ray_total"] = total_ray / rdenom
            train_metrics["ray_hit"] = total_ray_hit / rdenom
            train_metrics["ray_empty"] = total_ray_empty / rdenom
            train_metrics["ray_pre_free"] = total_ray_pre_free / rdenom
            train_metrics["ray_depth"] = total_ray_depth / rdenom
        for key, value in total_sup_loss.items():
            cnt = max(total_sup_count.get(key, 0), 1)
            train_metrics[key] = value / cnt
        sup_line = " ".join(
            f"{key}={float(train_metrics[key]):.4f}"
            for key in sorted(train_metrics.keys())
            if isinstance(key, str) and key.startswith("loss_t")
        )
        sup_part = f" {sup_line}" if sup_line else ""
        ray_part = ""
        if total_ray_sup_count > 0:
            ray_part = (
                f" ray_total={train_metrics['ray_total']:.4f} "
                f"ray_hit={train_metrics['ray_hit']:.4f} "
                f"ray_empty={train_metrics['ray_empty']:.4f} "
                f"ray_pre_free={train_metrics.get('ray_pre_free', 0.0):.4f} "
                f"ray_depth={train_metrics['ray_depth']:.4f}"
            )
        print(
            f"[overfit] epoch={epoch} "
            f"loss={train_metrics['loss']:.4f} "
            f"focal={train_metrics['focal']:.4f} "
            f"aux={train_metrics['aux']:.4f} "
            f"miou_present={train_metrics['miou_present']:.4f} "
            f"miou_active={train_metrics['miou_active']:.4f} "
            f"absent_fp_rate={train_metrics['absent_fp_rate']:.6f} "
            f"present_cls={train_metrics['present_class_count']} "
            f"active_cls={train_metrics['active_class_count']}"
            f"{ray_part}"
            f"{sup_part}"
        )


if __name__ == "__main__":
    main()
