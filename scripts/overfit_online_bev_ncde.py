#!/usr/bin/env python3
"""Online BEV NCDE 小样本过拟合脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "utils"))

from online_bev_ncde.data.occ3d_online_bev_ncde_dataset import Occ3DOnlineBevNcdeDataset  # noqa: E402
from online_bev_ncde.models.online_bev_ncde_aligner import build_model_from_config  # noqa: E402
from online_bev_ncde.trainer import Trainer, move_to_device, online_ncde_collate  # noqa: E402
from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.losses import build_loss, resize_labels_and_mask_to_logits  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402
from online_ncde.utils.reproducibility import set_seed  # noqa: E402


def compute_overfit_miou_from_hist(hist: np.ndarray, free_index: int) -> dict[str, float | int]:
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
    miou_present = float((inter[present] / np.maximum(union[present], eps)).mean() * 100.0) if present.any() else 0.0
    miou_active = float((inter[active] / np.maximum(union[active], eps)).mean() * 100.0) if active.any() else 0.0
    absent_fp = float(pred_cnt[absent].sum())
    total_non_free_pred = float(np.maximum(pred_cnt[non_free].sum(), eps))
    return {
        "miou_present": miou_present,
        "miou_active": miou_active,
        "absent_fp_rate": float(absent_fp / total_non_free_pred),
        "present_class_count": int(present.sum()),
        "active_class_count": int(active.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/online_bev_ncde/train.yaml"), help="配置文件路径")
    parser.add_argument("--sample-count", type=int, default=10, help="过拟合样本数")
    parser.add_argument("--epochs", type=int, default=300, help="过拟合轮数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser.parse_args()


def build_model(data_cfg: dict, model_cfg: dict):
    return build_model_from_config(data_cfg, model_cfg, amp_fp16=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config_with_base(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]
    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root 和 data.slow_logit_root 为必填项。")

    dataset = Occ3DOnlineBevNcdeDataset(
        info_path=data_cfg["info_path"],
        root_path=cfg["root_path"],
        fast_logits_root=fast_logits_root,
        slow_logit_root=slow_logit_root,
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        slow_noise_std=0.0,
        topk_other_fill_value=data_cfg.get("topk_other_fill_value", -5.0),
        topk_free_fill_value=data_cfg.get("topk_free_fill_value", 5.0),
        supervision_sidecar_path=data_cfg.get("train_supervision_sidecar_path", None),
    )
    sub = Subset(dataset, list(range(min(args.sample_count, len(dataset)))))
    loader = DataLoader(sub, batch_size=1, shuffle=True, num_workers=0, collate_fn=online_ncde_collate)

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(data_cfg, model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    loss_fn = build_loss(loss_cfg, num_classes=data_cfg["num_classes"])
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
        use_multistep_supervision=bool(train_cfg.get("use_multistep_supervision", False)),
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.15, 0.20, 0.25, 0.40])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(cfg.get("eval", {}).get("log_multistep_losses", True)),
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        metric = MetricMiouOcc3D(num_classes=data_cfg["num_classes"], use_image_mask=True)
        total_loss = 0.0
        total_focal = 0.0
        total_lovasz = 0.0
        total_sup_loss: dict[str, float] = {}
        total_sup_count: dict[str, int] = {}
        step_count = 0

        for sample in loader:
            sample = move_to_device(sample, device)
            sup_loss_batch: dict[str, float] = {}
            sup_count_batch: dict[str, int] = {}
            if loss_aggregator.use_multistep_supervision and loss_aggregator._has_multistep_fields(sample):  # noqa: SLF001
                outputs_step = model.forward_stepwise_train(
                    fast_logits=sample["fast_logits"],
                    slow_logits=sample["slow_logits"],
                    frame_ego2global=sample["frame_ego2global"],
                    frame_timestamps=sample.get("frame_timestamps", None),
                    frame_dt=sample.get("frame_dt", None),
                )
                step_logits = outputs_step["step_logits"]
                step_indices = outputs_step["step_indices"]
                loss_dict, sup_loss_batch, sup_count_batch = loss_aggregator._compute_multistep_loss(  # noqa: SLF001
                    step_logits=step_logits,
                    step_indices=step_indices,
                    sup_labels=sample["sup_labels"],
                    sup_masks=sample["sup_masks"],
                    sup_step_indices=sample["sup_step_indices"],
                    sup_valid_mask=sample["sup_valid_mask"],
                )
                logits = step_logits[:, -1] if step_logits.shape[1] > 0 else sample["fast_logits"][:, -1]
            else:
                outputs = model(
                    fast_logits=sample["fast_logits"],
                    slow_logits=sample["slow_logits"],
                    frame_ego2global=sample["frame_ego2global"],
                    frame_timestamps=sample.get("frame_timestamps", None),
                    frame_dt=sample.get("frame_dt", None),
                )
                logits = outputs["aligned"]
                loss_dict = loss_fn(logits, sample["gt_labels"], sample["gt_mask"])
            loss = loss_dict["total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("clip_norm", 5.0))
            optimizer.step()

            total_loss += float(loss.item())
            total_focal += float(loss_dict["focal"].item())
            total_lovasz += float(loss_dict["aux"].item())
            for key, value in sup_loss_batch.items():
                cnt = sup_count_batch.get(key, 0)
                if cnt <= 0:
                    continue
                total_sup_loss[key] = total_sup_loss.get(key, 0.0) + float(value) * float(cnt)
                total_sup_count[key] = total_sup_count.get(key, 0) + int(cnt)
            step_count += 1

            gt_labels_rs, gt_mask_rs = resize_labels_and_mask_to_logits(logits, sample["gt_labels"], sample["gt_mask"])
            preds_np = logits.argmax(dim=1).detach().cpu().numpy()
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
        miou_stats = compute_overfit_miou_from_hist(metric.hist, free_index=data_cfg["free_index"])
        train_metrics = {
            "loss": total_loss / denom,
            "focal": total_focal / denom,
            "aux": total_lovasz / denom,
            "miou_present": float(miou_stats["miou_present"]),
            "miou_active": float(miou_stats["miou_active"]),
            "absent_fp_rate": float(miou_stats["absent_fp_rate"]),
            "present_class_count": int(miou_stats["present_class_count"]),
            "active_class_count": int(miou_stats["active_class_count"]),
        }
        for key, value in total_sup_loss.items():
            cnt = max(total_sup_count.get(key, 0), 1)
            train_metrics[key] = value / cnt
        sup_line = " ".join(
            f"{key}={float(train_metrics[key]):.4f}"
            for key in sorted(train_metrics.keys())
            if isinstance(key, str) and key.startswith("loss_t")
        )
        sup_part = f" {sup_line}" if sup_line else ""
        print(
            f"[overfit] epoch={epoch} loss={train_metrics['loss']:.4f} focal={train_metrics['focal']:.4f} "
            f"lovasz={train_metrics['aux']:.4f} miou_present={train_metrics['miou_present']:.4f} "
            f"miou_active={train_metrics['miou_active']:.4f} absent_fp_rate={train_metrics['absent_fp_rate']:.6f} "
            f"present_cls={train_metrics['present_class_count']} active_cls={train_metrics['active_class_count']}{sup_part}"
        )


if __name__ == "__main__":
    main()
