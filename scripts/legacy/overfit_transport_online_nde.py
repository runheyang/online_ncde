#!/usr/bin/env python3
"""Small-sample overfit script for transport_online_nde."""

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

from transport_online_nde.config import load_config_with_base  # noqa: E402
from transport_online_nde.data.occ3d_transport_online_nde_dataset import (  # noqa: E402
    Occ3DTransportOnlineNdeDataset,
)
from transport_online_nde.losses import (  # noqa: E402
    TransportOnlineNdeLoss,
    resize_labels_and_mask_to_logits,
)
from transport_online_nde.metrics import MetricMiouOcc3D  # noqa: E402
from transport_online_nde.models.transport_online_nde_aligner import (  # noqa: E402
    TransportOnlineNdeAligner,
)
from transport_online_nde.trainer import (  # noqa: E402
    Trainer,
    move_to_device,
    transport_online_nde_collate,
)
from transport_online_nde.utils.reproducibility import set_seed  # noqa: E402


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

    if present.any():
        miou_present = float((inter[present] / np.maximum(union[present], eps)).mean() * 100.0)
    else:
        miou_present = 0.0

    if active.any():
        miou_active = float((inter[active] / np.maximum(union[active], eps)).mean() * 100.0)
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
        default=str(ROOT / "configs/transport_online_nde/train.yaml"),
        help="config file path",
    )
    parser.add_argument("--sample-count", type=int, default=10, help="number of samples")
    parser.add_argument("--epochs", type=int, default=300, help="overfit epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config_with_base(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    motion_cfg = model_cfg.get("motion", {})
    source_cfg = model_cfg.get("source", {})
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]

    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root and data.slow_logit_root are required")

    dataset = Occ3DTransportOnlineNdeDataset(
        info_path=data_cfg["info_path"],
        root_path=cfg["root_path"],
        fast_logits_root=fast_logits_root,
        slow_logit_root=slow_logit_root,
        gt_root=data_cfg["gt_root"],
        motion_gt_root=data_cfg["motion_gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        slow_noise_std=0.0,
        topk_other_fill_value=data_cfg.get("topk_other_fill_value", -5.0),
        topk_free_fill_value=data_cfg.get("topk_free_fill_value", 5.0),
        supervision_sidecar_path=data_cfg.get("train_supervision_sidecar_path", None),
        timestamp_scale=float(data_cfg.get("timestamp_scale", 1.0e-6)),
        downsample_xy=int(data_cfg.get("downsample_xy", 2)),
    )

    sub = Subset(dataset, list(range(min(args.sample_count, len(dataset)))))
    loader = DataLoader(
        sub,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=transport_online_nde_collate,
    )

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = TransportOnlineNdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        search_radius=int(motion_cfg.get("search_radius", 4)),
        bev_channels=int(motion_cfg.get("bev_channels", 64)),
        context_hidden=int(motion_cfg.get("context_hidden", 128)),
        decoder_init_scale=float(model_cfg.get("decoder_init_scale", 1.0e-2)),
        residual_readout=bool(model_cfg.get("residual_readout", False)),
        motion_mask_bias_init=float(motion_cfg.get("mask_bias_init", 1.0)),
        source_gn_groups=int(source_cfg.get("gn_groups", 8)),
        timestamp_scale=float(data_cfg.get("timestamp_scale", 1.0e-6)),
        amp_fp16=bool(model_cfg.get("amp_fp16", False)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    loss_fn = TransportOnlineNdeLoss(
        num_classes=data_cfg["num_classes"],
        gamma=loss_cfg.get("gamma", 2.0),
        class_weights=loss_cfg.get("class_weights", None),
        lambda_focal=loss_cfg.get("lambda_focal", 1.0),
        lambda_lovasz=loss_cfg.get("lambda_lovasz", 1.0),
        lambda_disp=loss_cfg.get("lambda_disp", 1.0),
        disp_group_weights=loss_cfg.get("disp_group_weights", {"fg": 1.0, "bg": 0.2, "free": 0.02}),
    )

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
        use_multistep_supervision=bool(train_cfg.get("use_multistep_supervision", True)),
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
        total_disp = 0.0
        total_sup_loss: dict[str, float] = {}
        total_sup_count: dict[str, int] = {}
        steps = 0

        for sample in loader:
            sample = move_to_device(sample, device)

            sup_loss_batch: dict[str, float] = {}
            sup_count_batch: dict[str, int] = {}
            if (
                loss_aggregator.use_multistep_supervision
                and loss_aggregator._has_multistep_fields(sample)  # noqa: SLF001
            ):
                outputs_step = model.forward_stepwise_train(
                    fast_logits=sample["fast_logits"],
                    slow_logits=sample["slow_logits"],
                    frame_ego2global=sample["frame_ego2global"],
                    frame_timestamps=sample.get("frame_timestamps", None),
                    frame_dt=sample.get("frame_dt", None),
                )
                step_logits = outputs_step["step_logits"]
                loss_dict, sup_loss_batch, sup_count_batch = loss_aggregator._compute_multistep_loss(  # noqa: SLF001
                    step_logits=step_logits,
                    step_delta_p_3d_final=outputs_step["step_delta_p_3d_final"],
                    step_valid=outputs_step["step_valid"],
                    sup_labels=sample["sup_labels"],
                    sup_masks=sample["sup_masks"],
                    sup_step_indices=sample["sup_step_indices"],
                    sup_valid_mask=sample["sup_valid_mask"],
                    sup_disp_gt_xy=sample["sup_disp_gt_xy"],
                    sup_group_mask=sample["sup_group_mask"],
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
                loss_dict = loss_fn(logits=logits, targets=sample["gt_labels"], occ_mask=sample["gt_mask"])

            loss = loss_dict["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("clip_norm", 5.0))
            optimizer.step()

            total_loss += float(loss.item())
            total_focal += float(loss_dict["focal"].item())
            total_lovasz += float(loss_dict["aux"].item())
            total_disp += float(loss_dict.get("disp", torch.tensor(0.0, device=device)).item())
            for key, value in sup_loss_batch.items():
                cnt = sup_count_batch.get(key, 0)
                if cnt <= 0:
                    continue
                total_sup_loss[key] = total_sup_loss.get(key, 0.0) + float(value) * float(cnt)
                total_sup_count[key] = total_sup_count.get(key, 0) + int(cnt)
            steps += 1

            gt_labels_rs, gt_mask_rs = resize_labels_and_mask_to_logits(logits, sample["gt_labels"], sample["gt_mask"])
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

        denom = max(steps, 1)
        miou_stats = compute_overfit_miou_from_hist(metric.hist, free_index=data_cfg["free_index"])
        line = (
            f"[overfit] epoch={epoch} loss={total_loss / denom:.4f} "
            f"focal={total_focal / denom:.4f} lovasz={total_lovasz / denom:.4f} "
            f"disp={total_disp / denom:.4f} "
            f"miou_present={float(miou_stats['miou_present']):.4f} "
            f"miou_active={float(miou_stats['miou_active']):.4f} "
            f"absent_fp_rate={float(miou_stats['absent_fp_rate']):.6f} "
            f"present_cls={int(miou_stats['present_class_count'])} "
            f"active_cls={int(miou_stats['active_class_count'])}"
        )
        sup_line = " ".join(
            f"{k}={float(total_sup_loss[k] / max(total_sup_count.get(k, 1), 1)):.4f}"
            for k in sorted(total_sup_loss.keys())
            if k.startswith("loss_t")
        )
        if sup_line:
            line = f"{line} {sup_line}"
        print(line)


if __name__ == "__main__":
    main()
