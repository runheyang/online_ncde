#!/usr/bin/env python3
"""Evaluate transport_online_nde."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "utils"))

from transport_online_nde.config import load_config_with_base  # noqa: E402
from transport_online_nde.data.occ3d_transport_online_nde_dataset import (  # noqa: E402
    Occ3DTransportOnlineNdeDataset,
)
from transport_online_nde.losses import TransportOnlineNdeLoss  # noqa: E402
from transport_online_nde.models.transport_online_nde_aligner import (  # noqa: E402
    TransportOnlineNdeAligner,
)
from transport_online_nde.trainer import Trainer, transport_online_nde_collate  # noqa: E402
from transport_online_nde.utils.checkpoints import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/transport_online_nde/eval.yaml"),
        help="config file path",
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    motion_cfg = model_cfg.get("motion", {})
    source_cfg = model_cfg.get("source", {})
    loss_cfg = cfg["loss"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})

    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root and data.slow_logit_root are required")

    dataset = Occ3DTransportOnlineNdeDataset(
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
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
        supervision_sidecar_path=data_cfg.get("val_supervision_sidecar_path", None),
        timestamp_scale=float(data_cfg.get("timestamp_scale", 1.0e-6)),
        downsample_xy=int(data_cfg.get("downsample_xy", 2)),
    )

    num_workers = int(eval_cfg.get("num_workers", 4))
    kwargs = dict(
        batch_size=int(eval_cfg.get("batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=transport_online_nde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(dataset, **kwargs)

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
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
        amp_fp16=bool(eval_cfg.get("amp_fp16", False)),
    ).to(device)

    load_checkpoint(args.checkpoint, model=model, strict=False)

    loss_fn = TransportOnlineNdeLoss(
        num_classes=data_cfg["num_classes"],
        gamma=loss_cfg.get("gamma", 2.0),
        class_weights=loss_cfg.get("class_weights", None),
        lambda_focal=loss_cfg.get("lambda_focal", 1.0),
        lambda_lovasz=loss_cfg.get("lambda_lovasz", 1.0),
        lambda_disp=loss_cfg.get("lambda_disp", 1.0),
        disp_group_weights=loss_cfg.get("disp_group_weights", {"fg": 1.0, "bg": 0.2, "free": 0.02}),
    )

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-4),
        loss_fn=loss_fn,
        device=device,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        free_conf_thresh=eval_cfg.get("free_conf_thresh", None),
        log_interval=eval_cfg.get("log_interval", 20),
        clip_norm=1.0,
        use_multistep_supervision=bool(cfg.get("train", {}).get("use_multistep_supervision", True)),
        supervision_labels=list(cfg.get("train", {}).get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])),
        supervision_weights=list(cfg.get("train", {}).get("supervision_weights", [0.15, 0.20, 0.25, 0.40])),
        supervision_weight_normalize=bool(cfg.get("train", {}).get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
    )

    metrics = trainer.evaluate(loader)
    print(
        f"[eval] loss={metrics['loss']:.4f} "
        f"focal={metrics['focal']:.4f} "
        f"lovasz={metrics['aux']:.4f} "
        f"disp={metrics['disp']:.4f} "
        f"miou={metrics['miou']:.4f}"
    )

    class_names = metrics.get("class_names", [])
    class_iou = metrics.get("per_class_iou", [])
    if isinstance(class_names, list) and isinstance(class_iou, list):
        for name, value in zip(class_names, class_iou):
            print(f"{name}: {float(value):.2f}")


if __name__ == "__main__":
    main()
