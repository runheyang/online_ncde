#!/usr/bin/env python3
"""少量样本过拟合测试脚本。"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "utils"))

from spconv_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from spconv_ncde.data.occ3d_ncde_dataset import Occ3DNcdeDataset  # noqa: E402
from spconv_ncde.losses import FocalLovaszLoss  # noqa: E402
from spconv_ncde.models.ncde_aligner import NcdeAligner  # noqa: E402
from spconv_ncde.trainer import Trainer, ncde_collate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/spconv_ncde/base.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="过拟合样本数量")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--output-dir",
        default="outputs/spconv_ncde/overfit",
        help="输出目录",
    )
    parser.add_argument("--save-every", type=int, default=10, help="保存间隔")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config_with_base(args.config)

    if torch.cuda.is_available():
        # 显式开启 TF32，利用 Tensor Core 加速 FP32 计算
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    root_path = cfg["root_path"]
    data_cfg = cfg["data"]
    loader_cfg = cfg.get("dataloader", {})

    sharing_strategy = loader_cfg.get("sharing_strategy", None)
    if sharing_strategy:
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    dataset = Occ3DNcdeDataset(
        info_path=data_cfg["info_path"],
        root_path=root_path,
        logits_root=data_cfg["logits_root"],
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        slow_noise_std=data_cfg.get("slow_noise_std", 0.0),
        fast_conf_thresh=data_cfg.get("fast_conf_thresh", None),
    )

    num_samples = min(args.num_samples, len(dataset))
    indices = list(range(num_samples))
    subset = Subset(dataset, indices)

    num_workers = cfg["train"]["num_workers"]
    loader_kwargs = dict(
        batch_size=cfg["train"]["batch_size"],
        num_workers=num_workers,
        shuffle=True,
        collate_fn=ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(subset, **loader_kwargs)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model = NcdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=cfg["model"]["feat_dim"],
        encoder_in_channels=cfg["model"]["encoder_in_channels"],
        decoder_init_scale=cfg["model"]["decoder_init_scale"],
        add_time_channel=cfg["ncde"]["add_time_channel"],
        time_scale=cfg["ncde"]["time_scale"],
        eps=cfg["ncde"]["eps"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    loss_fn = FocalLovaszLoss(
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        gamma=cfg["loss"]["gamma"],
        class_weights=cfg["loss"].get("class_weights", None),
        lovasz_weight=cfg["loss"].get("lovasz_weight", 1.0),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        fast_conf_thresh=data_cfg.get("fast_conf_thresh", None),
        free_conf_thresh=data_cfg.get("fast_conf_thresh", None),
        log_interval=cfg["train"]["log_interval"],
        clip_norm=cfg["train"]["clip_norm"],
    )

    output_dir = resolve_path(root_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[overfit] samples={num_samples} epochs={args.epochs} output={output_dir}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_one_epoch(loader, epoch)
        avg_loss = train_metrics["loss"]
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
            trainer.save_checkpoint(ckpt_path)
            extra = ""
            if "focal" in train_metrics and "lovasz" in train_metrics:
                extra = (
                    f" focal={train_metrics['focal']:.4f}"
                    f" lovasz={train_metrics['lovasz']:.4f}"
                )
            print(f"[overfit] epoch={epoch} avg_loss={avg_loss:.4f}{extra} -> {ckpt_path}")
        else:
            extra = ""
            if "focal" in train_metrics and "lovasz" in train_metrics:
                extra = (
                    f" focal={train_metrics['focal']:.4f}"
                    f" lovasz={train_metrics['lovasz']:.4f}"
                )
            print(f"[overfit] epoch={epoch} avg_loss={avg_loss:.4f}{extra}")


if __name__ == "__main__":
    main()


