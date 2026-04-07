#!/usr/bin/env python3
"""训练 SpConv NCDE 对齐器。"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
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

try:
    import wandb
except Exception:  # pragma: no cover - wandb 不是硬依赖
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/spconv_ncde/base.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--resume", default="", help="继续训练的权重路径")
    parser.add_argument("--train-limit", type=int, default=0, help="训练样本上限（0=全量）")
    parser.add_argument("--eval-every", type=int, default=1, help="每隔多少 epoch 评估一次")
    parser.add_argument("--val-info-path", default="", help="验证集 info_path（可覆盖配置）")
    parser.add_argument(
        "--val-scene-count",
        type=int,
        default=40,
        help="验证集使用的 scene 数量（默认 30，按 seed=0 打乱后取前 N）",
    )
    parser.add_argument("--wandb", action="store_true", help="启用 wandb 记录")
    parser.add_argument("--wandb-entity", default="runheyang", help="wandb entity")
    parser.add_argument("--wandb-project", default="neural-ode", help="wandb project")
    parser.add_argument("--wandb-name", default="", help="wandb run name（默认时间戳）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子，保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_subset(dataset, limit: int, start: int = 0):
    if limit is None or limit <= 0:
        return dataset
    end = min(start + limit, len(dataset))
    return Subset(dataset, list(range(start, end)))


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    set_seed(args.seed)

    if torch.cuda.is_available():
        # 显式开启 TF32，利用 Tensor Core 加速 FP32 计算
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    root_path = cfg["root_path"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})
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

    num_workers = train_cfg["num_workers"]
    loader_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        shuffle=True,
        collate_fn=ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    val_loader = None
    train_dataset = build_subset(dataset, args.train_limit)
    train_size = len(train_dataset)
    val_info_path = args.val_info_path or data_cfg.get("val_info_path", "")
    if val_info_path:
        val_dataset = Occ3DNcdeDataset(
            info_path=val_info_path,
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
        scene_names = [info.get("scene_name", "") for info in val_dataset.infos]
        unique_scenes = sorted({name for name in scene_names if name})
        if not unique_scenes:
            raise ValueError("未找到有效 scene_name，无法按 scene 划分验证集")
        rng = random.Random(0)
        rng.shuffle(unique_scenes)
        val_scene_count = min(args.val_scene_count, len(unique_scenes))
        val_scene_set = set(unique_scenes[:val_scene_count])
        val_indices = [i for i, name in enumerate(scene_names) if name in val_scene_set]
        if not val_indices:
            raise ValueError("验证集 scene 为空，请检查 val_scene_count 或数据")
        val_dataset = Subset(val_dataset, val_indices)
    else:
        val_dataset = None

    loader = DataLoader(train_dataset, **loader_kwargs)

    val_size = len(val_dataset) if val_dataset is not None else 0
    if val_dataset is not None:
        val_workers = eval_cfg.get("num_workers", train_cfg["num_workers"])
        val_kwargs = dict(
            batch_size=eval_cfg.get("batch_size", 1),
            num_workers=val_workers,
            shuffle=False,
            collate_fn=ncde_collate,
            pin_memory=loader_cfg.get("pin_memory", False),
        )
        if val_workers > 0:
            val_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
            val_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
        val_loader = DataLoader(val_dataset, **val_kwargs)

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = NcdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=cfg["model"]["feat_dim"],
        encoder_in_channels=cfg["model"]["encoder_in_channels"],
        decoder_init_scale=cfg["model"]["decoder_init_scale"],
        add_time_channel=cfg["ncde"]["add_time_channel"],
        time_scale=cfg["ncde"]["time_scale"],
        eps=cfg["ncde"]["eps"],
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
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
        log_interval=train_cfg["log_interval"],
        clip_norm=train_cfg["clip_norm"],
    )

    run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("未安装 wandb，无法启用日志记录。")
        run_name = args.wandb_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        config_payload = dict(
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            num_workers=train_cfg["num_workers"],
            train_size=train_size,
            val_size=val_size,
            amp_enabled=False,
            amp_dtype="fp32",
            eval_every=args.eval_every,
        )
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=config_payload,
        )

    output_dir = resolve_path(root_path, train_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = trainer.train_one_epoch(loader, epoch, warmup_start_lr=1e-5)
        avg_loss = train_metrics["loss"]
        current_lr = optimizer.param_groups[0].get("lr", train_cfg["lr"])
        if run is not None and wandb is not None:
            should_eval = (
                val_loader is not None
                and args.eval_every > 0
                and epoch % args.eval_every == 0
            )
            log_payload = {
                "train/loss": avg_loss,
                "lr": current_lr,
                "epoch": epoch,
            }
            if "focal" in train_metrics and "lovasz" in train_metrics:
                log_payload["train/focal"] = train_metrics["focal"]
                log_payload["train/lovasz"] = train_metrics["lovasz"]
            wandb.log(log_payload, step=epoch, commit=not should_eval)
        if val_loader is not None and args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = trainer.evaluate(val_loader)
            extra = ""
            if "focal" in metrics and "lovasz" in metrics:
                extra = f" focal={metrics['focal']:.4f} lovasz={metrics['lovasz']:.4f}"
            print(
                f"[eval] epoch={epoch} loss={metrics['loss']:.4f} "
                f"miou={metrics['miou']:.4f}{extra}"
            )
            class_names = metrics.get("class_names") or []
            per_class = metrics.get("per_class_iou") or []
            if isinstance(class_names, list) and isinstance(per_class, list):
                print(f"===> per class IoU of epoch {epoch}:")
                for name, value in zip(class_names, per_class):
                    print(f"===> {name} - IoU = {round(float(value), 2)}")
            if run is not None and wandb is not None:
                log_payload = {
                    "val/loss": metrics["loss"],
                    "val/miou": metrics["miou"],
                    "epoch": epoch,
                }
                if "focal" in metrics and "lovasz" in metrics:
                    log_payload["val/focal"] = metrics["focal"]
                    log_payload["val/lovasz"] = metrics["lovasz"]
                class_names = metrics.get("class_names") or []
                per_class = metrics.get("per_class_iou") or []
                if isinstance(class_names, list) and isinstance(per_class, list):
                    for name, value in zip(class_names, per_class):
                        log_payload[f"val/iou_{name}"] = value
                wandb.log(log_payload, step=epoch, commit=True)
        ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
        trainer.save_checkpoint(ckpt_path)
        extra = ""
        if "focal" in train_metrics and "lovasz" in train_metrics:
            extra = (
                f" focal={train_metrics['focal']:.4f}"
                f" lovasz={train_metrics['lovasz']:.4f}"
            )
        print(f"[train] epoch={epoch} avg_loss={avg_loss:.4f}{extra} -> {ckpt_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()


