#!/usr/bin/env python3
"""Train transport_online_nde."""

from __future__ import annotations

import argparse
import math
import numbers
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "utils"))

from transport_online_nde.config import load_config_with_base, resolve_path  # noqa: E402
from transport_online_nde.data.occ3d_transport_online_nde_dataset import (  # noqa: E402
    Occ3DTransportOnlineNdeDataset,
)
from transport_online_nde.losses import TransportOnlineNdeLoss  # noqa: E402
from transport_online_nde.models.transport_online_nde_aligner import (  # noqa: E402
    TransportOnlineNdeAligner,
)
from transport_online_nde.trainer import Trainer, transport_online_nde_collate  # noqa: E402
from transport_online_nde.utils.checkpoints import load_checkpoint  # noqa: E402
from transport_online_nde.utils.reproducibility import set_seed  # noqa: E402

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/transport_online_nde/train.yaml"),
        help="config file path",
    )
    parser.add_argument("--resume", default="", help="resume checkpoint")
    parser.add_argument("--train-limit", type=int, default=0, help="max train samples (0=all)")
    parser.add_argument("--eval-every", type=int, default=1, help="evaluate every N epochs")
    parser.add_argument(
        "--val-scene-count",
        type=int,
        default=40,
        help="number of shuffled validation scenes",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--wandb", action="store_true", help="enable wandb")
    parser.add_argument("--cosine-annealing", action="store_true", help="enable cosine lr schedule")
    parser.add_argument("--min-lr", type=float, default=1.0e-5, help="min lr for cosine")
    return parser.parse_args()


def build_subset(dataset, limit: int):
    if limit <= 0:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def build_scheduler(optimizer, train_cfg: dict, args):
    total_epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
    base_lr = float(train_cfg["lr"])

    if warmup_epochs == 0 and not args.cosine_annealing:
        return None

    warmup_start_lr = float(train_cfg.get("warmup_start_lr", 1e-5))
    start_factor = warmup_start_lr / base_lr
    min_factor = args.min_lr / base_lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return start_factor + (1.0 - start_factor) * epoch / max(warmup_epochs, 1)
        if not args.cosine_annealing:
            return 1.0
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def to_float(value):
    if isinstance(value, numbers.Real):
        return float(value)
    return None


def build_dataset(
    info_path: str,
    data_cfg: dict,
    root_path: str,
    fast_logits_root: str,
    slow_logit_root: str,
    supervision_sidecar_path: str | None = None,
) -> Occ3DTransportOnlineNdeDataset:
    return Occ3DTransportOnlineNdeDataset(
        info_path=info_path,
        root_path=root_path,
        fast_logits_root=fast_logits_root,
        slow_logit_root=slow_logit_root,
        gt_root=data_cfg["gt_root"],
        motion_gt_root=data_cfg["motion_gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        slow_noise_std=data_cfg.get("slow_noise_std", 0.0),
        topk_other_fill_value=data_cfg.get("topk_other_fill_value", -5.0),
        topk_free_fill_value=data_cfg.get("topk_free_fill_value", 5.0),
        supervision_sidecar_path=supervision_sidecar_path,
        timestamp_scale=float(data_cfg.get("timestamp_scale", 1.0e-6)),
        downsample_xy=int(data_cfg.get("downsample_xy", 2)),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    root_path = cfg["root_path"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    motion_cfg = model_cfg.get("motion", {})
    source_cfg = model_cfg.get("source", {})
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})
    loader_cfg = cfg.get("dataloader", {})
    wandb_cfg = cfg.get("wandb", {})

    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root and data.slow_logit_root are required")

    train_sidecar_path = data_cfg.get("train_supervision_sidecar_path", "")
    val_sidecar_path = data_cfg.get("val_supervision_sidecar_path", "")

    train_dataset = build_dataset(
        info_path=data_cfg["info_path"],
        data_cfg=data_cfg,
        root_path=root_path,
        fast_logits_root=fast_logits_root,
        slow_logit_root=slow_logit_root,
        supervision_sidecar_path=train_sidecar_path if train_sidecar_path else None,
    )
    train_dataset = build_subset(train_dataset, args.train_limit)

    num_workers = int(train_cfg["num_workers"])
    train_kwargs = dict(
        batch_size=int(train_cfg["batch_size"]),
        num_workers=num_workers,
        shuffle=True,
        collate_fn=transport_online_nde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        train_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        train_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    train_loader = DataLoader(train_dataset, **train_kwargs)

    val_loader = None
    val_info_path = data_cfg.get("val_info_path", "")
    if val_info_path:
        val_dataset = build_dataset(
            info_path=val_info_path,
            data_cfg=data_cfg,
            root_path=root_path,
            fast_logits_root=fast_logits_root,
            slow_logit_root=slow_logit_root,
            supervision_sidecar_path=val_sidecar_path if val_sidecar_path else None,
        )
        scene_names = [info.get("scene_name", "") for info in val_dataset.infos]
        unique_scenes = sorted({name for name in scene_names if name})
        if not unique_scenes:
            raise ValueError("No valid scene_name found for validation")
        rng = random.Random(0)
        rng.shuffle(unique_scenes)
        val_scene_count = min(args.val_scene_count, len(unique_scenes))
        val_scene_set = set(unique_scenes[:val_scene_count])
        val_indices = [i for i, name in enumerate(scene_names) if name in val_scene_set]
        if not val_indices:
            raise ValueError("Validation scene split is empty")
        val_dataset = Subset(val_dataset, val_indices)

        val_workers = int(eval_cfg.get("num_workers", num_workers))
        val_kwargs = dict(
            batch_size=int(eval_cfg.get("batch_size", 1)),
            num_workers=val_workers,
            shuffle=False,
            collate_fn=transport_online_nde_collate,
            pin_memory=loader_cfg.get("pin_memory", False),
        )
        if val_workers > 0:
            val_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
            val_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
        val_loader = DataLoader(val_dataset, **val_kwargs)

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

    start_epoch = 1
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, strict=False)
        start_epoch = payload.get("epoch", 0) + 1
        print(f"[resume] continue from epoch={start_epoch}")

    scheduler = build_scheduler(optimizer, train_cfg, args)
    if scheduler is not None and start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()

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
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        free_conf_thresh=eval_cfg.get("free_conf_thresh", None),
        log_interval=train_cfg.get("log_interval", 10),
        clip_norm=train_cfg.get("clip_norm", 5.0),
        use_multistep_supervision=bool(train_cfg.get("use_multistep_supervision", True)),
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.15, 0.20, 0.25, 0.40])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
    )

    run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed")
        run_name = wandb_cfg.get("name", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            entity=wandb_cfg.get("entity", "runheyang"),
            project=wandb_cfg.get("project", "neural-ode"),
            name=run_name,
            config={
                "epochs": int(train_cfg["epochs"]),
                "batch_size": int(train_cfg["batch_size"]),
                "lr": float(train_cfg["lr"]),
                "weight_decay": float(train_cfg["weight_decay"]),
            },
        )
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

    output_dir = resolve_path(root_path, train_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        train_metrics = trainer.train_one_epoch(train_loader, epoch=epoch)
        if scheduler is not None:
            scheduler.step()

        print(
            f"[train] epoch={epoch} "
            f"loss={train_metrics['loss']:.4f} "
            f"focal={train_metrics['focal']:.4f} "
            f"lovasz={train_metrics['aux']:.4f} "
            f"disp={train_metrics['disp']:.4f} "
            f"dyn={train_metrics['delta_dyn_abs_mean']:.4f} "
            f"adv={train_metrics['advect_disp_abs_mean']:.4f}"
        )

        if run is not None:
            payload = {f"train/{k}": float(v) for k, v in train_metrics.items()}
            payload["epoch"] = float(epoch)
            should_eval = val_loader is not None and args.eval_every > 0 and epoch % args.eval_every == 0
            run.log(payload, commit=not should_eval)

        if val_loader is not None and args.eval_every > 0 and epoch % args.eval_every == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(
                f"[eval] epoch={epoch} "
                f"loss={val_metrics['loss']:.4f} "
                f"focal={val_metrics['focal']:.4f} "
                f"lovasz={val_metrics['aux']:.4f} "
                f"disp={val_metrics['disp']:.4f} "
                f"miou={val_metrics['miou']:.4f}"
            )
            collapse_parts = []
            for key in (
                "collapse_moving_fg_pred_gt_norm_ratio",
                "collapse_moving_fg_epe",
                "collapse_moving_fg_near_zero_rate",
                "collapse_motion_mask_moving_fg_mean",
                "collapse_motion_mask_bgfree_mean",
                "collapse_motion_mask_fg_bg_gap",
            ):
                value = to_float(val_metrics.get(key, None))
                if value is not None:
                    collapse_parts.append(f"{key}={value:.4f}")
            voxels_value = to_float(val_metrics.get("collapse_moving_fg_voxels", None))
            if voxels_value is not None:
                collapse_parts.append(f"collapse_moving_fg_voxels={voxels_value:.0f}")
            if collapse_parts:
                print(f"[eval-collapse] epoch={epoch} " + " ".join(collapse_parts))
            class_names = val_metrics.get("class_names", [])
            class_iou = val_metrics.get("per_class_iou", [])
            if isinstance(class_names, list) and isinstance(class_iou, list):
                print(f"===> per class IoU of epoch {epoch}:")
                for name, value in zip(class_names, class_iou):
                    print(f"===> {name} - IoU = {round(float(value), 2)}")

            if run is not None:
                payload = {"epoch": float(epoch)}
                for key in ("loss", "focal", "aux", "disp", "miou"):
                    value = to_float(val_metrics.get(key, None))
                    if value is not None:
                        payload[f"val/{key}"] = value
                for key in (
                    "collapse_moving_fg_pred_gt_norm_ratio",
                    "collapse_moving_fg_epe",
                    "collapse_moving_fg_near_zero_rate",
                    "collapse_motion_mask_moving_fg_mean",
                    "collapse_motion_mask_bgfree_mean",
                    "collapse_motion_mask_fg_bg_gap",
                    "collapse_moving_fg_voxels",
                ):
                    value = to_float(val_metrics.get(key, None))
                    if value is not None:
                        payload[f"val/{key}"] = value
                if isinstance(class_names, list) and isinstance(class_iou, list):
                    for name, value in zip(class_names, class_iou):
                        score = to_float(value)
                        if score is not None:
                            payload[f"val/iou_{name}"] = score
                run.log(payload, commit=True)

        ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
        trainer.save_checkpoint(ckpt_path, epoch=epoch)
        print(f"[ckpt] saved -> {ckpt_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
