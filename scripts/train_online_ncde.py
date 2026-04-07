#!/usr/bin/env python3
"""训练 Online NCDE。"""

from __future__ import annotations

import argparse
import numbers
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import math

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "utils"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.losses import build_loss  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde.trainer import Trainer, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint  # noqa: E402
from online_ncde.utils.reproducibility import set_seed  # noqa: E402

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/online_ncde/fast_opusv1t__slow_opusv2l/train.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--resume", default="", help="恢复训练权重")
    parser.add_argument("--train-limit", type=int, default=0, help="训练样本上限（0=全量）")
    parser.add_argument("--eval-every", type=int, default=1, help="每隔多少 epoch 评估一次")
    parser.add_argument(
        "--val-scene-count",
        type=int,
        default=40,
        help="验证集使用的 scene 数量（按 seed=0 打乱后取前 N）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb")
    parser.add_argument("--cosine-annealing", action="store_true", help="启用余弦退火学习率调度")
    parser.add_argument("--min-lr", type=float, default=1.0e-5, help="余弦退火最低学习率")
    return parser.parse_args()


def build_subset(dataset, limit: int):
    if limit <= 0:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def build_scheduler(optimizer, train_cfg: dict, args):
    """构建学习率调度器（线性 warmup + 可选余弦退火），使用 LambdaLR 避免 SequentialLR 的废弃警告。"""
    total_epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
    base_lr = float(train_cfg["lr"])
    use_cosine = args.cosine_annealing

    if warmup_epochs == 0 and not use_cosine:
        return None

    warmup_start_lr = float(train_cfg.get("warmup_start_lr", 1e-5))
    start_factor = warmup_start_lr / base_lr
    min_factor = args.min_lr / base_lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # 线性 warmup：从 start_factor 线性增到 1.0
            return start_factor + (1.0 - start_factor) * epoch / max(warmup_epochs, 1)
        if not use_cosine:
            return 1.0
        # 余弦退火：从 1.0 降到 min_factor
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_dataset(
    info_path: str,
    data_cfg: dict,
    root_path: str,
    fast_logits_root: str,
    slow_logit_root: str,
    supervision_sidecar_path: str | None = None,
    logits_loader=None,
) -> Occ3DOnlineNcdeDataset:
    """根据 data_cfg 构造 Occ3DOnlineNcdeDataset。"""
    return Occ3DOnlineNcdeDataset(
        info_path=info_path,
        root_path=root_path,
        fast_logits_root=fast_logits_root,
        slow_logit_root=slow_logit_root,
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        slow_noise_std=data_cfg.get("slow_noise_std", 0.0),
        topk_other_fill_value=data_cfg.get("topk_other_fill_value", -5.0),
        topk_free_fill_value=data_cfg.get("topk_free_fill_value", 5.0),
        supervision_sidecar_path=supervision_sidecar_path,
        fast_logits_variant=data_cfg.get("fast_logits_variant", "topk"),
        slow_logit_variant=data_cfg.get("slow_logit_variant", "topk"),
        full_logits_clamp_min=data_cfg.get("full_logits_clamp_min", None),
        full_topk_k=data_cfg.get("full_topk_k", 3),
        logits_loader=logits_loader,
    )


def to_float(value):
    """将标量安全转换为 Python float，无法转换时返回 None。"""
    if isinstance(value, numbers.Real):
        return float(value)
    return None


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
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})
    loader_cfg = cfg.get("dataloader", {})
    wandb_cfg = cfg.get("wandb", {})
    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root 和 data.slow_logit_root 为必填项。")
    train_sidecar_path = data_cfg.get("train_supervision_sidecar_path", "")
    val_sidecar_path = data_cfg.get("val_supervision_sidecar_path", "")
    logits_loader = build_logits_loader(data_cfg, root_path)

    train_dataset = build_dataset(
        info_path=data_cfg["info_path"],
        data_cfg=data_cfg,
        root_path=root_path,
        fast_logits_root=fast_logits_root,
        slow_logit_root=slow_logit_root,
        supervision_sidecar_path=train_sidecar_path if train_sidecar_path else None,
        logits_loader=logits_loader,
    )
    train_dataset = build_subset(train_dataset, args.train_limit)

    num_workers = int(train_cfg["num_workers"])
    loader_kwargs = dict(
        batch_size=int(train_cfg["batch_size"]),
        num_workers=num_workers,
        shuffle=True,
        collate_fn=online_ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    train_loader = DataLoader(train_dataset, **loader_kwargs)

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
            logits_loader=logits_loader,
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

        val_workers = int(eval_cfg.get("num_workers", num_workers))
        val_kwargs = dict(
            batch_size=int(eval_cfg.get("batch_size", 1)),
            num_workers=val_workers,
            shuffle=False,
            collate_fn=online_ncde_collate,
            pin_memory=loader_cfg.get("pin_memory", False),
        )
        if val_workers > 0:
            val_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
            val_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
        val_loader = DataLoader(val_dataset, **val_kwargs)

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = OnlineNcdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        fast_occ_thresh=data_cfg.get("fast_occ_thresh", 0.25),
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

    start_epoch = 1
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, strict=False)
        start_epoch = payload.get("epoch", 0) + 1
        print(f"[resume] 从 epoch={start_epoch} 继续训练")

    scheduler = build_scheduler(optimizer, train_cfg, args)
    if scheduler is not None and start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()

    loss_fn = build_loss(loss_cfg, num_classes=data_cfg["num_classes"])
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
        use_multistep_supervision=bool(train_cfg.get("use_multistep_supervision", False)),
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.15, 0.20, 0.25, 0.40])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=str(train_cfg.get("rollout_mode", "full")),
        primary_supervision_label=str(eval_cfg.get("primary_supervision_label", "t-1.0")),
        stepwise_max_step_index=train_cfg.get("max_step_index", None),
    )

    run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("未安装 wandb，无法启用日志。")
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
        # 统一用 epoch 作为可视化横轴，而不是 wandb 默认的 _step。
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

    output_dir = resolve_path(root_path, train_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        train_metrics = trainer.train_one_epoch(
            train_loader,
            epoch=epoch,
        )
        if scheduler is not None:
            scheduler.step()
        train_sup_parts = []
        for key, value in train_metrics.items():
            if key.startswith("loss_t"):
                train_sup_parts.append(f"{key}={float(value):.4f}")
        train_sup_text = (" " + " ".join(train_sup_parts)) if train_sup_parts else ""
        print(
            f"[train] epoch={epoch} "
            f"loss={train_metrics['loss']:.4f} "
            f"focal={train_metrics['focal']:.4f} "
            f"aux={train_metrics['aux']:.4f} "
            f"delta={train_metrics['delta_scene_abs_mean']:.4f}"
            f"{train_sup_text}"
        )
        if run is not None:
            train_payload = {f"train/{k}": float(v) for k, v in train_metrics.items()}
            train_payload["epoch"] = float(epoch)
            should_eval = (
                val_loader is not None
                and args.eval_every > 0
                and epoch % args.eval_every == 0
            )
            run.log(train_payload, commit=not should_eval)

        if val_loader is not None and args.eval_every > 0 and epoch % args.eval_every == 0:
            val_metrics = trainer.evaluate(val_loader)
            val_sup_parts = []
            for key, value in val_metrics.items():
                if isinstance(key, str) and key.startswith("sup_loss_t"):
                    val_sup_parts.append(f"{key}={float(value):.4f}")
            val_sup_text = (" " + " ".join(val_sup_parts)) if val_sup_parts else ""
            print(
                f"[eval] epoch={epoch} "
                f"loss={val_metrics['loss']:.4f} "
                f"focal={val_metrics['focal']:.4f} "
                f"aux={val_metrics['aux']:.4f} "
                f"miou={val_metrics['miou']:.4f} "
                f"miou_d={val_metrics.get('miou_d', float('nan')):.4f}"
                f"{val_sup_text}"
            )
            class_names = val_metrics.get("class_names", [])
            per_class = val_metrics.get("per_class_iou", [])
            if isinstance(class_names, list) and isinstance(per_class, list):
                print(f"===> per class IoU of epoch {epoch}:")
                for name, value in zip(class_names, per_class):
                    print(f"===> {name} - IoU = {round(float(value), 2)}")
            if run is not None:
                payload = {"epoch": float(epoch)}
                for key in ("loss", "focal", "aux", "miou", "miou_d"):
                    value = to_float(val_metrics.get(key, None))
                    if value is not None:
                        payload[f"val/{key}"] = value
                for key, value in val_metrics.items():
                    if isinstance(key, str) and key.startswith("sup_loss_t"):
                        score = to_float(value)
                        if score is not None:
                            payload[f"val/{key}"] = score
                if isinstance(class_names, list) and isinstance(per_class, list):
                    for name, value in zip(class_names, per_class):
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
