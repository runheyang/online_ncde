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
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

# 使用 file_system 共享策略，避免低 ulimit 下多 epoch 重建 DataLoader 导致 fd 耗尽
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base  # noqa: E402
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
    parser.add_argument("--rayiou", action="store_true", help="评估时额外计算 RayIoU")
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
    logits_loader,
    ray_sidecar_split: str | None = None,
) -> Occ3DOnlineNcdeDataset:
    """根据 data_cfg 构造 Occ3DOnlineNcdeDataset。"""
    return Occ3DOnlineNcdeDataset(
        info_path=info_path,
        root_path=root_path,
        gt_root=data_cfg["gt_root"],
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        gt_mask_key=data_cfg["gt_mask_key"],
        logits_loader=logits_loader,
        ray_sidecar_dir=data_cfg.get("ray_sidecar_dir", None),
        ray_sidecar_split=ray_sidecar_split,
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
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
    logits_loader = build_logits_loader(data_cfg, root_path)

    train_dataset = build_dataset(
        info_path=data_cfg["info_path"],
        data_cfg=data_cfg,
        root_path=root_path,
        logits_loader=logits_loader,
        ray_sidecar_split="train",
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
            logits_loader=logits_loader,
            ray_sidecar_split="val",
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
    resumed_payload = None
    if args.resume:
        resumed_payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, strict=False)
        start_epoch = resumed_payload.get("epoch", 0) + 1
        print(f"[resume] 从 epoch={start_epoch} 继续训练")

    # EMA：默认开启，用于抑制 val 指标震荡
    ema = None
    ema_cfg = train_cfg.get("ema", {}) or {}
    if bool(ema_cfg.get("enabled", True)):
        from online_ncde.utils.ema import ModelEMA
        ema_decay = float(ema_cfg.get("decay", 0.999))
        ema = ModelEMA(model, decay=ema_decay, device=device)
        print(f"[ema] enabled, decay={ema_decay}")
        if resumed_payload is not None and "ema" in resumed_payload:
            ema.load_state_dict(resumed_payload["ema"])
            print(f"[ema] resumed num_updates={ema.num_updates}")

    scheduler = build_scheduler(optimizer, train_cfg, args)
    if scheduler is not None and start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()

    # 给 build_loss 注入 ray 需要的几何常量（yaml 里不重复填）。
    ray_cfg = loss_cfg.get("ray", None)
    if ray_cfg is not None:
        ray_cfg.setdefault("pc_range", list(data_cfg["pc_range"]))
        ray_cfg.setdefault("free_index", int(data_cfg["free_index"]))
    loss_fn = build_loss(loss_cfg, num_classes=data_cfg["num_classes"]).to(device)
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
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.15, 0.20, 0.25, 0.40])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=str(train_cfg.get("rollout_mode", "full")),
        primary_supervision_label=str(eval_cfg.get("primary_supervision_label", "t-1.0")),
        stepwise_max_step_index=train_cfg.get("max_step_index", None),
        ema=ema,
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

    # 从 config 路径推导输出目录：configs/X/Y/train.yaml → outputs/X/Y/{timestamp}_100x100x16
    config_rel = os.path.relpath(args.config, os.path.join(str(ROOT), "configs"))
    output_base = os.path.join(str(ROOT), "outputs", os.path.dirname(config_rel))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"{timestamp}_100x100x16")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[ckpt] output_dir: {output_dir}")

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
        # 仅在本 epoch 有 ray 监督样本时打印 ray 分项。
        ray_total_text = ""
        if "ray" in train_metrics:
            ray_total_text = (
                f" ray_total={float(train_metrics['ray']):.4f}"
                f" ray_hit={float(train_metrics['ray_hit']):.4f}"
                f" ray_empty={float(train_metrics['ray_empty']):.4f}"
                f" ray_pre_free={float(train_metrics.get('ray_pre_free', 0.0)):.4f}"
                f" ray_depth={float(train_metrics['ray_depth']):.4f}"
            )
        print(
            f"[train] epoch={epoch} "
            f"loss={train_metrics['loss']:.4f} "
            f"focal={train_metrics['focal']:.4f} "
            f"aux={train_metrics['aux']:.4f} "
            f"delta={train_metrics['delta_scene_abs_mean']:.4f}"
            f"{ray_total_text}"
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
            val_metrics = trainer.evaluate(val_loader, collect_predictions=args.rayiou)
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

            # --- RayIoU ---
            rayiou_result = None
            if args.rayiou:
                from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
                from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou

                sweep_rel = eval_cfg.get("sweep_pkl", "data/nuscenes/nuscenes_infos_val_sweep.pkl")
                sweep_path = Path(sweep_rel)
                sweep_pkl = str(sweep_path if sweep_path.is_absolute() else (ROOT / sweep_path).resolve())
                if epoch == start_epoch:
                    print(f"[rayiou] sweep pkl: {sweep_pkl}")

                origins_by_token = load_origins_from_sweep_pkl(sweep_pkl)
                predictions = val_metrics["predictions"]
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
                    print(f"[rayiou] epoch={epoch} 跳过 {skipped} 个样本（无对应 lidar origin）")
                print(f"[rayiou] epoch={epoch} {len(sem_pred_list)} 个样本参与计算")

                rayiou_result = calc_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list)
                print(
                    f"[rayiou] epoch={epoch} "
                    f"RayIoU={rayiou_result['RayIoU']:.4f} "
                    f"RayIoU@1={rayiou_result['RayIoU@1']:.4f} "
                    f"RayIoU@2={rayiou_result['RayIoU@2']:.4f} "
                    f"RayIoU@4={rayiou_result['RayIoU@4']:.4f}"
                )

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
                if rayiou_result is not None:
                    for key in ("RayIoU", "RayIoU@1", "RayIoU@2", "RayIoU@4"):
                        payload[f"val/{key}"] = float(rayiou_result[key])
                run.log(payload, commit=True)

        ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
        trainer.save_checkpoint(ckpt_path, epoch=epoch)
        print(f"[ckpt] saved -> {ckpt_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
