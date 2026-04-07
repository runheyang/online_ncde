#!/usr/bin/env python3
"""训练 DualBR-Node。"""

from __future__ import annotations

import argparse
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

from dualbr_node.config import load_config_with_base  # noqa: E402
from dualbr_node.data.occ3d_dualbr_node_dataset import Occ3DDualBrNodeDataset  # noqa: E402
from dualbr_node.losses import DualBrNodeLoss  # noqa: E402
from dualbr_node.models.dualbr_node_aligner import DualBrNodeAligner  # noqa: E402
from dualbr_node.trainer import Trainer, dualbr_node_collate  # noqa: E402
from online_ncde.config import resolve_path  # noqa: E402
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
        default=str(ROOT / "configs/dualbr_node/base_1s_tminus1.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--resume", default="", help="恢复训练权重")
    parser.add_argument("--train-limit", type=int, default=0, help="训练样本上限（0=全量）")
    parser.add_argument("--eval-every", type=int, default=1, help="每隔多少个 epoch 做一次验证")
    parser.add_argument(
        "--val-scene-count",
        type=int,
        default=40,
        help="验证集使用的 scene 数量（按 seed=0 打乱后取前 N）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb")
    return parser.parse_args()


def build_dataset(
    cfg: dict,
    info_path: str,
    supervision_sidecar_path: str | None,
) -> Occ3DDualBrNodeDataset:
    data_cfg = cfg["data"]
    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root 和 data.slow_logit_root 为必填项。")
    return Occ3DDualBrNodeDataset(
        info_path=info_path,
        root_path=cfg["root_path"],
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
        supervision_labels=cfg["train"]["supervision_labels"],
    )


def build_loader(dataset, batch_size: int, num_workers: int, loader_cfg: dict, shuffle: bool):
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dualbr_node_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    return DataLoader(dataset, **kwargs)


def build_model(cfg: dict, amp_fp16: bool = False) -> DualBrNodeAligner:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    return DualBrNodeAligner(
        num_classes=data_cfg["num_classes"],
        dense_feat_dim=model_cfg.get("dense_feat_dim", model_cfg.get("feat_dim", 32)),
        sparse_feat_dim=model_cfg.get("sparse_feat_dim", model_cfg.get("feat_dim", 32)),
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        sparse_class_ids=list(model_cfg["sparse_class_ids"]),
        fast_occ_thresh=data_cfg.get("fast_occ_thresh", 0.25),
        decoder_init_scale=model_cfg.get("decoder_init_scale", 1.0e-6),
        fusion_mode=model_cfg.get("fusion_mode", "masked_logit_blend"),
        mask_channels=model_cfg.get("mask_channels", 1),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
        amp_fp16=amp_fp16,
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

    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})
    loss_cfg = cfg["loss"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loader_cfg = cfg.get("dataloader", {})
    wandb_cfg = cfg.get("wandb", {})

    train_dataset = build_dataset(
        cfg=cfg,
        info_path=data_cfg["info_path"],
        supervision_sidecar_path=data_cfg.get("train_supervision_sidecar_path", None),
    )
    if args.train_limit > 0:
        train_dataset = Subset(train_dataset, list(range(min(args.train_limit, len(train_dataset)))))
    val_dataset = build_dataset(
        cfg=cfg,
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
        supervision_sidecar_path=data_cfg.get("val_supervision_sidecar_path", None),
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

    train_loader = build_loader(
        dataset=train_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        loader_cfg=loader_cfg,
        shuffle=True,
    )
    val_loader = build_loader(
        dataset=val_dataset,
        batch_size=int(eval_cfg.get("batch_size", 1)),
        num_workers=int(eval_cfg.get("num_workers", 0)),
        loader_cfg=loader_cfg,
        shuffle=False,
    )

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg=cfg, amp_fp16=bool(eval_cfg.get("amp_fp16", False))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-2)),
    )
    loss_fn = DualBrNodeLoss(
        num_classes=data_cfg["num_classes"],
        gamma=loss_cfg.get("gamma", 2.0),
        class_weights=loss_cfg.get("class_weights", None),
        lambda_focal=loss_cfg.get("lambda_focal", 1.0),
        lambda_lovasz=loss_cfg.get("lambda_lovasz", 1.0),
        lambda_dense_aux=loss_cfg.get("lambda_dense_aux", 0.0),
        lambda_sparse_aux=loss_cfg.get("lambda_sparse_aux", 0.0),
        lambda_mask=loss_cfg.get("lambda_mask", 0.0),
        mask_positive_class_ids=model_cfg.get(
            "mask_positive_class_ids",
            model_cfg.get("sparse_class_ids", []),
        ),
        mask_pos_weight=loss_cfg.get("mask_pos_weight", 1.0),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        free_conf_thresh=eval_cfg.get("free_conf_thresh", None),
        log_interval=int(train_cfg.get("log_interval", 10)),
        clip_norm=float(train_cfg.get("clip_norm", 5.0)),
        use_multistep_supervision=bool(train_cfg.get("use_multistep_supervision", True)),
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.5, 0.5])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=train_cfg.get("rollout_mode", "one_second_tminus1"),
        primary_supervision_label=eval_cfg.get("primary_supervision_label", "t-1.0"),
        stepwise_max_step_index=train_cfg.get("max_step_index", 6),
    )

    if args.resume:
        load_checkpoint(args.resume, model=model, optimizer=optimizer, strict=False)

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
                "batch_size": int(train_cfg.get("batch_size", 1)),
                "lr": float(train_cfg["lr"]),
                "weight_decay": float(train_cfg.get("weight_decay", 1.0e-2)),
            },
        )
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

    output_dir = resolve_path(cfg["root_path"], train_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    best_miou = float("-inf")
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_metrics = trainer.train_one_epoch(train_loader, epoch)
        train_msg = " ".join(f"{k}={float(v):.4f}" for k, v in train_metrics.items())
        print(f"[train][epoch={epoch}] {train_msg}")
        if run is not None:
            train_payload = {f"train/{k}": float(v) for k, v in train_metrics.items()}
            train_payload["epoch"] = float(epoch)
            should_eval = args.eval_every > 0 and epoch % args.eval_every == 0
            run.log(train_payload, commit=not should_eval)

        trainer.save_checkpoint(os.path.join(output_dir, "latest.pth"), epoch=epoch)
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            eval_metrics = trainer.evaluate(val_loader)
            eval_msg = " ".join(
                f"{k}={float(v):.4f}"
                for k, v in eval_metrics.items()
                if isinstance(v, (int, float))
            )
            print(f"[eval][epoch={epoch}] {eval_msg}")
            class_names = eval_metrics.get("class_names", [])
            per_class = eval_metrics.get("per_class_iou", [])
            if isinstance(class_names, list) and isinstance(per_class, list):
                print(f"===> per class IoU of epoch {epoch}:")
                for name, value in zip(class_names, per_class):
                    print(f"===> {name} - IoU = {round(float(value), 2)}")
            if run is not None:
                payload = {"epoch": float(epoch)}
                for key in ("loss", "focal", "aux", "miou"):
                    value = to_float(eval_metrics.get(key, None))
                    if value is not None:
                        payload[f"val/{key}"] = value
                for key, value in eval_metrics.items():
                    if isinstance(key, str) and key.startswith("sup_loss_t"):
                        score = to_float(value)
                        if score is not None:
                            payload[f"val/{key}"] = score
                class_names = eval_metrics.get("class_names", [])
                per_class = eval_metrics.get("per_class_iou", [])
                if isinstance(class_names, list) and isinstance(per_class, list):
                    for name, value in zip(class_names, per_class):
                        score = to_float(value)
                        if score is not None:
                            payload[f"val/iou_{name}"] = score
                run.log(payload, commit=True)
            miou = float(eval_metrics.get("miou", 0.0))
            if miou > best_miou:
                best_miou = miou
                trainer.save_checkpoint(os.path.join(output_dir, "best.pth"), epoch=epoch)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
