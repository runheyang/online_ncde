#!/usr/bin/env python3
"""评估 DualBR-Node。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "utils"))

from dualbr_node.config import load_config_with_base  # noqa: E402
from dualbr_node.data.occ3d_dualbr_node_dataset import Occ3DDualBrNodeDataset  # noqa: E402
from dualbr_node.losses import DualBrNodeLoss  # noqa: E402
from dualbr_node.models.dualbr_node_aligner import DualBrNodeAligner  # noqa: E402
from dualbr_node.trainer import Trainer, dualbr_node_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/dualbr_node/base_1s_tminus1.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root 和 data.slow_logit_root 为必填项。")

    dataset = Occ3DDualBrNodeDataset(
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
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
        supervision_sidecar_path=data_cfg.get("val_supervision_sidecar_path", None),
        supervision_labels=cfg["train"]["supervision_labels"],
    )

    num_workers = int(eval_cfg.get("num_workers", 0))
    kwargs = dict(
        batch_size=int(eval_cfg.get("batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=dualbr_node_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(dataset, **kwargs)

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = DualBrNodeAligner(
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
        amp_fp16=bool(eval_cfg.get("amp_fp16", False)),
    ).to(device)
    load_checkpoint(args.checkpoint, model=model, strict=False)

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
        optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-4),
        loss_fn=loss_fn,
        device=device,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        free_conf_thresh=eval_cfg.get("free_conf_thresh", None),
        log_interval=int(eval_cfg.get("log_interval", 20)),
        clip_norm=1.0,
        use_multistep_supervision=True,
        supervision_labels=list(cfg["train"]["supervision_labels"]),
        supervision_weights=list(cfg["train"]["supervision_weights"]),
        supervision_weight_normalize=bool(cfg["train"].get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=cfg["train"].get("rollout_mode", "one_second_tminus1"),
        primary_supervision_label=eval_cfg.get("primary_supervision_label", "t-1.0"),
        stepwise_max_step_index=cfg["train"].get("max_step_index", 6),
    )
    metrics = trainer.evaluate(loader)
    print(
        f"[eval] loss={metrics['loss']:.4f} "
        f"focal={metrics['focal']:.4f} "
        f"lovasz={metrics['aux']:.4f} "
        f"miou={metrics['miou']:.4f}"
    )
    class_names = metrics.get("class_names", [])
    class_iou = metrics.get("per_class_iou", [])
    if isinstance(class_names, list) and isinstance(class_iou, list):
        for name, value in zip(class_names, class_iou):
            print(f"{name}: {float(value):.2f}")


if __name__ == "__main__":
    main()
