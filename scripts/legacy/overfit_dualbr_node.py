#!/usr/bin/env python3
"""DualBR-Node 小样本过拟合脚本。"""

from __future__ import annotations

import argparse
import sys
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
from online_ncde.utils.reproducibility import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/dualbr_node/base_1s_tminus1.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--sample-count", type=int, default=10, help="过拟合样本数")
    parser.add_argument("--epochs", type=int, default=300, help="过拟合轮数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser.parse_args()


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

    dataset = Occ3DDualBrNodeDataset(
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
        supervision_labels=train_cfg["supervision_labels"],
    )
    sub = Subset(dataset, list(range(min(args.sample_count, len(dataset)))))
    loader = DataLoader(
        sub,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=dualbr_node_collate,
    )

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
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
    ).to(device)
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
        free_conf_thresh=cfg.get("eval", {}).get("free_conf_thresh", None),
        log_interval=int(train_cfg.get("log_interval", 10)),
        clip_norm=float(train_cfg.get("clip_norm", 5.0)),
        use_multistep_supervision=bool(train_cfg.get("use_multistep_supervision", True)),
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.5, 0.5])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(cfg.get("eval", {}).get("log_multistep_losses", True)),
        rollout_mode=train_cfg.get("rollout_mode", "one_second_tminus1"),
        primary_supervision_label=cfg.get("eval", {}).get("primary_supervision_label", "t-1.0"),
        stepwise_max_step_index=train_cfg.get("max_step_index", 6),
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_one_epoch(loader, epoch)
        eval_metrics = trainer.evaluate(loader)
        train_msg = " ".join(f"{k}={float(v):.4f}" for k, v in train_metrics.items())
        eval_msg = " ".join(
            f"{k}={float(v):.4f}"
            for k, v in eval_metrics.items()
            if isinstance(v, (int, float))
        )
        print(f"[overfit][epoch={epoch}] train {train_msg}")
        print(f"[overfit][epoch={epoch}] eval  {eval_msg}")


if __name__ == "__main__":
    main()
