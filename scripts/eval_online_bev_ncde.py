#!/usr/bin/env python3
"""评估 Online BEV NCDE。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "utils"))

from online_bev_ncde.data.occ3d_online_bev_ncde_dataset import Occ3DOnlineBevNcdeDataset  # noqa: E402
from online_bev_ncde.models.online_bev_ncde_aligner import build_model_from_config  # noqa: E402
from online_bev_ncde.trainer import Trainer, online_ncde_collate  # noqa: E402
from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.losses import build_loss  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/online_bev_ncde/eval.yaml"), help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    return parser.parse_args()


def build_model(data_cfg: dict, model_cfg: dict, eval_cfg: dict):
    return build_model_from_config(data_cfg, model_cfg, amp_fp16=bool(eval_cfg.get("amp_fp16", False)))


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    fast_logits_root = data_cfg.get("fast_logits_root", data_cfg.get("logits_root", ""))
    slow_logit_root = data_cfg.get("slow_logit_root", "")
    if not fast_logits_root or not slow_logit_root:
        raise KeyError("data.fast_logits_root 和 data.slow_logit_root 为必填项。")

    dataset = Occ3DOnlineBevNcdeDataset(
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
        supervision_sidecar_path=data_cfg.get("val_supervision_sidecar_path", "") or None,
    )

    num_workers = int(eval_cfg.get("num_workers", 4))
    kwargs = dict(
        batch_size=int(eval_cfg.get("batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=online_ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(dataset, **kwargs)

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(data_cfg, model_cfg, eval_cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, strict=False)

    loss_cfg = cfg["loss"]
    loss_fn = build_loss(loss_cfg, num_classes=data_cfg["num_classes"])
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
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
    )
    metrics = trainer.evaluate(loader)
    print(f"[eval] loss={metrics['loss']:.4f} focal={metrics['focal']:.4f} lovasz={metrics['aux']:.4f} miou={metrics['miou']:.4f}")
    class_names = metrics.get("class_names", [])
    class_iou = metrics.get("per_class_iou", [])
    if isinstance(class_names, list) and isinstance(class_iou, list):
        for name, value in zip(class_names, class_iou):
            print(f"{name}: {float(value):.2f}")


if __name__ == "__main__":
    main()
