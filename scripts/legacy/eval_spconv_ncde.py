#!/usr/bin/env python3
"""评估 SpConv NCDE 对齐器。"""

from __future__ import annotations

import argparse
import os
import sys
import random
from pathlib import Path

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
    parser.add_argument("--ckpt", required=True, help="模型权重路径")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制评估样本数量（用于快速测试）",
    )
    parser.add_argument(
        "--val-scene-count",
        type=int,
        default=None,
        help="验证集使用的 scene 数量（默认 30，按 seed=0 打乱后取前 N）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    if torch.cuda.is_available():
        # 显式开启 TF32，利用 Tensor Core 加速 FP32 计算
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    root_path = cfg["root_path"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})

    sharing_strategy = loader_cfg.get("sharing_strategy", None)
    if sharing_strategy:
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    info_path = data_cfg.get("val_info_path", data_cfg["info_path"])
    dataset = Occ3DNcdeDataset(
        info_path=info_path,
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

    if args.val_scene_count and args.val_scene_count > 0:
        scene_names = [info.get("scene_name", "") for info in dataset.infos]
        unique_scenes = list(dict.fromkeys(scene_names))
        rng = random.Random(0)
        rng.shuffle(unique_scenes)
        selected = set(unique_scenes[: args.val_scene_count])
        indices = [i for i, name in enumerate(scene_names) if name in selected]
        dataset = Subset(dataset, indices)
        print(f"[eval] 使用 {args.val_scene_count} 个 scene，共 {len(indices)} 个样本")

    if args.max_samples is not None:
        max_samples = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))
        print(f"[eval] 使用前 {max_samples} 个样本进行评估")

    num_workers = eval_cfg["num_workers"]
    loader_kwargs = dict(
        batch_size=eval_cfg["batch_size"],
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    loader = DataLoader(dataset, **loader_kwargs)

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = NcdeAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=cfg["model"]["feat_dim"],
        encoder_in_channels=cfg["model"]["encoder_in_channels"],
        decoder_init_scale=cfg["model"]["decoder_init_scale"],
        add_time_channel=cfg["ncde"]["add_time_channel"],
        time_scale=cfg["ncde"]["time_scale"],
        eps=cfg["ncde"]["eps"],
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    loss_fn = FocalLovaszLoss(
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        gamma=cfg["loss"]["gamma"],
        class_weights=cfg["loss"].get("class_weights", None),
        lovasz_weight=cfg["loss"].get("lovasz_weight", 1.0),
    )

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        loss_fn=loss_fn,
        device=device,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        fast_conf_thresh=data_cfg.get("fast_conf_thresh", None),
        free_conf_thresh=data_cfg.get("fast_conf_thresh", None),
        log_interval=10,
    )

    results = trainer.evaluate(loader)
    output_dir = resolve_path(root_path, eval_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    extra = ""
    if "focal" in results and "lovasz" in results:
        extra = f" focal={results['focal']:.4f} lovasz={results['lovasz']:.4f}"
    print(f"[eval] loss={results['loss']:.4f} miou={results['miou']:.2f}{extra}")
    class_names = results.get("class_names") or []
    per_class = results.get("per_class_iou") or []
    if isinstance(class_names, list) and isinstance(per_class, list):
        print("===> per class IoU:")
        for name, value in zip(class_names, per_class):
            print(f"===> {name} - IoU = {round(float(value), 2)}")


if __name__ == "__main__":
    main()


