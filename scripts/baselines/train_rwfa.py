#!/usr/bin/env python3
"""RWFA baseline 训练脚本（单卡）。

设计：复用 trainer / dataset / loss / RayIoU 的全套 wiring，只把 model 构造
换成 RecurrentWarpFusionAligner（fusion_kind=conv 或 attn）。RWFA 的 forward
接口与 OnlineNcdeAligner 完全一致（default / stepwise_train / stepwise_eval、
_fast_kl_active 协议），trainer 内部不需要任何分支改动。

精简掉的功能（baseline 实验场景一般不需要，需要时再扩）：
  - DDP（单卡训完即可）
  - wandb / 分箱 RayIoU / metrics json

CLI 与 train_online_ncde.py 兼容子集：相同 config 文件可直接复用，model 段
新增字段 fusion_inner_dim / fusion_body_dilations / fusion_attn_* 走默认即可。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

torch.backends.cudnn.benchmark = True
# 与 train_online_ncde 对齐：fp32 + TF32
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))  # 复用 train_online_ncde 中的 helper

# 复用主训练脚本里已经写好的 helper，避免重复实现
from train_online_ncde import (  # noqa: E402
    build_dataset,
    build_scheduler,
    build_subset,
    to_float,
)

from online_ncde.baselines import RecurrentWarpFusionAligner  # noqa: E402
from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.losses import build_loss  # noqa: E402
from online_ncde.trainer import Trainer, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint  # noqa: E402
from online_ncde.utils.ema import ModelEMA  # noqa: E402
from online_ncde.utils.reproducibility import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径（与 NCDE 同 config 兼容）")
    parser.add_argument(
        "--model-kind",
        choices=["rwfa-conv", "rwfa-attn"],
        default="rwfa-conv",
        help="RWFA 主干类型；conv = 3 dilated conv 残差块；attn = dilated + W-MSA + SW-MSA + dilated",
    )
    parser.add_argument("--resume", default="", help="恢复训练权重")
    parser.add_argument("--train-limit", type=int, default=0, help="训练样本上限（0=全量）")
    parser.add_argument("--val-scene-count", type=int, default=0,
                        help="按 scene 抽样验证集（0=全量），加速训练时小验）")
    parser.add_argument("--eval-every", type=int, default=1, help="每隔多少 epoch 评估一次")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cosine-annealing", action="store_true", help="启用余弦退火")
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--epochs", type=int, default=0, help="覆盖 config epochs（0=不覆盖）")
    parser.add_argument("--lambda-fast-kl", type=float, default=None, help="覆盖 train.lambda_fast_kl")
    parser.add_argument("--lambda-lovasz", type=float, default=None)
    parser.add_argument("--lambda-focal", type=float, default=None)
    parser.add_argument("--ray-override", type=str, default="", help="JSON 覆盖 loss.ray")
    parser.add_argument("--fast-logits-root", type=str, default=None, help="覆盖 data.fast_logits_root")
    parser.add_argument("--no-rayiou", action="store_true", help="评估时跳过 RayIoU（更快）")
    return parser.parse_args()


def _cleanup_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_model(model_kind: str, model_cfg: dict, data_cfg: dict, device: torch.device) -> RecurrentWarpFusionAligner:
    """根据 fusion_kind 构造 RWFA。conv/attn 共用同一个 Aligner 类。"""
    fusion_kind = "conv" if model_kind == "rwfa-conv" else "attn"
    return RecurrentWarpFusionAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=model_cfg.get("decoder_init_scale", 1.0e-3),
        use_fast_residual=bool(model_cfg.get("use_fast_residual", True)),
        fusion_kind=fusion_kind,
        fusion_inner_dim=int(model_cfg.get("fusion_inner_dim", 32)),
        fusion_body_dilations=tuple(model_cfg.get("fusion_body_dilations", [1, 2, 3])),
        fusion_gn_groups=int(model_cfg.get("fusion_gn_groups", 8)),
        fusion_attn_num_heads=int(model_cfg.get("fusion_attn_num_heads", 4)),
        fusion_attn_window_size=tuple(model_cfg.get("fusion_attn_window_size", [8, 8, 4])),
        fusion_attn_head_dilations=tuple(model_cfg.get("fusion_attn_head_dilations", [1, 2])),
        fusion_attn_mlp_ratio=float(model_cfg.get("fusion_attn_mlp_ratio", 2.0)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
    ).to(device)


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    if args.epochs > 0:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.ray_override:
        cfg.setdefault("loss", {}).setdefault("ray", {}).update(json.loads(args.ray_override))
        print(f"[ray-override] {args.ray_override}")
    if args.lambda_fast_kl is not None:
        cfg.setdefault("train", {})["lambda_fast_kl"] = float(args.lambda_fast_kl)
        print(f"[lambda-fast-kl] override = {args.lambda_fast_kl}")
    if args.lambda_lovasz is not None:
        cfg.setdefault("loss", {})["lambda_lovasz"] = float(args.lambda_lovasz)
    if args.lambda_focal is not None:
        cfg.setdefault("loss", {})["lambda_focal"] = float(args.lambda_focal)
    if args.fast_logits_root is not None:
        cfg.setdefault("data", {})["fast_logits_root"] = str(args.fast_logits_root)
        print(f"[fast-logits-root] override = {args.fast_logits_root}")

    set_seed(args.seed)

    root_path = cfg["root_path"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})
    loader_cfg = cfg.get("dataloader", {})

    logits_loader = build_logits_loader(data_cfg, root_path)
    train_dataset = build_dataset(
        info_path=data_cfg["info_path"], data_cfg=data_cfg, root_path=root_path,
        logits_loader=logits_loader, ray_sidecar_split="train",
    )
    train_dataset = build_subset(train_dataset, args.train_limit)

    val_dataset = None
    val_info_path = data_cfg.get("val_info_path", "")
    if val_info_path:
        val_dataset = build_dataset(
            info_path=val_info_path, data_cfg=data_cfg, root_path=root_path,
            logits_loader=logits_loader, ray_sidecar_split="val",
        )
        if args.val_scene_count > 0:
            scene_names = [info.get("scene_name", "") for info in val_dataset.infos]
            unique_scenes = sorted({n for n in scene_names if n})
            rng = random.Random(0)
            rng.shuffle(unique_scenes)
            keep = set(unique_scenes[: min(args.val_scene_count, len(unique_scenes))])
            val_indices = [i for i, n in enumerate(scene_names) if n in keep]
            val_dataset = Subset(val_dataset, val_indices)

    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = _build_model(args.model_kind, model_cfg, data_cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] kind={args.model_kind} params={n_params}")

    start_epoch = 1
    resumed_payload = None
    if args.resume:
        resumed_payload = load_checkpoint(args.resume, model=model, optimizer=None, strict=False)
        start_epoch = resumed_payload.get("epoch", 0) + 1
        print(f"[resume] 从 epoch={start_epoch} 继续训练")

    ema = None
    ema_cfg = train_cfg.get("ema", {}) or {}
    if bool(ema_cfg.get("enabled", True)):
        ema_decay = float(ema_cfg.get("decay", 0.999))
        ema = ModelEMA(model, decay=ema_decay, device=device)
        print(f"[ema] enabled, decay={ema_decay}")
        if resumed_payload is not None and "ema" in resumed_payload:
            ema.load_state_dict(resumed_payload["ema"])
            print(f"[ema] resumed num_updates={ema.num_updates}")

    num_workers = int(train_cfg["num_workers"])
    train_loader_kwargs = dict(
        batch_size=int(train_cfg["batch_size"]), num_workers=num_workers, shuffle=True,
        collate_fn=online_ncde_collate, pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        train_loader_kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    val_loader_kwargs = None
    if val_dataset is not None:
        val_workers = int(eval_cfg.get("num_workers", num_workers))
        val_loader_kwargs = dict(
            batch_size=int(eval_cfg.get("batch_size", 1)), num_workers=val_workers, shuffle=False,
            collate_fn=online_ncde_collate, pin_memory=loader_cfg.get("pin_memory", False),
        )
        if val_workers > 0:
            val_loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
            val_loader_kwargs["persistent_workers"] = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    if args.resume:
        opt_state = torch.load(args.resume, map_location="cpu").get("optimizer", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
    scheduler = build_scheduler(optimizer, train_cfg, args)
    if scheduler is not None and start_epoch > 1:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")
            for _ in range(start_epoch - 1):
                scheduler.step()

    ray_cfg = loss_cfg.get("ray", None)
    if ray_cfg is not None:
        ray_cfg.setdefault("pc_range", list(data_cfg["pc_range"]))
        ray_cfg.setdefault("free_index", int(data_cfg["free_index"]))
    loss_fn = build_loss(loss_cfg, num_classes=data_cfg["num_classes"]).to(device)

    trainer = Trainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
        num_classes=data_cfg["num_classes"], free_index=data_cfg["free_index"],
        free_conf_thresh=eval_cfg.get("free_conf_thresh", None),
        log_interval=train_cfg.get("log_interval", 10),
        clip_norm=train_cfg.get("clip_norm", 5.0),
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.15, 0.20, 0.25, 0.40])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        lambda_fast_kl=float(train_cfg.get("lambda_fast_kl", 0.0)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=str(train_cfg.get("rollout_mode", "full")),
        primary_supervision_label=str(eval_cfg.get("primary_supervision_label", "t-1.0")),
        stepwise_max_step_index=train_cfg.get("max_step_index", None),
        is_main=True,
        ema=ema,
    )

    # 输出目录：和主脚本规则一致（configs 相对路径 + 时间戳），但放在 outputs/baselines/ 下
    if args.resume:
        output_dir = os.path.dirname(os.path.abspath(args.resume))
    else:
        config_rel = os.path.relpath(args.config, os.path.join(str(ROOT), "configs"))
        output_base = os.path.join(str(ROOT), "outputs", "baselines", args.model_kind, os.path.dirname(config_rel))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[ckpt] output_dir: {output_dir}")

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs) if val_dataset is not None else None

    enable_rayiou = (not args.no_rayiou) and (val_loader is not None)
    sweep_pkl_abs = None
    if enable_rayiou:
        from online_ncde.config import resolve_path
        sweep_pkl_abs = resolve_path(
            root_path,
            eval_cfg.get("sweep_pkl", "data/nuscenes/nuscenes_infos_val_sweep.pkl"),
        )

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        train_metrics = trainer.train_one_epoch(train_loader, epoch=epoch)
        _cleanup_gpu_cache()
        if scheduler is not None:
            scheduler.step()

        sup_parts = [f"{k}={float(v):.4f}" for k, v in train_metrics.items() if k.startswith("loss_t")]
        sup_text = (" " + " ".join(sup_parts)) if sup_parts else ""
        ray_text = ""
        if "ray" in train_metrics:
            ray_text = (
                f" ray_total={float(train_metrics['ray']):.4f}"
                f" ray_hit={float(train_metrics['ray_hit']):.4f}"
                f" ray_empty={float(train_metrics['ray_empty']):.4f}"
                f" ray_depth={float(train_metrics['ray_depth']):.4f}"
            )
        kl_text = f" fast_kl={float(train_metrics['fast_kl']):.4f}" if "fast_kl" in train_metrics else ""
        print(
            f"[train] epoch={epoch} "
            f"loss={train_metrics['loss']:.4f} "
            f"focal={train_metrics['focal']:.4f} "
            f"aux={train_metrics['aux']:.4f} "
            f"delta={train_metrics['delta_scene_abs_mean']:.4f}"
            f"{kl_text}{ray_text}{sup_text}"
        )

        if val_loader is not None and args.eval_every > 0 and epoch % args.eval_every == 0:
            val_metrics = trainer.evaluate(val_loader, collect_predictions=enable_rayiou)
            _cleanup_gpu_cache()
            val_sup = " ".join(
                f"{k}={float(v):.4f}" for k, v in val_metrics.items()
                if isinstance(k, str) and k.startswith("sup_loss_t")
            )
            print(
                f"[eval] epoch={epoch} "
                f"loss={val_metrics['loss']:.4f} "
                f"miou={val_metrics['miou']:.4f} "
                f"miou_d={val_metrics.get('miou_d', float('nan')):.4f}"
                f"{(' ' + val_sup) if val_sup else ''}"
            )
            class_names = val_metrics.get("class_names", [])
            per_class = val_metrics.get("per_class_iou", [])
            if isinstance(class_names, list) and isinstance(per_class, list) and per_class:
                print(f"===> per class IoU of epoch {epoch}:")
                for name, value in zip(class_names, per_class):
                    print(f"===> {name} - IoU = {round(float(value), 2)}")

            rayiou_result = None
            if enable_rayiou and "predictions" in val_metrics:
                from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
                from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou
                origins_by_token = load_origins_from_sweep_pkl(sweep_pkl_abs)
                sem_pred_list, sem_gt_list, lidar_origin_list = [], [], []
                for item in val_metrics["predictions"]:
                    origin = origins_by_token.get(item["token"], None)
                    if origin is None:
                        continue
                    sem_pred_list.append(item["pred"])
                    sem_gt_list.append(item["gt"])
                    lidar_origin_list.append(origin)
                if sem_pred_list:
                    rayiou_result = calc_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list)
                    print(
                        f"[rayiou] epoch={epoch} "
                        f"RayIoU={rayiou_result['RayIoU']:.4f} "
                        f"@1={rayiou_result['RayIoU@1']:.4f} "
                        f"@2={rayiou_result['RayIoU@2']:.4f} "
                        f"@4={rayiou_result['RayIoU@4']:.4f}"
                    )

        ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
        trainer.save_checkpoint(ckpt_path, epoch=epoch)
        print(f"[ckpt] saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
