#!/usr/bin/env python3
"""StreamingFlow-style BEV GRU-ODE baseline 训练脚本。"""

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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.benchmark = True

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from train_online_ncde import (  # noqa: E402
    build_dataset,
    build_scheduler,
    build_subset,
    cleanup_ddp,
    setup_ddp_early,
    setup_ddp_init,
)

from online_ncde.baselines import StreamingFlowBEVOdeAligner  # noqa: E402
from online_ncde.config import config_output_subdir, load_config, load_config_with_base, merge_dict  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.evaluation import evaluate_dense_occ  # noqa: E402
from online_ncde.losses import build_loss  # noqa: E402
from online_ncde.trainer import Trainer, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from online_ncde.utils.reproducibility import set_seed  # noqa: E402

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


STREAMINGFLOW_OVERLAY = (
    ROOT / "src" / "online_ncde" / "baselines" / "streamingflow" / "occ3d_config.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--val-scene-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--ray-override", type=str, default="")
    parser.add_argument("--lambda-lovasz", type=float, default=None)
    parser.add_argument("--lambda-focal", type=float, default=None)
    parser.add_argument("--fast-logits-root", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-new-run", action="store_true")
    parser.add_argument("--cosine-annealing", action="store_true")
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--no-rayiou", action="store_true")
    return parser.parse_args()


def _cleanup_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _assert_occ3d(data_cfg: dict) -> None:
    if int(data_cfg["num_classes"]) != 18:
        raise ValueError("StreamingFlow baseline 只支持 Occ3D num_classes=18")
    if tuple(data_cfg["grid_size"]) != (200, 200, 16):
        raise ValueError("StreamingFlow baseline 只支持 Occ3D grid_size=(200,200,16)")


def _load_config_with_streamingflow_overlay(config_path: str) -> dict:
    cfg = load_config_with_base(config_path)
    overlay = load_config(str(STREAMINGFLOW_OVERLAY))
    return merge_dict(cfg, overlay)


def _build_model(data_cfg: dict, model_cfg: dict) -> StreamingFlowBEVOdeAligner:
    return StreamingFlowBEVOdeAligner(
        num_classes=int(data_cfg["num_classes"]),
        feat_dim=int(model_cfg.get("feat_dim", 192)),
        hidden_dim=int(model_cfg.get("hidden_dim", 192)),
        encoder_in_channels=int(model_cfg.get("encoder_in_channels", 18)),
        free_index=int(data_cfg["free_index"]),
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=model_cfg.get("decoder_init_scale", None),
        timestamp_scale=float(data_cfg.get("timestamp_scale", 1.0e-6)),
        streamingflow_cfg=dict(model_cfg.get("streamingflow", {}) or {}),
    )


def main() -> None:
    args = parse_args()
    local_rank, use_ddp = setup_ddp_early()

    cfg = _load_config_with_streamingflow_overlay(args.config)
    if args.epochs > 0:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.ray_override:
        cfg.setdefault("loss", {}).setdefault("ray", {}).update(json.loads(args.ray_override))
    if args.lambda_lovasz is not None:
        cfg.setdefault("loss", {})["lambda_lovasz"] = float(args.lambda_lovasz)
    if args.lambda_focal is not None:
        cfg.setdefault("loss", {})["lambda_focal"] = float(args.lambda_focal)
    if args.fast_logits_root is not None:
        cfg.setdefault("data", {})["fast_logits_root"] = str(args.fast_logits_root)

    set_seed(args.seed + local_rank)
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
    _assert_occ3d(data_cfg)

    logits_loader = build_logits_loader(data_cfg, root_path)
    train_dataset = build_dataset(
        info_path=data_cfg["info_path"],
        data_cfg=data_cfg,
        root_path=root_path,
        logits_loader=logits_loader,
        ray_sidecar_split="train",
    )
    train_dataset = build_subset(train_dataset, args.train_limit)

    val_dataset = None
    val_loader_kwargs = None
    if data_cfg.get("val_info_path", ""):
        val_dataset = build_dataset(
            info_path=data_cfg["val_info_path"],
            data_cfg=data_cfg,
            root_path=root_path,
            logits_loader=logits_loader,
            ray_sidecar_split="val",
        )
        if args.val_scene_count > 0:
            scene_names = [info.get("scene_name", "") for info in val_dataset.infos]
            unique_scenes = sorted({name for name in scene_names if name})
            rng = random.Random(0)
            rng.shuffle(unique_scenes)
            val_scene_set = set(unique_scenes[: min(args.val_scene_count, len(unique_scenes))])
            val_indices = [idx for idx, name in enumerate(scene_names) if name in val_scene_set]
            val_dataset = Subset(val_dataset, val_indices)
        val_workers = int(eval_cfg.get("num_workers", train_cfg["num_workers"]))
        val_loader_kwargs = dict(
            batch_size=int(eval_cfg.get("batch_size", 1)),
            num_workers=val_workers,
            shuffle=False,
            collate_fn=online_ncde_collate,
            pin_memory=loader_cfg.get("pin_memory", False),
        )
        if val_workers > 0:
            val_loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
            val_loader_kwargs["persistent_workers"] = False

    device = torch.device(
        f"cuda:{local_rank}" if use_ddp
        else (train_cfg["device"] if torch.cuda.is_available() else "cpu")
    )
    model = _build_model(data_cfg, model_cfg).to(device)

    start_epoch = 1
    resumed_payload = None
    if args.resume:
        resumed_payload = load_checkpoint(args.resume, model=model, optimizer=None, strict=False)
        start_epoch = int(resumed_payload.get("epoch", 0)) + 1

    rank, local_rank, _world_size = setup_ddp_init(local_rank)
    is_main = rank == 0
    if is_main:
        print(f"[model] kind=streamingflow-bev-ode params={sum(p.numel() for p in model.parameters())}")

    ema = None
    ema_cfg = train_cfg.get("ema", {}) or {}
    if bool(ema_cfg.get("enabled", True)):
        from online_ncde.utils.ema import ModelEMA
        ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999)), device=device)
        if resumed_payload is not None and "ema" in resumed_payload:
            ema.load_state_dict(resumed_payload["ema"])

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    num_workers = int(train_cfg["num_workers"])
    train_loader_kwargs = dict(
        batch_size=int(train_cfg["batch_size"]),
        num_workers=num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=online_ncde_collate,
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        train_loader_kwargs["persistent_workers"] = (
            False if use_ddp else loader_cfg.get("persistent_workers", False)
        )
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs) if val_dataset is not None else None

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
        # resume 时对齐主训练脚本的 LR 进度，避免学习率计划从头开始。
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")
            for _ in range(start_epoch - 1):
                scheduler.step()

    ray_cfg = loss_cfg.get("ray", None)
    if ray_cfg is not None:
        ray_cfg.setdefault("pc_range", list(data_cfg["pc_range"]))
        ray_cfg.setdefault("free_index", int(data_cfg["free_index"]))
    loss_fn = build_loss(loss_cfg, num_classes=int(data_cfg["num_classes"])).to(device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_classes=int(data_cfg["num_classes"]),
        free_index=int(data_cfg["free_index"]),
        free_conf_thresh=eval_cfg.get("free_conf_thresh", None),
        log_interval=train_cfg.get("log_interval", 10),
        clip_norm=train_cfg.get("clip_norm", 5.0),
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.15, 0.20, 0.25, 0.40])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        lambda_fast_kl=0.0,
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=str(train_cfg.get("rollout_mode", "full")),
        primary_supervision_label=str(eval_cfg.get("primary_supervision_label", "t-1.0")),
        stepwise_max_step_index=train_cfg.get("max_step_index", None),
        is_main=is_main,
        ema=ema,
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
    )

    if args.resume:
        output_dir = os.path.dirname(os.path.abspath(args.resume))
        run_timestamp = os.path.basename(output_dir)
    else:
        config_subdir = config_output_subdir(args.config, os.path.join(str(ROOT), "configs"))
        output_base = os.path.join(
            str(ROOT), "outputs", "baselines", "streamingflow_bev_ode", config_subdir
        )
        if use_ddp:
            if is_main:
                ts_tensor = torch.tensor(
                    [int(datetime.now().strftime("%Y%m%d%H%M%S"))],
                    dtype=torch.long,
                    device=device,
                )
            else:
                ts_tensor = torch.zeros(1, dtype=torch.long, device=device)
            dist.broadcast(ts_tensor, src=0)
            timestamp_raw = str(int(ts_tensor.item()))
            run_timestamp = f"{timestamp_raw[:8]}_{timestamp_raw[8:]}"
        else:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base, run_timestamp)
    os.makedirs(output_dir, exist_ok=True)
    if is_main:
        print(f"[ckpt] output_dir: {output_dir}")

    run = None
    if args.wandb and is_main:
        if wandb is None:
            raise ImportError("未安装 wandb")
        run = wandb.init(
            entity=wandb_cfg.get("entity", "runheyang"),
            project=wandb_cfg.get("project", "neural-ode"),
            name=wandb_cfg.get("name", "") or run_timestamp,
            config={"model_kind": "streamingflow-bev-ode", "config": args.config},
        )

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = trainer.train_one_epoch(train_loader, epoch=epoch)
        _cleanup_gpu_cache()
        if scheduler is not None:
            scheduler.step()

        if is_main:
            print(
                f"[train] epoch={epoch} loss={train_metrics['loss']:.4f} "
                f"focal={train_metrics['focal']:.4f} aux={train_metrics['aux']:.4f}"
            )
            if run is not None:
                payload = {f"train/{k}": float(v) for k, v in train_metrics.items()}
                payload["epoch"] = float(epoch)
                run.log(payload, commit=False)

        val_metrics = None
        if is_main and val_loader is not None and args.eval_every > 0 and epoch % args.eval_every == 0:
            enable_rayiou = not args.no_rayiou
            val_metrics = trainer.evaluate(
                val_loader,
                collect_predictions=True,
                compute_miou=False,
            )
            _cleanup_gpu_cache()
            sweep_rel = eval_cfg.get("sweep_pkl", "data/nuscenes/nuscenes_infos_val_sweep.pkl")
            sweep_path = Path(sweep_rel)
            sweep_pkl = str(
                sweep_path if sweep_path.is_absolute() else (Path(root_path) / sweep_path).resolve()
            )
            if enable_rayiou and epoch == start_epoch:
                print(f"[rayiou] sweep pkl: {sweep_pkl}")
            dense_eval = evaluate_dense_occ(
                val_metrics.get("predictions", []),
                num_classes=int(data_cfg["num_classes"]),
                enable_rayiou=enable_rayiou,
                sweep_pkl=sweep_pkl if enable_rayiou else None,
                print_rayiou_table=True,
            )
            dense_all = dense_eval["all"]
            val_metrics["miou"] = (
                dense_all["miou"] if dense_all.get("miou", None) is not None else float("nan")
            )
            val_metrics["miou_d"] = (
                dense_all["miou_d"] if dense_all.get("miou_d", None) is not None else float("nan")
            )
            val_metrics["per_class_iou"] = dense_all.get("per_class_iou", [])
            val_metrics["class_names"] = dense_all.get("class_names", [])
            print(
                f"[eval] epoch={epoch} loss={val_metrics['loss']:.4f} "
                f"miou={val_metrics['miou']:.4f} miou_d={val_metrics.get('miou_d', float('nan')):.4f}"
            )
            rayiou_result = dense_all.get("rayiou", None)
            if rayiou_result is not None:
                print(
                    f"[rayiou] epoch={epoch} "
                    f"RayIoU={rayiou_result['RayIoU']:.4f} "
                    f"RayIoU@1={rayiou_result['RayIoU@1']:.4f} "
                    f"RayIoU@2={rayiou_result['RayIoU@2']:.4f} "
                    f"RayIoU@4={rayiou_result['RayIoU@4']:.4f}"
                )
            rayiou_meta = dense_eval.get("rayiou_meta", None) or {}
            missing_origin_count = int(rayiou_meta.get("missing_origin_count", 0))
            if missing_origin_count:
                print(f"[rayiou] epoch={epoch} 跳过 {missing_origin_count} 个样本（无对应 lidar origin）")
            if run is not None:
                payload = {f"val/{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}
                payload["epoch"] = float(epoch)
                if rayiou_result is not None:
                    for key in ("RayIoU", "RayIoU@1", "RayIoU@2", "RayIoU@4"):
                        payload[f"val/{key}"] = float(rayiou_result[key])
                run.log(payload, commit=True)

        if is_main:
            raw_model = model.module if hasattr(model, "module") else model
            extra = {"ema": ema.state_dict()} if ema is not None else None
            epoch_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
            save_checkpoint(epoch_path, raw_model, optimizer=optimizer, epoch=epoch, extra=extra)
            latest_path = os.path.join(output_dir, "latest.pth")
            save_checkpoint(latest_path, raw_model, optimizer=optimizer, epoch=epoch, extra=extra)
            print(f"[ckpt] saved -> {epoch_path}")

        # DDP 下 rank0 做 eval/checkpoint，其他 rank 必须等待，避免下一轮 forward 挂死。
        if use_ddp:
            dist.barrier()

    if run is not None:
        run.finish()
    cleanup_ddp()


if __name__ == "__main__":
    main()
