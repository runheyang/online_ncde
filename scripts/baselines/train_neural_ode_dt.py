#!/usr/bin/env python3
"""Neural ODE 离散化 baseline 训练脚本（支持 DDP + wandb）。

与 train_online_ncde.py 同款 wiring：复用 trainer / dataset / loss / RayIoU /
EMA / scheduler / wandb / DDP 全套；只把模型换成 NeuralOdeDtAligner —— 控制
增量从 (Fast 差值 + 1x1x1 conv) 改为 标量 Δt 广播。

Config 完全兼容 train_online_ncde.py 的 yaml；NeuralOdeDtAligner 的超参
（func_g_inner_dim/func_g_body_dilations/func_g_gn_groups/use_fast_residual/
decoder_init_scale）走 model.* 字段，与 NCDE 同名同义。
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

torch.backends.cudnn.benchmark = True
from torch.utils.data.distributed import DistributedSampler  # noqa: E402

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))  # 复用 train_online_ncde 的 helper

# 复用主训练脚本里的 helper，避免重复实现
from train_online_ncde import (  # noqa: E402
    build_dataset,
    build_scheduler,
    build_subset,
    cleanup_ddp,
    setup_ddp_early,
    setup_ddp_init,
    to_float,
)

from online_ncde.baselines import NeuralOdeDtAligner  # noqa: E402
from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.losses import build_loss  # noqa: E402
from online_ncde.trainer import Trainer, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint  # noqa: E402
from online_ncde.utils.reproducibility import set_seed  # noqa: E402

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径（与 NCDE 同 config 兼容）")
    parser.add_argument("--resume", default="", help="恢复训练权重")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--val-scene-count", type=int, default=0,
                        help="验证集 scene 抽样（按 seed=0 打乱后取前 N，0=全量）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="启用 wandb")
    parser.add_argument("--wandb-new-run", action="store_true",
                        help="忽略 checkpoint 中的 wandb_run_id，强制新建 run")
    parser.add_argument("--cosine-annealing", action="store_true")
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--epochs", type=int, default=0, help="覆盖 config epochs（0=不覆盖）")
    parser.add_argument("--ray-override", type=str, default="", help="JSON 覆盖 loss.ray")
    parser.add_argument("--lambda-fast-kl", type=float, default=None)
    parser.add_argument("--lambda-lovasz", type=float, default=None)
    parser.add_argument("--lambda-focal", type=float, default=None)
    parser.add_argument("--save-metrics-json", action="store_true")
    parser.add_argument("--solver", choices=["heun", "euler"], default="euler",
                        help="ODE 求解器：euler（默认）或 heun")
    parser.add_argument("--fast-logits-root", type=str, default=None,
                        help="覆盖 data.fast_logits_root")
    return parser.parse_args()


def _cleanup_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    local_rank, use_ddp = setup_ddp_early()

    cfg = load_config_with_base(args.config)
    if args.epochs > 0:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.ray_override:
        ray_overrides = json.loads(args.ray_override)
        cfg.setdefault("loss", {}).setdefault("ray", {}).update(ray_overrides)
        if local_rank == 0:
            print(f"[ray-override] {ray_overrides}")
    if args.lambda_fast_kl is not None:
        cfg.setdefault("train", {})["lambda_fast_kl"] = float(args.lambda_fast_kl)
        if local_rank == 0:
            print(f"[lambda-fast-kl] override = {args.lambda_fast_kl}")
    if args.lambda_lovasz is not None:
        cfg.setdefault("loss", {})["lambda_lovasz"] = float(args.lambda_lovasz)
    if args.lambda_focal is not None:
        cfg.setdefault("loss", {})["lambda_focal"] = float(args.lambda_focal)
    if args.fast_logits_root is not None:
        cfg.setdefault("data", {})["fast_logits_root"] = str(args.fast_logits_root)
        if local_rank == 0:
            print(f"[fast-logits-root] override = {args.fast_logits_root}")

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

    logits_loader = build_logits_loader(data_cfg, root_path)
    train_dataset = build_dataset(
        info_path=data_cfg["info_path"], data_cfg=data_cfg, root_path=root_path,
        logits_loader=logits_loader, ray_sidecar_split="train",
    )
    train_dataset = build_subset(train_dataset, args.train_limit)

    num_workers = int(train_cfg["num_workers"])

    val_dataset = None
    val_loader_kwargs = None
    val_info_path = data_cfg.get("val_info_path", "")
    if val_info_path:
        val_dataset = build_dataset(
            info_path=val_info_path, data_cfg=data_cfg, root_path=root_path,
            logits_loader=logits_loader, ray_sidecar_split="val",
        )
        if args.val_scene_count > 0:
            scene_names = [info.get("scene_name", "") for info in val_dataset.infos]
            unique_scenes = sorted({n for n in scene_names if n})
            if not unique_scenes:
                raise ValueError("未找到有效 scene_name，无法按 scene 划分验证集")
            rng = random.Random(0)
            rng.shuffle(unique_scenes)
            val_scene_count = min(args.val_scene_count, len(unique_scenes))
            val_scene_set = set(unique_scenes[:val_scene_count])
            val_indices = [i for i, n in enumerate(scene_names) if n in val_scene_set]
            if not val_indices:
                raise ValueError("验证集 scene 为空，请检查 val_scene_count")
            val_dataset = Subset(val_dataset, val_indices)

        val_workers = int(eval_cfg.get("num_workers", num_workers))
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
    model = NeuralOdeDtAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=model_cfg.get("decoder_init_scale", 1.0e-3),
        use_fast_residual=bool(model_cfg.get("use_fast_residual", True)),
        func_g_inner_dim=int(model_cfg.get("func_g_inner_dim", 32)),
        func_g_body_dilations=tuple(model_cfg.get("func_g_body_dilations", [1, 2, 3])),
        func_g_gn_groups=int(model_cfg.get("func_g_gn_groups", 8)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
        solver_variant=args.solver,
    ).to(device)

    start_epoch = 1
    resumed_payload = None
    if args.resume:
        resumed_payload = load_checkpoint(args.resume, model=model, optimizer=None, strict=False)
        start_epoch = resumed_payload.get("epoch", 0) + 1

    rank, local_rank, world_size = setup_ddp_init(local_rank)
    is_main = rank == 0
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] kind=neural-ode-dt solver={args.solver} params={n_params}")
    if is_main and args.resume:
        print(f"[resume] 从 epoch={start_epoch} 继续训练")

    ema = None
    ema_cfg = train_cfg.get("ema", {}) or {}
    if bool(ema_cfg.get("enabled", True)):
        from online_ncde.utils.ema import ModelEMA
        ema_decay = float(ema_cfg.get("decay", 0.999))
        ema = ModelEMA(model, decay=ema_decay, device=device)
        if is_main:
            print(f"[ema] enabled, decay={ema_decay}")
        if resumed_payload is not None and "ema" in resumed_payload:
            ema.load_state_dict(resumed_payload["ema"])
            if is_main:
                print(f"[ema] resumed num_updates={ema.num_updates}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
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
        is_main=is_main,
        ema=ema,
    )

    # 输出目录：和主脚本规则一致，但放在 outputs/baselines/neural_ode_dt/ 下
    resumed_wandb_id = None
    if args.resume:
        output_dir = os.path.dirname(os.path.abspath(args.resume))
        _ckpt_payload = torch.load(args.resume, map_location="cpu")
        resumed_wandb_id = _ckpt_payload.get("wandb_run_id", None)
        if args.wandb_new_run:
            resumed_wandb_id = None
        del _ckpt_payload
    else:
        config_rel = os.path.relpath(args.config, os.path.join(str(ROOT), "configs"))
        output_base = os.path.join(
            str(ROOT), "outputs", "baselines", "neural_ode_dt",
            os.path.dirname(config_rel),
        )
        if use_ddp:
            if rank == 0:
                ts_tensor = torch.tensor(
                    [int(datetime.now().strftime("%Y%m%d%H%M%S"))], dtype=torch.long, device=device
                )
            else:
                ts_tensor = torch.zeros(1, dtype=torch.long, device=device)
            dist.broadcast(ts_tensor, src=0)
            timestamp = str(ts_tensor.item())
            timestamp = f"{timestamp[:8]}_{timestamp[8:]}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base, f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    if is_main:
        print(f"[ckpt] output_dir: {output_dir}")

    run = None
    if args.wandb and is_main:
        if wandb is None:
            raise ImportError("未安装 wandb，无法启用日志。")
        wandb_kwargs = dict(
            entity=wandb_cfg.get("entity", "runheyang"),
            project=wandb_cfg.get("project", "neural-ode"),
            config={
                "model_kind": "neural-ode-dt",
                "solver": args.solver,
                "epochs": int(train_cfg["epochs"]),
                "batch_size": int(train_cfg["batch_size"]),
                "lr": float(train_cfg["lr"]),
                "weight_decay": float(train_cfg["weight_decay"]),
            },
        )
        if resumed_wandb_id:
            wandb_kwargs["id"] = resumed_wandb_id
            wandb_kwargs["resume"] = "allow"
        else:
            base_name = wandb_cfg.get("name", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_kwargs["name"] = f"neural_ode_dt-{args.solver}-{base_name}"
        run = wandb.init(**wandb_kwargs)
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs) if val_dataset is not None else None

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = trainer.train_one_epoch(train_loader, epoch=epoch)
        _cleanup_gpu_cache()
        if scheduler is not None:
            scheduler.step()

        if is_main:
            sup_parts = [f"{k}={float(v):.4f}" for k, v in train_metrics.items() if k.startswith("loss_t")]
            sup_text = (" " + " ".join(sup_parts)) if sup_parts else ""
            ray_text = ""
            if "ray" in train_metrics:
                ray_text = (
                    f" ray_total={float(train_metrics['ray']):.4f}"
                    f" ray_hit={float(train_metrics['ray_hit']):.4f}"
                    f" ray_empty={float(train_metrics['ray_empty']):.4f}"
                    f" ray_pre_free={float(train_metrics.get('ray_pre_free', 0.0)):.4f}"
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
            if run is not None:
                train_payload = {f"train/{k}": float(v) for k, v in train_metrics.items()}
                train_payload["epoch"] = float(epoch)
                should_eval = (
                    val_dataset is not None
                    and args.eval_every > 0
                    and epoch % args.eval_every == 0
                )
                run.log(train_payload, commit=not should_eval)

        # eval / checkpoint 仅 rank 0 执行
        if is_main:
            val_metrics = None
            rayiou_result = None
            binned_ray_result = None
            if val_loader is not None and args.eval_every > 0 and epoch % args.eval_every == 0:
                val_metrics = trainer.evaluate(val_loader, collect_predictions=True)
                _cleanup_gpu_cache()
                val_sup = " ".join(
                    f"{k}={float(v):.4f}" for k, v in val_metrics.items()
                    if isinstance(k, str) and k.startswith("sup_loss_t")
                )
                print(
                    f"[eval] epoch={epoch} "
                    f"loss={val_metrics['loss']:.4f} "
                    f"focal={val_metrics['focal']:.4f} "
                    f"aux={val_metrics['aux']:.4f} "
                    f"miou={val_metrics['miou']:.4f} "
                    f"miou_d={val_metrics.get('miou_d', float('nan')):.4f}"
                    f"{(' ' + val_sup) if val_sup else ''}"
                )
                class_names = val_metrics.get("class_names", [])
                per_class = val_metrics.get("per_class_iou", [])
                if isinstance(class_names, list) and isinstance(per_class, list):
                    print(f"===> per class IoU of epoch {epoch}:")
                    for name, value in zip(class_names, per_class):
                        print(f"===> {name} - IoU = {round(float(value), 2)}")

                # --- RayIoU ---
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
                    print(f"[rayiou] epoch={epoch} 跳过 {skipped} 个样本")
                print(f"[rayiou] epoch={epoch} {len(sem_pred_list)} 个样本参与计算")

                need_pcds = args.save_metrics_json
                if need_pcds:
                    rayiou_result, raw_pcd_pred, raw_pcd_gt = calc_rayiou(
                        sem_pred_list, sem_gt_list, lidar_origin_list, return_pcds=True)
                else:
                    rayiou_result = calc_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list)
                print(
                    f"[rayiou] epoch={epoch} "
                    f"RayIoU={rayiou_result['RayIoU']:.4f} "
                    f"RayIoU@1={rayiou_result['RayIoU@1']:.4f} "
                    f"RayIoU@2={rayiou_result['RayIoU@2']:.4f} "
                    f"RayIoU@4={rayiou_result['RayIoU@4']:.4f}"
                )

                if need_pcds:
                    from online_ncde.ops.dvr.binned_ray_stats import compute_binned_ray_stats
                    binned_ray_result = compute_binned_ray_stats(raw_pcd_pred, raw_pcd_gt)
                    for bk in ["0-10m", "10-20m", "20-40m"]:
                        bs = binned_ray_result[bk]
                        print(
                            f"[rayiou-bin] {bk}: "
                            f"RayIoU={bs['RayIoU']:.4f} "
                            f"signed_err={bs['mean_signed_err']:+.4f} "
                            f"miss={bs['miss_rate']:.4f} "
                            f"false_hit={bs['false_hit_rate']:.4f}"
                        )
                    del raw_pcd_pred, raw_pcd_gt

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
                    for key in ("RayIoU", "RayIoU@1", "RayIoU@2", "RayIoU@4"):
                        payload[f"val/{key}"] = float(rayiou_result[key])
                    run.log(payload, commit=True)

            if (
                args.save_metrics_json
                and val_loader is not None
                and args.eval_every > 0
                and epoch % args.eval_every == 0
                and val_metrics is not None
                and rayiou_result is not None
            ):
                metrics_json = {
                    "epoch": epoch,
                    "model_kind": "neural-ode-dt",
                    "solver": args.solver,
                    "mIoU": float(val_metrics["miou"]),
                    "mIoU_d": float(val_metrics.get("miou_d", 0.0)),
                    "loss": float(val_metrics["loss"]),
                }
                if "fast_kl" in train_metrics:
                    metrics_json["fast_kl"] = {"train": float(train_metrics["fast_kl"])}
                if ray_cfg is not None:
                    metrics_json["ray_config"] = {
                        k: v for k, v in ray_cfg.items()
                        if isinstance(v, (int, float, str, bool))
                    }
                metrics_json["rayiou"] = {k: float(v) for k, v in rayiou_result.items()}
                if binned_ray_result is not None:
                    metrics_json["binned_ray"] = binned_ray_result
                json_path = os.path.join(output_dir, "metrics.json")
                with open(json_path, "w") as f:
                    json.dump(metrics_json, f, indent=2, ensure_ascii=False)
                print(f"[metrics] saved -> {json_path}")

            ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
            ckpt_extra = {}
            if run is not None:
                ckpt_extra["wandb_run_id"] = run.id
            trainer.save_checkpoint(ckpt_path, epoch=epoch, extra=ckpt_extra or None)
            print(f"[ckpt] saved -> {ckpt_path}")

        if use_ddp:
            dist.barrier()

    if run is not None:
        run.finish()
    cleanup_ddp()


if __name__ == "__main__":
    main()
