#!/usr/bin/env python3
"""训练 Online NCDE。"""

from __future__ import annotations

import argparse
import gc
import json
import numbers
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

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
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--resume", default="", help="恢复训练权重")
    parser.add_argument("--train-limit", type=int, default=0, help="训练样本上限（0=全量）")
    parser.add_argument("--eval-every", type=int, default=1, help="每隔多少 epoch 评估一次")
    parser.add_argument(
        "--val-scene-count",
        type=int,
        default=0,
        help="验证集使用的 scene 数量（按 seed=0 打乱后取前 N；0=全量评估，不做 subset）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb")
    parser.add_argument("--wandb-new-run", action="store_true", help="忽略 checkpoint 里的 wandb_run_id，强制新建 run")
    parser.add_argument("--cosine-annealing", action="store_true", help="启用余弦退火学习率调度")
    parser.add_argument("--min-lr", type=float, default=1.0e-5, help="余弦退火最低学习率")
    # 调参工作流参数
    parser.add_argument("--epochs", type=int, default=0, help="覆盖 config 中的 epochs（0=不覆盖）")
    parser.add_argument("--ray-override", type=str, default="",
                        help="JSON 字符串覆盖 loss.ray 参数，如 '{\"lambda_ray\": 0.3}'")
    parser.add_argument("--lambda-fast-kl", type=float, default=None,
                        help="覆盖 train.lambda_fast_kl（conf-weighted KL-to-fast 正则权重）")
    parser.add_argument("--lambda-lovasz", type=float, default=None,
                        help="覆盖 loss.lambda_lovasz（Lovasz IoU surrogate 权重，默认 1.0）")
    parser.add_argument("--lambda-focal", type=float, default=None,
                        help="覆盖 loss.lambda_focal（Focal loss 权重，默认 1.0）")
    parser.add_argument("--save-metrics-json", action="store_true",
                        help="eval 结束后将指标（含分箱 RayIoU）保存到 output_dir/metrics.json")
    parser.add_argument("--solver", choices=["heun", "euler"], default="euler",
                        help="ODE 求解器：euler（默认，Euler + next-fast 单次求值）或 heun")
    parser.add_argument("--fast-logits-root", type=str, default=None,
                        help="覆盖 data.fast_logits_root（如 data/logits_opusv1t_full 或 data/logits_opusv1t_full_postprocess）")
    parser.add_argument("--fast-kl-full-m", type=float, default=None,
                        help="距离加权 KL：体素 XY 距 ego < full_m 时 KL 权重=1（需与 --fast-kl-zero-m 同时给）")
    parser.add_argument("--fast-kl-zero-m", type=float, default=None,
                        help="距离加权 KL：体素 XY 距 ego > zero_m 时 KL 权重=0，full_m 到 zero_m 线性衰减")
    return parser.parse_args()


def build_subset(dataset, limit: int):
    if limit <= 0:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def build_scheduler(optimizer, train_cfg: dict, args):
    """构建学习率调度器（线性 warmup + 可选余弦退火）。"""
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
            return start_factor + (1.0 - start_factor) * epoch / max(warmup_epochs, 1)
        if not use_cosine:
            return 1.0
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
    """将标量安全转换为 Python float。"""
    if isinstance(value, numbers.Real):
        return float(value)
    return None


def _cleanup_gpu_cache():
    """强制回收 Python 垃圾并释放 CUDA 缓存。
    调用前须确保 DataLoader 引用已被 del 掉。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_ddp_early() -> tuple[int, bool]:
    """早期阶段：仅获取 local_rank 并设置 CUDA 设备，不初始化进程组。
    返回 (local_rank, use_ddp)。
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, True


def setup_ddp_init(local_rank: int) -> tuple[int, int, int]:
    """初始化 DDP 进程组。须在模型创建并 warmup 之后调用。

    默认使用 gloo 后端：Blackwell GPU + NCCL 2.26 存在兼容性问题
    （Conv3d 初始化产生的异步 CUDA error 被 NCCL watchdog 误判为致命错误），
    gloo 通过共享内存同步梯度，同节点双卡性能损失很小。
    如需切换后端可设置 DDP_BACKEND=nccl 环境变量。
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, local_rank, 1
    backend = os.environ.get("DDP_BACKEND", "gloo")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    local_rank, use_ddp = setup_ddp_early()

    cfg = load_config_with_base(args.config)
    # --epochs 覆盖
    if args.epochs > 0:
        cfg["train"]["epochs"] = args.epochs
    # --ray-override：JSON 覆盖 loss.ray 参数
    if args.ray_override:
        ray_overrides = json.loads(args.ray_override)
        if "loss" not in cfg:
            cfg["loss"] = {}
        if "ray" not in cfg["loss"]:
            cfg["loss"]["ray"] = {}
        cfg["loss"]["ray"].update(ray_overrides)
        if local_rank == 0:
            print(f"[ray-override] {ray_overrides}")
    # --lambda-fast-kl：覆盖 train.lambda_fast_kl
    if args.lambda_fast_kl is not None:
        if "train" not in cfg:
            cfg["train"] = {}
        cfg["train"]["lambda_fast_kl"] = float(args.lambda_fast_kl)
        if local_rank == 0:
            print(f"[lambda-fast-kl] override = {args.lambda_fast_kl}")
    # --lambda-lovasz：覆盖 loss.lambda_lovasz
    if args.lambda_lovasz is not None:
        if "loss" not in cfg:
            cfg["loss"] = {}
        cfg["loss"]["lambda_lovasz"] = float(args.lambda_lovasz)
        if local_rank == 0:
            print(f"[lambda-lovasz] override = {args.lambda_lovasz}")
    # --lambda-focal：覆盖 loss.lambda_focal
    if args.lambda_focal is not None:
        if "loss" not in cfg:
            cfg["loss"] = {}
        cfg["loss"]["lambda_focal"] = float(args.lambda_focal)
        if local_rank == 0:
            print(f"[lambda-focal] override = {args.lambda_focal}")
    # --fast-logits-root：覆盖 data.fast_logits_root
    if args.fast_logits_root is not None:
        if "data" not in cfg:
            cfg["data"] = {}
        cfg["data"]["fast_logits_root"] = str(args.fast_logits_root)
        if local_rank == 0:
            print(f"[fast-logits-root] override = {args.fast_logits_root}")
    # --fast-kl-full-m / --fast-kl-zero-m：覆盖 model.fast_kl_full_m / model.fast_kl_zero_m
    if args.fast_kl_full_m is not None:
        if "model" not in cfg:
            cfg["model"] = {}
        cfg["model"]["fast_kl_full_m"] = float(args.fast_kl_full_m)
    if args.fast_kl_zero_m is not None:
        if "model" not in cfg:
            cfg["model"] = {}
        cfg["model"]["fast_kl_zero_m"] = float(args.fast_kl_zero_m)
    if (args.fast_kl_full_m is not None or args.fast_kl_zero_m is not None) and local_rank == 0:
        print(f"[fast-kl-dist-mask] full_m={args.fast_kl_full_m} zero_m={args.fast_kl_zero_m}")
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
        info_path=data_cfg["info_path"],
        data_cfg=data_cfg,
        root_path=root_path,
        logits_loader=logits_loader,
        ray_sidecar_split="train",
    )
    train_dataset = build_subset(train_dataset, args.train_limit)

    num_workers = int(train_cfg["num_workers"])

    # --- 构建 val dataset 和 dataloader 参数 ---
    val_dataset = None
    val_loader_kwargs = None
    val_info_path = data_cfg.get("val_info_path", "")
    if val_info_path:
        val_dataset = build_dataset(
            info_path=val_info_path,
            data_cfg=data_cfg,
            root_path=root_path,
            logits_loader=logits_loader,
            ray_sidecar_split="val",
        )
        # val_scene_count=0 时走全量评估，不做 Subset
        if args.val_scene_count > 0:
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
        val_loader_kwargs = dict(
            batch_size=int(eval_cfg.get("batch_size", 1)),
            num_workers=val_workers,
            shuffle=False,
            collate_fn=online_ncde_collate,
            pin_memory=loader_cfg.get("pin_memory", False),
        )
        if val_workers > 0:
            val_loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
            # val 强制关闭 persistent_workers：eval 频率低，worker 用完即释放，
            # 避免与 train workers 同时常驻导致内存峰值过高
            val_loader_kwargs["persistent_workers"] = False

    # DDP 模式下按 local_rank 分配 GPU，否则用配置值
    device = torch.device(f"cuda:{local_rank}" if use_ddp else (train_cfg["device"] if torch.cuda.is_available() else "cpu"))
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
        solver_variant=args.solver,
        fast_kl_full_m=model_cfg.get("fast_kl_full_m", None),
        fast_kl_zero_m=model_cfg.get("fast_kl_zero_m", None),
    ).to(device)

    # 先加载权重
    start_epoch = 1
    resumed_payload = None
    if args.resume:
        resumed_payload = load_checkpoint(args.resume, model=model, optimizer=None, strict=False)
        start_epoch = resumed_payload.get("epoch", 0) + 1

    # 初始化进程组（默认 gloo，同节点双卡性能损失小）
    rank, local_rank, world_size = setup_ddp_init(local_rank)
    is_main = rank == 0
    if is_main:
        if args.solver == "euler":
            print(f"[solver] {args.solver} (next-fast only, 单次 func_g 求值)")
        else:
            print(f"[solver] {args.solver}")
    if is_main and args.resume:
        print(f"[resume] 从 epoch={start_epoch} 继续训练")

    # EMA：在 DDP 包装前基于原始权重构建，避免 deepcopy DDP 带来的复杂性
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

    # DDP 包装
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # --- DistributedSampler 需要进程组已初始化，放在 init_ddp 之后 ---
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
        # DDP 多进程时关闭 persistent_workers，避免 train/val 切换期间
        # worker 同时常驻导致内存峰值过高（3 卡 × 16 workers × 2 loader = 96 进程）
        train_loader_kwargs["persistent_workers"] = False if use_ddp else loader_cfg.get("persistent_workers", False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    # resume 时恢复 optimizer 状态
    if args.resume:
        opt_state = torch.load(args.resume, map_location="cpu").get("optimizer", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    scheduler = build_scheduler(optimizer, train_cfg, args)
    if scheduler is not None and start_epoch > 1:
        # resume 快进 scheduler 到当前 epoch，无需对应 optimizer.step()，suppress 警告
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")
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
        lambda_fast_kl=float(train_cfg.get("lambda_fast_kl", 0.0)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=str(train_cfg.get("rollout_mode", "full")),
        primary_supervision_label=str(eval_cfg.get("primary_supervision_label", "t-1.0")),
        stepwise_max_step_index=train_cfg.get("max_step_index", None),
        is_main=is_main,
        ema=ema,
    )

    # --- 推导 output_dir：resume 时复用原目录，否则新建时间戳目录 ---
    resumed_wandb_id = None
    if args.resume:
        # checkpoint 路径形如 .../20260412_225327/epoch_4.pth，取父目录
        output_dir = os.path.dirname(os.path.abspath(args.resume))
        # 从 checkpoint 中读取 wandb_run_id 用于续接
        _ckpt_payload = torch.load(args.resume, map_location="cpu")
        resumed_wandb_id = _ckpt_payload.get("wandb_run_id", None)
        if args.wandb_new_run:
            resumed_wandb_id = None
        del _ckpt_payload
    else:
        config_rel = os.path.relpath(args.config, os.path.join(str(ROOT), "configs"))
        output_base = os.path.join(str(ROOT), "outputs", os.path.dirname(config_rel))
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

    # --- wandb 初始化：resume 时续接同一个 run ---
    run = None
    if args.wandb and is_main:
        if wandb is None:
            raise ImportError("未安装 wandb，无法启用日志。")
        wandb_kwargs = dict(
            entity=wandb_cfg.get("entity", "runheyang"),
            project=wandb_cfg.get("project", "neural-ode"),
            config={
                "epochs": int(train_cfg["epochs"]),
                "batch_size": int(train_cfg["batch_size"]),
                "lr": float(train_cfg["lr"]),
                "weight_decay": float(train_cfg["weight_decay"]),
            },
        )
        if resumed_wandb_id:
            # 续接之前的 run
            wandb_kwargs["id"] = resumed_wandb_id
            wandb_kwargs["resume"] = "allow"
        else:
            wandb_kwargs["name"] = wandb_cfg.get("name", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
        run = wandb.init(**wandb_kwargs)
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

    # 循环外创建 train_loader（persistent_workers 常驻），避免每轮重建开销
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    # val_loader 在循环外创建一次，persistent_workers=False 故 worker 每次迭代结束自动回收
    val_loader = DataLoader(val_dataset, **val_loader_kwargs) if val_dataset is not None else None

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        # DDP 模式下每 epoch 设置 sampler epoch，保证数据打乱不同
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = trainer.train_one_epoch(
            train_loader,
            epoch=epoch,
        )
        _cleanup_gpu_cache()
        if scheduler is not None:
            scheduler.step()

        if is_main:
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
            fast_kl_text = ""
            if "fast_kl" in train_metrics:
                fast_kl_text = f" fast_kl={float(train_metrics['fast_kl']):.4f}"
            print(
                f"[train] epoch={epoch} "
                f"loss={train_metrics['loss']:.4f} "
                f"focal={train_metrics['focal']:.4f} "
                f"aux={train_metrics['aux']:.4f} "
                f"delta={train_metrics['delta_scene_abs_mean']:.4f}"
                f"{fast_kl_text}"
                f"{ray_total_text}"
                f"{train_sup_text}"
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

        # eval / checkpoint 仅 rank 0 执行，其他 rank 在 barrier 处等待
        if is_main:
            if val_loader is not None and args.eval_every > 0 and epoch % args.eval_every == 0:
                val_metrics = trainer.evaluate(val_loader, collect_predictions=True)
                _cleanup_gpu_cache()
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
                binned_ray_result = None
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

                # return_pcds=True 以便分箱统计复用 raycasting 结果
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

                # 分箱 ray 统计
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

            # 保存指标 JSON（每次 eval 覆盖，最终保留最后 epoch 结果）
            if args.save_metrics_json and val_loader is not None and args.eval_every > 0 and epoch % args.eval_every == 0:
                metrics_json = {
                    "epoch": epoch,
                    "mIoU": float(val_metrics["miou"]),
                    "mIoU_d": float(val_metrics.get("miou_d", 0.0)),
                    "loss": float(val_metrics["loss"]),
                }
                # Fast-KL 诊断
                if "fast_kl" in train_metrics:
                    metrics_json["fast_kl"] = {
                        "train": float(train_metrics["fast_kl"]),
                    }
                # ray loss 配置记录
                if ray_cfg is not None:
                    metrics_json["ray_config"] = {
                        k: v for k, v in ray_cfg.items()
                        if isinstance(v, (int, float, str, bool))
                    }
                metrics_json["rayiou"] = {
                    k: float(v) for k, v in rayiou_result.items()
                }
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

        # 同步所有 rank，等待 rank 0 完成 eval/checkpoint
        if use_ddp:
            dist.barrier()

    if run is not None:
        run.finish()
    cleanup_ddp()


if __name__ == "__main__":
    main()
