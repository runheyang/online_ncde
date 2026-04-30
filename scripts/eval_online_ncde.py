#!/usr/bin/env python3
"""评估 Online NCDE（同时计算 mIoU 和 RayIoU）。"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

# 输入形状固定，开启 benchmark 让 cuDNN 自动选 conv 算法
torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.data.build_dataset import build_online_ncde_dataset  # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.losses import build_loss  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner          # noqa: E402
from online_ncde.trainer import Trainer, online_ncde_collate  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--limit", type=int, default=None,
                        help="只使用前 N 条样本进行评估")
    parser.add_argument("--sweep-pkl", default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
                        help="sweep pkl 路径（相对于项目根目录）")
    parser.add_argument("--solver", choices=["heun", "euler"], default="euler",
                        help="ODE 求解器：euler（默认，Euler + next-fast 单次求值）或 heun")
    parser.add_argument(
        "--exclude-short-history",
        action="store_true",
        help=(
            "只评估满足 config.min_history_completeness（通常 4）的完整历史样本；"
            "默认包含全部短历史样本（min_history_completeness=0，h=0 走 aligner 退化分支）。"
        ),
    )
    return parser.parse_args()


def resolve_sweep_pkl(args, cfg) -> str:
    """确定 source sweep pkl 路径。"""
    if args.sweep_pkl:
        p = Path(args.sweep_pkl)
        return str(p if p.is_absolute() else (ROOT / p).resolve())
    # 从 canonical pkl 的 metadata 中找 source_info_path
    info_path = cfg["data"].get("val_info_path", cfg["data"]["info_path"])
    info_abs = Path(info_path) if Path(info_path).is_absolute() else (ROOT / info_path).resolve()
    with open(info_abs, "rb") as f:
        meta = pickle.load(f).get("metadata", {})
    src = meta.get("source_info_path", "")
    if src and Path(src).exists():
        return src
    raise FileNotFoundError(
        f"无法推断 sweep pkl 路径（canonical metadata.source_info_path={src}），"
        "请通过 --sweep-pkl 指定。"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    train_cfg = cfg.get("train", {})
    loader_cfg = cfg.get("dataloader", {})
    logits_loader = build_logits_loader(data_cfg, cfg["root_path"])

    # 默认 min_history_completeness=0，含全部短历史样本（h=0 走 aligner 退化分支）。
    # --exclude-short-history 才回退到 config 的阈值（通常 4）过滤短历史。
    min_hc = int(data_cfg.get("min_history_completeness", 4)) if args.exclude_short_history else 0
    print(f"[eval] min_history_completeness={min_hc}"
          + (f"  (--exclude-short-history 使用 config 阈值 {min_hc})" if args.exclude_short_history else ""))
    dataset = build_online_ncde_dataset(
        data_cfg,
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
        root_path=cfg["root_path"],
        logits_loader=logits_loader,
        ray_sidecar_dir=data_cfg.get("ray_sidecar_dir", None),
        ray_sidecar_split="val",
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
        min_history_completeness=min_hc,
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
    if args.limit is not None:
        n = min(args.limit, len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"[eval] --limit={args.limit}，使用前 {n} 条样本")
    loader = DataLoader(dataset, **kwargs)

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
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
    ).to(device)
    if args.solver == "euler":
        print(f"[solver] {args.solver} (next-fast only, 单次 func_g 求值)")
    else:
        print(f"[solver] {args.solver}")
    load_checkpoint_for_eval(args.checkpoint, model=model, strict=False)

    loss_cfg = cfg["loss"]
    # build_loss 在带 ray_cfg 时需要 pc_range / free_index，保持与 train 脚本一致
    ray_cfg = loss_cfg.get("ray", None)
    if ray_cfg is not None:
        ray_cfg.setdefault("pc_range", list(data_cfg["pc_range"]))
        ray_cfg.setdefault("free_index", int(data_cfg["free_index"]))
    loss_fn = build_loss(loss_cfg, num_classes=data_cfg["num_classes"]).to(device)
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
        supervision_labels=list(train_cfg.get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])),
        supervision_weights=list(train_cfg.get("supervision_weights", [0.15, 0.20, 0.25, 0.40])),
        supervision_weight_normalize=bool(train_cfg.get("supervision_weight_normalize", True)),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=str(train_cfg.get("rollout_mode", "full")),
        primary_supervision_label=str(eval_cfg.get("primary_supervision_label", "t-1.0")),
        stepwise_max_step_index=train_cfg.get("max_step_index", None),
    )

    # 单次推理：mIoU + 收集 predictions 用于 RayIoU
    metrics = trainer.evaluate(loader, collect_predictions=True)

    # --- mIoU 结果 ---
    print(
        f"[eval] loss={metrics['loss']:.4f} "
        f"focal={metrics['focal']:.4f} "
        f"aux={metrics['aux']:.4f} "
        f"miou={metrics['miou']:.4f} "
        f"miou_d={metrics.get('miou_d', float('nan')):.4f}"
    )
    class_names = metrics.get("class_names", [])
    class_iou = metrics.get("per_class_iou", [])
    if isinstance(class_names, list) and isinstance(class_iou, list):
        for name, value in zip(class_names, class_iou):
            print(f"{name}: {float(value):.2f}")

    # --- RayIoU ---
    print("\n[rayiou] 加载 lidar origins...")
    sweep_pkl = resolve_sweep_pkl(args, cfg)
    print(f"[rayiou] sweep pkl: {sweep_pkl}")

    from online_ncde.ops.dvr.ego_pose import load_origins_from_sweep_pkl
    origins_by_token = load_origins_from_sweep_pkl(sweep_pkl)
    print(f"[rayiou] 共 {len(origins_by_token)} 个 token 的 origin")

    from online_ncde.ops.dvr.ray_metrics import main as calc_rayiou

    predictions = metrics["predictions"]
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
        print(f"[rayiou] 跳过 {skipped} 个样本（无对应 lidar origin）")
    print(f"[rayiou] {len(sem_pred_list)} 个样本参与计算")

    rayiou_result = calc_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list)
    print(f"\n[rayiou] RayIoU={rayiou_result['RayIoU']:.4f}")
    print(f"[rayiou] RayIoU@1={rayiou_result['RayIoU@1']:.4f}")
    print(f"[rayiou] RayIoU@2={rayiou_result['RayIoU@2']:.4f}")
    print(f"[rayiou] RayIoU@4={rayiou_result['RayIoU@4']:.4f}")


if __name__ == "__main__":
    main()
