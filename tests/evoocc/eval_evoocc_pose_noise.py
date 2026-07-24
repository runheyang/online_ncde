#!/usr/bin/env python3
"""评估 EvoOcc 对 ego pose 高斯扰动的敏感度。

评估逻辑与 ``scripts/eval_evoocc.py`` 保持一致，唯一差别是在 dataset 输出后，
向每帧 ``frame_ego2global`` 的局部平移 (x, y) 和 yaw 加入独立高斯噪声：

    P_noisy = P_clean @ E(dx, dy, dyaw)

预设档位沿用常见的 0.1 m / 0.01 rad 参数化。噪声由 scene、帧 token 和 seed
稳定生成，因此同一帧出现在不同重叠样本中时会得到相同扰动。

示例：
    conda run -n neural_ode python tests/evoocc/eval_evoocc_pose_noise.py \
        --config configs/evoocc/fast_alocc2dmini__slow_alocc3d.yaml \
        --checkpoint ckpts/xxx.pt \
        --noise-seeds 0 1 2 \
        --output-json outputs/pose_noise_eval.json
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# 输入形状固定，开启 benchmark 让 cuDNN 自动选择卷积算法
torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from evoocc.config import load_config_with_base  # noqa: E402
from evoocc.data.build_dataset import build_evoocc_dataset  # noqa: E402
from evoocc.data.build_logits_loader import build_logits_loader  # noqa: E402
from evoocc.evaluation import evaluate_dense_occ  # noqa: E402
from evoocc.losses import build_loss  # noqa: E402
from evoocc.models.evoocc_aligner import EvoOccAligner  # noqa: E402
from evoocc.trainer import Trainer, evoocc_collate  # noqa: E402
from evoocc.utils.checkpoints import load_checkpoint_for_eval  # noqa: E402


@dataclass(frozen=True)
class PoseNoiseLevel:
    """单个 ego pose 扰动档位。"""

    name: str
    translation_std_m: float
    yaw_std_rad: float

    @property
    def yaw_std_deg(self) -> float:
        return math.degrees(self.yaw_std_rad)


POSE_NOISE_LEVELS: dict[str, PoseNoiseLevel] = {
    "clean": PoseNoiseLevel("clean", 0.0, 0.0),
    "moderate": PoseNoiseLevel("moderate", 0.10, 0.01),
    "severe": PoseNoiseLevel("severe", 0.20, 0.02),
    "extreme": PoseNoiseLevel("extreme", 0.50, 0.05),
}


def _sequence_item(value: Any, index: int) -> str:
    if isinstance(value, (list, tuple)) and index < len(value):
        item = str(value[index])
        if item:
            return item
    return ""


def _stable_frame_seed(
    *,
    base_seed: int,
    sample: dict[str, Any],
    frame_index: int,
    pose: torch.Tensor,
) -> int:
    """为物理帧构造稳定 seed，保证重叠样本中的扰动一致。"""
    meta = cast("dict[str, Any]", sample.get("meta", {}))
    scene_name = str(meta.get("scene_name", ""))

    frame_id = ""
    for key in ("frame_tokens", "frame_sample_tokens"):
        frame_id = _sequence_item(meta.get(key, None), frame_index)
        if frame_id:
            frame_id = f"{key}:{frame_id}"
            break

    if not frame_id:
        timestamps = sample.get("frame_timestamps", None)
        if torch.is_tensor(timestamps) and frame_index < int(timestamps.numel()):
            frame_id = f"timestamp:{int(timestamps.reshape(-1)[frame_index].item())}"

    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(base_seed)).encode("utf-8"))
    digest.update(b"|")
    digest.update(scene_name.encode("utf-8"))
    digest.update(b"|")
    if frame_id:
        digest.update(frame_id.encode("utf-8"))
    else:
        # 旧格式缺少 token/timestamp 时，以 clean pose 作为跨样本稳定标识。
        pose_bytes = pose.detach().cpu().contiguous().numpy().tobytes()
        digest.update(b"pose:")
        digest.update(pose_bytes)
    return int.from_bytes(digest.digest(), byteorder="little", signed=False) & ((1 << 63) - 1)


def perturb_frame_ego2global(
    sample: dict[str, Any],
    level: PoseNoiseLevel,
    seed: int,
) -> torch.Tensor:
    """在 ego 局部坐标系对每帧 pose 施加 SE(2) 高斯扰动。"""
    poses = cast(torch.Tensor, sample["frame_ego2global"])
    if poses.ndim != 3 or tuple(poses.shape[-2:]) != (4, 4):
        raise ValueError(
            "frame_ego2global 必须为 (T,4,4)，"
            f"实际 shape={tuple(poses.shape)}"
        )
    if level.translation_std_m == 0.0 and level.yaw_std_rad == 0.0:
        return poses.clone()

    num_frames = int(poses.shape[0])
    standard_noise = torch.empty((num_frames, 3), dtype=torch.float64)
    for frame_index in range(num_frames):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_frame_seed(
                base_seed=seed,
                sample=sample,
                frame_index=frame_index,
                pose=poses[frame_index],
            )
        )
        standard_noise[frame_index] = torch.randn(
            (3,), generator=generator, dtype=torch.float64
        )

    dx = standard_noise[:, 0] * float(level.translation_std_m)
    dy = standard_noise[:, 1] * float(level.translation_std_m)
    dyaw = standard_noise[:, 2] * float(level.yaw_std_rad)

    error_transform = torch.eye(4, dtype=torch.float64).repeat(num_frames, 1, 1)
    cos_yaw = torch.cos(dyaw)
    sin_yaw = torch.sin(dyaw)
    error_transform[:, 0, 0] = cos_yaw
    error_transform[:, 0, 1] = -sin_yaw
    error_transform[:, 1, 0] = sin_yaw
    error_transform[:, 1, 1] = cos_yaw
    error_transform[:, 0, 3] = dx
    error_transform[:, 1, 3] = dy

    # 右乘表示误差定义在每帧 ego 局部坐标系中。
    noisy_poses = poses.detach().cpu().to(torch.float64) @ error_transform
    return noisy_poses.to(dtype=poses.dtype)


class PoseNoiseEvalDataset(Dataset):
    """只替换 frame_ego2global，其余字段原样复用 base dataset。"""

    def __init__(
        self,
        base_dataset: Dataset,
        level: PoseNoiseLevel,
        seed: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.level = level
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = cast("dict[str, Any]", self.base_dataset[index])
        noisy_sample = dict(sample)
        noisy_sample["frame_ego2global"] = perturb_frame_ego2global(
            sample=sample,
            level=self.level,
            seed=self.seed,
        )

        meta = dict(cast("dict[str, Any]", sample.get("meta", {})))
        meta["pose_noise"] = {
            "level": self.level.name,
            "seed": self.seed,
            "translation_std_m": self.level.translation_std_m,
            "yaw_std_rad": self.level.yaw_std_rad,
        }
        noisy_sample["meta"] = meta
        return noisy_sample


def _parse_level_names(spec: str) -> list[str]:
    names = [item.strip().lower() for item in str(spec).split(",") if item.strip()]
    if not names:
        raise ValueError("--levels 不能为空")
    unknown = [name for name in names if name not in POSE_NOISE_LEVELS]
    if unknown:
        raise ValueError(
            f"未知噪声档位 {unknown}；可选值={list(POSE_NOISE_LEVELS)}"
        )
    # 去重但保持用户指定顺序。
    return list(dict.fromkeys(names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EvoOcc ego pose 高斯扰动敏感度评估（mIoU + RayIoU）"
    )
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument(
        "--levels",
        default="clean,moderate,severe",
        help="要评估的档位，逗号分隔",
    )
    parser.add_argument(
        "--noise-seeds",
        type=int,
        nargs="+",
        default=[0],
        help="非 clean 档位的随机种子；论文结果建议使用 0 1 2",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="只使用前 N 条样本进行评估")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="覆盖 eval.batch_size")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="覆盖 eval.num_workers")
    parser.add_argument(
        "--sweep-pkl",
        default=None,
        help="覆盖 RayIoU 使用的 sweep pkl；相对路径按项目根目录解析",
    )
    parser.add_argument("--solver", choices=["heun", "euler"], default="euler")
    parser.add_argument(
        "--exclude-short-history",
        action="store_true",
        help="只评估满足 config.min_history_completeness 的完整历史样本",
    )
    parser.add_argument(
        "--skip-rayiou",
        action="store_true",
        help="跳过 RayIoU，用于快速检查",
    )
    parser.add_argument(
        "--print-per-class",
        action="store_true",
        help="为每次运行打印逐类别 IoU",
    )
    parser.add_argument(
        "--print-rayiou-table",
        action="store_true",
        help="为每次运行打印完整 RayIoU 表",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="保存逐次结果和跨 seed 汇总；相对路径按项目根目录解析",
    )
    return parser.parse_args()


def resolve_sweep_pkl(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    """优先使用 CLI/config，最后从 canonical metadata 推断 sweep pkl。"""
    sweep_path = args.sweep_pkl or cfg.get("eval", {}).get("sweep_pkl", None)
    if sweep_path:
        path = Path(str(sweep_path))
        return str(path if path.is_absolute() else (ROOT / path).resolve())

    data_cfg = cfg["data"]
    info_path = data_cfg.get("val_info_path", data_cfg["info_path"])
    info_abs = Path(info_path) if Path(info_path).is_absolute() else (ROOT / info_path).resolve()
    with open(info_abs, "rb") as file:
        metadata = pickle.load(file).get("metadata", {})
    source_path = str(metadata.get("source_info_path", ""))
    if source_path and Path(source_path).exists():
        return source_path
    raise FileNotFoundError(
        "无法推断 sweep pkl 路径，请通过 --sweep-pkl 指定；"
        f"metadata.source_info_path={source_path!r}"
    )


def build_base_eval_dataset(
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> Dataset:
    data_cfg = cfg["data"]
    logits_loader = build_logits_loader(data_cfg, cfg["root_path"])
    min_hc = (
        int(data_cfg.get("min_history_completeness", 4))
        if args.exclude_short_history
        else 0
    )
    print(
        f"[eval] min_history_completeness={min_hc}"
        + ("  (--exclude-short-history)" if args.exclude_short_history else "")
    )
    dataset: Dataset = build_evoocc_dataset(
        data_cfg,
        info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
        root_path=cfg["root_path"],
        logits_loader=logits_loader,
        ray_sidecar_dir=data_cfg.get("ray_sidecar_dir", None),
        ray_sidecar_split="val",
        fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
        min_history_completeness=min_hc,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(f"--limit 必须 > 0，当前为 {args.limit}")
        keep = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(keep))
        print(f"[eval] --limit={args.limit}，实际使用 {keep} 条样本")
    return dataset


def build_eval_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> DataLoader:
    eval_cfg = cfg["eval"]
    loader_cfg = cfg.get("dataloader", {})
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(eval_cfg.get("num_workers", 4))
    )
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(eval_cfg.get("batch_size", 1))
    )
    if num_workers < 0:
        raise ValueError(f"--num-workers 必须 >= 0，当前为 {num_workers}")
    if batch_size <= 0:
        raise ValueError(f"--batch-size 必须 > 0，当前为 {batch_size}")

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": False,
        "collate_fn": evoocc_collate,
        "pin_memory": loader_cfg.get("pin_memory", False),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    return DataLoader(dataset, **kwargs)


def build_eval_trainer(
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> Trainer:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    train_cfg = cfg.get("train", {})

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = EvoOccAligner(
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
        print("[solver] euler (next-fast only, 单次 func_g 求值)")
    else:
        print("[solver] heun")
    load_checkpoint_for_eval(args.checkpoint, model=model, strict=False)

    loss_cfg = copy.deepcopy(cfg["loss"])
    ray_cfg = loss_cfg.get("ray", None)
    if ray_cfg is not None:
        ray_cfg.setdefault("pc_range", list(data_cfg["pc_range"]))
        ray_cfg.setdefault("free_index", int(data_cfg["free_index"]))
    loss_fn = build_loss(loss_cfg, num_classes=data_cfg["num_classes"]).to(device)

    return Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-4),
        loss_fn=loss_fn,
        device=device,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        free_conf_thresh=eval_cfg.get("free_conf_thresh", None),
        log_interval=eval_cfg.get("log_interval", 20),
        clip_norm=1.0,
        supervision_labels=list(
            train_cfg.get("supervision_labels", ["t-1.5", "t-1.0", "t-0.5", "t"])
        ),
        supervision_weights=list(
            train_cfg.get("supervision_weights", [0.15, 0.20, 0.25, 0.40])
        ),
        supervision_weight_normalize=bool(
            train_cfg.get("supervision_weight_normalize", True)
        ),
        log_multistep_losses=bool(eval_cfg.get("log_multistep_losses", True)),
        rollout_mode=str(train_cfg.get("rollout_mode", "full")),
        primary_supervision_label=str(
            eval_cfg.get("primary_supervision_label", "t-1.0")
        ),
        stepwise_max_step_index=train_cfg.get("max_step_index", None),
        metric_variant=str(
            data_cfg.get("metric_variant", data_cfg.get("dataset_variant", "occ3d"))
        ),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def evaluate_one_run(
    *,
    trainer: Trainer,
    loader: DataLoader,
    cfg: dict[str, Any],
    level: PoseNoiseLevel,
    seed: int,
    enable_rayiou: bool,
    origins_by_token: dict[str, Any] | None,
    print_per_class: bool,
    print_rayiou_table: bool,
) -> dict[str, Any]:
    data_cfg = cfg["data"]
    metrics = trainer.evaluate(loader, collect_predictions=True, compute_miou=False)
    predictions = metrics.pop("predictions")
    dense_eval = evaluate_dense_occ(
        predictions,
        num_classes=int(data_cfg["num_classes"]),
        metric_variant=str(
            data_cfg.get("metric_variant", data_cfg.get("dataset_variant", "occ3d"))
        ),
        enable_rayiou=enable_rayiou,
        origins_by_token=origins_by_token,
        print_rayiou_table=print_rayiou_table,
    )
    dense_all = dense_eval["all"]
    rayiou = dense_all.get("rayiou", None)

    result_metrics: dict[str, Any] = {
        "loss": _optional_float(metrics.get("loss", None)),
        "focal": _optional_float(metrics.get("focal", None)),
        "aux": _optional_float(metrics.get("aux", None)),
        "miou": _optional_float(dense_all.get("miou", None)),
        "miou_d": _optional_float(dense_all.get("miou_d", None)),
        "occupied_iou": _optional_float(dense_all.get("occupied_iou", None)),
        "rayiou": _optional_float(rayiou.get("RayIoU", None)) if rayiou else None,
        "rayiou_at_1": _optional_float(rayiou.get("RayIoU@1", None)) if rayiou else None,
        "rayiou_at_2": _optional_float(rayiou.get("RayIoU@2", None)) if rayiou else None,
        "rayiou_at_4": _optional_float(rayiou.get("RayIoU@4", None)) if rayiou else None,
        "num_predictions": int(dense_eval["num_predictions"]),
        "rayiou_num_samples": int(rayiou.get("num_samples", 0)) if rayiou else 0,
        "per_class_iou": [float(value) for value in dense_all.get("per_class_iou", [])],
        "class_names": list(dense_all.get("class_names", [])),
    }
    rayiou_meta = dense_eval.get("rayiou_meta", None)
    if rayiou_meta is not None:
        result_metrics["rayiou_missing_origin_count"] = int(
            rayiou_meta.get("missing_origin_count", 0)
        )

    print(
        f"[result] level={level.name} seed={seed} "
        f"sigma_xy={level.translation_std_m:.3f}m "
        f"sigma_yaw={level.yaw_std_rad:.3f}rad/{level.yaw_std_deg:.2f}deg "
        f"mIoU={result_metrics['miou']:.4f} "
        f"mIoU_D={result_metrics['miou_d']:.4f}"
        + (
            f" RayIoU={result_metrics['rayiou']:.4f}"
            if result_metrics["rayiou"] is not None
            else ""
        )
    )
    if print_per_class:
        for name, value in zip(
            result_metrics["class_names"], result_metrics["per_class_iou"]
        ):
            print(f"{name}: {float(value):.2f}")

    return {
        "level": level.name,
        "seed": int(seed),
        "translation_std_m": level.translation_std_m,
        "yaw_std_rad": level.yaw_std_rad,
        "yaw_std_deg": level.yaw_std_deg,
        "metrics": result_metrics,
    }


def summarize_runs(
    runs: list[dict[str, Any]],
    level_names: list[str],
) -> dict[str, Any]:
    metric_names = ("miou", "miou_d", "rayiou", "rayiou_at_1", "rayiou_at_2", "rayiou_at_4")
    rayiou_metric_names = {"rayiou", "rayiou_at_1", "rayiou_at_2", "rayiou_at_4"}
    summary: dict[str, Any] = {}
    for level_name in level_names:
        level_runs = [run for run in runs if run["level"] == level_name]
        if not level_runs:
            continue
        level_summary: dict[str, Any] = {
            "num_runs": len(level_runs),
            "translation_std_m": level_runs[0]["translation_std_m"],
            "yaw_std_rad": level_runs[0]["yaw_std_rad"],
            "yaw_std_deg": level_runs[0]["yaw_std_deg"],
            "metrics": {},
        }
        for metric_name in metric_names:
            values = [
                float(run["metrics"][metric_name])
                for run in level_runs
                if run["metrics"].get(metric_name, None) is not None
            ]
            if not values:
                continue
            # 底层 RayIoU 保持 0--1；仅汇总时转成百分数，与 mIoU/mIoU_D 对齐。
            if metric_name in rayiou_metric_names:
                values = [value * 100.0 for value in values]
            level_summary["metrics"][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        summary[level_name] = level_summary

    clean_summary = summary.get("clean", None)
    if clean_summary is not None:
        for level_summary in summary.values():
            for metric_name, stats in level_summary["metrics"].items():
                clean_stats = clean_summary["metrics"].get(metric_name, None)
                if clean_stats is not None:
                    stats["delta_vs_clean"] = float(stats["mean"] - clean_stats["mean"])
    return summary


def _format_mean_std(stats: dict[str, float] | None) -> str:
    if not stats:
        return "--"
    mean = float(stats["mean"])
    std = float(stats["std"])
    delta = float(stats.get("delta_vs_clean", 0.0))
    return f"{mean:.3f}±{std:.3f} ({delta:+.3f})"


def print_summary(summary: dict[str, Any], level_names: list[str]) -> None:
    print("\n[summary] mean±std，括号内为相对 clean 的绝对变化")
    print(
        f"{'level':<10} {'sigma_xy':>9} {'sigma_yaw':>19} "
        f"{'mIoU':>24} {'mIoU_D':>24} {'RayIoU':>24}"
    )
    for level_name in level_names:
        item = summary.get(level_name, None)
        if item is None:
            continue
        print(
            f"{level_name:<10} "
            f"{item['translation_std_m']:>8.3f}m "
            f"{item['yaw_std_rad']:>7.3f}rad/{item['yaw_std_deg']:>5.2f}deg "
            f"{_format_mean_std(item['metrics'].get('miou')):>24} "
            f"{_format_mean_std(item['metrics'].get('miou_d')):>24} "
            f"{_format_mean_std(item['metrics'].get('rayiou')):>24}"
        )


def save_results(
    output_path: str,
    *,
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    path = Path(output_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "noise_model": "independent Gaussian SE(2) pose jitter; P_noisy=P_clean@E",
        "levels": {
            name: {
                **asdict(level),
                "yaw_std_deg": level.yaw_std_deg,
            }
            for name, level in POSE_NOISE_LEVELS.items()
        },
        "args": vars(args),
        "runs": runs,
        "summary": summary,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[output] 已保存: {path}")


def main() -> None:
    args = parse_args()
    level_names = _parse_level_names(args.levels)
    noise_seeds = list(dict.fromkeys(int(seed) for seed in args.noise_seeds))
    if not noise_seeds:
        raise ValueError("--noise-seeds 不能为空")

    cfg = load_config_with_base(args.config)
    base_dataset = build_base_eval_dataset(args, cfg)
    print(f"[dataset] num_samples={len(base_dataset)}")
    trainer = build_eval_trainer(args, cfg)

    enable_rayiou = bool(cfg["eval"].get("enable_rayiou", True)) and not args.skip_rayiou
    origins_by_token: dict[str, Any] | None = None
    if enable_rayiou:
        sweep_pkl = resolve_sweep_pkl(args, cfg)
        print(f"[rayiou] sweep pkl: {sweep_pkl}")
        from evoocc.ops.dvr.ego_pose import load_origins_from_sweep_pkl

        origins_by_token = load_origins_from_sweep_pkl(sweep_pkl)
        print(f"[rayiou] 已加载 {len(origins_by_token)} 个 token 的 lidar origin")
    else:
        print("[rayiou] disabled")

    runs: list[dict[str, Any]] = []
    for level_name in level_names:
        level = POSE_NOISE_LEVELS[level_name]
        # clean 与 seed 无关，只运行一次，避免无意义的重复评估。
        seeds_for_level = [noise_seeds[0]] if level_name == "clean" else noise_seeds
        for seed in seeds_for_level:
            print(
                f"\n[run] level={level.name} seed={seed} "
                f"sigma_xy={level.translation_std_m:.3f}m "
                f"sigma_yaw={level.yaw_std_rad:.3f}rad/{level.yaw_std_deg:.2f}deg"
            )
            noisy_dataset = PoseNoiseEvalDataset(
                base_dataset=base_dataset,
                level=level,
                seed=seed,
            )
            loader = build_eval_loader(noisy_dataset, args, cfg)
            runs.append(
                evaluate_one_run(
                    trainer=trainer,
                    loader=loader,
                    cfg=cfg,
                    level=level,
                    seed=seed,
                    enable_rayiou=enable_rayiou,
                    origins_by_token=origins_by_token,
                    print_per_class=args.print_per_class,
                    print_rayiou_table=args.print_rayiou_table,
                )
            )
            # persistent worker 在多档位循环中应随 DataLoader 及时释放。
            del loader
            del noisy_dataset
            gc.collect()

    summary = summarize_runs(runs, level_names)
    print_summary(summary, level_names)
    if args.output_json:
        save_results(
            args.output_json,
            args=args,
            runs=runs,
            summary=summary,
        )


if __name__ == "__main__":
    main()
