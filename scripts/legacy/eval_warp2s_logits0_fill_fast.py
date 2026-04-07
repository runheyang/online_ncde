#!/usr/bin/env python3
"""评估基线：warp 慢系统 slow_logit.npz，并用快系统末帧 logits 填充未知区域。"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.ego_warp_list import backward_warp_dense_trilinear  # noqa: E402
from online_ncde.data.logits_io import (  # noqa: E402
    decode_single_frame_sparse_topk,
    load_logits_npz,
)
from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


FAST_FILL_SIGMOID_THR = 0.5
FAST_FILL_KERNEL_SIZE = 3
FINAL_OPUSV2_SIGMOID_THR = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/online_ncde/fast_opusv1t__slow_opusv2l/eval.yaml"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--info-path",
        default="",
        help="可选覆盖 info_path（默认用 data.val_info_path）",
    )
    parser.add_argument(
        "--slow-logits-root",
        default="data/logits_opusv2l",
        help="慢系统 logits 根目录（优先按 info.slow_logit_path 解析，否则回退到 scene/token/slow_logit.npz）",
    )
    parser.add_argument(
        "--fast-logits-root",
        default="data/logits_opusv1t",
        help="快系统 logits 根目录（按 scene/token/logits.npz 组织）",
    )
    parser.add_argument(
        "--slow-frame-index",
        type=int,
        default=0,
        help="保留参数；slow_logit.npz 仅有单帧，因此只能指向该单帧",
    )
    parser.add_argument(
        "--fast-frame-index",
        type=int,
        default=-1,
        help="快系统使用的帧索引（默认最后一帧）",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="运行设备，默认 cuda（不可用时自动回退 cpu）",
    )
    parser.add_argument(
        "--zero-eps",
        type=float,
        default=1.0e-6,
        help="判定为 zero padding 的阈值（max(abs(logits)) <= eps）",
    )
    parser.add_argument(
        "--free-conf-thresh",
        type=float,
        default=FINAL_OPUSV2_SIGMOID_THR,
        help="最终预测阶段的 OPUSv2 风格阈值：若 sigmoid(max_non_free_logit) < thresh，则预测为 free",
    )
    parser.add_argument(
        "--padding-mode",
        default="zeros",
        choices=["zeros", "border", "reflection"],
        help="grid_sample 的 padding_mode（建议 zeros 以显式标记越界未知）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评估前 N 个 valid 样本（0 表示全量）",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="可选输出结果 json 路径",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4) // 2)),
        help="npz 异步读取线程数；0 表示关闭异步预取",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=8,
        help="I/O 预取窗口大小（最多在后台排队的样本数）",
    )
    return parser.parse_args()


def load_infos(info_path: str) -> list[dict[str, Any]]:
    with open(info_path, "rb") as f:
        payload = pickle.load(f)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    valid_infos = [info for info in infos if info.get("valid", True)]
    return valid_infos


def safe_logit(prob: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """数值稳定的 logit 变换。"""
    prob = prob.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.log(prob) - torch.log1p(-prob)


def predict_opusv2_style(
    logits: torch.Tensor,
    free_index: int,
    conf_thresh: float,
) -> torch.Tensor:
    """
    按 OPUSv2 口径从 dense logits 生成语义预测。

    只看非 free 语义通道的最大 logit；
    sigmoid(max_logit) >= conf_thresh 则输出该类别，否则输出 free。
    """
    num_classes = int(logits.shape[0])
    class_mask = torch.ones(num_classes, dtype=torch.bool, device=logits.device)
    class_mask[free_index] = False

    sem_logits = logits[class_mask]
    max_logits, sem_pred = sem_logits.max(dim=0)
    pred = sem_pred.clone()
    pred[torch.sigmoid(max_logits) < float(conf_thresh)] = int(free_index)
    return pred


def postprocess_fast_logits_opusv1(
    logits: torch.Tensor,
    free_index: int,
    score_thr: float,
    other_fill_value: float,
    free_fill_value: float,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    将快系统 dense logits 按 OPUSv1 的 occupancy 后处理规则转成可填充 logits。

    逻辑对齐 third_party/OPUS/models/opusv1/opus_head.py:
    1. 仅保留 sigmoid(max_non_free_logit) > score_thr 的体素；
    2. 对非 free 类 score 体执行 max_pool3d dilation + erosion；
    3. 原始高置信体素保持不变；
    4. 将保留结果写回 dense logits，其余体素回退为 free 先验。
    """
    num_classes = int(logits.shape[0])
    class_mask = torch.ones(num_classes, dtype=torch.bool, device=logits.device)
    class_mask[free_index] = False

    sem_logits = logits[class_mask]
    sem_scores = torch.sigmoid(sem_logits)
    keep_mask = sem_scores.amax(dim=0) > float(score_thr)

    processed = logits.new_full(logits.shape, fill_value=float(other_fill_value))
    processed[free_index] = float(free_fill_value)
    if not torch.any(keep_mask):
        return processed

    occ = sem_scores * keep_mask.unsqueeze(0).to(dtype=sem_scores.dtype)
    occ = occ.unsqueeze(0)
    pad = int(kernel_size) // 2
    dilated = F.max_pool3d(occ, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool3d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)

    original_mask = (occ > float(score_thr)).any(dim=1, keepdim=True).expand_as(eroded)
    eroded[original_mask] = occ[original_mask]
    eroded = eroded.squeeze(0)
    occupied_mask = eroded.amax(dim=0) > float(score_thr)
    if not torch.any(occupied_mask):
        return processed

    processed[free_index, occupied_mask] = float(other_fill_value)
    processed_non_free = processed[class_mask]
    processed_non_free[:, occupied_mask] = safe_logit(eroded[:, occupied_mask]).to(
        dtype=processed.dtype
    )
    processed[class_mask] = processed_non_free
    return processed


def load_labels_selected_npz(
    path: str,
    mask_key: str,
    default_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """仅读取 labels.npz 的必要字段，减少不必要的解压与拷贝。"""
    with np.load(path, allow_pickle=False) as data:
        semantics = data["semantics"]
        mask = data[mask_key] if mask_key in data.files else default_mask
    return semantics, mask


def normalize_frame_index(frame_index: int, num_frames: int) -> int:
    """将负索引转换为正索引，并做合法性校验。"""
    idx = frame_index if frame_index >= 0 else num_frames + frame_index
    if idx < 0 or idx >= num_frames:
        raise IndexError(
            f"frame_index={frame_index} 越界，num_frames={num_frames}，合法范围=[0, {num_frames - 1}]"
        )
    return idx


def decode_one_frame_sparse_topk(
    sparse_coords: np.ndarray,
    sparse_topk_values: np.ndarray,
    sparse_topk_indices: np.ndarray,
    frame_splits: np.ndarray,
    frame_index: int,
    grid_size: tuple[int, int, int],
    num_classes: int,
    free_index: int,
    other_fill_value: float,
    free_fill_value: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """仅解码指定帧 top-k logits，返回 (C, X, Y, Z)。"""
    x_size, y_size, z_size = grid_size
    num_frames = int(frame_splits.shape[0] - 1)
    if num_frames <= 0:
        raise ValueError(f"frame_splits 非法，无法解码: shape={frame_splits.shape}")
    idx = normalize_frame_index(frame_index, num_frames)
    start = int(frame_splits[idx])
    end = int(frame_splits[idx + 1])

    dense = torch.full(
        (num_classes, x_size, y_size, z_size),
        fill_value=other_fill_value,
        dtype=dtype,
        device=device,
    )
    hit_mask = torch.zeros((x_size, y_size, z_size), dtype=torch.bool, device=device)

    if end > start:
        coords_t = torch.from_numpy(sparse_coords[start:end]).to(device=device, dtype=torch.long)
        values_t = torch.from_numpy(sparse_topk_values[start:end]).to(device=device, dtype=dtype)
        indices_t = torch.from_numpy(sparse_topk_indices[start:end]).to(device=device, dtype=torch.long)

        x_idx = coords_t[:, 0]
        y_idx = coords_t[:, 1]
        z_idx = coords_t[:, 2]
        hit_mask[x_idx, y_idx, z_idx] = True

        topk = values_t.shape[1]
        cls_flat = indices_t.reshape(-1)
        x_flat = x_idx[:, None].expand(-1, topk).reshape(-1)
        y_flat = y_idx[:, None].expand(-1, topk).reshape(-1)
        z_flat = z_idx[:, None].expand(-1, topk).reshape(-1)
        dense[cls_flat, x_flat, y_flat, z_flat] = values_t.reshape(-1)

    dense[free_index, ~hit_mask] = torch.as_tensor(free_fill_value, device=device, dtype=dtype)
    return dense


def resolve_relative_sample_file(
    root_path: str,
    root_rel: str,
    rel_path: str,
) -> str:
    """按 root_rel + 相对路径解析样本文件。"""
    if not rel_path:
        raise FileNotFoundError("样本缺少相对路径字段，无法读取文件。")
    path = resolve_path(root_path, os.path.join(root_rel, rel_path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return path


def resolve_sample_logits_path(
    root_path: str,
    logits_root: str,
    info: dict[str, Any],
    info_key: str,
    default_name: str,
) -> str:
    """优先使用 info 中的相对路径字段，缺失时回退到 scene/token 命名。"""
    rel_path = str(info.get(info_key, ""))
    if rel_path:
        return resolve_relative_sample_file(root_path, logits_root, rel_path)

    scene_name = info.get("scene_name", "")
    token = info.get("token", "")
    if not scene_name or not token:
        raise KeyError("info 缺少 scene_name/token，无法拼接 logits 路径。")
    rel = os.path.join(logits_root, scene_name, token, default_name)
    path = resolve_path(root_path, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"logits 文件不存在: {path}")
    return path


def load_sample_payload(
    info: dict[str, Any],
    root_path: str,
    slow_logits_root: str,
    fast_logits_root: str,
    gt_root: str,
    gt_mask_key: str,
    default_mask_np: np.ndarray,
) -> dict[str, Any]:
    """读取单样本评估所需的全部 CPU 数据。"""
    slow_logits_path = resolve_sample_logits_path(
        root_path=root_path,
        logits_root=slow_logits_root,
        info=info,
        info_key="slow_logit_path",
        default_name="slow_logit.npz",
    )
    fast_logits_path = resolve_sample_logits_path(
        root_path=root_path,
        logits_root=fast_logits_root,
        info=info,
        info_key="logits_path",
        default_name="logits.npz",
    )
    slow_logits_npz = load_logits_npz(slow_logits_path)
    fast_logits_npz = load_logits_npz(fast_logits_path)

    curr_gt_path = os.path.join(gt_root, info["scene_name"], info["token"], "labels.npz")
    gt_semantics, gt_mask = load_labels_selected_npz(
        curr_gt_path,
        mask_key=gt_mask_key,
        default_mask=default_mask_np,
    )
    return {
        "T_slow_to_curr": info["T_slow_to_curr"],
        "slow_logits_npz": slow_logits_npz,
        "fast_logits_npz": fast_logits_npz,
        "gt_semantics": gt_semantics,
        "gt_mask": gt_mask,
    }


def iter_prefetched(
    infos: list[dict[str, Any]],
    loader: Callable[[dict[str, Any]], dict[str, Any]],
    io_workers: int,
    prefetch: int,
) -> Iterator[dict[str, Any]]:
    """按顺序产出样本，后台并发读取后续样本以隐藏磁盘 I/O。"""
    total = len(infos)
    if io_workers <= 0 or total <= 1:
        for info in infos:
            yield loader(info)
        return

    max_inflight = max(1, max(prefetch, io_workers))
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        pending: dict[int, Future[dict[str, Any]]] = {}
        submit_idx = 0
        yield_idx = 0

        initial = min(total, max_inflight)
        while submit_idx < initial:
            pending[submit_idx] = pool.submit(loader, infos[submit_idx])
            submit_idx += 1

        while yield_idx < total:
            fut = pending.pop(yield_idx)
            yield fut.result()
            if submit_idx < total:
                pending[submit_idx] = pool.submit(loader, infos[submit_idx])
                submit_idx += 1
            yield_idx += 1


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    data_cfg = cfg["data"]

    root_path = cfg["root_path"]
    info_path_cfg = data_cfg.get("val_info_path", data_cfg["info_path"])
    info_path = resolve_path(root_path, args.info_path or info_path_cfg)
    infos = load_infos(info_path)
    if args.limit > 0:
        infos = infos[: args.limit]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    num_classes = int(data_cfg["num_classes"])
    free_index = int(data_cfg["free_index"])
    grid_size = tuple(int(v) for v in data_cfg["grid_size"])
    pc_range = tuple(float(v) for v in data_cfg["pc_range"])
    voxel_size = tuple(float(v) for v in data_cfg["voxel_size"])
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    gt_mask_key = data_cfg.get("gt_mask_key", "mask_camera")
    topk_other_fill_value = float(data_cfg.get("topk_other_fill_value", -5.0))
    topk_free_fill_value = float(data_cfg.get("topk_free_fill_value", 5.0))
    io_workers = max(0, int(args.io_workers))
    prefetch = max(1, int(args.prefetch))
    default_mask_np = np.ones(grid_size, dtype=np.float32)

    metric = MetricMiouOcc3D(
        num_classes=num_classes,
        use_image_mask=True,
        use_lidar_mask=False,
    )

    sample_iter = iter_prefetched(
        infos=infos,
        loader=lambda info: load_sample_payload(
            info=info,
            root_path=root_path,
            slow_logits_root=args.slow_logits_root,
            fast_logits_root=args.fast_logits_root,
            gt_root=gt_root,
            gt_mask_key=gt_mask_key,
            default_mask_np=default_mask_np,
        ),
        io_workers=io_workers,
        prefetch=prefetch,
    )
    iterator = (
        tqdm(sample_iter, total=len(infos), desc="[eval warp2s_logits0_fill_fast]")
        if tqdm is not None
        else sample_iter
    )

    with torch.inference_mode():
        for sample in iterator:
            # 1) 慢系统单帧 slow_logit.npz 解码
            if normalize_frame_index(int(args.slow_frame_index), 1) != 0:
                raise IndexError("slow_logit.npz 仅包含单帧，slow-frame-index 只能指向该单帧。")
            slow_logits = decode_single_frame_sparse_topk(
                sparse_coords=sample["slow_logits_npz"]["sparse_coords"],
                sparse_topk_values=sample["slow_logits_npz"]["sparse_topk_values"],
                sparse_topk_indices=sample["slow_logits_npz"]["sparse_topk_indices"],
                grid_size=grid_size,
                num_classes=num_classes,
                free_index=free_index,
                other_fill_value=topk_other_fill_value,
                free_fill_value=topk_free_fill_value,
                device=device,
                dtype=torch.float32,
            )

            # 2) 快系统末帧 logits 解码（用于填充）
            fast_now = decode_one_frame_sparse_topk(
                sparse_coords=sample["fast_logits_npz"]["sparse_coords"],
                sparse_topk_values=sample["fast_logits_npz"]["sparse_topk_values"],
                sparse_topk_indices=sample["fast_logits_npz"]["sparse_topk_indices"],
                frame_splits=sample["fast_logits_npz"]["frame_splits"],
                frame_index=int(args.fast_frame_index),
                grid_size=grid_size,
                num_classes=num_classes,
                free_index=free_index,
                other_fill_value=topk_other_fill_value,
                free_fill_value=topk_free_fill_value,
                device=device,
                dtype=torch.float32,
            )
            fast_now = postprocess_fast_logits_opusv1(
                logits=fast_now,
                free_index=free_index,
                score_thr=FAST_FILL_SIGMOID_THR,
                other_fill_value=topk_other_fill_value,
                free_fill_value=topk_free_fill_value,
                kernel_size=FAST_FILL_KERNEL_SIZE,
            )

            # 3) 直接使用 grid_sample warp 慢系统 logits（不依赖 image_mask）
            transform = torch.from_numpy(sample["T_slow_to_curr"]).to(
                device=device, dtype=torch.float32
            )
            warped_slow = backward_warp_dense_trilinear(
                dense_prev_feat=slow_logits,
                transform_prev_to_curr=transform,
                spatial_shape_xyz=grid_size,
                pc_range=pc_range,
                voxel_size=voxel_size,
                padding_mode=args.padding_mode,
            )

            # 4) zero padding 区域用快系统末帧 logits 填充，其余区域使用 warp 结果
            padded_mask = warped_slow.abs().amax(dim=0) <= float(args.zero_eps)
            merged = warped_slow.clone()
            merged[:, padded_mask] = fast_now[:, padded_mask]
            pred = predict_opusv2_style(
                merged,
                free_index=free_index,
                conf_thresh=float(args.free_conf_thresh),
            ).cpu().numpy()

            # 5) 累计 IoU（评估口径保持与当前脚本一致）
            metric.add_batch(
                semantics_pred=pred,
                semantics_gt=sample["gt_semantics"],
                mask_lidar=None,
                mask_camera=sample["gt_mask"],
            )

    miou = metric.count_miou(verbose=False)
    per_class = metric.get_per_class_iou()
    per_class = np.nan_to_num(per_class, nan=0.0)

    print(f"[eval] samples={len(infos)} miou={miou:.4f}")
    for name, iou in zip(metric.class_names, per_class.tolist()):
        print(f"{name}: {float(iou):.2f}")

    if args.output_json:
        payload = {
            "num_samples": len(infos),
            "miou": float(miou),
            "class_names": list(metric.class_names),
            "per_class_iou": [float(v) for v in per_class.tolist()],
            "config": args.config,
            "info_path": info_path,
            "slow_logits_root": args.slow_logits_root,
            "fast_logits_root": args.fast_logits_root,
            "slow_frame_index": int(args.slow_frame_index),
            "fast_frame_index": int(args.fast_frame_index),
            "padding_mode": args.padding_mode,
            "zero_eps": float(args.zero_eps),
            "free_conf_thresh": float(args.free_conf_thresh),
            "io_workers": int(io_workers),
            "prefetch": int(prefetch),
        }
        out_path = resolve_path(root_path, args.output_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
