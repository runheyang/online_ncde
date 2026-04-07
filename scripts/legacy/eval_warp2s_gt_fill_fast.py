#!/usr/bin/env python3
"""评估基线：2s 前 GT warp 到当前，并用当前 fast logits 填充未知区域。"""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.ego_warp_list import backward_warp_dense_trilinear  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


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
        "--device",
        default="cuda",
        help="运行设备，默认 cuda（不可用时自动回退 cpu）",
    )
    parser.add_argument(
        "--mask-thresh",
        type=float,
        default=0.5,
        help="warp 后可见性阈值，> thresh 视为已覆盖",
    )
    parser.add_argument(
        "--padding-mode",
        default="zeros",
        choices=["zeros", "border", "reflection"],
        help="grid_sample 的 padding_mode（建议 zeros 以显式标记未知）",
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


def load_logits_last_frame_inputs(path: str) -> dict[str, np.ndarray]:
    """仅读取解码最后一帧需要的稀疏字段。"""
    with np.load(path, allow_pickle=False) as data:
        return {
            "sparse_coords": data["sparse_coords"],
            "sparse_topk_values": data["sparse_topk_values"],
            "sparse_topk_indices": data["sparse_topk_indices"],
            "frame_splits": data["frame_splits"],
        }


def labels_to_logits_on_device(
    semantics: np.ndarray,
    num_classes: int,
    device: torch.device,
    pos_value: float = 5.0,
    neg_value: float = -5.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """在目标设备直接构造 one-hot logits，避免 CPU one-hot 后再整块搬运。"""
    labels = torch.from_numpy(semantics.astype(np.int64, copy=False)).to(
        device=device, dtype=torch.long
    )
    logits = torch.full(
        (num_classes, *labels.shape),
        fill_value=neg_value,
        dtype=dtype,
        device=device,
    )
    logits.scatter_(0, labels.unsqueeze(0), pos_value)
    return logits


def load_sample_payload(
    info: dict[str, Any],
    root_path: str,
    gt_root: str,
    gt_mask_key: str,
    default_mask_np: np.ndarray,
) -> dict[str, Any]:
    """读取单样本评估所需的全部 CPU 数据。"""
    slow_gt_path = resolve_path(root_path, info["slow_gt_path"])
    slow_semantics, slow_mask = load_labels_selected_npz(
        slow_gt_path,
        mask_key=gt_mask_key,
        default_mask=default_mask_np,
    )

    logits_path = resolve_path(root_path, info["logits_path"])
    logits_npz = load_logits_last_frame_inputs(logits_path)

    curr_gt_path = os.path.join(gt_root, info["scene_name"], info["token"], "labels.npz")
    gt_semantics, gt_mask = load_labels_selected_npz(
        curr_gt_path,
        mask_key=gt_mask_key,
        default_mask=default_mask_np,
    )
    return {
        "T_slow_to_curr": info["T_slow_to_curr"],
        "slow_semantics": slow_semantics,
        "slow_mask": slow_mask,
        "logits": logits_npz,
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


def decode_last_frame_sparse_topk(
    sparse_coords: np.ndarray,
    sparse_topk_values: np.ndarray,
    sparse_topk_indices: np.ndarray,
    frame_splits: np.ndarray,
    grid_size: tuple[int, int, int],
    num_classes: int,
    free_index: int,
    other_fill_value: float,
    free_fill_value: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """仅解码最后一帧 fast logits，返回 (C, X, Y, Z)。"""
    x_size, y_size, z_size = grid_size
    num_frames = int(frame_splits.shape[0] - 1)
    if num_frames <= 0:
        raise ValueError(f"frame_splits 非法，无法解码最后一帧: shape={frame_splits.shape}")

    start = int(frame_splits[num_frames - 1])
    end = int(frame_splits[num_frames])

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
            gt_root=gt_root,
            gt_mask_key=gt_mask_key,
            default_mask_np=default_mask_np,
        ),
        io_workers=io_workers,
        prefetch=prefetch,
    )
    iterator = (
        progressbar.progressbar(sample_iter, max_value=len(infos), prefix="[eval warp2s_gt_fill_fast] ")
        if progressbar is not None
        else sample_iter
    )

    with torch.inference_mode():
        for sample in iterator:
            # 1) slow(2s 前) GT -> logits（直接在目标设备构造）
            slow_logits = labels_to_logits_on_device(
                semantics=sample["slow_semantics"],
                num_classes=num_classes,
                device=device,
                dtype=torch.float32,
            )
            slow_mask = torch.from_numpy(
                sample["slow_mask"].astype(np.float32, copy=False)
            ).to(device=device)
            slow_logits.mul_(slow_mask.unsqueeze(0))

            # 2) 解码当前时刻 fast logits（logits.npz 最后一帧）
            logits_npz = sample["logits"]
            fast_now = decode_last_frame_sparse_topk(
                sparse_coords=logits_npz["sparse_coords"],
                sparse_topk_values=logits_npz["sparse_topk_values"],
                sparse_topk_indices=logits_npz["sparse_topk_indices"],
                frame_splits=logits_npz["frame_splits"],
                grid_size=grid_size,
                num_classes=num_classes,
                free_index=free_index,
                other_fill_value=topk_other_fill_value,
                free_fill_value=topk_free_fill_value,
                device=device,
                dtype=torch.float32,
            )

            # 3) 2s 前 GT warp 到当前时刻（logits+cover 一次性 warp）
            transform = torch.from_numpy(sample["T_slow_to_curr"]).to(
                device=device, dtype=torch.float32
            )
            packed = torch.cat([slow_logits, slow_mask.unsqueeze(0)], dim=0)
            warped = backward_warp_dense_trilinear(
                dense_prev_feat=packed,
                transform_prev_to_curr=transform,
                spatial_shape_xyz=grid_size,
                pc_range=pc_range,
                voxel_size=voxel_size,
                padding_mode=args.padding_mode,
            )
            warped_slow = warped[:num_classes]
            warped_cover = warped[num_classes]
            known_mask = warped_cover > float(args.mask_thresh)

            # 4) 未覆盖区域用当前 fast logits 填充
            fast_now[:, known_mask] = warped_slow[:, known_mask]
            pred = fast_now.argmax(dim=0).cpu().numpy()

            # 5) 累计 IoU
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
            "padding_mode": args.padding_mode,
            "mask_thresh": float(args.mask_thresh),
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

