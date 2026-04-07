#!/usr/bin/env python3
"""评估保存的 logits 列表：对所有可匹配 keyframe 帧计算各类别 IoU 与 mIoU。"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.keyframe_mapping import NuScenesKeyFrameResolver  # noqa: E402
from online_ncde.data.labels_io import load_labels_npz  # noqa: E402
from online_ncde.metrics import MetricMiouOcc3D  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved logits.npz on all keyframe steps in saved list."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/online_ncde/fast_opusv1t__slow_opusv2l/eval.yaml"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--saved-list",
        required=True,
        help="保存 logits 的列表文件（pkl/json，支持 dict['infos']/dict['entries']/list）",
    )
    parser.add_argument(
        "--pred-path-key",
        default="pred_logits_path",
        help="列表中保存预测 logits 相对/绝对路径的字段名（默认 pred_logits_path）",
    )
    parser.add_argument(
        "--pred-root",
        default="",
        help="预测 logits 根目录；当 pred_path_key 是相对路径时会拼接该目录",
    )
    parser.add_argument(
        "--pred-format",
        choices=("topk", "full"),
        default="topk",
        help="预测 logits 文件格式：topk=logits.npz，full=logits_full.npz",
    )
    parser.add_argument(
        "--pred-postprocess",
        choices=("none", "opusv1"),
        default="none",
        help="评估前对 dense logits 施加的可选后处理",
    )
    parser.add_argument(
        "--postprocess-score-thr",
        type=float,
        default=0.5,
        help="OPUSv1 后处理的 sigmoid occupancy 阈值",
    )
    parser.add_argument(
        "--postprocess-kernel-size",
        type=int,
        default=3,
        help="OPUSv1 后处理的 max_pool3d kernel size",
    )
    parser.add_argument(
        "--scene-token-layout",
        action="store_true",
        help="忽略 pred_path_key，直接按 pred_root/scene_name/token/logits.npz 查找",
    )
    parser.add_argument(
        "--sidecar-path",
        default="",
        help="可选 sidecar（当 frame_tokens 缺失时回退用 4 时刻监督映射）",
    )
    parser.add_argument(
        "--nusc-dataroot",
        default="data/nuscenes",
        help="NuScenes dataroot（用于 frame_token -> keyframe 映射）",
    )
    parser.add_argument(
        "--nusc-version",
        default="v1.0-trainval",
        help="NuScenes 版本（默认 v1.0-trainval）",
    )
    parser.add_argument(
        "--sweep-info-path",
        default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
        help="sweep pkl，用于约束可评估 sample token",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评估前 N 条列表样本，0 表示全量",
    )
    parser.add_argument(
        "--deduplicate-gt",
        action="store_true",
        help="按 (scene_name, gt_token) 去重，避免滑窗重复统计同一 keyframe GT",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="可选：将统计结果写入 json",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 4) // 2)),
        help="并行线程数（读取预测/GT+解码+统计）",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=64,
        help="预取窗口大小（最多并发中的样本 future 数）",
    )
    return parser.parse_args()


def load_saved_list(path: str) -> list[dict[str, Any]]:
    """读取保存列表，统一返回 list[dict]。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"saved-list 不存在: {p}")

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with p.open("rb") as f:
            payload = pickle.load(f)

    if isinstance(payload, dict):
        if isinstance(payload.get("infos", None), list):
            infos = payload["infos"]
        elif isinstance(payload.get("entries", None), list):
            infos = payload["entries"]
        else:
            raise KeyError("saved-list dict 需包含 'infos' 或 'entries' 列表字段。")
    elif isinstance(payload, list):
        infos = payload
    else:
        raise TypeError(f"saved-list 类型不支持: {type(payload)}")

    out = []
    for item in infos:
        if isinstance(item, dict):
            out.append(item)
    return out


def load_sidecar_index(path: str) -> dict[str, dict[str, Any]]:
    """读取 sidecar，并按 token 建立索引。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"sidecar 不存在: {p}")
    with p.open("rb") as f:
        payload = pickle.load(f)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        raise TypeError("sidecar.entries 需为 list")

    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        token = str(entry.get("token", ""))
        if token:
            index[token] = entry
    return index


def resolve_pred_logits_path(
    info: dict[str, Any],
    root_path: str,
    pred_root: str,
    pred_path_key: str,
    scene_token_layout: bool,
    pred_format: str,
) -> str | None:
    """解析当前样本预测 logits 文件路径。"""
    scene_name = str(info.get("scene_name", ""))
    token = str(info.get("token", ""))
    pred_root_abs = resolve_path(root_path, pred_root) if pred_root else ""
    pred_filename = "logits_full.npz" if pred_format == "full" else "logits.npz"

    # 模式1：按 scene/token 固定目录组织
    if scene_token_layout:
        if not scene_name or not token or not pred_root_abs:
            return None
        path = os.path.join(pred_root_abs, scene_name, token, pred_filename)
        return path if os.path.exists(path) else None

    # 模式2：优先使用指定字段
    path_str = str(info.get(pred_path_key, ""))
    candidates: list[str] = []
    if path_str:
        path_str = rewrite_pred_rel_or_abs_path(path_str=path_str, pred_filename=pred_filename)
        if os.path.isabs(path_str):
            candidates.append(path_str)
        else:
            if pred_root_abs:
                candidates.append(os.path.join(pred_root_abs, path_str))
            candidates.append(resolve_path(root_path, path_str))

    # 兼容常见字段名，避免用户每次手动指定。
    if not candidates:
        for fallback_key in ("aligned_logits_path", "saved_logits_path", "logits_path"):
            p = str(info.get(fallback_key, ""))
            if not p:
                continue
            p = rewrite_pred_rel_or_abs_path(path_str=p, pred_filename=pred_filename)
            if os.path.isabs(p):
                candidates.append(p)
            else:
                if pred_root_abs:
                    candidates.append(os.path.join(pred_root_abs, p))
                candidates.append(resolve_path(root_path, p))
            break

    # 再兜底一次 scene/token 目录
    if scene_name and token and pred_root_abs:
        candidates.append(os.path.join(pred_root_abs, scene_name, token, pred_filename))

    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def rewrite_pred_rel_or_abs_path(path_str: str, pred_filename: str) -> str:
    """将路径中的 logits 文件名按 pred_format 改写为目标文件名。"""
    parent, basename = os.path.split(path_str)
    if basename in {"logits.npz", "logits_full.npz"}:
        return os.path.join(parent, pred_filename) if parent else pred_filename
    return path_str


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
) -> torch.Tensor:
    """仅解码指定帧 top-k 为 dense logits，返回 (C, X, Y, Z)。"""
    x_size, y_size, z_size = grid_size
    num_frames = int(frame_splits.shape[0] - 1)
    if frame_index < 0 or frame_index >= num_frames:
        raise IndexError(f"frame_index 越界: {frame_index}, num_frames={num_frames}")

    start = int(frame_splits[frame_index])
    end = int(frame_splits[frame_index + 1])

    dense = torch.full(
        (num_classes, x_size, y_size, z_size),
        fill_value=float(other_fill_value),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    hit_mask = torch.zeros((x_size, y_size, z_size), dtype=torch.bool, device=torch.device("cpu"))

    if end > start:
        coords_t = torch.from_numpy(sparse_coords[start:end]).to(dtype=torch.long)
        values_t = torch.from_numpy(sparse_topk_values[start:end]).to(dtype=torch.float32)
        indices_t = torch.from_numpy(sparse_topk_indices[start:end]).to(dtype=torch.long)

        x_idx = coords_t[:, 0]
        y_idx = coords_t[:, 1]
        z_idx = coords_t[:, 2]
        hit_mask[x_idx, y_idx, z_idx] = True

        n_hit, topk = values_t.shape
        if n_hit > 0 and topk > 0:
            x_rep = x_idx.repeat_interleave(topk)
            y_rep = y_idx.repeat_interleave(topk)
            z_rep = z_idx.repeat_interleave(topk)
            c_rep = indices_t.reshape(-1)
            v_rep = values_t.reshape(-1)
            dense[c_rep, x_rep, y_rep, z_rep] = v_rep

    dense[free_index, ~hit_mask] = torch.as_tensor(float(free_fill_value), dtype=torch.float32)
    return dense


def decode_one_frame_sparse_full(
    sparse_coords: np.ndarray,
    sparse_values: np.ndarray,
    frame_splits: np.ndarray,
    frame_index: int,
    grid_size: tuple[int, int, int],
    num_classes: int,
    free_index: int,
    other_fill_value: float,
    free_fill_value: float,
) -> torch.Tensor:
    """仅解码指定帧 full sparse logits 为 dense logits，返回 (C, X, Y, Z)。"""
    x_size, y_size, z_size = grid_size
    num_frames = int(frame_splits.shape[0] - 1)
    if frame_index < 0 or frame_index >= num_frames:
        raise IndexError(f"frame_index 越界: {frame_index}, num_frames={num_frames}")

    semantic_indices = [c for c in range(num_classes) if c != int(free_index)]
    if sparse_values.ndim != 2 or sparse_values.shape[1] != len(semantic_indices):
        raise ValueError(
            "sparse_values 维度异常，"
            f"期望第二维为 {len(semantic_indices)}，实际为 {tuple(sparse_values.shape)}"
        )

    start = int(frame_splits[frame_index])
    end = int(frame_splits[frame_index + 1])

    dense = torch.full(
        (num_classes, x_size, y_size, z_size),
        fill_value=float(other_fill_value),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    hit_mask = torch.zeros((x_size, y_size, z_size), dtype=torch.bool, device=torch.device("cpu"))

    if end > start:
        coords_t = torch.from_numpy(sparse_coords[start:end]).to(dtype=torch.long)
        values_t = torch.from_numpy(sparse_values[start:end]).to(dtype=torch.float32)

        x_idx = coords_t[:, 0]
        y_idx = coords_t[:, 1]
        z_idx = coords_t[:, 2]
        hit_mask[x_idx, y_idx, z_idx] = True

        semantic_tensor = torch.as_tensor(semantic_indices, dtype=torch.long)
        dense[
            semantic_tensor[:, None],
            x_idx[None, :],
            y_idx[None, :],
            z_idx[None, :],
        ] = values_t.transpose(0, 1)

    dense[free_index, ~hit_mask] = torch.as_tensor(float(free_fill_value), dtype=torch.float32)
    return dense


def postprocess_fast_logits_opusv1(
    logits: torch.Tensor,
    free_index: int,
    score_thr: float,
    other_fill_value: float,
    free_fill_value: float,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    将 dense logits 按 OPUSv1 的 occupancy 后处理规则转成可评估 logits。

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
    processed_non_free[:, occupied_mask] = torch.special.logit(eroded[:, occupied_mask], eps=1e-6).to(
        dtype=processed.dtype
    )
    processed[class_mask] = processed_non_free
    return processed


def logits_to_pred(
    logits: torch.Tensor,
    pred_postprocess: str,
    free_index: int,
    other_fill_value: float,
    free_fill_value: float,
    postprocess_score_thr: float,
    postprocess_kernel_size: int,
) -> np.ndarray:
    """将 dense logits 转成最终语义预测。"""
    dense = logits
    if pred_postprocess == "opusv1":
        dense = postprocess_fast_logits_opusv1(
            logits=dense,
            free_index=free_index,
            score_thr=postprocess_score_thr,
            other_fill_value=other_fill_value,
            free_fill_value=free_fill_value,
            kernel_size=postprocess_kernel_size,
        )
    return dense.argmax(dim=0).cpu().numpy()


def build_step_map_from_sidecar(entry: dict[str, Any]) -> dict[int, str]:
    """当 frame_tokens 缺失时，使用 sidecar 里的监督映射回退。"""
    out: dict[int, str] = {}
    sup_mask = entry.get("supervision_mask", [])
    sup_steps = entry.get("supervision_step_indices", [])
    sup_tokens = entry.get("supervision_gt_tokens", [])
    if not isinstance(sup_mask, list) or not isinstance(sup_steps, list) or not isinstance(sup_tokens, list):
        return out
    n = min(len(sup_mask), len(sup_steps), len(sup_tokens))
    for i in range(n):
        valid = int(sup_mask[i]) if i < len(sup_mask) else 0
        if valid <= 0:
            continue
        step_idx = int(sup_steps[i])
        gt_token = str(sup_tokens[i])
        if step_idx >= 0 and gt_token:
            out[step_idx] = gt_token
    return out


def hist_info(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    """构建混淆矩阵（与 MetricMiouOcc3D.hist_info 逻辑一致）。"""
    assert pred.shape == gt.shape
    k = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[k].astype(int) + pred[k].astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def compute_hist_for_pair(
    preds: np.ndarray,
    gt_semantics: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """按 image mask 过滤后统计单对(pred, gt)的混淆矩阵。"""
    mask = gt_mask.astype(bool)
    pred_use = preds[mask]
    gt_use = gt_semantics[mask]
    return hist_info(pred_use.flatten(), gt_use.flatten(), num_classes=num_classes)


def process_one_info(
    info: dict[str, Any],
    *,
    root_path: str,
    pred_root: str,
    pred_path_key: str,
    scene_token_layout: bool,
    pred_format: str,
    pred_postprocess: str,
    sidecar_index: dict[str, dict[str, Any]],
    keyframe_resolver: NuScenesKeyFrameResolver,
    gt_root: str,
    gt_mask_key: str,
    grid_size: tuple[int, int, int],
    num_classes: int,
    free_index: int,
    other_fill_value: float,
    free_fill_value: float,
    postprocess_score_thr: float,
    postprocess_kernel_size: int,
    deduplicate_gt: bool,
) -> dict[str, Any]:
    """线程 worker：处理单个 info，返回局部统计和元信息。"""
    ret: dict[str, Any] = {
        "all_hist": np.zeros((num_classes, num_classes), dtype=np.float64),
        "all_count": 0,
        "step_hists": {},
        "step_counts": {},
        "pairs": [],
        "missing_pred_path": 0,
        "missing_scene_or_token": 0,
        "missing_frame_tokens": 0,
        "missing_gt": 0,
        "step_out_of_range": 0,
        "no_keyframe_steps": 0,
        "corrupted_pred": 0,
    }

    scene_name = str(info.get("scene_name", ""))
    token = str(info.get("token", ""))
    if not scene_name or not token:
        ret["missing_scene_or_token"] = 1
        return ret

    pred_path = resolve_pred_logits_path(
        info=info,
        root_path=root_path,
        pred_root=pred_root,
        pred_path_key=pred_path_key,
        scene_token_layout=bool(scene_token_layout),
        pred_format=pred_format,
    )
    if not pred_path:
        ret["missing_pred_path"] = 1
        return ret

    frame_tokens = info.get("frame_tokens", [])
    frame_tokens_list = [str(tok) for tok in frame_tokens] if isinstance(frame_tokens, list) else []
    if frame_tokens_list:
        keyframe_steps = keyframe_resolver.resolve_keyframe_steps(frame_tokens_list)
    else:
        sidecar_entry = sidecar_index.get(token, None)
        if sidecar_entry is None:
            ret["missing_frame_tokens"] = 1
            return ret
        keyframe_steps = build_step_map_from_sidecar(sidecar_entry)
    if not keyframe_steps:
        ret["no_keyframe_steps"] = 1
        return ret

    try:
        with np.load(pred_path, allow_pickle=False) as pred_npz:
            if pred_format == "full":
                required = ["sparse_coords", "sparse_values", "frame_splits"]
            else:
                required = ["sparse_coords", "sparse_topk_values", "sparse_topk_indices", "frame_splits"]
            if any(k not in pred_npz.files for k in required):
                ret["missing_pred_path"] = 1
                return ret
            sparse_coords = pred_npz["sparse_coords"]
            frame_splits = pred_npz["frame_splits"]
            sparse_values = pred_npz["sparse_values"] if pred_format == "full" else None
            sparse_topk_values = pred_npz["sparse_topk_values"] if pred_format == "topk" else None
            sparse_topk_indices = pred_npz["sparse_topk_indices"] if pred_format == "topk" else None
    except Exception:
        ret["corrupted_pred"] = 1
        return ret

    pred_num_frames = int(frame_splits.shape[0] - 1)
    for step_idx in sorted(keyframe_steps.keys()):
        gt_token = str(keyframe_steps[step_idx])
        if not gt_token:
            continue
        if step_idx < 0 or step_idx >= pred_num_frames:
            ret["step_out_of_range"] += 1
            continue

        gt_path = os.path.join(gt_root, scene_name, gt_token, "labels.npz")
        if not os.path.exists(gt_path):
            ret["missing_gt"] += 1
            continue
        gt_npz = load_labels_npz(gt_path)
        gt_semantics = gt_npz["semantics"]
        gt_mask = gt_npz.get(gt_mask_key, np.ones(gt_semantics.shape, dtype=np.float32))

        if pred_format == "full":
            dense_logits = decode_one_frame_sparse_full(
                sparse_coords=sparse_coords,
                sparse_values=sparse_values,
                frame_splits=frame_splits,
                frame_index=int(step_idx),
                grid_size=grid_size,
                num_classes=num_classes,
                free_index=free_index,
                other_fill_value=other_fill_value,
                free_fill_value=free_fill_value,
            )
        else:
            dense_logits = decode_one_frame_sparse_topk(
                sparse_coords=sparse_coords,
                sparse_topk_values=sparse_topk_values,
                sparse_topk_indices=sparse_topk_indices,
                frame_splits=frame_splits,
                frame_index=int(step_idx),
                grid_size=grid_size,
                num_classes=num_classes,
                free_index=free_index,
                other_fill_value=other_fill_value,
                free_fill_value=free_fill_value,
            )
        preds = logits_to_pred(
            logits=dense_logits,
            pred_postprocess=pred_postprocess,
            free_index=free_index,
            other_fill_value=other_fill_value,
            free_fill_value=free_fill_value,
            postprocess_score_thr=postprocess_score_thr,
            postprocess_kernel_size=postprocess_kernel_size,
        )
        pair_hist = compute_hist_for_pair(
            preds=preds,
            gt_semantics=gt_semantics,
            gt_mask=gt_mask,
            num_classes=num_classes,
        )
        ret["all_hist"] += pair_hist
        ret["all_count"] += 1
        ret["pairs"].append(
            {
                "scene_name": scene_name,
                "gt_token": gt_token,
                "step_idx": int(step_idx),
                "hist": pair_hist,
            }
        )

        step_key = int(step_idx)
        if step_key not in ret["step_hists"]:
            ret["step_hists"][step_key] = np.zeros((num_classes, num_classes), dtype=np.float64)
            ret["step_counts"][step_key] = 0
        ret["step_hists"][step_key] += pair_hist
        ret["step_counts"][step_key] += 1

    return ret


def iter_threaded_results(
    infos: list[dict[str, Any]],
    worker_fn,
    io_workers: int,
    prefetch: int,
):
    """并行执行 worker_fn(info)，按完成顺序产出结果。"""
    if io_workers <= 1:
        for info in infos:
            yield worker_fn(info)
        return

    total = len(infos)
    max_inflight = max(io_workers, prefetch)
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        futures: set[Future] = set()
        next_idx = 0

        def _submit_one(idx: int) -> None:
            futures.add(pool.submit(worker_fn, infos[idx]))

        while next_idx < total and len(futures) < max_inflight:
            _submit_one(next_idx)
            next_idx += 1

        while futures:
            done_set, futures = wait(futures, return_when=FIRST_COMPLETED)
            for done in done_set:
                yield done.result()
            if next_idx < total:
                _submit_one(next_idx)
                next_idx += 1


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    data_cfg = cfg["data"]
    root_path = cfg["root_path"]

    num_classes = int(data_cfg["num_classes"])
    free_index = int(data_cfg["free_index"])
    grid_size = tuple(data_cfg["grid_size"])
    gt_root = resolve_path(root_path, data_cfg["gt_root"])
    gt_mask_key = str(data_cfg.get("gt_mask_key", "mask_camera"))
    other_fill_value = float(data_cfg.get("topk_other_fill_value", -5.0))
    free_fill_value = float(data_cfg.get("topk_free_fill_value", 5.0))

    infos = load_saved_list(resolve_path(root_path, args.saved_list))
    infos = [x for x in infos if bool(x.get("valid", True))]
    if args.limit > 0:
        infos = infos[: int(args.limit)]

    sidecar_index: dict[str, dict[str, Any]] = {}
    if args.sidecar_path:
        sidecar_index = load_sidecar_index(resolve_path(root_path, args.sidecar_path))

    nusc_dataroot = resolve_path(root_path, args.nusc_dataroot)
    sweep_info_path = resolve_path(root_path, args.sweep_info_path)
    keyframe_resolver = NuScenesKeyFrameResolver(
        dataroot=nusc_dataroot,
        version=args.nusc_version,
        sweep_info_path=sweep_info_path,
    )

    metric_all = MetricMiouOcc3D(
        num_classes=num_classes,
        use_image_mask=True,
        use_lidar_mask=False,
    )
    per_step_metrics: dict[int, MetricMiouOcc3D] = {}

    missing_pred_path = 0
    missing_scene_or_token = 0
    missing_frame_tokens = 0
    missing_gt = 0
    step_out_of_range = 0
    no_keyframe_steps = 0
    used_keyframe_pairs = 0
    dedup_skipped = 0
    seen_gt: set[tuple[str, str]] = set()
    corrupted_pred = 0

    io_workers = max(1, int(args.io_workers))
    prefetch = max(1, int(args.prefetch))
    worker_fn = lambda info: process_one_info(
        info,
        root_path=root_path,
        pred_root=args.pred_root,
        pred_path_key=args.pred_path_key,
        scene_token_layout=bool(args.scene_token_layout),
        pred_format=str(args.pred_format),
        pred_postprocess=str(args.pred_postprocess),
        sidecar_index=sidecar_index,
        keyframe_resolver=keyframe_resolver,
        gt_root=gt_root,
        gt_mask_key=gt_mask_key,
        grid_size=grid_size,
        num_classes=num_classes,
        free_index=free_index,
        other_fill_value=other_fill_value,
        free_fill_value=free_fill_value,
        postprocess_score_thr=float(args.postprocess_score_thr),
        postprocess_kernel_size=int(args.postprocess_kernel_size),
        deduplicate_gt=bool(args.deduplicate_gt),
    )
    result_iter = iter_threaded_results(
        infos=infos,
        worker_fn=worker_fn,
        io_workers=io_workers,
        prefetch=prefetch,
    )
    iterator = tqdm(result_iter, total=len(infos), desc="[eval saved keyframes]") if tqdm is not None else result_iter
    for result in iterator:
        missing_pred_path += int(result.get("missing_pred_path", 0))
        missing_scene_or_token += int(result.get("missing_scene_or_token", 0))
        missing_frame_tokens += int(result.get("missing_frame_tokens", 0))
        missing_gt += int(result.get("missing_gt", 0))
        step_out_of_range += int(result.get("step_out_of_range", 0))
        no_keyframe_steps += int(result.get("no_keyframe_steps", 0))
        corrupted_pred += int(result.get("corrupted_pred", 0))

        pairs = result.get("pairs", [])
        if not isinstance(pairs, list):
            continue
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            scene_name = str(pair.get("scene_name", ""))
            gt_token = str(pair.get("gt_token", ""))
            step_idx = int(pair.get("step_idx", -1))
            step_hist = pair.get("hist", None)
            if step_hist is None or step_idx < 0:
                continue

            if args.deduplicate_gt:
                dedup_key = (scene_name, gt_token)
                if dedup_key in seen_gt:
                    dedup_skipped += 1
                    continue
                seen_gt.add(dedup_key)

            step_hist_np = np.asarray(step_hist, dtype=np.float64)
            metric_all.hist += step_hist_np
            metric_all.cnt += 1
            used_keyframe_pairs += 1
            metric_step = per_step_metrics.setdefault(
                int(step_idx),
                MetricMiouOcc3D(num_classes=num_classes, use_image_mask=True, use_lidar_mask=False),
            )
            metric_step.hist += step_hist_np
            metric_step.cnt += 1

    class_names = metric_all.class_names
    per_step_results: dict[str, Any] = {}
    for step_idx in sorted(per_step_metrics.keys()):
        m = per_step_metrics[step_idx]
        step_miou = float(m.count_miou(verbose=False)) if m.cnt > 0 else float("nan")
        step_per_class = np.nan_to_num(m.get_per_class_iou(), nan=0.0).tolist() if m.cnt > 0 else []
        print(f"[keyframe][step={step_idx}] num={m.cnt} miou={step_miou:.2f}")
        for name, value in zip(class_names, step_per_class):
            print(f"  {name}: {float(value):.2f}")
        per_step_results[str(step_idx)] = {
            "num_keyframes": int(m.cnt),
            "miou": None if not np.isfinite(step_miou) else float(step_miou),
            "per_class_iou": [float(v) for v in step_per_class],
            "class_names": class_names,
        }

    if metric_all.cnt > 0:
        all_miou = float(metric_all.count_miou(verbose=False))
        all_per_class = np.nan_to_num(metric_all.get_per_class_iou(), nan=0.0).tolist()
        print(f"[keyframe][all] num={metric_all.cnt} miou={all_miou:.2f}")
        for name, value in zip(class_names, all_per_class):
            print(f"  {name}: {float(value):.2f}")
    else:
        all_miou = float("nan")
        all_per_class = []
        print("[keyframe][all] no samples")

    print(
        "[meta] "
        f"infos={len(infos)} "
        f"used_keyframes={used_keyframe_pairs} "
        f"missing_pred_path={missing_pred_path} "
        f"missing_scene_or_token={missing_scene_or_token} "
        f"missing_frame_tokens={missing_frame_tokens} "
        f"no_keyframe_steps={no_keyframe_steps} "
        f"step_out_of_range={step_out_of_range} "
        f"missing_gt={missing_gt} "
        f"dedup_skipped={dedup_skipped} "
        f"corrupted_pred={corrupted_pred} "
        f"pred_format={args.pred_format} "
        f"pred_postprocess={args.pred_postprocess} "
        f"postprocess_score_thr={float(args.postprocess_score_thr):.3f} "
        f"postprocess_kernel_size={int(args.postprocess_kernel_size)} "
        f"io_workers={io_workers} "
        f"prefetch={prefetch}"
    )

    if args.output_json:
        payload = {
            "keyframe_all": {
                "num_keyframes": int(metric_all.cnt),
                "miou": None if not np.isfinite(all_miou) else float(all_miou),
                "per_class_iou": [float(v) for v in all_per_class],
                "class_names": class_names,
            },
            "keyframe_per_step": per_step_results,
            "meta": {
                "num_infos": int(len(infos)),
                "used_keyframes": int(used_keyframe_pairs),
                "missing_pred_path": int(missing_pred_path),
                "missing_scene_or_token": int(missing_scene_or_token),
                "missing_frame_tokens": int(missing_frame_tokens),
                "no_keyframe_steps": int(no_keyframe_steps),
                "step_out_of_range": int(step_out_of_range),
                "missing_gt": int(missing_gt),
                "dedup_skipped": int(dedup_skipped),
                "corrupted_pred": int(corrupted_pred),
                "pred_format": str(args.pred_format),
                "pred_postprocess": str(args.pred_postprocess),
                "postprocess_score_thr": float(args.postprocess_score_thr),
                "postprocess_kernel_size": int(args.postprocess_kernel_size),
                "io_workers": int(io_workers),
                "prefetch": int(prefetch),
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
