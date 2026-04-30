"""快/慢系统 logits.npz 读写与稀疏解码。"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def load_logits_npz(path: str) -> Dict[str, np.ndarray]:
    """读取 logits.npz 并返回字段字典。"""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def decode_sparse_topk(
    sparse_coords: np.ndarray,
    sparse_topk_values: np.ndarray,
    sparse_topk_indices: np.ndarray,
    frame_splits: np.ndarray,
    grid_size: Tuple[int, int, int],
    num_classes: int,
    free_index: int,
    other_fill_value: float = -8.0,
    free_fill_value: float = 10.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    将 top-k 稀疏表示还原为 dense logits: (T, C, X, Y, Z)。

    契约:
    1. sparse_topk_values 读取后立刻转 FP32。
    2. 稀疏命中体素仅写入 top-k 类，其余类别保持 -8.0。
    3. 非稀疏体素默认 free 类为 free_fill_value，其余类别为 -8.0。
    """
    x_size, y_size, z_size = grid_size
    num_frames = int(frame_splits.shape[0] - 1)
    use_device = device or torch.device("cpu")

    # torch.from_numpy 不支持 uint16（OpenOccupancy 大网格的 sparse_coords dtype），
    # 在最外层统一升宽到 int32 后再切片，避免逐帧重复转换。
    if sparse_coords.dtype == np.uint16:
        sparse_coords = sparse_coords.astype(np.int32, copy=False)

    outputs: list[torch.Tensor] = []
    for t in range(num_frames):
        start = int(frame_splits[t])
        end = int(frame_splits[t + 1])

        dense = torch.full(
            (num_classes, x_size, y_size, z_size),
            fill_value=other_fill_value,
            dtype=dtype,
            device=use_device,
        )
        hit_mask = torch.zeros((x_size, y_size, z_size), dtype=torch.bool, device=use_device)
        if end > start:
            coords_t = torch.from_numpy(sparse_coords[start:end]).to(
                device=use_device, dtype=torch.long
            )
            values_t = torch.from_numpy(sparse_topk_values[start:end]).to(
                device=use_device, dtype=torch.float32
            )
            indices_t = torch.from_numpy(sparse_topk_indices[start:end]).to(
                device=use_device, dtype=torch.long
            )

            x_idx = coords_t[:, 0]
            y_idx = coords_t[:, 1]
            z_idx = coords_t[:, 2]
            hit_mask[x_idx, y_idx, z_idx] = True

            # clamp_min：确保 topk values 不低于 other_fill_value，
            # 避免第 K 大 logit 反而小于非 topk 位置的填充值
            values_t = values_t.clamp_min(other_fill_value)

            N_hit, K = values_t.shape
            x_rep = x_idx.repeat_interleave(K)
            y_rep = y_idx.repeat_interleave(K)
            z_rep = z_idx.repeat_interleave(K)
            c_rep = indices_t.reshape(-1)
            v_rep = values_t.reshape(-1).to(dtype)
            dense[c_rep, x_rep, y_rep, z_rep] = v_rep

        # 仅对未命中体素注入 free 先验；命中体素仍保持“仅写 top-k，其余 -8.0”。
        unhit_mask = ~hit_mask
        dense[free_index, unhit_mask] = torch.as_tensor(
            free_fill_value, device=use_device, dtype=dtype
        )

        outputs.append(dense)
    return torch.stack(outputs, dim=0)


def decode_sparse_full(
    sparse_coords: np.ndarray,
    sparse_values: np.ndarray,
    frame_splits: np.ndarray,
    grid_size: Tuple[int, int, int],
    num_classes: int,
    free_index: int,
    other_fill_value: float = -8.0,
    free_fill_value: float = 10.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    将 full sparse 表示还原为 dense logits: (T, C, X, Y, Z)。

    契约:
    1. sparse_values 只包含非 free 的语义 logits，读取后立刻转 FP32。
    2. 稀疏命中体素写入全部 17 维语义 logits，free 类保持 other_fill_value。
    3. 非稀疏体素默认 free 类为 free_fill_value，其余类别为 other_fill_value。
    """
    x_size, y_size, z_size = grid_size
    num_frames = int(frame_splits.shape[0] - 1)
    use_device = device or torch.device("cpu")
    semantic_indices = [c for c in range(num_classes) if c != int(free_index)]
    if sparse_values.ndim != 2 or sparse_values.shape[1] != len(semantic_indices):
        raise ValueError(
            "sparse_values 维度异常，"
            f"期望第二维为 {len(semantic_indices)}，实际为 {tuple(sparse_values.shape)}"
        )

    # uint16 → int32（torch.from_numpy 不支持 uint16）。
    if sparse_coords.dtype == np.uint16:
        sparse_coords = sparse_coords.astype(np.int32, copy=False)

    outputs: list[torch.Tensor] = []
    for t in range(num_frames):
        start = int(frame_splits[t])
        end = int(frame_splits[t + 1])

        dense = torch.full(
            (num_classes, x_size, y_size, z_size),
            fill_value=other_fill_value,
            dtype=dtype,
            device=use_device,
        )
        hit_mask = torch.zeros((x_size, y_size, z_size), dtype=torch.bool, device=use_device)
        if end > start:
            coords_t = torch.from_numpy(sparse_coords[start:end]).to(
                device=use_device, dtype=torch.long
            )
            values_t = torch.from_numpy(sparse_values[start:end]).to(
                device=use_device, dtype=torch.float32
            )

            x_idx = coords_t[:, 0]
            y_idx = coords_t[:, 1]
            z_idx = coords_t[:, 2]
            hit_mask[x_idx, y_idx, z_idx] = True

            semantic_tensor = torch.as_tensor(semantic_indices, device=use_device, dtype=torch.long)
            dense[
                semantic_tensor[:, None],
                x_idx[None, :],
                y_idx[None, :],
                z_idx[None, :],
            ] = values_t.transpose(0, 1).to(dtype)

        dense[free_index, ~hit_mask] = torch.as_tensor(
            free_fill_value, device=use_device, dtype=dtype
        )
        outputs.append(dense)
    return torch.stack(outputs, dim=0)


def sparse_full_to_topk(
    sparse_values: np.ndarray,
    num_classes: int,
    free_index: int,
    k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 sparse 阶段将 full logits 转为 top-k 格式。

    sparse_values: (N, num_classes-1)，不含 free 类的语义 logits。
    返回:
        topk_values: (N, k)  float16
        topk_indices: (N, k) uint8 —— 索引为原始 num_classes 空间中的类别 id
    """
    semantic_indices = torch.as_tensor(
        [c for c in range(num_classes) if c != int(free_index)],
        dtype=torch.long,
    )
    num_sem = sparse_values.shape[1]
    use_k = min(int(k), num_sem)
    if use_k <= 0:
        raise ValueError(f"k 必须为正整数，当前为 {k}")

    # 保持 sparse 中间结果为 fp16/uint8，避免在 top-k 阶段放大 worker 侧内存占用。
    sparse_tensor = torch.from_numpy(sparse_values)
    topk_vals, topk_local = torch.topk(sparse_tensor, k=use_k, dim=1, largest=True, sorted=True)
    topk_global = semantic_indices[topk_local]
    return (
        topk_vals.numpy(),
        topk_global.numpy().astype(np.uint8, copy=False),
    )


def decode_single_frame_sparse_topk(
    sparse_coords: np.ndarray,
    sparse_topk_values: np.ndarray,
    sparse_topk_indices: np.ndarray,
    grid_size: Tuple[int, int, int],
    num_classes: int,
    free_index: int,
    other_fill_value: float = -8.0,
    free_fill_value: float = 10.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    将单帧 top-k 稀疏表示还原为 dense logits: (C, X, Y, Z)。

    slow_logit.npz 只包含一帧，因此不需要 frame_splits。
    """
    frame_splits = np.array([0, sparse_coords.shape[0]])
    return decode_sparse_topk(
        sparse_coords=sparse_coords,
        sparse_topk_values=sparse_topk_values,
        sparse_topk_indices=sparse_topk_indices,
        frame_splits=frame_splits,
        grid_size=grid_size,
        num_classes=num_classes,
        free_index=free_index,
        other_fill_value=other_fill_value,
        free_fill_value=free_fill_value,
        device=device,
        dtype=dtype,
    )[0]


def decode_single_frame_sparse_full(
    sparse_coords: np.ndarray,
    sparse_values: np.ndarray,
    grid_size: Tuple[int, int, int],
    num_classes: int,
    free_index: int,
    other_fill_value: float = -8.0,
    free_fill_value: float = 10.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    将单帧 full sparse 表示还原为 dense logits: (C, X, Y, Z)。

    slow_logit_full.npz 只保存一帧，且 sparse_values 仅包含非 free 的语义 logits。
    """
    frame_splits = np.array([0, sparse_coords.shape[0]])
    return decode_sparse_full(
        sparse_coords=sparse_coords,
        sparse_values=sparse_values,
        frame_splits=frame_splits,
        grid_size=grid_size,
        num_classes=num_classes,
        free_index=free_index,
        other_fill_value=other_fill_value,
        free_fill_value=free_fill_value,
        device=device,
        dtype=dtype,
    )[0]
