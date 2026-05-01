"""Dense top-k → (C, X, Y, Z) scatter, GPU/CPU 通用."""
from __future__ import annotations
import torch


def scatter_topk_to_dense(
    topk_vals: torch.Tensor,
    topk_idx: torch.Tensor,
    num_classes: int,
    fill_value: float,
) -> torch.Tensor:
    """topk_vals (X,Y,Z,K) + topk_idx (X,Y,Z,K) → dense (C,X,Y,Z) on the same device.

    所有 tensor 自动跟 topk_vals 在同一 device 上 (无来回搬运).
    非 top-k 位置填 fill_value. 调用方负责 clamp_min / max_centering.
    """
    X, Y, Z, K = topk_vals.shape
    device = topk_vals.device
    dense = torch.full(
        (num_classes, X, Y, Z), fill_value=fill_value,
        dtype=topk_vals.dtype, device=device,
    )
    x_idx = torch.arange(X, device=device).view(X, 1, 1, 1).expand(-1, Y, Z, K)
    y_idx = torch.arange(Y, device=device).view(1, Y, 1, 1).expand(X, -1, Z, K)
    z_idx = torch.arange(Z, device=device).view(1, 1, Z, 1).expand(X, Y, -1, K)
    c_idx = topk_idx.long()
    dense[c_idx, x_idx, y_idx, z_idx] = topk_vals
    return dense


def decode_topk_npz_to_dense_gpu(
    npz_path: str,
    device: torch.device,
    num_classes: int,
    clamp_min: float,
    fill_value: float,
    max_centering: bool = False,
) -> torch.Tensor:
    """读取 alocc dense topk npz, 全程 GPU 解码 → (C, X, Y, Z) fp32.

    流程: CPU 读 npz fp16/uint8 → 直接 .to(device) → GPU 上 clamp_min + scatter.
    跟 logits_loader.AloccDenseTopkLoader._decode_dense_topk_frame 等价, 但 scatter 在 GPU.
    """
    import numpy as np
    with np.load(npz_path, allow_pickle=False) as data:
        topk_vals_cpu = torch.from_numpy(data["topk_values"])     # fp16 (X,Y,Z,K)
        topk_idx_cpu = torch.from_numpy(data["topk_indices"])     # uint8 (X,Y,Z,K)
    topk_vals = topk_vals_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    topk_idx = topk_idx_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    if max_centering:
        topk_vals = topk_vals - topk_vals.max(dim=-1, keepdim=True).values
    topk_vals = topk_vals.clamp_min(clamp_min)
    return scatter_topk_to_dense(topk_vals, topk_idx, num_classes, fill_value)
