"""Slow logits 预加载缓存.

把单 scene 的 slow logits 全部预加载成 dense (C, X, Y, Z) fp16 放在 GPU 上,
用时 .float() 仅 ~0.1ms, 避开磁盘 IO + zlib 解压噪声.

通过注入 decoder_fn 同时支持:
  - alocc_dense_topk:  topk_values + topk_indices (alocc3d_wo_mask)
  - opus_sparse_full:  sparse_coords + sparse_values (logits_opusv2l_full)
"""
from __future__ import annotations
import os
from typing import Callable, Dict, Iterable

import torch

from ._dense_decode import (
    decode_opus_sparse_full_npz_to_dense_gpu,
    decode_topk_npz_to_dense_gpu,
)


class SlowLogitsGPUCache:
    """path → dense (C, X, Y, Z) fp16 GPU tensor.

    decoder_fn(path) 必须返回 (C, X, Y, Z) fp32 GPU tensor.
    """

    def __init__(
        self,
        device: torch.device,
        decoder_fn: Callable[[str], torch.Tensor],
    ) -> None:
        self.device = device
        self.decoder_fn = decoder_fn
        self._cache: Dict[str, torch.Tensor] = {}

    def preload(
        self, paths: Iterable[str], skip_missing: bool = True, verbose: bool = True
    ) -> None:
        unique = sorted(set(paths))
        n = len(unique)
        miss = 0
        for i, p in enumerate(unique):
            if not os.path.exists(p):
                miss += 1
                if not skip_missing:
                    raise FileNotFoundError(p)
                continue
            dense = self.decoder_fn(p)
            self._cache[p] = dense.to(torch.float16)
            if verbose and (i + 1) % 50 == 0:
                print(f"  preloaded {i+1}/{n} (miss so far: {miss})")
        gpu_mb = sum(t.numel() * t.element_size() for t in self._cache.values()) / 1024 / 1024
        if verbose:
            print(f"  preload done: {len(self._cache)} cached, {miss} missing, GPU mem {gpu_mb:.1f} MB")

    def get(self, path: str) -> torch.Tensor:
        """返回 (C, X, Y, Z) fp32, 已上 GPU."""
        return self._cache[path].float()

    def has(self, path: str) -> bool:
        return path in self._cache

    def __len__(self) -> int:
        return len(self._cache)


def build_slow_decoder_fn(
    data_cfg: dict, device: torch.device
) -> Callable[[str], torch.Tensor]:
    """根据 cfg.data.slow_logit_format 选择 slow logits 解码函数.

    返回的闭包: path -> (C, X, Y, Z) fp32 GPU tensor.

    支持:
      - alocc_dense_topk:  data/alocc3d_wo_mask 等
      - opus_sparse_full:  data/logits_opusv2l_full 等

    复合 cfg (logits_format=composite) 用 slow_logit_format 区分.
    """
    fmt = data_cfg.get("slow_logit_format")
    if fmt is None:
        # 兼容 logits_format=alocc_dense_topk 这种非复合写法
        fmt = data_cfg.get("logits_format", "alocc_dense_topk")
    fmt = str(fmt)
    num_classes = int(data_cfg["num_classes"])

    if fmt == "alocc_dense_topk":
        clamp_min = float(data_cfg.get("alocc_clamp_min", -5.0))
        fill_value = float(data_cfg.get("alocc_fill_value", -5.0))
        max_centering = bool(data_cfg.get("alocc_max_centering", False))

        def _decode(path: str) -> torch.Tensor:
            return decode_topk_npz_to_dense_gpu(
                path, device=device, num_classes=num_classes,
                clamp_min=clamp_min, fill_value=fill_value,
                max_centering=max_centering,
            )
        return _decode

    if fmt == "opus_sparse_full":
        free_index = int(data_cfg["free_index"])
        grid_size = tuple(data_cfg["grid_size"])
        topk_k = int(data_cfg.get("opus_full_topk_k", 3))
        other_fill = float(data_cfg.get("opus_other_fill_value", -5.0))
        free_fill = float(data_cfg.get("opus_free_fill_value", 5.0))

        def _decode(path: str) -> torch.Tensor:
            return decode_opus_sparse_full_npz_to_dense_gpu(
                path, device=device, num_classes=num_classes,
                free_index=free_index, grid_size=grid_size, topk_k=topk_k,
                other_fill_value=other_fill, free_fill_value=free_fill,
            )
        return _decode

    raise ValueError(f"slow_logit_format 不支持: {fmt!r}")
