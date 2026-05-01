"""Slow logits 预加载缓存.

alocc3d_wo_mask 的 npz 是 zlib compressed, 每次读 ~26ms 解压 fp16 topk_values.
评估时往返同一组 keyframe 多次 (多个 slow_interval 共享), 在线推理时 reset 帧
也只是少数, 但 zlib 每次都要重新解压.

把 slow 全部预加载成 dense (C, X, Y, Z) fp16 放在 GPU 上, 用时 .float() 仅 ~0.1ms.

显存预算: 1 帧 dense fp16 = 18 * 200 * 200 * 16 * 2 / 2^20 = 21.97 MB.
5 scenes × ~40 keyframe ≈ 4.6 GB. 全 val (150 scenes ≈ 6000 kf) ≈ 132 GB (放不下).
所以这个 cache 仅适合中小规模评估; 大规模评估应在线解码或换 fp32 numpy on CPU.
"""
from __future__ import annotations
import os
from typing import Iterable, Dict

import numpy as np
import torch

from ._dense_decode import scatter_topk_to_dense


class SlowLogitsGPUCache:
    """path → dense (C, X, Y, Z) fp16 GPU tensor."""

    def __init__(
        self,
        device: torch.device,
        num_classes: int = 18,
        clamp_min: float = -5.0,
        fill_value: float = -5.0,
        max_centering: bool = False,
    ) -> None:
        self.device = device
        self.num_classes = int(num_classes)
        self.clamp_min = float(clamp_min)
        self.fill_value = float(fill_value)
        self.max_centering = bool(max_centering)
        self._cache: Dict[str, torch.Tensor] = {}

    def preload(self, paths: Iterable[str], skip_missing: bool = True, verbose: bool = True) -> None:
        """一次性加载并 decode 到 GPU dense fp16."""
        unique = sorted(set(paths))
        n = len(unique)
        miss = 0
        for i, p in enumerate(unique):
            if not os.path.exists(p):
                miss += 1
                if not skip_missing:
                    raise FileNotFoundError(p)
                continue
            with np.load(p, allow_pickle=False) as d:
                vals = torch.from_numpy(d["topk_values"]).to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                idx = torch.from_numpy(d["topk_indices"]).to(
                    device=self.device, dtype=torch.long, non_blocking=True
                )
            if self.max_centering:
                vals = vals - vals.max(dim=-1, keepdim=True).values
            vals = vals.clamp_min(self.clamp_min)
            dense = scatter_topk_to_dense(vals, idx, self.num_classes, self.fill_value)
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
