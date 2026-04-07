"""Logits 加载策略接口与 ALOCC dense top-k 实现。

本模块与 logits_io.py（OPUS 稀疏格式）完全解耦，
删除旧代码时不需要修改本文件。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import torch

from online_ncde.config import resolve_path


class LogitsLoader(ABC):
    """logits 加载策略接口。

    子类负责：从 canonical info dict 中提取路径、读取文件、
    返回与 Occ3DOnlineNcdeDataset 约定形状一致的 tensor。
    """

    @abstractmethod
    def load_fast_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """加载快系统 logits，返回 (T, C, X, Y, Z)。"""
        ...

    @abstractmethod
    def load_slow_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """加载慢系统 logits，返回 (C, X, Y, Z)。"""
        ...


class AloccDenseTopkLoader(LogitsLoader):
    """ALOCC 格式 dense top-k logits 加载器。

    每帧 npz 包含：
      - topk_values:  (X, Y, Z, K)  float16  —— top-K logit 值
      - topk_indices: (X, Y, Z, K)  uint8    —— top-K 类别 id

    处理流程（逐帧）：
      1. max-centering：每体素减去 K 个 logits 中的最大值，使最大位置变 0
      2. clamp_min(clamp_min)：截断过小值
      3. scatter 到 (num_classes, X, Y, Z)，未命中位置填 fill_value
    """

    def __init__(
        self,
        root_path: str,
        fast_logits_root: str,
        slow_logit_root: str,
        num_classes: int,
        grid_size: Tuple[int, int, int],
        fill_value: float = -12.0,
        clamp_min: float = -12.0,
        topk_k: int = 3,
        max_centering: bool = True,
    ) -> None:
        self.root_path = root_path
        self.fast_logits_root = fast_logits_root
        self.slow_logit_root = slow_logit_root
        self.num_classes = int(num_classes)
        self.grid_size = tuple(grid_size)
        self.fill_value = float(fill_value)
        self.clamp_min = float(clamp_min)
        self.topk_k = int(topk_k)
        self.max_centering = bool(max_centering)

        # 预计算 scatter 用的坐标索引，所有帧复用，避免重复分配
        X, Y, Z = self.grid_size
        K = self.topk_k
        # (X*Y*Z*K,) 的展平坐标
        self._x_idx = torch.arange(X).view(X, 1, 1, 1).expand(X, Y, Z, K).reshape(-1)
        self._y_idx = torch.arange(Y).view(1, Y, 1, 1).expand(X, Y, Z, K).reshape(-1)
        self._z_idx = torch.arange(Z).view(1, 1, Z, 1).expand(X, Y, Z, K).reshape(-1)

    def _decode_dense_topk_frame(
        self,
        path: str,
        device: torch.device,
    ) -> torch.Tensor:
        """单帧 dense top-k → (C, X, Y, Z)。

        流程：
          1. 读取 topk_values / topk_indices
          2. max-centering（每体素减最大 logit）
          3. clamp_min
          4. scatter 到全类别维度
        """
        with np.load(path, allow_pickle=False) as data:
            topk_values = torch.from_numpy(data["topk_values"].astype(np.float32))   # (X, Y, Z, K)
            topk_indices = torch.from_numpy(data["topk_indices"].astype(np.int64))    # (X, Y, Z, K)

        # max-centering：每体素的 top-K logits 减去其中最大值（可选）
        if self.max_centering:
            max_vals = topk_values.max(dim=-1, keepdim=True).values  # (X, Y, Z, 1)
            centered = topk_values - max_vals
        else:
            centered = topk_values
        # 截断：clamp_min 避免极端负值
        centered = centered.clamp_min(self.clamp_min)

        # 构造 dense tensor，默认填充 fill_value（非 top-K 类别的背景值）
        X, Y, Z = self.grid_size
        dense = torch.full(
            (self.num_classes, X, Y, Z),
            fill_value=self.fill_value,
            dtype=torch.float32,
            device=device,
        )

        # scatter：将 centered top-K 值写入对应类别位置
        c_idx = topk_indices.reshape(-1)
        v_flat = centered.reshape(-1)
        dense[c_idx, self._x_idx, self._y_idx, self._z_idx] = v_flat

        return dense

    def _resolve(self, logits_root: str, rel_path: str) -> str:
        """拼接完整路径。"""
        full = resolve_path(self.root_path, os.path.join(logits_root, rel_path))
        if not os.path.exists(full):
            raise FileNotFoundError(f"ALOCC logits 文件不存在: {full}")
        return full

    def load_fast_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """加载 13 帧快系统 logits，返回 (T, C, X, Y, Z)。"""
        frame_rel_paths = info["frame_rel_paths"]
        frames = []
        for rel_path in frame_rel_paths:
            full_path = self._resolve(self.fast_logits_root, rel_path)
            frames.append(self._decode_dense_topk_frame(full_path, device))
        return torch.stack(frames, dim=0)

    def load_slow_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """加载慢系统单帧 logits，返回 (C, X, Y, Z)。"""
        rel_path = info["slow_logit_path"]
        full_path = self._resolve(self.slow_logit_root, rel_path)
        return self._decode_dense_topk_frame(full_path, device)
