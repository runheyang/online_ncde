"""Logits 加载策略接口与具体实现。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import torch

from online_ncde.config import resolve_path
from online_ncde.data.logits_io import (
    decode_single_frame_sparse_topk,
    sparse_full_to_topk,
)


def _resolve_relative(root_path: str, logits_root: str, rel_path: str) -> str:
    """统一拼接 logits 文件路径。"""
    if not rel_path:
        raise ValueError("rel_path 为空，无法定位 logits 文件")
    return resolve_path(root_path, os.path.join(logits_root, rel_path))


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
        label_id_offset: int = 0,
        path_token_type: str = "rel_path",
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
        self.label_id_offset = int(label_id_offset)
        self.path_token_type = str(path_token_type).strip().lower()
        if self.path_token_type not in {"rel_path", "sample_token"}:
            raise ValueError(
                "alocc path_token_type 仅支持 {'rel_path', 'sample_token'}，"
                f"当前为 {path_token_type!r}"
            )

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

        if int(topk_values.shape[-1]) != self.topk_k or int(topk_indices.shape[-1]) != self.topk_k:
            raise ValueError(
                f"{path} 的 top-k 维度与配置不一致："
                f"values={tuple(topk_values.shape)}, indices={tuple(topk_indices.shape)}, "
                f"配置 alocc_topk_k={self.topk_k}"
            )
        if self.label_id_offset:
            topk_indices = topk_indices + int(self.label_id_offset)
        min_idx = int(topk_indices.min().item())
        max_idx = int(topk_indices.max().item())
        if min_idx < 0 or max_idx >= self.num_classes:
            raise ValueError(
                f"{path} 的 topk_indices 经 offset={self.label_id_offset} 后越界："
                f"min={min_idx}, max={max_idx}, num_classes={self.num_classes}"
            )

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
        c_idx = topk_indices.reshape(-1).to(device=device, dtype=torch.long)
        v_flat = centered.reshape(-1).to(device=device)
        dense[
            c_idx,
            self._x_idx.to(device=device),
            self._y_idx.to(device=device),
            self._z_idx.to(device=device),
        ] = v_flat

        return dense

    def _resolve(self, logits_root: str, rel_path: str) -> str:
        """拼接完整路径。"""
        return resolve_path(self.root_path, os.path.join(logits_root, rel_path))

    def _make_sample_token_rel_path(self, info: Dict[str, Any], token: str) -> str:
        """从 scene/sample_token 组装逐帧 logits 相对路径。"""
        if not token:
            return ""
        scene_name = str(info.get("scene_name", ""))
        if not scene_name:
            raise KeyError("path_token_type=sample_token 需要 info['scene_name']")
        return os.path.join(scene_name, str(token), "logits.npz")

    def _iter_fast_rel_paths(self, info: Dict[str, Any]) -> list[str]:
        """按配置返回 fast logits 的逐帧相对路径。"""
        if self.path_token_type == "sample_token":
            if "frame_sample_tokens" not in info:
                raise KeyError("path_token_type=sample_token 需要 info['frame_sample_tokens']")
            return [
                self._make_sample_token_rel_path(info, str(token))
                for token in info["frame_sample_tokens"]
            ]
        return list(info["frame_rel_paths"])

    def _slow_rel_path(self, info: Dict[str, Any]) -> str:
        """按配置返回 slow logits 相对路径。"""
        if self.path_token_type == "sample_token":
            token = str(info.get("slow_sample_token", "") or info.get("token", ""))
            return self._make_sample_token_rel_path(info, token)
        return str(info["slow_logit_path"])

    def _empty_frame(self, device: torch.device) -> torch.Tensor:
        """pad 帧占位：全 fill_value 的 (C, X, Y, Z) 张量。"""
        X, Y, Z = self.grid_size
        return torch.full(
            (self.num_classes, X, Y, Z),
            fill_value=self.fill_value,
            dtype=torch.float32,
            device=device,
        )

    def load_fast_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """加载 13 帧快系统 logits，返回 (T, C, X, Y, Z)。"""
        frame_rel_paths = self._iter_fast_rel_paths(info)
        frames = []
        for rel_path in frame_rel_paths:
            if not rel_path:
                # 短历史 pad 帧：空路径直接返回占位张量，不走 IO
                frames.append(self._empty_frame(device))
                continue
            full_path = self._resolve(self.fast_logits_root, rel_path)
            frames.append(self._decode_dense_topk_frame(full_path, device))
        return torch.stack(frames, dim=0)

    def load_slow_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """加载慢系统单帧 logits，返回 (C, X, Y, Z)。"""
        rel_path = self._slow_rel_path(info)
        full_path = self._resolve(self.slow_logit_root, rel_path)
        return self._decode_dense_topk_frame(full_path, device)


class OpusSparseFullLoader(LogitsLoader):
    """OPUS 逐帧 sparse full logits 加载器。

    每帧 npz 包含：
      - sparse_coords: (N, 3) uint8 —— 体素坐标
      - sparse_values: (N, 17) float16 —— 17 维语义 logits（不含 free）

    处理流程（逐帧）：
      1. 读取 sparse_coords / sparse_values
      2. sparse_full_to_topk：保留 top-k 大 logits
      3. decode_single_frame_sparse_topk：散射到 (C, X, Y, Z) dense tensor
    """

    def __init__(
        self,
        root_path: str,
        fast_logits_root: str,
        slow_logit_root: str,
        num_classes: int,
        free_index: int,
        grid_size: Tuple[int, int, int],
        topk_k: int = 3,
        other_fill_value: float = -5.0,
        free_fill_value: float = 5.0,
    ) -> None:
        self.root_path = root_path
        self.fast_logits_root = fast_logits_root
        self.slow_logit_root = slow_logit_root
        self.num_classes = int(num_classes)
        self.free_index = int(free_index)
        self.grid_size = tuple(grid_size)
        self.topk_k = int(topk_k)
        self.other_fill_value = float(other_fill_value)
        self.free_fill_value = float(free_fill_value)

    def _decode_frame(self, path: str, device: torch.device) -> torch.Tensor:
        """单帧 sparse full → top-k → dense (C, X, Y, Z)。"""
        with np.load(path, allow_pickle=False) as data:
            sparse_coords = data["sparse_coords"]    # (N, 3) uint8
            sparse_values = data["sparse_values"]     # (N, 17) float16

        # full → top-k（sparse 阶段，仅操作 N 个命中体素）
        topk_values, topk_indices = sparse_full_to_topk(
            sparse_values,
            num_classes=self.num_classes,
            free_index=self.free_index,
            k=self.topk_k,
        )

        # top-k sparse → dense (C, X, Y, Z)
        return decode_single_frame_sparse_topk(
            sparse_coords=sparse_coords,
            sparse_topk_values=topk_values,
            sparse_topk_indices=topk_indices,
            grid_size=self.grid_size,
            num_classes=self.num_classes,
            free_index=self.free_index,
            other_fill_value=self.other_fill_value,
            free_fill_value=self.free_fill_value,
            device=device,
        )

    def _resolve(self, logits_root: str, rel_path: str) -> str:
        """拼接完整路径。"""
        if not rel_path:
            raise ValueError("rel_path 为空，无法定位 logits 文件")
        return resolve_path(self.root_path, os.path.join(logits_root, rel_path))

    def _empty_frame(self, device: torch.device) -> torch.Tensor:
        """pad 帧占位：free 通道为 free_fill_value，其它为 other_fill_value。"""
        X, Y, Z = self.grid_size
        frame = torch.full(
            (self.num_classes, X, Y, Z),
            fill_value=self.other_fill_value,
            dtype=torch.float32,
            device=device,
        )
        frame[self.free_index] = self.free_fill_value
        return frame

    def load_fast_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """加载 13 帧快系统 logits，返回 (T, C, X, Y, Z)。"""
        frame_rel_paths = info["frame_rel_paths"]
        frames = []
        for rel_path in frame_rel_paths:
            if not rel_path:
                # 短历史 pad 帧：空路径直接返回占位张量，不走 IO
                frames.append(self._empty_frame(device))
                continue
            full_path = self._resolve(self.fast_logits_root, rel_path)
            frames.append(self._decode_frame(full_path, device))
        return torch.stack(frames, dim=0)

    def load_slow_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """加载慢系统单帧 logits，返回 (C, X, Y, Z)。"""
        rel_path = info["slow_logit_path"]
        full_path = self._resolve(self.slow_logit_root, rel_path)
        return self._decode_frame(full_path, device)


class CompositeLogitsLoader(LogitsLoader):
    """组合 loader：fast/slow 使用不同格式的子 loader。

    例如 fast=opus_sparse_full + slow=alocc_dense_topk。
    """

    def __init__(self, fast_loader: LogitsLoader, slow_loader: LogitsLoader) -> None:
        self.fast_loader = fast_loader
        self.slow_loader = slow_loader

    def load_fast_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        return self.fast_loader.load_fast_logits(info, device)

    def load_slow_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        return self.slow_loader.load_slow_logits(info, device)
