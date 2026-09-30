from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import torch

from evoocc.config import resolve_path
from evoocc.data.logits_io import (
    decode_single_frame_sparse_topk,
    sparse_full_to_topk,
)


def _resolve_relative(root_path: str, logits_root: str, rel_path: str) -> str:
    if not rel_path:
        raise ValueError("rel_path 为空，无法定位 logits 文件")
    return resolve_path(root_path, os.path.join(logits_root, rel_path))


class LogitsLoader(ABC):
    @abstractmethod
    def load_fast_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Returns (T,C,X,Y,Z)."""
        ...

    @abstractmethod
    def load_slow_logits(
        self,
        info: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Returns (C,X,Y,Z)."""
        ...


class AloccDenseTopkLoader(LogitsLoader):
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

        X, Y, Z = self.grid_size
        K = self.topk_k
        # Flattened (X*Y*Z*K,) coordinate indices for scatter, shared across frames.
        self._x_idx = torch.arange(X).view(X, 1, 1, 1).expand(X, Y, Z, K).reshape(-1)
        self._y_idx = torch.arange(Y).view(1, Y, 1, 1).expand(X, Y, Z, K).reshape(-1)
        self._z_idx = torch.arange(Z).view(1, 1, Z, 1).expand(X, Y, Z, K).reshape(-1)

    def _decode_dense_topk_frame(
        self,
        path: str,
        device: torch.device,
    ) -> torch.Tensor:
        with np.load(path, allow_pickle=False) as data:
            topk_values = torch.from_numpy(data["topk_values"].astype(np.float32))
            topk_indices = torch.from_numpy(data["topk_indices"].astype(np.int64))

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

        if self.max_centering:
            # Max-centering: subtract the per-voxel max so the top-1 logit becomes 0.
            max_vals = topk_values.max(dim=-1, keepdim=True).values
            centered = topk_values - max_vals
        else:
            centered = topk_values
        centered = centered.clamp_min(self.clamp_min)

        X, Y, Z = self.grid_size
        dense = torch.full(
            (self.num_classes, X, Y, Z),
            fill_value=self.fill_value,
            dtype=torch.float32,
            device=device,
        )

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
        return resolve_path(self.root_path, os.path.join(logits_root, rel_path))

    def _make_sample_token_rel_path(self, info: Dict[str, Any], token: str) -> str:
        if not token:
            return ""
        scene_name = str(info.get("scene_name", ""))
        if not scene_name:
            raise KeyError("path_token_type=sample_token 需要 info['scene_name']")
        return os.path.join(scene_name, str(token), "logits.npz")

    def _iter_fast_rel_paths(self, info: Dict[str, Any]) -> list[str]:
        if self.path_token_type == "sample_token":
            if "frame_sample_tokens" not in info:
                raise KeyError("path_token_type=sample_token 需要 info['frame_sample_tokens']")
            return [
                self._make_sample_token_rel_path(info, str(token))
                for token in info["frame_sample_tokens"]
            ]
        return list(info["frame_rel_paths"])

    def _slow_rel_path(self, info: Dict[str, Any]) -> str:
        if self.path_token_type == "sample_token":
            token = str(info.get("slow_sample_token", "") or info.get("token", ""))
            return self._make_sample_token_rel_path(info, token)
        return str(info["slow_logit_path"])

    def _empty_frame(self, device: torch.device) -> torch.Tensor:
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
        frame_rel_paths = self._iter_fast_rel_paths(info)
        frames = []
        for rel_path in frame_rel_paths:
            if not rel_path:
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
        rel_path = self._slow_rel_path(info)
        full_path = self._resolve(self.slow_logit_root, rel_path)
        return self._decode_dense_topk_frame(full_path, device)


class OpusSparseFullLoader(LogitsLoader):
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
        with np.load(path, allow_pickle=False) as data:
            sparse_coords = data["sparse_coords"]
            sparse_values = data["sparse_values"]

        topk_values, topk_indices = sparse_full_to_topk(
            sparse_values,
            num_classes=self.num_classes,
            free_index=self.free_index,
            k=self.topk_k,
        )

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
        if not rel_path:
            raise ValueError("rel_path 为空，无法定位 logits 文件")
        return resolve_path(self.root_path, os.path.join(logits_root, rel_path))

    def _empty_frame(self, device: torch.device) -> torch.Tensor:
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
        frame_rel_paths = info["frame_rel_paths"]
        frames = []
        for rel_path in frame_rel_paths:
            if not rel_path:
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
        rel_path = info["slow_logit_path"]
        full_path = self._resolve(self.slow_logit_root, rel_path)
        return self._decode_frame(full_path, device)


class CompositeLogitsLoader(LogitsLoader):
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
