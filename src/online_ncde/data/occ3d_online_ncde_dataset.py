"""Occ3D-nuScenes online_ncde 数据集。"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from online_ncde.config import resolve_path
from online_ncde.data.labels_io import load_labels_npz
from online_ncde.data.logits_io import (
    decode_single_frame_sparse_full,
    decode_single_frame_sparse_topk,
    decode_sparse_full,
    decode_sparse_topk,
    load_logits_npz,
    sparse_full_to_topk,
)
from online_ncde.data.logits_loader import LogitsLoader
from online_ncde.data.ray_sidecar import RaySidecar

# 进程级 flag：缺 token warning 每个 worker 进程只打一次
_RAY_MISSING_WARNED = False


class Occ3DOnlineNcdeDataset(Dataset):
    """读取 online_ncde 训练/评估样本。"""

    def __init__(
        self,
        info_path: str,
        root_path: str,
        fast_logits_root: str,
        slow_logit_root: str,
        gt_root: str,
        num_classes: int,
        free_index: int,
        grid_size: Tuple[int, int, int],
        gt_mask_key: str = "mask_camera",
        slow_noise_std: float = 0.0,
        topk_other_fill_value: float = -8.0,
        topk_free_fill_value: float = 10.0,
        supervision_sidecar_path: str | None = None,
        fast_logits_variant: str = "topk",
        slow_logit_variant: str = "topk",
        full_logits_clamp_min: float | None = None,
        full_topk_k: int = 3,
        logits_loader: LogitsLoader | None = None,
        ray_sidecar_dir: str | None = None,
        ray_sidecar_split: str | None = None,
    ) -> None:
        self.root_path = root_path
        self.info_path = resolve_path(root_path, info_path)
        self.fast_logits_root = fast_logits_root
        self.slow_logit_root = slow_logit_root
        self.gt_root = resolve_path(root_path, gt_root)
        self.num_classes = int(num_classes)
        self.free_index = int(free_index)
        self.grid_size = tuple(grid_size)
        self.gt_mask_key = gt_mask_key
        self.slow_noise_std = float(slow_noise_std)
        self.topk_other_fill_value = float(topk_other_fill_value)
        self.topk_free_fill_value = float(topk_free_fill_value)
        self.fast_logits_variant = self._normalize_logits_variant(
            fast_logits_variant,
            field_name="fast_logits_variant",
        )
        self.slow_logit_variant = self._normalize_logits_variant(
            slow_logit_variant,
            field_name="slow_logit_variant",
        )
        self.full_logits_clamp_min = (
            None if full_logits_clamp_min is None else float(full_logits_clamp_min)
        )
        self.full_topk_k = int(full_topk_k)
        self.logits_loader = logits_loader
        self.supervision_labels = ["t-1.5", "t-1.0", "t-0.5", "t"]
        self.supervision_by_token: Dict[str, Dict[str, Any]] | None = None
        # ray sidecar 延后到 supervision_labels 定版后再构造，方便做顺序校验
        self.ray_sidecar: RaySidecar | None = None
        self._ray_sidecar_dir = ray_sidecar_dir
        self._ray_sidecar_split = ray_sidecar_split

        with open(self.info_path, "rb") as f:
            payload = pickle.load(f)
        infos = payload["infos"] if isinstance(payload, dict) else payload
        self.infos: List[Dict[str, Any]] = [info for info in infos if info.get("valid", True)]

        if supervision_sidecar_path:
            sidecar_path = resolve_path(root_path, supervision_sidecar_path)
            if not os.path.exists(sidecar_path):
                raise FileNotFoundError(f"监督 sidecar 不存在: {sidecar_path}")
            with open(sidecar_path, "rb") as f:
                sidecar = pickle.load(f)
            metadata = sidecar.get("metadata", {}) if isinstance(sidecar, dict) else {}
            labels = metadata.get("supervision_labels", self.supervision_labels)
            if isinstance(labels, list) and labels:
                self.supervision_labels = [str(x) for x in labels]
            entries = sidecar.get("entries", []) if isinstance(sidecar, dict) else []
            if not isinstance(entries, list):
                raise TypeError(f"sidecar.entries 类型异常: {type(entries)}")
            self.supervision_by_token = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                token = str(entry.get("token", ""))
                if token:
                    self.supervision_by_token[token] = entry
        elif self.infos and "supervision_mask" in self.infos[0]:
            # canonical pkl 直接包含多帧监督字段，无需 sidecar
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            labels = metadata.get("supervision_labels", self.supervision_labels)
            if isinstance(labels, list) and labels:
                self.supervision_labels = [str(x) for x in labels]
            self.supervision_by_token = {}
            for info in self.infos:
                token = str(info.get("token", ""))
                if token:
                    self.supervision_by_token[token] = info

        # supervision_labels 已定版，构造 ray sidecar 并校验时刻顺序
        if self._ray_sidecar_dir:
            split = self._ray_sidecar_split or "train"
            self.ray_sidecar = RaySidecar(
                sidecar_dir=resolve_path(root_path, self._ray_sidecar_dir),
                split=split,
            )
            ray_labels = list(self.ray_sidecar.supervision_labels)
            if ray_labels and ray_labels != self.supervision_labels:
                raise ValueError(
                    "ray sidecar 与 dataset 的 supervision_labels 不一致，"
                    "会把 ray GT 监督到错误时刻：\n"
                    f"  dataset: {self.supervision_labels}\n"
                    f"  ray    : {ray_labels}"
                )

    def __len__(self) -> int:
        return len(self.infos)

    def _resolve_logits_path(
        self,
        root_rel: str,
        info: Dict[str, Any],
        info_key: str,
        variant: str,
        default_name: str,
    ) -> str:
        """
        统一路径拼接规则：
        root_path + 对应 logits_root + pkl 中相对路径字段。
        当 pkl 缺少相对路径时，回退到 scene/token/default_name。
        """
        rel_path = str(info.get(info_key, ""))
        if rel_path:
            rel_path = self._rewrite_rel_path_for_variant(
                rel_path=rel_path,
                variant=variant,
                default_name=default_name,
            )
            full_path = resolve_path(self.root_path, os.path.join(root_rel, rel_path))
        else:
            scene_name = str(info.get("scene_name", ""))
            token = str(info.get("token", ""))
            if not scene_name or not token:
                raise FileNotFoundError(
                    f"{info_key} 为空，且 info 缺少 scene_name/token，无法读取 logits 文件。"
                )
            full_path = resolve_path(
                self.root_path,
                os.path.join(root_rel, scene_name, token, default_name),
            )
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"logits 文件不存在: {full_path}")
        return full_path

    @staticmethod
    def _normalize_logits_variant(variant: str, field_name: str) -> str:
        value = str(variant).strip().lower()
        if value not in {"topk", "full", "full_topk"}:
            raise ValueError(f"{field_name} 仅支持 topk/full/full_topk，当前为 {variant!r}")
        return value

    @staticmethod
    def _rewrite_rel_path_for_variant(rel_path: str, variant: str, default_name: str) -> str:
        if variant not in ("full", "full_topk"):
            return rel_path
        rel_dir, _ = os.path.split(rel_path)
        return os.path.join(rel_dir, default_name) if rel_dir else default_name

    @staticmethod
    def _validate_npz_keys(path: str, logits_npz: Dict[str, Any], required_keys: List[str]) -> None:
        missing = [key for key in required_keys if key not in logits_npz]
        if missing:
            raise KeyError(f"{path} 缺少字段 {missing}，实际字段为 {sorted(logits_npz.keys())}")

    def _decode_logits(
        self,
        path: str,
        npz: Dict[str, Any],
        variant: str,
        multi_frame: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """统一解码 topk/full 格式的 logits。

        multi_frame=True  → 返回 (T, C, X, Y, Z)，需要 frame_splits
        multi_frame=False → 返回 (C, X, Y, Z)，单帧
        """
        common = dict(
            grid_size=self.grid_size,
            num_classes=self.num_classes,
            free_index=self.free_index,
            other_fill_value=self.topk_other_fill_value,
            free_fill_value=self.topk_free_fill_value,
            device=device,
            dtype=torch.float32,
        )
        if variant in ("full", "full_topk"):
            required = ["sparse_coords", "sparse_values"]
            if multi_frame:
                required.append("frame_splits")
            self._validate_npz_keys(path, npz, required)

            if variant == "full_topk":
                topk_values, topk_indices = sparse_full_to_topk(
                    sparse_values=npz["sparse_values"],
                    num_classes=self.num_classes,
                    free_index=self.free_index,
                    k=self.full_topk_k,
                )
                if multi_frame:
                    return decode_sparse_topk(
                        sparse_coords=npz["sparse_coords"],
                        sparse_topk_values=topk_values,
                        sparse_topk_indices=topk_indices,
                        frame_splits=npz["frame_splits"],
                        **common,
                    )
                return decode_single_frame_sparse_topk(
                    sparse_coords=npz["sparse_coords"],
                    sparse_topk_values=topk_values,
                    sparse_topk_indices=topk_indices,
                    **common,
                )

            if multi_frame:
                decoded = decode_sparse_full(
                    sparse_coords=npz["sparse_coords"],
                    sparse_values=npz["sparse_values"],
                    frame_splits=npz["frame_splits"],
                    **common,
                )
            else:
                decoded = decode_single_frame_sparse_full(
                    sparse_coords=npz["sparse_coords"],
                    sparse_values=npz["sparse_values"],
                    **common,
                )
            if self.full_logits_clamp_min is not None:
                decoded = decoded.clamp_min(self.full_logits_clamp_min)
            return decoded
        else:
            required = ["sparse_coords", "sparse_topk_values", "sparse_topk_indices"]
            if multi_frame:
                required.append("frame_splits")
            self._validate_npz_keys(path, npz, required)
            if multi_frame:
                return decode_sparse_topk(
                    sparse_coords=npz["sparse_coords"],
                    sparse_topk_values=npz["sparse_topk_values"],
                    sparse_topk_indices=npz["sparse_topk_indices"],
                    frame_splits=npz["frame_splits"],
                    **common,
                )
            return decode_single_frame_sparse_topk(
                sparse_coords=npz["sparse_coords"],
                sparse_topk_values=npz["sparse_topk_values"],
                sparse_topk_indices=npz["sparse_topk_indices"],
                **common,
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.infos[idx]
        device = torch.device("cpu")
        label_cache: Dict[str, Dict[str, Any]] = {}

        def _load_label_cached(path: str) -> Dict[str, Any]:
            if path not in label_cache:
                label_cache[path] = load_labels_npz(path)
            return label_cache[path]

        if self.logits_loader is not None:
            # 新路径：通过注入的 LogitsLoader 加载（如 ALOCC dense top-k）
            fast_logits = self.logits_loader.load_fast_logits(info, device)
            slow_logits = self.logits_loader.load_slow_logits(info, device)
        else:
            # 旧路径：OPUS 稀疏格式内联解码
            fast_logits_path = self._resolve_logits_path(
                root_rel=self.fast_logits_root,
                info=info,
                info_key="logits_path",
                variant=self.fast_logits_variant,
                default_name="logits_full.npz" if self.fast_logits_variant in ("full", "full_topk") else "logits.npz",
            )
            fast_logits = self._decode_logits(
                path=fast_logits_path,
                npz=load_logits_npz(fast_logits_path),
                variant=self.fast_logits_variant,
                multi_frame=True,
                device=device,
            )
            slow_logits_path = self._resolve_logits_path(
                root_rel=self.slow_logit_root,
                info=info,
                info_key="slow_logit_path",
                variant=self.slow_logit_variant,
                default_name="slow_logit_full.npz" if self.slow_logit_variant in ("full", "full_topk") else "slow_logit.npz",
            )
            slow_logits = self._decode_logits(
                path=slow_logits_path,
                npz=load_logits_npz(slow_logits_path),
                variant=self.slow_logit_variant,
                multi_frame=False,
                device=device,
            )
        num_frames = fast_logits.shape[0]

        frame_ego2global = torch.from_numpy(info["frame_ego2global"]).float()

        if self.slow_noise_std > 0:
            slow_logits = slow_logits + torch.randn_like(slow_logits) * self.slow_noise_std

        scene_name = info.get("scene_name", "")
        token = info.get("token", "")
        curr_gt_path = os.path.join(self.gt_root, scene_name, token, "labels.npz")
        curr_npz = _load_label_cached(curr_gt_path)
        gt_labels = torch.from_numpy(curr_npz["semantics"].astype("int64"))
        gt_mask_np = curr_npz.get(self.gt_mask_key, None)
        if gt_mask_np is None:
            gt_mask_np = torch.ones(self.grid_size, dtype=torch.float32).numpy()
        gt_mask = torch.from_numpy(gt_mask_np.astype("float32"))

        sup_labels = None
        sup_masks = None
        sup_step_indices = None
        sup_valid_mask = None
        if self.supervision_by_token is not None:
            sidecar_entry = self.supervision_by_token.get(token, None)
            if sidecar_entry is None:
                raise KeyError(f"sidecar 缺少 token={token} 的监督条目。")
            num_sup = len(self.supervision_labels)
            sup_labels = torch.zeros((num_sup, *self.grid_size), dtype=torch.long)
            sup_masks = torch.zeros((num_sup, *self.grid_size), dtype=torch.float32)
            sup_step_indices = torch.full((num_sup,), -1, dtype=torch.long)
            sup_valid_mask = torch.zeros((num_sup,), dtype=torch.float32)

            sidecar_mask = sidecar_entry.get("supervision_mask", [0] * num_sup)
            sidecar_steps = sidecar_entry.get("supervision_step_indices", [-1] * num_sup)
            sidecar_rel_paths = sidecar_entry.get("supervision_gt_rel_paths", [""] * num_sup)

            for sup_i in range(num_sup):
                valid = int(sidecar_mask[sup_i]) if sup_i < len(sidecar_mask) else 0
                if valid <= 0:
                    continue
                step_idx = int(sidecar_steps[sup_i]) if sup_i < len(sidecar_steps) else -1
                gt_rel = str(sidecar_rel_paths[sup_i]) if sup_i < len(sidecar_rel_paths) else ""
                if step_idx < 0 or not gt_rel:
                    continue
                gt_path = gt_rel if os.path.isabs(gt_rel) else os.path.join(self.gt_root, gt_rel)
                if not os.path.exists(gt_path):
                    continue
                sup_npz = _load_label_cached(gt_path)
                sup_sem = torch.from_numpy(sup_npz["semantics"].astype("int64"))
                sup_mask_np = sup_npz.get(self.gt_mask_key, None)
                if sup_mask_np is None:
                    sup_mask_np = torch.ones(self.grid_size, dtype=torch.float32).numpy()
                sup_m = torch.from_numpy(sup_mask_np.astype("float32"))

                sup_labels[sup_i] = sup_sem
                sup_masks[sup_i] = sup_m
                sup_step_indices[sup_i] = step_idx
                sup_valid_mask[sup_i] = 1.0

        frame_timestamps = info.get("frame_timestamps", None)
        if frame_timestamps is not None:
            frame_timestamps = torch.from_numpy(frame_timestamps).long()
        frame_dt = info.get("frame_dt", None)
        if frame_dt is not None:
            frame_dt = torch.from_numpy(frame_dt).float()

        ray_gt_dist = None
        ray_origin = None
        ray_sup_valid = None
        if self.ray_sidecar is not None:
            hit = self.ray_sidecar.query(token)
            if hit is None:
                # token 缺失：整条样本跳过 ray loss。每个 worker 进程首次遇到时
                # 打印一次 warning（DataLoader 多 worker 下无法做到全局唯一）。
                global _RAY_MISSING_WARNED
                if not _RAY_MISSING_WARNED:
                    pid = os.getpid()
                    print(
                        f"[ray_sidecar pid={pid}] WARN: token={token} 缺失，"
                        f"该样本将跳过 ray loss；本 worker 后续缺失不再提示。"
                    )
                    _RAY_MISSING_WARNED = True
                num_sup = len(self.supervision_labels)
                num_rays = self.ray_sidecar.num_rays
                ray_gt_dist = torch.full((num_sup, num_rays), float("nan"), dtype=torch.float32)
                ray_origin = torch.zeros((num_sup, 3), dtype=torch.float32)
                ray_sup_valid = torch.zeros((num_sup,), dtype=torch.float32)
            else:
                dist_np, origin_np, mask_np = hit
                ray_gt_dist = torch.from_numpy(dist_np)
                ray_origin = torch.from_numpy(origin_np)
                ray_sup_valid = torch.from_numpy(mask_np.astype("float32"))

        return {
            "fast_logits": fast_logits,
            "slow_logits": slow_logits,
            "frame_ego2global": frame_ego2global,
            "frame_timestamps": frame_timestamps,
            "frame_dt": frame_dt,
            "gt_labels": gt_labels,
            "gt_mask": gt_mask,
            "sup_labels": sup_labels,
            "sup_masks": sup_masks,
            "sup_step_indices": sup_step_indices,
            "sup_valid_mask": sup_valid_mask,
            "ray_gt_dist": ray_gt_dist,
            "ray_origin": ray_origin,
            "ray_sup_valid": ray_sup_valid,
            "meta": {
                "scene_name": scene_name,
                "token": token,
                "frame_tokens": info.get("frame_tokens", []),
                "logits_path": info.get("logits_path", ""),
                "slow_logit_path": info.get("slow_logit_path", ""),
                "fast_logits_variant": self.fast_logits_variant,
                "slow_logit_variant": self.slow_logit_variant,
                "supervision_labels": self.supervision_labels,
            },
        }
