"""OPUSv1-T 在线 fast runner.

只服务 scripts/streaming/opusv1 下的入口，不改动现有 alocc2d_mini FastRunner。
runner 复用 OPUS 自己的 dataset/model 构建逻辑，
但不调用 dataset.evaluate / RayIoU，只把当前 keyframe 转成 EvoOcc 需要的
dense logits: (C, X, Y, Z)。
"""
from __future__ import annotations

import importlib
import os
import queue
import sys
from typing import Iterable, Optional

import torch
from mmcv.parallel import collate, scatter
from mmcv.runner.fp16_utils import cast_tensor_type


def _add_opus_path(opus_root: str) -> None:
    """切到 OPUS 根目录并注册其 models/loaders."""
    if opus_root not in sys.path:
        sys.path.insert(0, opus_root)
    os.chdir(opus_root)
    importlib.import_module("models")
    importlib.import_module("loaders")


def _unwrap(x):
    return x.data[0] if hasattr(x, "data") else x


def _take_single_aug(x):
    """把 OPUS test_mode 下的 [DataContainer(...)] 规整成实际对象."""
    x = _unwrap(x)
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = _unwrap(x[0])
    return x


def _decode_points(points: torch.Tensor, pc_range: torch.Tensor) -> torch.Tensor:
    """等价于 OPUS models.bbox.utils.decode_points，避免额外跨包引用."""
    points = points.clone()
    points[..., 0] = points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    points[..., 1] = points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    points[..., 2] = points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return points


class OpusV1FastRunner:
    """OPUSv1 单帧在线推理 + dense logits 封装.

    输出契约和 offline OPUS sparse-full loader 对齐：
      - 只用 sigmoid(raw_logits)>0.5 和中心距离过滤 query；
      - 命中体素对 17 个非 free raw logits 取 top-k，并 clamp_min；
      - 未命中体素 free 通道为 free_fill_value，其余为 other_fill_value；
      - 返回 (num_classes, X, Y, Z) fp32 CUDA tensor。
    """

    def __init__(
        self,
        opus_root: str,
        config_path: str,
        ckpt_path: str,
        num_classes: int = 18,
        free_index: int = 17,
        grid_size: Iterable[int] = (200, 200, 16),
        other_fill_value: float = -5.0,
        free_fill_value: float = 5.0,
        topk_k: int = 3,
        clamp_min: float = -5.0,
        max_cached_frames: Optional[int] = None,
        device: str = "cuda:0",
    ) -> None:
        self.opus_root = opus_root
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.num_classes = int(num_classes)
        self.free_index = int(free_index)
        self.grid_size = tuple(int(v) for v in grid_size)
        self.other_fill_value = float(other_fill_value)
        self.free_fill_value = float(free_fill_value)
        self.topk_k = int(topk_k)
        self.clamp_min = float(clamp_min)
        self.max_cached_frames = max_cached_frames
        self.device = torch.device(device)
        self.model = None
        self.cfg = None
        self._dataset = None
        self._semantic_indices = [
            c for c in range(self.num_classes) if c != self.free_index
        ]

    def build(self) -> None:
        _add_opus_path(self.opus_root)
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model

        cfg = Config.fromfile(self.config_path)
        self.cfg = cfg
        if self.max_cached_frames is None:
            self.max_cached_frames = int(cfg.model.pts_bbox_head.transformer.num_frames)
        self.model = build_model(cfg.model)
        self.model.to(self.device).eval()
        self.model.fp16_enabled = True
        load_checkpoint(self.model, self.ckpt_path, map_location="cuda", strict=True)

        cfg.data.val.test_mode = True
        self._dataset = build_dataset(cfg.data.val)

        head_grid = tuple(int(v) for v in self.model.pts_bbox_head.voxel_num.tolist())
        if head_grid != self.grid_size:
            raise ValueError(f"OPUS grid_size={head_grid}, cfg grid_size={self.grid_size}")

    @property
    def dataset(self):
        if self._dataset is None:
            raise RuntimeError("call build() first")
        return self._dataset

    def reset_history(self) -> None:
        """按 scene 清空 OPUS online feature cache."""
        if self.model is None:
            return
        if hasattr(self.model, "memory"):
            self.model.memory = {}
        if hasattr(self.model, "queue"):
            self.model.queue = queue.Queue()

    @torch.no_grad()
    def forward_keyframe(self, sample: dict) -> torch.Tensor:
        """对单个 OPUS sample 跑在线 forward，返回 dense logits."""
        if self.model is None:
            raise RuntimeError("call build() first")
        img = _take_single_aug(sample["img"])
        img_metas = _take_single_aug(sample["img_metas"])
        if isinstance(img_metas, dict):
            img_metas = [img_metas]
        outs = self._forward_online_raw(img=img, img_metas=img_metas)
        return self._outs_to_dense_logits(outs)

    def _forward_online_raw(self, img: torch.Tensor, img_metas: list[dict]) -> dict:
        """复刻 OPUSV1.simple_test_online，但保留 head raw outputs."""
        model = self.model
        model.fp16_enabled = False
        if len(img_metas) != 1:
            raise ValueError(f"OPUS online 只支持 batch_size=1, got {len(img_metas)}")

        if img.dim() == 4:
            img = img.unsqueeze(0)
        B, N, C, H, W = img.shape
        img = img.reshape(B, N // 6, 6, C, H, W)

        img_filenames = img_metas[0]["filename"]
        num_frames = len(img_filenames) // 6
        img_shape = (H, W, C)
        img_metas[0]["img_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["ori_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["pad_shape"] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []
        for i in range(num_frames):
            img_indices = list(range(i * 6, (i + 1) * 6))
            img_metas_curr = [{}]
            for k, item in img_metas[0].items():
                if isinstance(item, list) and len(item) == 6 * num_frames:
                    img_metas_curr[0][k] = [item[j] for j in img_indices]
                else:
                    img_metas_curr[0][k] = item

            cache_key = img_filenames[img_indices[0]]
            if cache_key in model.memory:
                img_feats_curr = model.memory[cache_key]
            else:
                img_feats_curr = model.extract_feat(img[:, i], img_metas_curr)
                model.memory[cache_key] = img_feats_curr
                model.queue.put(cache_key)
                # OPUS 原版保留 15 帧特征；8f 流式只需要保留当前窗口，降低 12G 卡峰值。
                while model.queue.qsize() > int(self.max_cached_frames):
                    pop_key = model.queue.get()
                    model.memory.pop(pop_key, None)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat(
                [img_feats_list[i][j] for i in range(len(img_feats_list))],
                dim=0,
            )
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = cast_tensor_type(img_feats_reorganized, torch.half, torch.float32)
        return model.pts_bbox_head(img_feats, img_metas_reorganized)

    def _empty_dense(self) -> torch.Tensor:
        X, Y, Z = self.grid_size
        dense = torch.full(
            (self.num_classes, X, Y, Z),
            fill_value=self.other_fill_value,
            dtype=torch.float32,
            device=self.device,
        )
        dense[self.free_index] = self.free_fill_value
        return dense

    def _outs_to_dense_logits(self, pred_dicts: dict) -> torch.Tensor:
        """把 OPUS sparse set raw logits 转成 (C,X,Y,Z) dense top-k logits."""
        head = self.model.pts_bbox_head
        all_cls_scores = pred_dicts["all_cls_scores"]
        all_refine_pts = pred_dicts["all_refine_pts"]
        raw_cls_logits = all_cls_scores[-1]
        cls_scores = raw_cls_logits.sigmoid()
        refine_pts = all_refine_pts[-1]
        if refine_pts.shape[0] != 1:
            raise ValueError("OPUS streaming runner 只支持 batch_size=1")

        dense = self._empty_dense()
        refine_pts = _decode_points(refine_pts[0], head.pc_range)
        scores = cls_scores[0]
        raw_logits = raw_cls_logits[0]

        ctr_dist_thr = float(head.test_cfg.get("ctr_dist_thr", 3.0))
        score_thr = float(head.test_cfg.get("score_thr", 0.5))
        centers = refine_pts.mean(dim=1, keepdim=True)
        ctr_dists = torch.norm(refine_pts - centers, dim=-1)
        mask_dist = ctr_dists < ctr_dist_thr
        mask_score = (scores > score_thr).any(dim=-1)
        mask = mask_dist & mask_score
        if not torch.any(mask):
            return dense

        refine_pts = refine_pts[mask]
        raw_logits = raw_logits[mask]
        # 只用 sigmoid 分数做过滤；voxel 内聚合的是原始 logits。
        pts = torch.cat([refine_pts, raw_logits], dim=-1)
        pts_infos, voxels, num_pts = head.voxel_generator(pts)
        if voxels.numel() == 0:
            return dense

        voxels = torch.flip(voxels, [1]).long()
        voxel_logits = pts_infos[..., 3:].sum(dim=1) / num_pts[..., None]
        k = min(self.topk_k, int(voxel_logits.shape[-1]))
        topk_vals, topk_local_idx = torch.topk(
            voxel_logits, k=k, dim=-1, largest=True, sorted=True
        )
        topk_vals = topk_vals.clamp_min(self.clamp_min).to(dtype=dense.dtype)
        semantic_idx = torch.as_tensor(
            self._semantic_indices, device=self.device, dtype=torch.long
        )
        topk_global_idx = semantic_idx[topk_local_idx.long()]
        x_idx, y_idx, z_idx = voxels[:, 0], voxels[:, 1], voxels[:, 2]
        dense[self.free_index, x_idx, y_idx, z_idx] = self.other_fill_value
        n_hit = int(voxels.shape[0])
        hit_idx = torch.arange(n_hit, device=self.device).view(n_hit, 1).expand(-1, k)
        dense[
            topk_global_idx.reshape(-1),
            x_idx[hit_idx].reshape(-1),
            y_idx[hit_idx].reshape(-1),
            z_idx[hit_idx].reshape(-1),
        ] = topk_vals.reshape(-1)
        return dense

    def prepare_batch(self, raw_sample) -> dict:
        """把 dataset[i] 输出包装成 forward_keyframe 期望的 batch dict."""
        batch = collate([raw_sample], samples_per_gpu=1)
        batch = scatter(batch, [0])[0]
        return batch
