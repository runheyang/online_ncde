"""快系统 alocc2d mini 在线推理 + dense topk 压缩封装.

负责:
  1. build OccStudio ALOCC 模型, load checkpoint
  2. 接受单个 OccStudio sample (经 pipeline + collate 后), 输出 dense (C, X, Y, Z) fast logits
  3. 数值与离线 alocc2d_mini_wo_mask logits.npz 完全一致 (POC 已验证 top1 一致率 99.9997%)
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Optional

import torch
from mmcv.parallel import collate, scatter


def _add_occstudio_path(occstudio_root: str) -> None:
    if occstudio_root not in sys.path:
        sys.path.insert(0, occstudio_root)
    # 切到 OccStudio 根目录, 配置里 ann_file/data_root 都是 'data/...' 相对路径
    os.chdir(occstudio_root)
    # 触发 mmcv registry 注册
    import mmdet3d.models.alocc      # noqa
    import mmdet3d.datasets          # noqa


def _unwrap(x):
    return x.data[0] if hasattr(x, "data") else x


class FastRunner:
    """alocc2d mini 单帧推理 + dense topk 压缩.

    Usage:
        runner = FastRunner(occstudio_root, config_path, ckpt_path,
                            num_classes=18, free_index=17, topk_k=3,
                            clamp_min=-5.0, fill_value=-5.0, max_centering=False)
        runner.build()
        runner.reset_history()
        for sample_dict in dataset_samples:
            fast_logits = runner.forward_keyframe(sample_dict)  # (C, X, Y, Z) fp32
    """

    def __init__(
        self,
        occstudio_root: str,
        config_path: str,
        ckpt_path: str,
        num_classes: int = 18,
        free_index: int = 17,
        topk_k: int = 3,
        clamp_min: float = -5.0,
        fill_value: float = -5.0,
        max_centering: bool = False,
        label_id_offset: int = 0,
        drop_others_label: bool = False,
        device: str = "cuda:0",
    ) -> None:
        self.occstudio_root = occstudio_root
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.num_classes = int(num_classes)
        self.free_index = int(free_index)
        self.topk_k = int(topk_k)
        self.clamp_min = float(clamp_min)
        self.fill_value = float(fill_value)
        self.max_centering = bool(max_centering)
        self.label_id_offset = int(label_id_offset)
        self.drop_others_label = bool(drop_others_label)
        self.device = torch.device(device)
        self.model = None
        self._dataset = None

    # ---- build & dataset ----

    def build(self) -> None:
        _add_occstudio_path(self.occstudio_root)
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.models import build_model
        from mmdet3d.datasets import build_dataset
        cfg = Config.fromfile(self.config_path)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        self.cfg = cfg
        self.model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        load_checkpoint(self.model, self.ckpt_path, map_location="cpu")
        self.model.to(self.device).eval()
        cfg.data.test.test_mode = True
        self._dataset = build_dataset(cfg.data.test)

    @property
    def dataset(self):
        if self._dataset is None:
            raise RuntimeError("call build() first")
        return self._dataset

    def reset_history(self) -> None:
        """跨 scene 时重置 alocc 内部 history buffer.

        alocc 的 do_history=True 模式下, 模型用 internal cache 存过去 history_cat_num 帧.
        一个 scene 跑完后必须清空, 否则下一 scene 前几帧会被上一 scene 的 history 污染.
        """
        if self.model is None:
            return
        # 严格按 ALOCC 类里 self.history_* 的初始化字段清零, 跟 alocc.py:461-465/195 对齐.
        for name in ("history_bev", "history_bev_before_encoder",
                     "history_seq_ids", "history_forward_augs",
                     "history_sweep_time", "history_stereo",
                     "history_grid", "history_feat"):
            if hasattr(self.model, name):
                setattr(self.model, name, None)

    # ---- inference ----

    @torch.no_grad()
    def forward_keyframe(self, sample: dict) -> torch.Tensor:
        """对单个 OccStudio sample 跑 forward, 返回 dense (C, X, Y, Z) fp32 logits.

        sample 是经过 pipeline + collate([sample], samples_per_gpu=1) + scatter([0])[0]
        后的 dict. 调用方负责 sample 的准备.
        """
        img_inputs = sample["img_inputs"]
        img_metas = sample["img_metas"]
        points = sample.get("points", None)
        if not isinstance(img_inputs, list):
            img_inputs = [img_inputs]
        if not isinstance(img_metas, list):
            img_metas = [img_metas]
        if points is not None and not isinstance(points, list):
            points = [points]
        img = _unwrap(img_inputs[0])
        metas = _unwrap(img_metas[0])
        pts = [None] if points is None else _unwrap(points[0])
        self.model.do_history = True

        # ALOcc3D depth-stereo 分支还需要当前帧和相邻帧的相机参数。
        extra_kwargs = {
            key: sample[key]
            for key in ("aux_cam_params", "adj_aux_cam_params")
            if key in sample
        }
        results = self.model.extract_feat(
            points=pts,
            img=img,
            img_metas=metas,
            **extra_kwargs,
        )
        if self.model.with_specific_component("occupancy_head"):
            output_test = self.model.occupancy_head(results["img_bev_feat"], results=results)
        elif self.model.with_specific_component("alocc_head"):
            output_test = self.model.alocc_head.simple_test(
                results["img_bev_feat"], img_metas=metas, class_prototype=None, bda=img[6])
        else:
            raise RuntimeError("model has neither occupancy_head nor alocc_head")
        pred = output_test["output_voxels"][0]            # (1, C, X, Y, Z) raw pre-softmax
        pred = pred.permute(0, 2, 3, 4, 1)[0]             # (X, Y, Z, C)
        if self.model.fix_void:
            pred = pred[..., 1:]
        # CVPR2023 坐标系 (复刻 simple_test, 不做 softmax)
        pred = pred.permute(3, 2, 0, 1)
        pred = torch.flip(pred, [2])
        pred = torch.rot90(pred, -1, [2, 3])
        pred = pred.permute(2, 3, 1, 0)                   # (X, Y, Z, C)
        if self.drop_others_label:
            if pred.shape[-1] < self.num_classes + 1:
                raise ValueError(
                    "drop_others_label=True 需要比内部类别数多 1 个 others 通道: "
                    f"logits={pred.shape[-1]}, num_classes={self.num_classes}"
                )
            pred = pred[..., 1:]

        # topk(3) 沿 class 维
        topk_vals, topk_idx = torch.topk(
            pred, k=self.topk_k, dim=-1, largest=True, sorted=True
        )
        if self.drop_others_label:
            topk_idx = topk_idx + 1
        if self.label_id_offset:
            topk_idx = topk_idx + self.label_id_offset
        # 复刻 logits_loader 的 dense decode: max-center (可选) + clamp_min + scatter
        if self.max_centering:
            max_vals = topk_vals.max(dim=-1, keepdim=True).values
            centered = topk_vals - max_vals
        else:
            centered = topk_vals
        centered = centered.clamp_min(self.clamp_min)
        # scatter 回 dense (C, X, Y, Z)
        return self._scatter_to_dense(centered, topk_idx)

    def _scatter_to_dense(
        self, topk_vals: torch.Tensor, topk_idx: torch.Tensor
    ) -> torch.Tensor:
        """复刻 logits_loader.AloccDenseTopkLoader 的 dense decode.

        Args:
            topk_vals: (X, Y, Z, K) float
            topk_idx:  (X, Y, Z, K) long/int
        Returns:
            dense:     (C, X, Y, Z) float, 非 top-k 位置填 fill_value, free 索引位置填默认
        """
        X, Y, Z, K = topk_vals.shape
        C = self.num_classes
        # 全部初始化为 fill_value
        dense = torch.full(
            (C, X, Y, Z), self.fill_value, dtype=topk_vals.dtype, device=topk_vals.device
        )
        # scatter: dense[idx[..., k], x, y, z] = vals[..., k]
        # 用 advanced indexing: 一次性把 K 个 top 位置写进 dense
        x_idx = torch.arange(X, device=topk_vals.device).view(X, 1, 1, 1).expand(-1, Y, Z, K)
        y_idx = torch.arange(Y, device=topk_vals.device).view(1, Y, 1, 1).expand(X, -1, Z, K)
        z_idx = torch.arange(Z, device=topk_vals.device).view(1, 1, Z, 1).expand(X, Y, -1, K)
        c_idx = topk_idx.long()
        min_idx = int(c_idx.min().item())
        max_idx = int(c_idx.max().item())
        if min_idx < 0 or max_idx >= C:
            raise ValueError(
                f"FastRunner topk index 越界: min={min_idx}, max={max_idx}, num_classes={C}"
            )
        dense[c_idx, x_idx, y_idx, z_idx] = topk_vals
        return dense.float()

    # ---- helpers for callers ----

    def prepare_batch(self, raw_sample) -> dict:
        """把 dataset[i] 直接喂出的 sample 包装成 forward_keyframe 期望的 batch dict."""
        batch = collate([raw_sample], samples_per_gpu=1)
        batch = scatter(batch, [0])[0]
        return batch
