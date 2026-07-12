"""Backend 专用 benchmark mode."""
from __future__ import annotations

import numpy as np
import torch

from evoocc.streaming.benchmark_loop import benchmark_callable
from evoocc.streaming.streaming_loader import scatter_to_device


def _unwrap(x):
    return x.data[0] if hasattr(x, "data") else x


def _unwrap_alocc_batch_for_simple_test(batch):
    """把 streaming_loader.scatter_to_device 后的 batch unwrap 成 ALOcc simple_test 参数."""
    img_inputs = batch["img_inputs"]
    img_metas = batch["img_metas"]
    points = batch.get("points", None)
    if not isinstance(img_inputs, list):
        img_inputs = [img_inputs]
    if not isinstance(img_metas, list):
        img_metas = [img_metas]
    if points is not None and not isinstance(points, list):
        points = [points]
    img = _unwrap(img_inputs[0])
    metas = _unwrap(img_metas[0])
    pts = [None] if points is None else _unwrap(points[0])
    return pts, metas, img


def benchmark_alocc_fast_only(fast, raw_batches, warmup: int, samples: int):
    """ALOcc 官方 simple_test 路径，包含 softmax/argmax/cpu."""
    fast.reset_history()
    old_cal_metric = getattr(fast.model, "cal_metric_in_model", None)
    if old_cal_metric is not None:
        # benchmark 只统计模型前向，不把 OccStudio 内部 mIoU 计算算进耗时。
        fast.model.cal_metric_in_model = False

    def step(raw):
        batch = scatter_to_device(raw, 0)
        pts, metas, img = _unwrap_alocc_batch_for_simple_test(batch)
        return fast.model.simple_test(pts, metas, img)

    try:
        return benchmark_callable(
            "fast-only",
            raw_batches,
            warmup,
            samples,
            step,
        )
    finally:
        if old_cal_metric is not None:
            fast.model.cal_metric_in_model = old_cal_metric


def _take_single_aug(x):
    x = _unwrap(x)
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = _unwrap(x[0])
    return x


def _unwrap_opus_batch(batch):
    img = _take_single_aug(batch["img"])
    img_metas = _take_single_aug(batch["img_metas"])
    if isinstance(img_metas, dict):
        img_metas = [img_metas]
    return img_metas, img


def _sparse_occ_to_dense_uint8(result, grid_size=(200, 200, 16), free_index=17):
    """OPUS 原生 sparse sem_pred/occ_loc -> dense uint8."""
    dense = np.full(tuple(grid_size), fill_value=int(free_index), dtype=np.uint8)
    if not result:
        return dense
    item = result[0]
    occ_loc = item["occ_loc"]
    sem_pred = item["sem_pred"]
    if occ_loc.size > 0:
        dense[occ_loc[:, 0], occ_loc[:, 1], occ_loc[:, 2]] = sem_pred.astype(np.uint8)
    return dense


def benchmark_opus_native_only(fast, raw_batches, warmup: int, samples: int, data_cfg: dict):
    """OPUS 原生 simple_test + sparse 转 dense uint8."""
    fast.reset_history()

    def step(raw):
        batch = scatter_to_device(raw, 0)
        img_metas, img = _unwrap_opus_batch(batch)
        result = fast.model.simple_test(img_metas, img)
        return _sparse_occ_to_dense_uint8(
            result,
            grid_size=tuple(data_cfg["grid_size"]),
            free_index=int(data_cfg["free_index"]),
        )

    return benchmark_callable(
        "native-only",
        raw_batches,
        warmup,
        samples,
        step,
    )


def benchmark_opus_fast_only(fast, raw_batches, warmup: int, samples: int):
    """OPUS raw-top3 dense logits -> argmax/cpu."""
    fast.reset_history()

    def step(raw):
        batch = scatter_to_device(raw, 0)
        fast_logits = fast.forward_keyframe(batch)
        return fast_logits.argmax(0).to(torch.uint8).cpu().numpy()

    return benchmark_callable(
        "fast-only(raw-top3)",
        raw_batches,
        warmup,
        samples,
        step,
    )
