#!/usr/bin/env python3
"""实时 benchmark 的评估适配工具。"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OPUS_ROOT = ROOT / "third_party" / "OPUS"
if str(OPUS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPUS_ROOT))


def get_dataset_infos(dataset: Any) -> list[dict[str, Any]]:
    """兼容旧版 data_infos 与新版 data_list。"""
    infos = getattr(dataset, "data_infos", None)
    if infos is None:
        infos = getattr(dataset, "data_list", None)
    if infos is None:
        raise AttributeError("dataset 既没有 data_infos，也没有 data_list。")
    return list(infos)


class FilteredValidationDataset:
    """只为 benchmark 评估保留子集视图，不改原始 dataset。"""

    def __init__(self, base_dataset: Any, keep_indices: list[int]) -> None:
        self.base_dataset = base_dataset
        self.keep_indices = list(keep_indices)
        base_infos = get_dataset_infos(base_dataset)
        self.subset_infos = [base_infos[i] for i in self.keep_indices]

    def __len__(self) -> int:
        return len(self.keep_indices)

    def __getitem__(self, index: int) -> Any:
        return self.base_dataset[self.keep_indices[index]]

    @property
    def data_infos(self) -> list[dict[str, Any]]:
        return self.subset_infos

    @property
    def data_list(self) -> list[dict[str, Any]]:
        return self.subset_infos

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        return type(self.base_dataset).evaluate(self, *args, **kwargs)

    def eval_miou(self, *args: Any, **kwargs: Any) -> Any:
        return type(self.base_dataset).eval_miou(self, *args, **kwargs)

    def eval_riou(self, *args: Any, **kwargs: Any) -> Any:
        return type(self.base_dataset).eval_riou(self, *args, **kwargs)

    def format_results(self, *args: Any, **kwargs: Any) -> Any:
        return type(self.base_dataset).format_results(self, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)


def dense_logits_to_occ_result(
    dense_logits: torch.Tensor,
    free_index: int,
) -> dict[str, np.ndarray]:
    """把 dense logits 适配回 OPUS evaluate() 需要的稀疏结果格式。"""
    sem_grid = dense_logits.argmax(dim=0)
    occ_loc = torch.nonzero(sem_grid != int(free_index), as_tuple=False)
    if occ_loc.numel() == 0:
        return {
            "sem_pred": np.zeros((0,), dtype=np.int64),
            "occ_loc": np.zeros((0, 3), dtype=np.int64),
        }
    sem_pred = sem_grid[occ_loc[:, 0], occ_loc[:, 1], occ_loc[:, 2]]
    return {
        "sem_pred": sem_pred.detach().cpu().numpy().astype(np.int64, copy=False),
        "occ_loc": occ_loc.detach().cpu().numpy().astype(np.int64, copy=False),
    }


def evaluate_final_predictions(
    base_dataset: Any,
    token_to_dataset_idx: dict[str, int],
    results_by_token: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """按 dataset index 排序后，复用 OPUS dataset.evaluate() 出主指标。"""
    keep_indices: list[int] = []
    sorted_results: list[dict[str, np.ndarray]] = []

    for token, dataset_idx in sorted(token_to_dataset_idx.items(), key=lambda item: item[1]):
        result = results_by_token.get(token, None)
        if result is None:
            continue
        keep_indices.append(int(dataset_idx))
        sorted_results.append(result)

    if not sorted_results:
        return {}

    _ensure_opus_ray_metrics_loaded()
    filtered_dataset = FilteredValidationDataset(base_dataset=base_dataset, keep_indices=keep_indices)
    metrics = filtered_dataset.evaluate(sorted_results, jsonfile_prefix="submission_realtime_benchmark")
    return dict(metrics)


@contextmanager
def _pushd(path: Path):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def _ensure_opus_ray_metrics_loaded() -> None:
    """OPUS 的 ray_metrics 会按 cwd 查找 lib/dvr，需要先在 OPUS 根目录导入。"""
    if "loaders.ray_metrics" in sys.modules:
        return
    with _pushd(OPUS_ROOT):
        importlib.import_module("loaders.ray_metrics")
