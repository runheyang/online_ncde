"""GT labels 读取与编码工具。"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def load_labels_npz(path: str) -> Dict[str, np.ndarray]:
    """读取 labels.npz。"""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def labels_to_logits(
    semantics: np.ndarray,
    num_classes: int,
    pos_value: float = 10.0,
    neg_value: float = -10.0,
) -> torch.Tensor:
    """将语义标签转为 one-hot logits，作为慢系统输入。"""
    labels = torch.from_numpy(semantics.astype(np.int64))
    one_hot = F.one_hot(labels, num_classes=num_classes).permute(3, 0, 1, 2).float()
    logits = one_hot * (pos_value - neg_value) + neg_value
    return logits


