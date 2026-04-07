from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from online_ncde.data.logits_io import (
    decode_single_frame_sparse_full,
    decode_sparse_full,
    sparse_full_to_topk,
)
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset


def test_decode_sparse_full_restores_semantic_logits_and_free_fill() -> None:
    sparse_coords = np.asarray([[0, 0, 0], [1, 1, 0]], dtype=np.uint8)
    sparse_values = np.asarray(
        [
            np.arange(1, 18, dtype=np.float16),
            np.arange(101, 118, dtype=np.float16),
        ],
        dtype=np.float16,
    )
    frame_splits = np.asarray([0, 1, 2], dtype=np.int32)

    dense = decode_sparse_full(
        sparse_coords=sparse_coords,
        sparse_values=sparse_values,
        frame_splits=frame_splits,
        grid_size=(2, 2, 1),
        num_classes=18,
        free_index=17,
        other_fill_value=-5.0,
        free_fill_value=5.0,
        dtype=torch.float32,
    )

    assert dense.shape == (2, 18, 2, 2, 1)
    assert torch.allclose(dense[0, :17, 0, 0, 0], torch.arange(1, 18, dtype=torch.float32))
    assert torch.allclose(dense[1, :17, 1, 1, 0], torch.arange(101, 118, dtype=torch.float32))
    assert dense[0, 17, 0, 0, 0].item() == -5.0
    assert dense[1, 17, 1, 1, 0].item() == -5.0
    assert dense[0, 17, 1, 1, 0].item() == 5.0
    assert dense[0, 3, 1, 1, 0].item() == -5.0


def test_decode_single_frame_sparse_full_restores_dense_logits() -> None:
    sparse_coords = np.asarray([[1, 0, 0]], dtype=np.uint8)
    sparse_values = np.asarray([np.linspace(-2.0, 2.0, 17, dtype=np.float16)], dtype=np.float16)

    dense = decode_single_frame_sparse_full(
        sparse_coords=sparse_coords,
        sparse_values=sparse_values,
        grid_size=(2, 2, 1),
        num_classes=18,
        free_index=17,
        other_fill_value=-5.0,
        free_fill_value=5.0,
        dtype=torch.float32,
    )

    assert dense.shape == (18, 2, 2, 1)
    assert torch.allclose(
        dense[:17, 1, 0, 0],
        torch.linspace(-2.0, 2.0, 17, dtype=torch.float32),
        atol=1.0e-3,
    )
    assert dense[17, 1, 0, 0].item() == -5.0
    assert dense[17, 0, 0, 0].item() == 5.0
    assert dense[4, 0, 0, 0].item() == -5.0


def test_sparse_full_to_topk_selects_descending_semantic_classes() -> None:
    sparse_values = np.asarray(
        [
            [1.0, 4.0, -2.0, 3.0],
            [8.0, 8.5, 7.5, 7.0],
        ],
        dtype=np.float16,
    )

    topk_values, topk_indices = sparse_full_to_topk(
        sparse_values=sparse_values,
        num_classes=5,
        free_index=2,
        k=2,
    )

    assert topk_values.dtype == np.float16
    assert topk_indices.dtype == np.uint8
    np.testing.assert_allclose(
        topk_values,
        np.asarray(
            [
                [4.0, 3.0],
                [8.5, 8.0],
            ],
            dtype=np.float16,
        ),
    )
    np.testing.assert_array_equal(
        topk_indices,
        np.asarray(
            [
                [1, 4],
                [1, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_resolve_logits_path_supports_full_filename_rewrite_and_slow_fallback(tmp_path: Path) -> None:
    dataset = Occ3DOnlineNcdeDataset.__new__(Occ3DOnlineNcdeDataset)
    dataset.root_path = str(tmp_path)

    fast_dir = tmp_path / "fast_root" / "scene-a" / "token-a"
    slow_dir = tmp_path / "slow_root" / "scene-a" / "token-a"
    fast_dir.mkdir(parents=True)
    slow_dir.mkdir(parents=True)
    (fast_dir / "logits.npz").touch()
    (fast_dir / "logits_full.npz").touch()
    (slow_dir / "slow_logit.npz").touch()
    (slow_dir / "slow_logit_full.npz").touch()

    info = {
        "scene_name": "scene-a",
        "token": "token-a",
        "logits_path": "scene-a/token-a/logits.npz",
        "slow_logit_path": "",
    }

    fast_topk = dataset._resolve_logits_path(
        root_rel="fast_root",
        info=info,
        info_key="logits_path",
        variant="topk",
        default_name="logits.npz",
    )
    fast_full = dataset._resolve_logits_path(
        root_rel="fast_root",
        info=info,
        info_key="logits_path",
        variant="full",
        default_name="logits_full.npz",
    )
    slow_full = dataset._resolve_logits_path(
        root_rel="slow_root",
        info=info,
        info_key="slow_logit_path",
        variant="full",
        default_name="slow_logit_full.npz",
    )

    assert fast_topk.endswith("fast_root/scene-a/token-a/logits.npz")
    assert fast_full.endswith("fast_root/scene-a/token-a/logits_full.npz")
    assert slow_full.endswith("slow_root/scene-a/token-a/slow_logit_full.npz")
