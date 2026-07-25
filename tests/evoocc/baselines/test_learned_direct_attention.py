"""Learned direct window-attention baseline 单元测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from evoocc.baselines.learned_direct_attention import (
    DirectWindowAttentionNet,
    LearnedDirectAttentionAligner,
)
from evoocc.baselines.learned_direct_attention.modules import (
    _WindowCrossAttention3D,
)
from evoocc.baselines.learned_direct_attention.profiling import (
    estimate_learned_direct_attention_stepwise,
)
from evoocc.baselines.learned_direct_fusion.profiling import (
    estimate_learned_direct_fusion_stepwise,
)
from evoocc.config import load_config


ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG = (
    ROOT
    / "src"
    / "evoocc"
    / "baselines"
    / "learned_direct_attention"
    / "occ3d_config.yaml"
)


def test_occ3d_config_fixes_architecture_and_training_protocol() -> None:
    cfg = load_config(str(BASELINE_CONFIG))
    assert cfg["train"]["epochs"] == 10
    assert cfg["train"]["gradient_accumulation_steps"] == 4

    model_cfg = cfg["model"]["learned_direct_attention"]
    assert model_cfg["latent_grid_size"] == [50, 50, 16]
    assert model_cfg["latent_dim"] == 288
    assert model_cfg["attention_inner_dim"] == 96
    assert model_cfg["attention_num_heads"] == 8
    assert model_cfg["attention_window_size"] == [5, 5, 4]
    assert model_cfg["attention_local_dilations"] == [1, 2]


def test_shifted_window_mask_blocks_cyclic_boundaries() -> None:
    attention = _WindowCrossAttention3D(
        dim=8,
        num_heads=2,
        window_size=(2, 2, 2),
        shift_size=(1, 1, 0),
    )
    mask = attention._build_shift_mask(
        spatial_shape=(4, 4, 2),
        batch=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is not None
    assert mask.shape == (8, 1, 8, 8)
    assert torch.isneginf(mask).any()
    assert (mask == 0).any()


def test_attention_fusion_shape_and_backward() -> None:
    fusion = DirectWindowAttentionNet(
        feature_dim=16,
        inner_dim=8,
        num_heads=2,
        window_size=(2, 2, 2),
        local_dilations=(1, 2),
        gn_groups=2,
        mlp_ratio=2.0,
    )
    slow = torch.randn(2, 16, 4, 4, 2, requires_grad=True)
    fast = torch.randn(2, 16, 4, 4, 2, requires_grad=True)
    fused = fusion(slow, fast)
    assert fused.shape == slow.shape
    fused.square().mean().backward()
    assert slow.grad is not None
    assert fast.grad is not None
    assert torch.isfinite(slow.grad).all()
    assert torch.isfinite(fast.grad).all()


def test_aligner_reuses_direct_fusion_dataflow() -> None:
    model = LearnedDirectAttentionAligner(
        num_classes=18,
        encoder_in_channels=18,
        free_index=17,
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        voxel_size=(0.4, 0.4, 0.4),
    )
    assert model.input_grid_size == (200, 200, 16)
    assert model.latent_grid_size == (50, 50, 16)
    assert model.latent_voxel_size == (1.6, 1.6, 0.4)
    assert isinstance(model.fusion, DirectWindowAttentionNet)
    assert model.fusion.q_stem[0].in_channels == 288
    assert model.fusion.q_stem[0].out_channels == 96
    assert model.fusion.head.out_channels == 288


def test_flops_are_between_80_and_90_percent_of_convolution() -> None:
    convolution = estimate_learned_direct_fusion_stepwise()
    attention = estimate_learned_direct_attention_stepwise()
    ratio = attention.macs / convolution.macs
    assert 0.80 <= ratio <= 0.90
