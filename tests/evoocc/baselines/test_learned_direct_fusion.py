"""Learned direct fusion baseline 单元测试。"""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn

from evoocc.baselines.learned_direct_fusion.aligner import (
    LearnedDirectFusionAligner,
)
from evoocc.baselines.learned_direct_fusion.modules import (
    DirectFusionNet,
    XYDownsampleEncoder,
    XYUpsampleResidualDecoder,
)
from evoocc.baselines.learned_direct_fusion.profiling import (
    estimate_evoocc_stepwise,
    estimate_learned_direct_fusion_stepwise,
)
from evoocc.config import load_config


ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG = (
    ROOT
    / "src"
    / "evoocc"
    / "baselines"
    / "learned_direct_fusion"
    / "occ3d_config.yaml"
)


def _build_small_aligner() -> LearnedDirectFusionAligner:
    return LearnedDirectFusionAligner(
        num_classes=3,
        encoder_in_channels=3,
        free_index=2,
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        voxel_size=(0.4, 0.4, 0.4),
        latent_dim=2,
        fusion_inner_dim=2,
        fusion_body_dilations=(1,),
        fusion_gn_groups=1,
        decoder_channels=2,
        decoder_init_scale=0.0,
        input_grid_size=(200, 200, 16),
        latent_grid_size=(100, 100, 16),
    )


def test_occ3d_config_fixes_training_protocol() -> None:
    cfg = load_config(str(BASELINE_CONFIG))
    assert cfg["train"]["epochs"] == 10
    assert cfg["train"]["gradient_accumulation_steps"] == 2
    model_cfg = cfg["model"]["learned_direct_fusion"]
    assert model_cfg["input_grid_size"] == [200, 200, 16]
    assert model_cfg["latent_grid_size"] == [100, 100, 16]
    assert model_cfg["latent_dim"] == 128
    assert model_cfg["fusion_inner_dim"] == 48


def test_xy_encoder_only_downsamples_xy() -> None:
    encoder = XYDownsampleEncoder(
        in_channels=3,
        out_channels=8,
        gn_groups=4,
    )
    logits = torch.randn(2, 3, 8, 10, 4)
    encoded = encoder(logits)
    assert encoded.shape == (2, 8, 4, 5, 4)


def test_fusion_decoder_shape_and_backward() -> None:
    fusion = DirectFusionNet(
        feature_dim=8,
        inner_dim=4,
        body_dilations=(1, 2),
        gn_groups=2,
    )
    decoder = XYUpsampleResidualDecoder(
        in_channels=8,
        decoder_channels=4,
        out_channels=3,
        init_scale=None,
        gn_groups=2,
    )
    slow = torch.randn(2, 8, 4, 5, 4, requires_grad=True)
    fast = torch.randn(2, 8, 4, 5, 4, requires_grad=True)
    fused = fusion(slow, fast)
    logits = decoder(fused, output_shape_xyz=(8, 10, 4))
    assert fused.shape == slow.shape
    assert logits.shape == (2, 3, 8, 10, 4)
    logits.square().mean().backward()
    assert slow.grad is not None
    assert fast.grad is not None


def test_each_target_reuses_original_slow_anchor() -> None:
    aligner = _build_small_aligner()
    anchor = torch.ones(2, 2, 2, 1)
    slow_encode_calls = 0
    seen_anchor_ptrs: list[int] = []
    seen_anchor_poses: list[float] = []
    seen_target_poses: list[float] = []

    def validate_shapes(self, fast_logits, slow_logits):
        del self, fast_logits, slow_logits

    def encode_slow(self, slow_logits):
        nonlocal slow_encode_calls
        del self, slow_logits
        slow_encode_calls += 1
        return anchor

    def encode_fast(self, fast_logits):
        del self
        return torch.zeros(
            fast_logits.shape[0],
            2,
            2,
            2,
            1,
            dtype=fast_logits.dtype,
        )

    def warp(self, slow_anchor, pose_anchor, pose_target):
        del self
        seen_anchor_ptrs.append(slow_anchor.data_ptr())
        seen_anchor_poses.append(float(pose_anchor[0, 3]))
        seen_target_poses.append(float(pose_target[0, 3]))
        return slow_anchor + pose_target[0, 3]

    def decode(self, fused, target_fast_logits):
        del self, fused
        return target_fast_logits

    aligner._validate_sample_shapes = MethodType(validate_shapes, aligner)
    aligner._encode_slow_anchor = MethodType(encode_slow, aligner)
    aligner._encode_target_fast = MethodType(encode_fast, aligner)
    aligner._warp_anchor_to_target = MethodType(warp, aligner)
    aligner._decode_targets = MethodType(decode, aligner)
    aligner.fusion = nn.Identity()

    class _PairIdentity(nn.Module):
        def forward(self, warped_slow, current_fast):
            return warped_slow + current_fast

    aligner.fusion = _PairIdentity()

    fast_logits = torch.randn(5, 3, 2, 2, 1)
    slow_logits = torch.randn(3, 2, 2, 1)
    poses = torch.eye(4).repeat(5, 1, 1)
    poses[:, 0, 3] = torch.arange(5, dtype=torch.float32)

    outputs, _ = aligner._predict_targets(
        fast_logits=fast_logits,
        slow_logits=slow_logits,
        frame_ego2global=poses,
        target_indices=[1, 2, 3, 4],
        anchor_index=0,
    )

    assert outputs.shape == (4, 3, 2, 2, 1)
    assert slow_encode_calls == 1
    assert seen_anchor_ptrs == [anchor.data_ptr()] * 4
    assert seen_anchor_poses == [0.0, 0.0, 0.0, 0.0]
    assert seen_target_poses == [1.0, 2.0, 3.0, 4.0]


def test_target_indices_follow_supervision_steps() -> None:
    indices = LearnedDirectFusionAligner._target_indices(
        num_frames=5,
        rollout_start_step=0,
        max_step_index=None,
    )
    assert indices == [1, 2, 3, 4]


def test_stepwise_conv_flops_match_evoocc_within_five_percent() -> None:
    evoocc = estimate_evoocc_stepwise()
    baseline = estimate_learned_direct_fusion_stepwise()
    ratio = baseline.macs / evoocc.macs
    assert 0.95 <= ratio <= 1.05
