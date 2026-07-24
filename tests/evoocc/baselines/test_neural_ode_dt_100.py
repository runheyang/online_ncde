"""100×100×16 Neural ODE Δt baseline 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from evoocc.baselines.neural_ode_dt_100 import NeuralOdeDt100Aligner
from evoocc.baselines.neural_ode_dt_100.profiling import (
    estimate_neural_ode_dt_100_stepwise,
)
from evoocc.baselines.neural_ode_dt_100.rollout import NeuralOdeDtSolver
from evoocc.baselines.learned_direct_fusion.modules import XYDownsampleEncoder
from evoocc.baselines.learned_direct_fusion.profiling import (
    estimate_evoocc_stepwise,
)
from evoocc.config import load_config


ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG = (
    ROOT
    / "src"
    / "evoocc"
    / "baselines"
    / "neural_ode_dt_100"
    / "occ3d_config.yaml"
)


def _build_aligner() -> NeuralOdeDt100Aligner:
    return NeuralOdeDt100Aligner(
        num_classes=18,
        encoder_in_channels=18,
        free_index=17,
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        voxel_size=(0.4, 0.4, 0.4),
    )


def test_occ3d_config_fixes_grid_channels_and_training_protocol() -> None:
    cfg = load_config(str(BASELINE_CONFIG))
    assert cfg["train"]["epochs"] == 10
    assert cfg["train"]["gradient_accumulation_steps"] == 4

    model_cfg = cfg["model"]["neural_ode_dt_100"]
    assert model_cfg["input_grid_size"] == [200, 200, 16]
    assert model_cfg["latent_grid_size"] == [100, 100, 16]
    assert model_cfg["latent_dim"] == 120
    assert model_cfg["func_g_inner_dim"] == 48
    assert model_cfg["solver_variant"] == "euler"


def test_model_uses_fixed_100_grid_and_raised_feature_dim() -> None:
    model = _build_aligner()
    assert model.input_grid_size == (200, 200, 16)
    assert model.latent_grid_size == (100, 100, 16)
    assert model.latent_voxel_size == (0.8, 0.8, 0.4)
    assert model.feat_dim == 120
    assert model.hidden_dim == 120
    assert model.solver_variant == "euler"

    assert model.fast_encoder.pool.kernel_size == (1, 2, 2)
    assert model.fast_encoder.pool.stride == (1, 2, 2)
    assert model.fast_encoder.conv.out_channels == 120
    assert model.func_g.stem_conv.in_channels == 240
    assert model.func_g.stem_conv.out_channels == 48
    assert model.func_g.head_conv.out_channels == 120
    assert model.decoder.project.in_channels == 120
    assert model.decoder.project.out_channels == 32
    assert not hasattr(model, "ctrl_proj")


def test_encoder_only_downsamples_xy() -> None:
    encoder = XYDownsampleEncoder(
        in_channels=3,
        out_channels=8,
        gn_groups=4,
    )
    logits = torch.randn(2, 3, 8, 10, 4)
    encoded = encoder(logits)
    assert encoded.shape == (2, 8, 4, 5, 4)


def test_invalid_latent_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="演化空间固定"):
        NeuralOdeDt100Aligner(
            num_classes=3,
            encoder_in_channels=3,
            free_index=2,
            pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
            voxel_size=(0.4, 0.4, 0.4),
            latent_grid_size=(50, 50, 16),
        )


class _ConstantVectorField(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        fast: torch.Tensor,
    ) -> torch.Tensor:
        del fast
        return torch.full_like(hidden, 2.0)


def test_euler_solver_broadcasts_dt_without_control_increment() -> None:
    solver = NeuralOdeDtSolver(
        func_g=_ConstantVectorField(),
        variant="euler",
    )
    hidden = torch.zeros(4, 3, 2, 1)
    fast = torch.randn_like(hidden)
    next_hidden, delta_scene = solver.step(
        h_adv=hidden,
        f_prev_adv=fast,
        f_t=fast,
        dt=torch.tensor(0.25),
    )
    expected = torch.full_like(hidden, 0.5)
    torch.testing.assert_close(delta_scene, expected)
    torch.testing.assert_close(next_hidden, expected)


def test_stepwise_conv_flops_match_evoocc_within_one_percent() -> None:
    evoocc = estimate_evoocc_stepwise()
    baseline = estimate_neural_ode_dt_100_stepwise()
    ratio = baseline.macs / evoocc.macs
    assert 0.99 <= ratio <= 1.01
