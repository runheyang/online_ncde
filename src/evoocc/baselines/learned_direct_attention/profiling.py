"""Direct window attention 与卷积 direct fusion 的 FLOPs 估算。"""

from __future__ import annotations

from typing import Tuple

from evoocc.baselines.learned_direct_fusion.profiling import (
    ComputeEstimate,
    estimate_learned_direct_fusion_stepwise,
)


def _conv3d_macs(
    output_voxels: int,
    in_channels: int,
    out_channels: int,
    kernel_volume: int,
    groups: int = 1,
) -> int:
    return (
        int(output_voxels)
        * int(out_channels)
        * (int(in_channels) // int(groups))
        * int(kernel_volume)
    )


def estimate_learned_direct_attention_stepwise(
    *,
    num_targets: int = 4,
    input_grid_size: Tuple[int, int, int] = (200, 200, 16),
    latent_grid_size: Tuple[int, int, int] = (50, 50, 16),
    num_classes: int = 18,
    latent_dim: int = 288,
    attention_inner_dim: int = 96,
    attention_window_size: Tuple[int, int, int] = (5, 5, 4),
    attention_mlp_ratio: float = 2.0,
    decoder_channels: int = 32,
    num_attention_blocks: int = 2,
    num_local_blocks: int = 2,
) -> ComputeEstimate:
    """估算四目标window cross-attention的乘加计算量。"""
    input_voxels = int(
        input_grid_size[0] * input_grid_size[1] * input_grid_size[2]
    )
    latent_voxels = int(
        latent_grid_size[0]
        * latent_grid_size[1]
        * latent_grid_size[2]
    )
    window_tokens = int(
        attention_window_size[0]
        * attention_window_size[1]
        * attention_window_size[2]
    )
    ffn_dim = max(
        int(round(attention_inner_dim * float(attention_mlp_ratio))),
        attention_inner_dim,
    )

    # 一个slow anchor，加num_targets个current fast。
    encoders = (num_targets + 1) * _conv3d_macs(
        latent_voxels,
        num_classes,
        latent_dim,
        kernel_volume=27,
    )

    # q/kv stems与输出head。
    fusion_per_target = 3 * _conv3d_macs(
        latent_voxels,
        latent_dim,
        attention_inner_dim,
        kernel_volume=1,
    )
    # 每层包含Q/K/V/out projection、FFN以及QK^T/AV。
    fusion_per_target += num_attention_blocks * (
        4
        * _conv3d_macs(
            latent_voxels,
            attention_inner_dim,
            attention_inner_dim,
            kernel_volume=1,
        )
        + 2
        * _conv3d_macs(
            latent_voxels,
            attention_inner_dim,
            ffn_dim,
            kernel_volume=1,
        )
        + 2
        * latent_voxels
        * window_tokens
        * attention_inner_dim
    )
    fusion_per_target += num_local_blocks * _conv3d_macs(
        latent_voxels,
        attention_inner_dim,
        attention_inner_dim,
        kernel_volume=27,
    )

    decoder_per_target = (
        _conv3d_macs(
            latent_voxels,
            latent_dim,
            decoder_channels,
            kernel_volume=1,
        )
        + _conv3d_macs(
            input_voxels,
            decoder_channels,
            decoder_channels,
            kernel_volume=27,
        )
        + _conv3d_macs(
            input_voxels,
            decoder_channels,
            decoder_channels,
            kernel_volume=9,
            groups=decoder_channels,
        )
        + _conv3d_macs(
            input_voxels,
            decoder_channels,
            num_classes,
            kernel_volume=1,
        )
    )
    return ComputeEstimate(
        macs=encoders
        + num_targets * (fusion_per_target + decoder_per_target)
    )


def main() -> None:
    convolution = estimate_learned_direct_fusion_stepwise()
    attention = estimate_learned_direct_attention_stepwise()
    print(
        "Learned direct fusion: "
        f"{convolution.gmacs:.5f} GMACs / {convolution.gflops:.5f} GFLOPs"
    )
    print(
        "Learned direct attention: "
        f"{attention.gmacs:.5f} GMACs / {attention.gflops:.5f} GFLOPs"
    )
    print(f"ratio={attention.macs / convolution.macs:.6f}")


if __name__ == "__main__":
    main()
