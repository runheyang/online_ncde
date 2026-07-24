"""Neural ODE Δt 100×100×16 baseline 的卷积 FLOPs 估算。"""

from __future__ import annotations

from evoocc.baselines.learned_direct_fusion.profiling import (
    ComputeEstimate,
    estimate_evoocc_stepwise,
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


def estimate_neural_ode_dt_100_stepwise(
    *,
    num_frames: int = 5,
    num_targets: int = 4,
    input_grid_size: tuple[int, int, int] = (200, 200, 16),
    latent_grid_size: tuple[int, int, int] = (100, 100, 16),
    num_classes: int = 18,
    latent_dim: int = 120,
    func_g_inner_dim: int = 48,
    decoder_channels: int = 32,
    num_body_blocks: int = 3,
) -> ComputeEstimate:
    """估算四步 Euler rollout 的 Conv3d MACs。

    与 EvoOcc 统计口径一致，不计 warp、插值、归一化和激活。
    """
    input_voxels = int(
        input_grid_size[0] * input_grid_size[1] * input_grid_size[2]
    )
    latent_voxels = int(
        latent_grid_size[0] * latent_grid_size[1] * latent_grid_size[2]
    )

    # 五个 fast frame 加一个 slow anchor。
    encoders = (num_frames + 1) * _conv3d_macs(
        latent_voxels,
        num_classes,
        latent_dim,
        kernel_volume=27,
    )
    func_g_per_step = (
        _conv3d_macs(
            latent_voxels,
            2 * latent_dim,
            func_g_inner_dim,
            kernel_volume=1,
        )
        + num_body_blocks
        * _conv3d_macs(
            latent_voxels,
            func_g_inner_dim,
            func_g_inner_dim,
            kernel_volume=27,
        )
        + _conv3d_macs(
            latent_voxels,
            func_g_inner_dim,
            latent_dim,
            kernel_volume=1,
        )
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
        macs=(
            encoders
            + num_targets * func_g_per_step
            + num_targets * decoder_per_target
        )
    )


def main() -> None:
    evoocc = estimate_evoocc_stepwise()
    baseline = estimate_neural_ode_dt_100_stepwise()
    print(
        f"EvoOcc: {evoocc.gmacs:.5f} GMACs / "
        f"{evoocc.gflops:.5f} GFLOPs"
    )
    print(
        f"Neural ODE dt 100: {baseline.gmacs:.5f} GMACs / "
        f"{baseline.gflops:.5f} GFLOPs"
    )
    print(f"ratio={baseline.macs / evoocc.macs:.6f}")


if __name__ == "__main__":
    main()
