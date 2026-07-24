"""Learned direct fusion 与 EvoOcc 的卷积 FLOPs 估算。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeEstimate:
    """统一记录 MACs；FLOPs 按一次乘加等于 2 FLOPs 计算。"""

    macs: int

    @property
    def flops(self) -> int:
        return 2 * self.macs

    @property
    def gmacs(self) -> float:
        return self.macs / 1.0e9

    @property
    def gflops(self) -> float:
        return self.flops / 1.0e9


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


def estimate_evoocc_stepwise(
    *,
    num_frames: int = 5,
    num_targets: int = 4,
    grid_size: tuple[int, int, int] = (200, 200, 16),
    num_classes: int = 18,
    feature_dim: int = 32,
    inner_dim: int = 24,
    num_body_blocks: int = 3,
) -> ComputeEstimate:
    """估算 Euler EvoOcc 在四时刻 stepwise 输出下的 Conv3d MACs。"""
    voxels = int(grid_size[0] * grid_size[1] * grid_size[2])

    # fast 序列编码一次，slow anchor 编码一次。
    encoder = (num_frames + 1) * _conv3d_macs(
        voxels,
        num_classes,
        feature_dim,
        kernel_volume=27,
    )

    func_g_per_step = (
        _conv3d_macs(
            voxels,
            2 * feature_dim,
            inner_dim,
            kernel_volume=1,
        )
        + num_body_blocks
        * _conv3d_macs(
            voxels,
            inner_dim,
            inner_dim,
            kernel_volume=27,
        )
        + _conv3d_macs(
            voxels,
            inner_dim,
            feature_dim,
            kernel_volume=1,
        )
    )
    ctrl_per_step = _conv3d_macs(
        voxels,
        feature_dim + 1,
        feature_dim,
        kernel_volume=1,
    )
    dynamics = num_targets * (func_g_per_step + ctrl_per_step)

    decoder_per_target = (
        _conv3d_macs(
            voxels,
            feature_dim,
            feature_dim,
            kernel_volume=27,
        )
        + _conv3d_macs(
            voxels,
            feature_dim,
            feature_dim,
            kernel_volume=9,
            groups=feature_dim,
        )
        + _conv3d_macs(
            voxels,
            feature_dim,
            num_classes,
            kernel_volume=1,
        )
    )
    return ComputeEstimate(
        macs=encoder + dynamics + num_targets * decoder_per_target
    )


def estimate_learned_direct_fusion_stepwise(
    *,
    num_targets: int = 4,
    input_grid_size: tuple[int, int, int] = (200, 200, 16),
    latent_grid_size: tuple[int, int, int] = (50, 50, 16),
    num_classes: int = 18,
    latent_dim: int = 288,
    fusion_inner_dim: int = 104,
    decoder_channels: int = 32,
    num_body_blocks: int = 3,
) -> ComputeEstimate:
    """估算 direct fusion 四个独立目标输出的 Conv3d MACs。"""
    input_voxels = int(
        input_grid_size[0] * input_grid_size[1] * input_grid_size[2]
    )
    latent_voxels = int(
        latent_grid_size[0] * latent_grid_size[1] * latent_grid_size[2]
    )

    # 一个 slow anchor，加 num_targets 个 current fast。
    encoders = (num_targets + 1) * _conv3d_macs(
        latent_voxels,
        num_classes,
        latent_dim,
        kernel_volume=27,
    )
    fusion_per_target = (
        _conv3d_macs(
            latent_voxels,
            2 * latent_dim,
            fusion_inner_dim,
            kernel_volume=1,
        )
        + num_body_blocks
        * _conv3d_macs(
            latent_voxels,
            fusion_inner_dim,
            fusion_inner_dim,
            kernel_volume=27,
        )
        + _conv3d_macs(
            latent_voxels,
            fusion_inner_dim,
            latent_dim,
            kernel_volume=1,
        )
        + _conv3d_macs(
            latent_voxels,
            latent_dim,
            decoder_channels,
            kernel_volume=1,
        )
    )
    decoder_per_target = (
        _conv3d_macs(
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
    evoocc = estimate_evoocc_stepwise()
    baseline = estimate_learned_direct_fusion_stepwise()
    print(
        "EvoOcc: "
        f"{evoocc.gmacs:.5f} GMACs / {evoocc.gflops:.5f} GFLOPs"
    )
    print(
        "Learned direct fusion: "
        f"{baseline.gmacs:.5f} GMACs / {baseline.gflops:.5f} GFLOPs"
    )
    print(f"ratio={baseline.macs / evoocc.macs:.6f}")


if __name__ == "__main__":
    main()
