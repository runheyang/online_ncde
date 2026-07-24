"""在 100×100×16 latent 中演化的 Neural ODE Δt baseline。"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, cast

import torch

from evoocc.baselines.learned_direct_fusion.modules import (
    XYDownsampleEncoder,
    XYUpsampleResidualDecoder,
)
from evoocc.baselines.neural_ode_dt_100.rollout import (
    _NeuralOdeDtRolloutBase,
)


class NeuralOdeDt100Aligner(_NeuralOdeDtRolloutBase):
    """仅以真实 ``Δt`` 驱动、在固定低分辨率空间递归演化的对齐器。

    继承父类的双状态初始化、相邻帧 ego-warp、ODE solver、stepwise 接口和
    Fast-KL 协议，只替换空间编码与解码路径。
    """

    def __init__(
        self,
        num_classes: int,
        encoder_in_channels: int,
        free_index: int,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        latent_dim: int = 120,
        func_g_inner_dim: int = 48,
        func_g_body_dilations: Sequence[int] = (1, 3, 5),
        func_g_gn_groups: int = 8,
        decoder_channels: int = 32,
        decoder_init_scale: Optional[float] = 1.0e-6,
        use_fast_residual: bool = True,
        input_grid_size: Tuple[int, int, int] = (200, 200, 16),
        latent_grid_size: Tuple[int, int, int] = (100, 100, 16),
        timestamp_scale: float = 1.0e-6,
        solver_variant: str = "euler",
    ) -> None:
        input_grid = tuple(int(value) for value in input_grid_size)
        latent_grid = tuple(int(value) for value in latent_grid_size)
        if input_grid != (200, 200, 16):
            raise ValueError(
                "Neural ODE Δt baseline 的输入固定为 (200,200,16)，"
                f"当前 {input_grid}"
            )
        if latent_grid != (100, 100, 16):
            raise ValueError(
                "Neural ODE Δt baseline 的演化空间固定为 (100,100,16)，"
                f"当前 {latent_grid}"
            )

        input_voxel_size = tuple(float(value) for value in voxel_size)
        if len(input_voxel_size) != 3:
            raise ValueError(
                f"voxel_size 必须长度为 3，当前 {input_voxel_size}"
            )
        latent_voxel_size = (
            2.0 * input_voxel_size[0],
            2.0 * input_voxel_size[1],
            input_voxel_size[2],
        )

        # 父类负责完整 rollout；其 voxel_size 应对应实际演化网格。
        super().__init__(
            num_classes=int(num_classes),
            feat_dim=int(latent_dim),
            hidden_dim=int(latent_dim),
            encoder_in_channels=int(encoder_in_channels),
            free_index=int(free_index),
            pc_range=pc_range,
            voxel_size=latent_voxel_size,
            decoder_init_scale=decoder_init_scale,
            use_fast_residual=bool(use_fast_residual),
            func_g_inner_dim=int(func_g_inner_dim),
            func_g_body_dilations=tuple(
                int(value) for value in func_g_body_dilations
            ),
            func_g_gn_groups=int(func_g_gn_groups),
            timestamp_scale=float(timestamp_scale),
            solver_variant=str(solver_variant),
        )

        self.input_grid_size = input_grid
        self.latent_grid_size = latent_grid
        self.input_voxel_size = cast(
            Tuple[float, float, float],
            input_voxel_size,
        )
        self.latent_voxel_size = cast(
            Tuple[float, float, float],
            latent_voxel_size,
        )
        self.latent_dim = int(latent_dim)
        self.decoder_channels = int(decoder_channels)

        # 仅下采样 XY；Z 方向始终保持 16。
        self.fast_encoder = XYDownsampleEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=self.latent_dim,
            gn_groups=int(func_g_gn_groups),
        )
        self.slow_encoder = XYDownsampleEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=self.latent_dim,
            gn_groups=int(func_g_gn_groups),
        )
        self.decoder = XYUpsampleResidualDecoder(
            in_channels=self.latent_dim,
            decoder_channels=self.decoder_channels,
            out_channels=self.num_classes,
            init_scale=decoder_init_scale,
            gn_groups=int(func_g_gn_groups),
        )

    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        expected = (
            self.encoder_in_channels,
            *self.input_grid_size,
        )
        if tuple(fast_logits.shape[1:]) != expected:
            raise ValueError(
                "fast logits 必须为 "
                f"(T,{expected[0]},200,200,16)，当前 {tuple(fast_logits.shape)}"
            )
        encoded = self.fast_encoder(fast_logits)
        if tuple(encoded.shape[2:]) != self.latent_grid_size:
            raise RuntimeError(
                "fast encoder 未产生固定的 100×100×16 latent，"
                f"当前 {tuple(encoded.shape)}"
            )
        return encoded

    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        expected = (
            self.encoder_in_channels,
            *self.input_grid_size,
        )
        if tuple(slow_logits.shape) != expected:
            raise ValueError(
                f"slow logits 必须为 {expected}，当前 {tuple(slow_logits.shape)}"
            )
        encoded = self.slow_encoder(slow_logits.unsqueeze(0))[0]
        if tuple(encoded.shape[1:]) != self.latent_grid_size:
            raise RuntimeError(
                "slow encoder 未产生固定的 100×100×16 latent，"
                f"当前 {tuple(encoded.shape)}"
            )
        return encoded

    def _decode_dense_state(self, z_dense: torch.Tensor) -> torch.Tensor:
        """只在状态演化完成后恢复到 200×200×16 logits。"""
        if tuple(z_dense.shape) != (
            self.latent_dim,
            *self.latent_grid_size,
        ):
            raise ValueError(
                "待解码状态必须位于 100×100×16，"
                f"当前 {tuple(z_dense.shape)}"
            )
        return self.decoder(
            z_dense.unsqueeze(0),
            output_shape_xyz=self.input_grid_size,
        )[0]
