"""50×50×16 learned direct window-attention aligner。"""

from __future__ import annotations

from typing import Optional, Tuple

from evoocc.baselines.learned_direct_attention.modules import (
    DirectWindowAttentionNet,
)
from evoocc.baselines.learned_direct_fusion.aligner import (
    LearnedDirectFusionAligner,
)


class LearnedDirectAttentionAligner(LearnedDirectFusionAligner):
    """复用direct fusion数据流，仅将卷积融合替换为窗口交叉注意力。"""

    def __init__(
        self,
        num_classes: int,
        encoder_in_channels: int,
        free_index: int,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        latent_dim: int = 288,
        attention_inner_dim: int = 96,
        attention_num_heads: int = 8,
        attention_window_size: Tuple[int, int, int] = (5, 5, 4),
        attention_local_dilations: Tuple[int, int] = (1, 2),
        attention_gn_groups: int = 8,
        attention_mlp_ratio: float = 2.0,
        decoder_channels: int = 32,
        decoder_init_scale: Optional[float] = 1.0e-6,
        use_fast_residual: bool = True,
        input_grid_size: Tuple[int, int, int] = (200, 200, 16),
        latent_grid_size: Tuple[int, int, int] = (50, 50, 16),
        timestamp_scale: float = 1.0e-6,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            encoder_in_channels=encoder_in_channels,
            free_index=free_index,
            pc_range=pc_range,
            voxel_size=voxel_size,
            latent_dim=latent_dim,
            fusion_gn_groups=attention_gn_groups,
            decoder_channels=decoder_channels,
            decoder_init_scale=decoder_init_scale,
            use_fast_residual=use_fast_residual,
            input_grid_size=input_grid_size,
            latent_grid_size=latent_grid_size,
            timestamp_scale=timestamp_scale,
        )
        self.fusion = DirectWindowAttentionNet(
            feature_dim=int(latent_dim),
            inner_dim=int(attention_inner_dim),
            num_heads=int(attention_num_heads),
            window_size=tuple(int(value) for value in attention_window_size),
            local_dilations=tuple(
                int(value) for value in attention_local_dilations
            ),
            gn_groups=int(attention_gn_groups),
            mlp_ratio=float(attention_mlp_ratio),
        )
