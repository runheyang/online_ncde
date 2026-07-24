"""对比方法 baseline 集合。

每个 baseline 以类或函数形式暴露，统一接受 dataset 样本 + pc_range/voxel_size
等静态配置，返回 dense 预测 (X, Y, Z) long，供 mIoU/RayIoU 评估。
"""

from evoocc.baselines.neural_ode_dt_aligner import (
    NeuralOdeDtAligner,
    NeuralOdeDtSolver,
)
from evoocc.baselines.learned_direct_fusion import LearnedDirectFusionAligner
from evoocc.baselines.no_warp_motion_attn import (
    NoWarpMotionBiasAttnAligner,
    NoWarpMotionBiasAttnFusion,
)
from evoocc.baselines.recurrent_warp_fusion import (
    FusionAttnNet,
    FusionNet,
    RecurrentWarpFusionAligner,
)
from evoocc.baselines.streamingflow import StreamingFlowBEVOdeAligner
from evoocc.baselines.warp_slow_fill_fast import WarpSlowFillFastBaseline

__all__ = [
    "WarpSlowFillFastBaseline",
    "LearnedDirectFusionAligner",
    "RecurrentWarpFusionAligner",
    "FusionNet",
    "FusionAttnNet",
    "NeuralOdeDtAligner",
    "NeuralOdeDtSolver",
    "NoWarpMotionBiasAttnAligner",
    "NoWarpMotionBiasAttnFusion",
    "StreamingFlowBEVOdeAligner",
]
