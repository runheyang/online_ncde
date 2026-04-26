"""对比方法 baseline 集合。

每个 baseline 以类或函数形式暴露，统一接受 dataset 样本 + pc_range/voxel_size
等静态配置，返回 dense 预测 (X, Y, Z) long，供 mIoU/RayIoU 评估。
"""

from online_ncde.baselines.recurrent_warp_fusion import (
    FusionAttnNet,
    FusionNet,
    RecurrentWarpFusionAligner,
)
from online_ncde.baselines.warp_slow_fill_fast import WarpSlowFillFastBaseline

__all__ = [
    "WarpSlowFillFastBaseline",
    "RecurrentWarpFusionAligner",
    "FusionNet",
    "FusionAttnNet",
]
