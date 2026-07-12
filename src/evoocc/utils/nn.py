"""神经网络通用工具函数。"""

from __future__ import annotations


def resolve_group_norm_groups(num_channels: int, preferred_groups: int) -> int:
    """选择可整除通道数的 GN 组数。"""
    if preferred_groups <= 0:
        return 1
    upper = min(int(preferred_groups), int(num_channels))
    for groups in range(upper, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1
