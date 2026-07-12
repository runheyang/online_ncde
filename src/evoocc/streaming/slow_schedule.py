"""慢系统注入调度: 给 keyframe timestamp 流和间隔 t 秒, 决定哪些 step 注入 slow logits."""
from __future__ import annotations

from typing import Iterable, List, Set


def schedule_slow_steps(timestamps_us: Iterable[int], interval_sec: float) -> Set[int]:
    """决定 slow 注入的 step 集合.

    Args:
        timestamps_us: 每个 keyframe 的绝对 timestamp(微秒, 来自 nuscenes pkl 的 timestamp 字段).
        interval_sec: 慢系统注入间隔(秒). 0 表示每帧都注入(等价于直接用 slow). 负数表示只首帧注入.

    Returns:
        slow_steps: set[int], 在哪些 step (0-indexed) 注入 slow logits.
                    首帧(step=0)总是包含, 后续按 interval 累计.
    """
    ts = list(timestamps_us)
    if not ts:
        return set()
    used: Set[int] = {0}
    if interval_sec < 0:
        return used
    last_t = ts[0] / 1e6
    for i in range(1, len(ts)):
        t = ts[i] / 1e6
        if t - last_t + 1e-3 >= interval_sec:
            used.add(i)
            last_t = t
    return used


def slow_step_indices(timestamps_us: Iterable[int], interval_sec: float) -> List[int]:
    """sorted list 形式, 便于打印/调试."""
    return sorted(schedule_slow_steps(timestamps_us, interval_sec))
