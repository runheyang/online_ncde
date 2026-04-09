"""progressbar2 辅助：提供 it/s 速度显示的 widget 和快捷构造函数。"""

from __future__ import annotations

import time
import progressbar


class IterSpeed(progressbar.widgets.WidgetBase):
    """显示迭代速度的 widget，类似 tqdm 的 it/s。"""

    def __init__(self, fmt: str = "{:.2f} it/s"):
        super().__init__()
        self.fmt = fmt
        self._start_time: float | None = None

    def __call__(self, progress, data, **kwargs):
        if self._start_time is None:
            self._start_time = time.monotonic()
            return "? it/s"
        elapsed = time.monotonic() - self._start_time
        value = data.get("value", 0) or 0
        if elapsed <= 0 or value <= 0:
            return "? it/s"
        return self.fmt.format(value / elapsed)


def make_pbar(max_value: int, prefix: str = "") -> progressbar.ProgressBar:
    """创建带 it/s 显示的 ProgressBar。"""
    widgets = [
        prefix,
        progressbar.SimpleProgress(), " ",
        progressbar.Bar(), " ",
        progressbar.Percentage(), " ",
        IterSpeed(), " ",
        progressbar.ETA(),
    ]
    return progressbar.ProgressBar(max_value=max_value, widgets=widgets)
