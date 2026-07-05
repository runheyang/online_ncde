"""模型 EMA（指数移动平均），用于抑制训练后期指标震荡。"""

from __future__ import annotations

import copy
from typing import Any, Dict

import torch


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    # DDP/Compiled 等包装一律剥到原始模块，保证 state_dict 对齐。
    return getattr(model, "module", model)


class ModelEMA:
    """维护模型参数的指数移动平均，供 eval 时使用。"""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        device: torch.device | None = None,
    ) -> None:
        target = _unwrap(model)
        self.module = copy.deepcopy(target)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)
        self.decay = float(decay)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """每个 optimizer.step() 后调用一次。

        采用 `min(decay, (1+n)/(10+n))` 的 warmup，避免 EMA 在训练早期被初始
        权重拖住。num_updates 较大时自然退化为固定 decay。
        """
        self.num_updates += 1
        n = self.num_updates
        effective_decay = min(self.decay, (1.0 + n) / (10.0 + n))

        src_sd = _unwrap(model).state_dict()
        ema_sd = self.module.state_dict()
        for key, ema_v in ema_sd.items():
            src_v = src_sd[key].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(effective_decay).add_(
                    src_v.to(ema_v.device, dtype=ema_v.dtype),
                    alpha=1.0 - effective_decay,
                )
            else:
                # BN running stats 等整型/long 缓冲直接 copy，不做平均。
                ema_v.copy_(src_v.to(ema_v.device))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module.state_dict(),
            "num_updates": self.num_updates,
            "decay": self.decay,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.module.load_state_dict(sd["module"])
        self.num_updates = int(sd.get("num_updates", 0))
        self.decay = float(sd.get("decay", self.decay))
