"""StreamingFlow-style timestamp event loop。"""

from __future__ import annotations

import time
from typing import Sequence

import torch
import torch.nn as nn

from evoocc.baselines.streamingflow.gru_ode import (
    DualGRUCell2D,
    DualGRUODECell2D,
    PModel2D,
    infer_state_deterministic,
)


class StreamingFlowODECore(nn.Module):
    """BEV latent 上的 GRU-ODE + observation jump 主循环。"""

    def __init__(
        self,
        channels: int = 64,
        impute: bool = True,
        deterministic_impute: bool = True,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.impute = bool(impute)
        self.deterministic_impute = bool(deterministic_impute)
        if not self.deterministic_impute:
            raise ValueError("主 baseline 只支持 deterministic_impute=True")

        self.ode_cell = DualGRUODECell2D(
            input_size=self.channels, hidden_size=self.channels, gn_groups=gn_groups
        )
        self.obs_cell = DualGRUCell2D(
            input_size=self.channels, hidden_size=self.channels, gn_groups=gn_groups
        )
        self.p_model = PModel2D(channels=self.channels, gn_groups=gn_groups)

    def infer_state(self, state: torch.Tensor) -> torch.Tensor:
        q_params = self.p_model(state)
        return infer_state_deterministic(q_params)

    def ode_step(
        self,
        state: torch.Tensor,
        input_feat: torch.Tensor,
        delta_t: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if delta_t <= 0.0:
            return state, input_feat
        ode_input = input_feat if self.impute else torch.zeros_like(input_feat)
        dt = state.new_tensor(float(delta_t))
        state = state + dt * self.ode_cell(ode_input, state)
        input_feat = self.infer_state(state)
        return state, input_feat

    @staticmethod
    def _time_to_float(t: float | int | torch.Tensor) -> float:
        if isinstance(t, torch.Tensor):
            return float(t.detach().cpu().item())
        return float(t)

    def forward(
        self,
        initial_input: torch.Tensor,
        observations: Sequence[tuple[float | torch.Tensor, torch.Tensor]],
        target_times: torch.Tensor,
        return_step_times: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """按 StreamingFlow 事件顺序输出 target BEV states。

        observation 与 target 时间相同时，先 jump 再保存 target state。
        """
        if initial_input.dim() != 4:
            raise ValueError(
                f"initial_input 需要 (B,C,H,W)，当前: {tuple(initial_input.shape)}"
            )
        if initial_input.shape[1] != self.channels:
            raise ValueError(
                f"initial_input channel={initial_input.shape[1]}，期望 {self.channels}"
            )

        sorted_obs = sorted(
            [(self._time_to_float(t), obs) for t, obs in observations],
            key=lambda item: item[0],
        )
        target_list = [self._time_to_float(t) for t in target_times.reshape(-1)]

        state = torch.zeros_like(initial_input)
        input_feat = initial_input
        current_time = sorted_obs[0][0] if sorted_obs else 0.0

        saved: list[torch.Tensor] = []
        step_time_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        step_time_cpu: list[float] = []
        use_cuda_timing = bool(return_step_times and initial_input.is_cuda)
        use_cpu_timing = bool(return_step_times and not initial_input.is_cuda)
        if use_cuda_timing:
            segment_start = torch.cuda.Event(enable_timing=True)
            segment_start.record()
        elif use_cpu_timing:
            segment_start_cpu = time.perf_counter()

        target_idx = 0
        eps = 1.0e-6

        def save_target() -> None:
            nonlocal segment_start, segment_start_cpu  # type: ignore[name-defined]
            saved.append(state)
            if use_cuda_timing:
                segment_end = torch.cuda.Event(enable_timing=True)
                segment_end.record()
                step_time_events.append((segment_start, segment_end))
                segment_start = torch.cuda.Event(enable_timing=True)
                segment_start.record()
            elif use_cpu_timing:
                now = time.perf_counter()
                step_time_cpu.append((now - segment_start_cpu) * 1000.0)
                segment_start_cpu = now

        for obs_time, obs in sorted_obs:
            if obs.dim() != 4:
                raise ValueError(f"observation 需要 (B,C,H,W)，当前: {tuple(obs.shape)}")

            # target 严格早于下一个 observation：只做 ODE evolve 后保存。
            while target_idx < len(target_list) and target_list[target_idx] < obs_time - eps:
                state, input_feat = self.ode_step(
                    state, input_feat, target_list[target_idx] - current_time
                )
                current_time = target_list[target_idx]
                save_target()
                target_idx += 1

            state, input_feat = self.ode_step(state, input_feat, obs_time - current_time)
            current_time = obs_time

            # observation jump 到达后更新 imputed input。
            state = self.obs_cell(obs, state)
            input_feat = self.infer_state(state)

            # target 与 observation 同时刻时，保存 jump 后状态。
            while target_idx < len(target_list) and abs(target_list[target_idx] - obs_time) <= eps:
                save_target()
                target_idx += 1

        while target_idx < len(target_list):
            state, input_feat = self.ode_step(
                state, input_feat, target_list[target_idx] - current_time
            )
            current_time = target_list[target_idx]
            save_target()
            target_idx += 1

        if saved:
            states = torch.stack(saved, dim=1)
        else:
            b, c, h, w = initial_input.shape
            states = initial_input.new_zeros((b, 0, c, h, w))

        if not return_step_times:
            return states

        if use_cuda_timing:
            torch.cuda.synchronize(device=initial_input.device)
            step_ms = [start.elapsed_time(end) for start, end in step_time_events]
        else:
            step_ms = step_time_cpu
        step_time = torch.tensor(step_ms, device=initial_input.device, dtype=torch.float32)
        return states, step_time
