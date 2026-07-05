"""StreamingFlow-style BEV GRU-ODE baseline aligner。"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple, cast

import torch
import torch.nn as nn

from evoocc.baselines.streamingflow.bridge import (
    BEVTo3DDecoder,
    LightNoPoolSmallDecoder,
    LightNoPoolSmallEncoder,
    LogitsToBEVAdapter,
    SameTimeGatedFusion,
    SpatialGRURefiner2D,
    StreamingFlowSmallDecoder2D,
    StreamingFlowSmallEncoder2D,
)
from evoocc.baselines.streamingflow.core import StreamingFlowODECore
from evoocc.data.time_series import compute_segment_dt, cumulative_tau


class StreamingFlowBEVOdeAligner(nn.Module):
    """Occ3D-only StreamingFlow BEV GRU-ODE baseline。"""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        hidden_dim: int,
        encoder_in_channels: int,
        free_index: int,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        decoder_init_scale: float | None = None,
        timestamp_scale: float = 1.0e-6,
        streamingflow_cfg: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        del pc_range, voxel_size, decoder_init_scale  # 本 baseline 不走几何 warp / 残差初始化。
        cfg = dict(streamingflow_cfg or {})

        self.num_classes = int(num_classes)
        self.height_bins = 16
        self.grid_size = (200, 200, self.height_bins)
        self.free_index = int(free_index)
        self.encoder_in_channels = int(encoder_in_channels)
        self.timestamp_scale = float(timestamp_scale)
        self._fast_kl_active: bool = False

        if self.num_classes != 18:
            raise ValueError(f"StreamingFlow baseline 只支持 Occ3D num_classes=18，当前 {num_classes}")
        if self.encoder_in_channels != 18:
            raise ValueError(
                f"StreamingFlow baseline 只支持 encoder_in_channels=18，当前 {encoder_in_channels}"
            )

        self.bev_channels = int(cfg.get("bev_channels", hidden_dim))
        if int(feat_dim) != self.bev_channels or int(hidden_dim) != self.bev_channels:
            raise ValueError(
                "StreamingFlow baseline 要求 feat_dim=hidden_dim=bev_channels，"
                f"当前 feat_dim={feat_dim}, hidden_dim={hidden_dim}, bev_channels={self.bev_channels}"
            )
        if self.bev_channels != 96:
            raise ValueError(f"StreamingFlow 100x100 baseline 固定 bev_channels=96，当前 {self.bev_channels}")

        small_encoder_kind = str(cfg.get("small_encoder_kind", "streamingflow_downsample"))
        small_decoder_kind = str(cfg.get("small_decoder_kind", "streamingflow_upsample"))
        if small_encoder_kind not in {"streamingflow_downsample", "light_no_pool"}:
            raise ValueError(f"未知 small_encoder_kind: {small_encoder_kind!r}")
        if small_decoder_kind not in {"streamingflow_upsample", "light_no_pool"}:
            raise ValueError(f"未知 small_decoder_kind: {small_decoder_kind!r}")
        if (small_encoder_kind, small_decoder_kind) not in {
            ("streamingflow_downsample", "streamingflow_upsample"),
            ("light_no_pool", "light_no_pool"),
        }:
            raise ValueError(
                f"small encoder/decoder kind 必须成对使用，当前 "
                f"{small_encoder_kind!r}/{small_decoder_kind!r}"
            )

        bev_resolution = tuple(cfg.get("bev_resolution", [200, 200]))
        temporal_resolution = tuple(cfg.get("temporal_state_resolution", [100, 100]))
        expected_temporal_resolution = (
            (100, 100) if small_encoder_kind == "streamingflow_downsample" else (200, 200)
        )
        if bev_resolution != (200, 200) or temporal_resolution != expected_temporal_resolution:
            raise ValueError(
                f"StreamingFlow baseline 固定 BEV/temporal resolution 为 "
                f"200x200/{expected_temporal_resolution[0]}x{expected_temporal_resolution[1]}，"
                f"当前 {bev_resolution}/{temporal_resolution}"
            )
        self.bev_resolution = (int(bev_resolution[0]), int(bev_resolution[1]))
        self.temporal_state_resolution = (
            int(temporal_resolution[0]),
            int(temporal_resolution[1]),
        )

        bev_stride_xy = int(cfg.get("bev_stride_xy", 1))
        if bev_stride_xy != 1:
            raise ValueError(f"StreamingFlow 100x100 baseline 要求 adapter bev_stride_xy=1，当前 {bev_stride_xy}")
        decoder_upsample_scale = self.grid_size[0] // self.bev_resolution[0]
        if (
            self.grid_size[0] % self.bev_resolution[0] != 0
            or self.grid_size[1] % self.bev_resolution[1] != 0
            or decoder_upsample_scale != self.grid_size[1] // self.bev_resolution[1]
        ):
            raise ValueError(
                f"BEV resolution {bev_resolution} 无法整除输出 grid {self.grid_size[:2]}"
            )

        self.observation_channels = int(cfg.get("observation_channels", 64))
        self.decoded_bev_channels = int(cfg.get("decoded_bev_channels", 64))
        adapter_mid = int(cfg.get("adapter_mid_channels", 64))
        decoder_mid = int(cfg.get("decoder_mid_channels", 64))
        decoder_high = int(cfg.get("decoder_high_channels", 64))
        small_encoder_blocks = int(cfg.get("small_encoder_blocks", 3))
        small_decoder_blocks = int(cfg.get("small_decoder_blocks", 3))
        small_encoder_filter = int(cfg.get("small_encoder_filter_size", 32))
        small_decoder_filter = int(cfg.get("small_decoder_filter_size", 32))
        gn_groups = int(cfg.get("gn_groups", 8))

        self.fast_adapter = LogitsToBEVAdapter(
            num_classes=self.num_classes,
            height_bins=self.height_bins,
            mid_channels=adapter_mid,
            out_channels=self.observation_channels,
            stride_xy=bev_stride_xy,
            gn_groups=gn_groups,
        )
        self.slow_adapter = LogitsToBEVAdapter(
            num_classes=self.num_classes,
            height_bins=self.height_bins,
            mid_channels=adapter_mid,
            out_channels=self.observation_channels,
            stride_xy=bev_stride_xy,
            gn_groups=gn_groups,
        )
        if small_encoder_kind == "streamingflow_downsample":
            self.fast_small_encoder = StreamingFlowSmallEncoder2D(
                in_channels=self.observation_channels,
                latent_channels=self.bev_channels,
                filter_size=small_encoder_filter,
                gn_groups=gn_groups,
            )
            self.slow_small_encoder = StreamingFlowSmallEncoder2D(
                in_channels=self.observation_channels,
                latent_channels=self.bev_channels,
                filter_size=small_encoder_filter,
                gn_groups=gn_groups,
            )
        elif small_encoder_kind == "light_no_pool":
            if self.observation_channels != self.bev_channels:
                raise ValueError(
                    "light_no_pool small encoder 要求 observation_channels == bev_channels，"
                    f"当前 {self.observation_channels} vs {self.bev_channels}"
                )
            self.fast_small_encoder = LightNoPoolSmallEncoder(
                channels=self.bev_channels, num_blocks=small_encoder_blocks, gn_groups=gn_groups
            )
            self.slow_small_encoder = LightNoPoolSmallEncoder(
                channels=self.bev_channels, num_blocks=small_encoder_blocks, gn_groups=gn_groups
            )
        else:
            raise AssertionError("unreachable small_encoder_kind")
        self.same_time_fusion = SameTimeGatedFusion(channels=self.bev_channels)

        self.core = StreamingFlowODECore(
            channels=self.bev_channels,
            impute=bool(cfg.get("impute", True)),
            deterministic_impute=bool(cfg.get("deterministic_impute", True)),
            gn_groups=gn_groups,
        )
        self.sequence_refiner = SpatialGRURefiner2D(
            channels=self.bev_channels,
            num_gru_blocks=int(cfg.get("n_spatial_gru", 1)),
            num_res_layers=int(cfg.get("n_res_layers", 1)),
            gn_groups=gn_groups,
        )
        if small_decoder_kind == "streamingflow_upsample":
            self.small_decoder = StreamingFlowSmallDecoder2D(
                latent_channels=self.bev_channels,
                out_channels=self.decoded_bev_channels,
                filter_size=small_decoder_filter,
                gn_groups=gn_groups,
            )
        elif small_decoder_kind == "light_no_pool":
            if self.decoded_bev_channels != self.bev_channels:
                raise ValueError(
                    "light_no_pool small decoder 要求 decoded_bev_channels == bev_channels，"
                    f"当前 {self.decoded_bev_channels} vs {self.bev_channels}"
                )
            self.small_decoder = LightNoPoolSmallDecoder(
                channels=self.bev_channels, num_blocks=small_decoder_blocks, gn_groups=gn_groups
            )
        else:
            raise AssertionError("unreachable small_decoder_kind")
        self.bev_to_3d = BEVTo3DDecoder(
            in_channels=self.decoded_bev_channels,
            mid_channels=decoder_mid,
            high_channels=decoder_high,
            upsample_scale=decoder_upsample_scale,
            num_classes=self.num_classes,
            height_bins=self.height_bins,
            gn_groups=gn_groups,
        )

    def _validate_logits(self, fast_logits: torch.Tensor, slow_logits: torch.Tensor) -> None:
        if fast_logits.dim() != 5 or slow_logits.dim() != 4:
            raise ValueError(
                f"单样本输入应为 fast=(T,C,X,Y,Z), slow=(C,X,Y,Z)，"
                f"当前 {tuple(fast_logits.shape)}, {tuple(slow_logits.shape)}"
            )
        expected = (self.num_classes, *self.grid_size)
        if tuple(fast_logits.shape[1:]) != expected or tuple(slow_logits.shape) != expected:
            raise ValueError(
                f"StreamingFlow baseline 只支持 Occ3D logits shape C,X,Y,Z={expected}，"
                f"当前 fast={tuple(fast_logits.shape[1:])}, slow={tuple(slow_logits.shape)}"
            )

    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        return self.fast_small_encoder(self.fast_adapter(fast_logits))

    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        slow_batch = slow_logits.unsqueeze(0)
        return self.slow_small_encoder(self.slow_adapter(slow_batch))[0]

    def _compute_tau(
        self,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if frame_timestamps is None and frame_dt is None:
            dt = torch.full((max(num_frames - 1, 0),), 0.5, device=device, dtype=torch.float32)
        else:
            dt = compute_segment_dt(
                frame_timestamps=frame_timestamps,
                frame_dt=frame_dt,
                num_frames=num_frames,
                timestamp_scale=self.timestamp_scale,
            ).to(device=device)
        return cumulative_tau(dt).to(device=device)

    def _prepare_single_rollout(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: int,
        rollout_steps: int,
    ) -> tuple[torch.Tensor, list[tuple[float, torch.Tensor]], torch.Tensor]:
        fast_bev = self._encode_fast(fast_logits)
        slow_bev = self._encode_slow(slow_logits)
        tau = self._compute_tau(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=fast_logits.shape[0],
            device=fast_logits.device,
        )
        t0 = tau[rollout_start_step]

        obs0 = self.same_time_fusion(
            slow_bev.unsqueeze(0), fast_bev[rollout_start_step].unsqueeze(0)
        )
        observations: list[tuple[float, torch.Tensor]] = [(0.0, obs0)]

        last_step = rollout_start_step + rollout_steps
        for step in range(rollout_start_step + 1, last_step + 1):
            rel_t = float((tau[step] - t0).detach().cpu().item())
            observations.append((rel_t, fast_bev[step].unsqueeze(0)))

        target_times = (tau[rollout_start_step + 1 : last_step + 1] - t0).float()
        return obs0, observations, target_times

    def _decode_bev_states(self, bev_states: torch.Tensor) -> torch.Tensor:
        if bev_states.shape[1] == 0:
            b = bev_states.shape[0]
            return bev_states.new_zeros((b, 0, self.num_classes, *self.grid_size))
        bev = self.sequence_refiner(bev_states)
        bev = self.small_decoder(bev)
        return self.bev_to_3d(bev)

    def _diagnostics(self, bev_states: torch.Tensor) -> dict[str, torch.Tensor]:
        value = bev_states.detach().abs().mean() if bev_states.numel() > 0 else bev_states.sum() * 0.0
        return {
            "delta_scene_abs_mean": value.float(),
            "bev_state_abs_mean": value.float(),
        }

    def _forward_single_stepwise_train(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        max_step_index: int | None = None,
        rollout_start_step: int = 0,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        self._validate_logits(fast_logits, slow_logits)
        num_frames = int(fast_logits.shape[0])
        if rollout_start_step >= num_frames - 1:
            step_indices = torch.tensor([num_frames - 1], device=fast_logits.device, dtype=torch.long)
            return {
                "step_logits": slow_logits.unsqueeze(0).float(),
                "step_indices": step_indices,
                "diagnostics": {
                    "delta_scene_abs_mean": torch.tensor(0.0, device=fast_logits.device),
                    "bev_state_abs_mean": torch.tensor(0.0, device=fast_logits.device),
                },
            }

        rollout_steps = (num_frames - 1) - rollout_start_step
        if max_step_index is not None:
            rollout_steps = min(rollout_steps, max(int(max_step_index), 0))
        if rollout_steps <= 0:
            return {
                "step_logits": fast_logits.new_zeros((0, self.num_classes, *self.grid_size)),
                "step_indices": torch.zeros((0,), device=fast_logits.device, dtype=torch.long),
                "diagnostics": {
                    "delta_scene_abs_mean": torch.tensor(0.0, device=fast_logits.device),
                    "bev_state_abs_mean": torch.tensor(0.0, device=fast_logits.device),
                },
            }

        initial_input, observations, target_times = self._prepare_single_rollout(
            fast_logits=fast_logits,
            slow_logits=slow_logits,
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            rollout_start_step=rollout_start_step,
            rollout_steps=rollout_steps,
        )
        bev_states = cast(
            torch.Tensor,
            self.core(
                initial_input=initial_input,
                observations=observations,
                target_times=target_times,
                return_step_times=False,
            ),
        )
        step_logits = self._decode_bev_states(bev_states)[0].float()
        step_indices = torch.arange(
            rollout_start_step + 1,
            rollout_start_step + 1 + rollout_steps,
            device=fast_logits.device,
            dtype=torch.long,
        )
        return {
            "step_logits": step_logits,
            "step_indices": step_indices,
            "diagnostics": self._diagnostics(bev_states),
        }

    def _forward_single(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: int = 0,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        out = self._forward_single_stepwise_train(
            fast_logits=fast_logits,
            slow_logits=slow_logits,
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            max_step_index=None,
            rollout_start_step=rollout_start_step,
        )
        step_logits = cast(torch.Tensor, out["step_logits"])
        aligned = step_logits[-1] if step_logits.shape[0] > 0 else slow_logits.float()
        return {"aligned": aligned, "diagnostics": cast(dict[str, torch.Tensor], out["diagnostics"])}

    def _measure_decode_ms(self, bev_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps = int(bev_states.shape[1])
        if steps <= 0:
            empty_logits = bev_states.new_zeros((bev_states.shape[0], 0, self.num_classes, *self.grid_size))
            return empty_logits, bev_states.new_zeros((0,), dtype=torch.float32)

        if bev_states.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            logits = self._decode_bev_states(bev_states).float()
            end.record()
            torch.cuda.synchronize(device=bev_states.device)
            total_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            logits = self._decode_bev_states(bev_states).float()
            total_ms = (time.perf_counter() - t0) * 1000.0
        per_step = logits.new_full((steps,), float(total_ms) / max(steps, 1))
        return logits, per_step

    def _forward_single_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: int = 0,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        self._validate_logits(fast_logits, slow_logits)
        num_frames = int(fast_logits.shape[0])
        if rollout_start_step >= num_frames - 1:
            step_indices = torch.tensor([num_frames - 1], device=fast_logits.device, dtype=torch.long)
            zeros = torch.zeros((1,), device=fast_logits.device, dtype=torch.float32)
            return {
                "step_logits": slow_logits.unsqueeze(0).float(),
                "step_time_ms": zeros,
                "step_warp_ms": zeros,
                "step_solver_ms": zeros,
                "step_decode_ms": zeros,
                "step_indices": step_indices,
                "diagnostics": {
                    "delta_scene_abs_mean": torch.tensor(0.0, device=fast_logits.device),
                    "bev_state_abs_mean": torch.tensor(0.0, device=fast_logits.device),
                },
            }

        rollout_steps = (num_frames - 1) - rollout_start_step
        initial_input, observations, target_times = self._prepare_single_rollout(
            fast_logits=fast_logits,
            slow_logits=slow_logits,
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            rollout_start_step=rollout_start_step,
            rollout_steps=rollout_steps,
        )
        bev_states, solver_ms = cast(
            tuple[torch.Tensor, torch.Tensor],
            self.core(
                initial_input=initial_input,
                observations=observations,
                target_times=target_times,
                return_step_times=True,
            ),
        )
        logits, decode_ms = self._measure_decode_ms(bev_states)
        solver_ms = solver_ms.to(device=fast_logits.device, dtype=torch.float32)
        step_warp_ms = torch.zeros_like(solver_ms)
        step_time_ms = step_warp_ms + solver_ms + decode_ms
        step_indices = torch.arange(
            rollout_start_step + 1, num_frames, device=fast_logits.device, dtype=torch.long
        )
        return {
            "step_logits": logits[0],
            "step_time_ms": step_time_ms,
            "step_warp_ms": step_warp_ms,
            "step_solver_ms": solver_ms,
            "step_decode_ms": decode_ms,
            "step_indices": step_indices,
            "diagnostics": self._diagnostics(bev_states),
        }

    def _unsqueeze_inputs(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ):
        if fast_logits.dim() == 5:
            fast_logits = fast_logits.unsqueeze(0)
            slow_logits = slow_logits.unsqueeze(0)
            frame_ego2global = frame_ego2global.unsqueeze(0)
            if frame_timestamps is not None:
                frame_timestamps = frame_timestamps.unsqueeze(0)
            if frame_dt is not None:
                frame_dt = frame_dt.unsqueeze(0)
        return fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt

    def forward(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        mode: str = "default",
        max_step_index: int | None = None,
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt = (
            self._unsqueeze_inputs(
                fast_logits, slow_logits, frame_ego2global, frame_timestamps, frame_dt
            )
        )
        del frame_ego2global  # StreamingFlow baseline 不使用 ego pose。
        if mode == "default":
            return self._forward_batched_default(
                fast_logits, slow_logits, frame_timestamps, frame_dt, rollout_start_step
            )
        if mode == "stepwise_train":
            return self._forward_batched_stepwise_train(
                fast_logits,
                slow_logits,
                frame_timestamps,
                frame_dt,
                max_step_index=max_step_index,
                rollout_start_step=rollout_start_step,
            )
        if mode == "stepwise_eval":
            return self._forward_batched_stepwise_eval(
                fast_logits, slow_logits, frame_timestamps, frame_dt, rollout_start_step
            )
        raise ValueError(f"未知 forward mode: {mode!r}")

    def forward_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        return self.forward(
            fast_logits=fast_logits,
            slow_logits=slow_logits,
            frame_ego2global=frame_ego2global,
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            mode="stepwise_eval",
            rollout_start_step=rollout_start_step,
        )

    def _forward_batched_default(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        aligned_list = []
        diag_list: list[dict[str, torch.Tensor]] = []
        for b in range(fast_logits.shape[0]):
            rss_b = int(rollout_start_step[b].item()) if rollout_start_step is not None else 0
            out = self._forward_single(
                fast_logits=fast_logits[b],
                slow_logits=slow_logits[b],
                frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                frame_dt=frame_dt[b] if frame_dt is not None else None,
                rollout_start_step=rss_b,
            )
            aligned_list.append(cast(torch.Tensor, out["aligned"]))
            diag_list.append(cast(dict[str, torch.Tensor], out["diagnostics"]))
        return {"aligned": torch.stack(aligned_list, dim=0), "diagnostics": diag_list}

    def _forward_batched_stepwise_train(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        max_step_index: int | None = None,
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        step_logits_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        step_indices: torch.Tensor | None = None
        for b in range(fast_logits.shape[0]):
            rss_b = int(rollout_start_step[b].item()) if rollout_start_step is not None else 0
            out = self._forward_single_stepwise_train(
                fast_logits=fast_logits[b],
                slow_logits=slow_logits[b],
                frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                frame_dt=frame_dt[b] if frame_dt is not None else None,
                max_step_index=max_step_index,
                rollout_start_step=rss_b,
            )
            sample_step_indices = cast(torch.Tensor, out["step_indices"])
            if step_indices is None:
                step_indices = sample_step_indices
            elif sample_step_indices.shape != step_indices.shape:
                raise ValueError(
                    f"batch 内 step 数不一致: {sample_step_indices.shape} vs {step_indices.shape}"
                )
            step_logits_list.append(cast(torch.Tensor, out["step_logits"]))
            diag_list.append(cast(dict[str, torch.Tensor], out["diagnostics"]))

        if step_indices is None:
            step_indices = torch.zeros((0,), device=fast_logits.device, dtype=torch.long)
        return {
            "step_logits": torch.stack(step_logits_list, dim=0),
            "step_indices": step_indices,
            "diagnostics": diag_list,
        }

    def _forward_batched_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        rollout_start_step: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        step_logits_list: list[torch.Tensor] = []
        step_time_list: list[torch.Tensor] = []
        step_warp_list: list[torch.Tensor] = []
        step_solver_list: list[torch.Tensor] = []
        step_decode_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        step_indices: torch.Tensor | None = None
        for b in range(fast_logits.shape[0]):
            rss_b = int(rollout_start_step[b].item()) if rollout_start_step is not None else 0
            out = self._forward_single_stepwise_eval(
                fast_logits=fast_logits[b],
                slow_logits=slow_logits[b],
                frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                frame_dt=frame_dt[b] if frame_dt is not None else None,
                rollout_start_step=rss_b,
            )
            sample_step_indices = cast(torch.Tensor, out["step_indices"])
            if step_indices is None:
                step_indices = sample_step_indices
            elif sample_step_indices.shape != step_indices.shape:
                raise ValueError(
                    f"batch 内 step 数不一致: {sample_step_indices.shape} vs {step_indices.shape}"
                )
            step_logits_list.append(cast(torch.Tensor, out["step_logits"]))
            step_time_list.append(cast(torch.Tensor, out["step_time_ms"]))
            step_warp_list.append(cast(torch.Tensor, out["step_warp_ms"]))
            step_solver_list.append(cast(torch.Tensor, out["step_solver_ms"]))
            step_decode_list.append(cast(torch.Tensor, out["step_decode_ms"]))
            diag_list.append(cast(dict[str, torch.Tensor], out["diagnostics"]))

        if step_indices is None:
            step_indices = torch.zeros((0,), device=fast_logits.device, dtype=torch.long)
        return {
            "step_logits": torch.stack(step_logits_list, dim=0),
            "step_time_ms": torch.stack(step_time_list, dim=0),
            "step_warp_ms": torch.stack(step_warp_list, dim=0),
            "step_solver_ms": torch.stack(step_solver_list, dim=0),
            "step_decode_ms": torch.stack(step_decode_list, dim=0),
            "step_indices": step_indices,
            "diagnostics": diag_list,
        }
