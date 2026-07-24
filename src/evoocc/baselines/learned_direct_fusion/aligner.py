"""100×100×16 learned direct fusion aligner。"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from evoocc.baselines.learned_direct_fusion.modules import (
    DirectFusionNet,
    XYDownsampleEncoder,
    XYUpsampleResidualDecoder,
)
from evoocc.data.ego_warp_list import (
    backward_warp_dense_trilinear,
    build_sampling_grid,
    compute_transform_prev_to_curr,
)


def _compute_m_occ(fast_logits: torch.Tensor, free_index: int) -> torch.Tensor:
    """计算 fast occupied confidence，与 EvoOcc 保持一致。"""
    masked = fast_logits.clone()
    masked.narrow(-4, free_index, 1).fill_(float("-inf"))
    max_non_free = masked.amax(dim=-4, keepdim=True)
    free_logit = fast_logits.narrow(-4, free_index, 1)
    return max_non_free - free_logit


class LearnedDirectFusionAligner(nn.Module):
    """从同一个 delayed slow anchor 独立预测各目标时刻。

    本 baseline 不维护递归状态：每个目标时刻都重新 warp 原始 slow feature，
    再与该时刻的 fast feature 做卷积融合。
    """

    def __init__(
        self,
        num_classes: int,
        encoder_in_channels: int,
        free_index: int,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        latent_dim: int = 128,
        fusion_inner_dim: int = 48,
        fusion_body_dilations: Sequence[int] = (1, 3, 5),
        fusion_gn_groups: int = 8,
        decoder_channels: int = 32,
        decoder_init_scale: Optional[float] = 1.0e-6,
        use_fast_residual: bool = True,
        input_grid_size: Tuple[int, int, int] = (200, 200, 16),
        latent_grid_size: Tuple[int, int, int] = (100, 100, 16),
        timestamp_scale: float = 1.0e-6,
    ) -> None:
        super().__init__()
        del timestamp_scale  # direct fusion 不使用物理时间输入。
        self.num_classes = int(num_classes)
        self.encoder_in_channels = int(encoder_in_channels)
        self.free_index = int(free_index)
        self.latent_dim = int(latent_dim)
        self.use_fast_residual = bool(use_fast_residual)
        self.input_grid_size = tuple(int(value) for value in input_grid_size)
        self.latent_grid_size = tuple(int(value) for value in latent_grid_size)

        if self.input_grid_size != (200, 200, 16):
            raise ValueError(
                "learned direct fusion 的输入空间固定为 (200,200,16)，"
                f"当前 {self.input_grid_size}"
            )
        if self.latent_grid_size != (100, 100, 16):
            raise ValueError(
                "learned direct fusion 的演化空间固定为 (100,100,16)，"
                f"当前 {self.latent_grid_size}"
            )

        pc_range_tuple = tuple(float(value) for value in pc_range)
        if len(pc_range_tuple) != 6:
            raise ValueError(f"pc_range 必须长度为 6，当前 {pc_range_tuple}")
        self.pc_range = cast(
            Tuple[float, float, float, float, float, float],
            pc_range_tuple,
        )
        voxel_size_tuple = tuple(float(value) for value in voxel_size)
        if len(voxel_size_tuple) != 3:
            raise ValueError(f"voxel_size 必须长度为 3，当前 {voxel_size_tuple}")
        self.voxel_size = cast(Tuple[float, float, float], voxel_size_tuple)
        self.latent_voxel_size = (
            2.0 * self.voxel_size[0],
            2.0 * self.voxel_size[1],
            self.voxel_size[2],
        )

        self.fast_encoder = XYDownsampleEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=self.latent_dim,
            gn_groups=fusion_gn_groups,
        )
        self.slow_encoder = XYDownsampleEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=self.latent_dim,
            gn_groups=fusion_gn_groups,
        )
        self.fusion = DirectFusionNet(
            feature_dim=self.latent_dim,
            inner_dim=int(fusion_inner_dim),
            body_dilations=fusion_body_dilations,
            gn_groups=fusion_gn_groups,
        )
        self.decoder = XYUpsampleResidualDecoder(
            in_channels=self.latent_dim,
            decoder_channels=int(decoder_channels),
            out_channels=self.num_classes,
            init_scale=decoder_init_scale,
            gn_groups=fusion_gn_groups,
        )
        self._fast_kl_active = False

    def _validate_sample_shapes(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
    ) -> None:
        expected_fast_tail = (
            self.encoder_in_channels,
            *self.input_grid_size,
        )
        if tuple(fast_logits.shape[1:]) != expected_fast_tail:
            raise ValueError(
                "fast logits 形状必须为 "
                f"(T,{expected_fast_tail[0]},200,200,16)，当前 {tuple(fast_logits.shape)}"
            )
        if tuple(slow_logits.shape) != expected_fast_tail:
            raise ValueError(
                f"slow logits 形状必须为 {expected_fast_tail}，当前 {tuple(slow_logits.shape)}"
            )

    def _encode_slow_anchor(self, slow_logits: torch.Tensor) -> torch.Tensor:
        feature = self.slow_encoder(slow_logits.unsqueeze(0))[0]
        if tuple(feature.shape[1:]) != self.latent_grid_size:
            raise RuntimeError(
                "slow encoder 未产生固定的 100×100×16 latent，"
                f"当前 {tuple(feature.shape)}"
            )
        return feature

    def _encode_target_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        feature = self.fast_encoder(fast_logits)
        if tuple(feature.shape[2:]) != self.latent_grid_size:
            raise RuntimeError(
                "fast encoder 未产生固定的 100×100×16 latent，"
                f"当前 {tuple(feature.shape)}"
            )
        return feature

    def _warp_anchor_to_target(
        self,
        slow_anchor: torch.Tensor,
        pose_anchor: torch.Tensor,
        pose_target: torch.Tensor,
    ) -> torch.Tensor:
        """始终从原始 slow anchor 直接 warp 到目标时刻。"""
        transform = compute_transform_prev_to_curr(
            pose_prev_ego2global=pose_anchor,
            pose_curr_ego2global=pose_target,
        )
        grid = build_sampling_grid(
            transform_prev_to_curr=transform,
            spatial_shape_xyz=self.latent_grid_size,
            pc_range=self.pc_range,
            voxel_size=self.latent_voxel_size,
        )
        return backward_warp_dense_trilinear(
            dense_prev_feat=slow_anchor,
            transform_prev_to_curr=None,
            spatial_shape_xyz=self.latent_grid_size,
            pc_range=self.pc_range,
            voxel_size=self.latent_voxel_size,
            padding_mode="border",
            prebuilt_grid=grid,
        )

    def _decode_targets(
        self,
        fused: torch.Tensor,
        target_fast_logits: torch.Tensor,
    ) -> torch.Tensor:
        logits_delta = self.decoder(
            fused,
            output_shape_xyz=self.input_grid_size,
        )
        if self.use_fast_residual:
            return logits_delta + target_fast_logits
        return logits_delta

    def _compute_fast_kl(
        self,
        fast_logits: torch.Tensor,
        aligned_logits: torch.Tensor,
    ) -> torch.Tensor:
        aligned_f = aligned_logits.float()
        fast_f = fast_logits.float()
        weights = _compute_m_occ(fast_f, self.free_index).clamp(min=0.0)
        log_fast = F.log_softmax(fast_f, dim=1)
        log_aligned = F.log_softmax(aligned_f, dim=1)
        kl = F.kl_div(
            log_fast,
            log_aligned,
            log_target=True,
            reduction="none",
        ).sum(dim=1, keepdim=True)
        return (weights * kl).mean()

    @staticmethod
    def _target_indices(
        num_frames: int,
        rollout_start_step: int,
        max_step_index: int | None,
    ) -> list[int]:
        rollout_steps = max(num_frames - 1 - rollout_start_step, 0)
        if max_step_index is not None:
            rollout_steps = min(rollout_steps, max(int(max_step_index), 0))
        return list(
            range(
                rollout_start_step + 1,
                rollout_start_step + 1 + rollout_steps,
            )
        )

    def _predict_targets(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        target_indices: Sequence[int],
        anchor_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """批量预测目标时刻；各目标共享编码后的 slow anchor，但不存在递归。"""
        self._validate_sample_shapes(fast_logits, slow_logits)
        if not target_indices:
            empty = fast_logits.new_zeros(
                (0, self.num_classes, *self.input_grid_size)
            )
            return empty, fast_logits.new_tensor(0.0)

        slow_anchor = self._encode_slow_anchor(slow_logits)
        target_index_tensor = torch.tensor(
            list(target_indices),
            device=fast_logits.device,
            dtype=torch.long,
        )
        target_fast_logits = fast_logits.index_select(0, target_index_tensor)
        target_fast_features = self._encode_target_fast(target_fast_logits)

        warped_slow = torch.stack(
            [
                self._warp_anchor_to_target(
                    slow_anchor=slow_anchor,
                    pose_anchor=frame_ego2global[anchor_index],
                    pose_target=frame_ego2global[target_index],
                )
                for target_index in target_indices
            ],
            dim=0,
        )
        fused = self.fusion(warped_slow, target_fast_features)
        refined = self._decode_targets(fused, target_fast_logits).float()
        fusion_magnitude = (fused - warped_slow).abs().mean()
        return refined, fusion_magnitude

    def _forward_single(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        rollout_start_step: int,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        num_frames = int(fast_logits.shape[0])
        if rollout_start_step >= num_frames - 1:
            return {
                "aligned": slow_logits.float(),
                "diagnostics": {
                    "fusion_abs_mean": fast_logits.new_tensor(0.0),
                },
            }
        target_index = num_frames - 1
        logits, magnitude = self._predict_targets(
            fast_logits=fast_logits,
            slow_logits=slow_logits,
            frame_ego2global=frame_ego2global,
            target_indices=[target_index],
            anchor_index=rollout_start_step,
        )
        return {
            "aligned": logits[0],
            "diagnostics": {"fusion_abs_mean": magnitude.float()},
        }

    def _forward_single_stepwise_train(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        max_step_index: int | None,
        rollout_start_step: int,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        num_frames = int(fast_logits.shape[0])
        if rollout_start_step >= num_frames - 1:
            return {
                "step_logits": slow_logits.unsqueeze(0).float(),
                "step_indices": torch.tensor(
                    [num_frames - 1],
                    device=fast_logits.device,
                    dtype=torch.long,
                ),
                "diagnostics": {
                    "fusion_abs_mean": fast_logits.new_tensor(0.0),
                },
            }

        target_indices = self._target_indices(
            num_frames=num_frames,
            rollout_start_step=rollout_start_step,
            max_step_index=max_step_index,
        )
        logits, magnitude = self._predict_targets(
            fast_logits=fast_logits,
            slow_logits=slow_logits,
            frame_ego2global=frame_ego2global,
            target_indices=target_indices,
            anchor_index=rollout_start_step,
        )
        output: Dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "step_logits": logits,
            "step_indices": torch.tensor(
                target_indices,
                device=fast_logits.device,
                dtype=torch.long,
            ),
            "diagnostics": {"fusion_abs_mean": magnitude.float()},
        }
        if self._fast_kl_active and self.use_fast_residual and target_indices:
            index_tensor = torch.tensor(
                target_indices,
                device=fast_logits.device,
                dtype=torch.long,
            )
            output["fast_kl"] = self._compute_fast_kl(
                fast_logits=fast_logits.index_select(0, index_tensor).detach(),
                aligned_logits=logits,
            )
        return output

    def _forward_single_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        rollout_start_step: int,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        num_frames = int(fast_logits.shape[0])
        if rollout_start_step >= num_frames - 1:
            zero = torch.zeros(1, device=fast_logits.device, dtype=torch.float32)
            return {
                "step_logits": slow_logits.unsqueeze(0).float(),
                "step_indices": torch.tensor(
                    [num_frames - 1],
                    device=fast_logits.device,
                    dtype=torch.long,
                ),
                "step_time_ms": zero,
                "step_warp_ms": zero,
                "step_solver_ms": zero,
                "step_decode_ms": zero,
                "diagnostics": {
                    "fusion_abs_mean": fast_logits.new_tensor(0.0),
                },
            }

        self._validate_sample_shapes(fast_logits, slow_logits)
        target_indices = self._target_indices(
            num_frames=num_frames,
            rollout_start_step=rollout_start_step,
            max_step_index=None,
        )
        slow_anchor = self._encode_slow_anchor(slow_logits)
        target_tensor = torch.tensor(
            target_indices,
            device=fast_logits.device,
            dtype=torch.long,
        )
        target_fast_logits = fast_logits.index_select(0, target_tensor)
        target_fast_features = self._encode_target_fast(target_fast_logits)

        step_logits: list[torch.Tensor] = []
        warp_times: list[float] = []
        fusion_times: list[float] = []
        decode_times: list[float] = []
        magnitudes: list[torch.Tensor] = []
        use_cuda_events = fast_logits.is_cuda

        for local_index, target_index in enumerate(target_indices):
            if use_cuda_events:
                warp_start = torch.cuda.Event(enable_timing=True)
                warp_end = torch.cuda.Event(enable_timing=True)
                fusion_start = torch.cuda.Event(enable_timing=True)
                fusion_end = torch.cuda.Event(enable_timing=True)
                decode_start = torch.cuda.Event(enable_timing=True)
                decode_end = torch.cuda.Event(enable_timing=True)

                warp_start.record()
                warped = self._warp_anchor_to_target(
                    slow_anchor,
                    frame_ego2global[rollout_start_step],
                    frame_ego2global[target_index],
                )
                warp_end.record()
                fusion_start.record()
                fused = self.fusion(
                    warped.unsqueeze(0),
                    target_fast_features[local_index : local_index + 1],
                )
                fusion_end.record()
                decode_start.record()
                refined = self._decode_targets(
                    fused,
                    target_fast_logits[local_index : local_index + 1],
                )
                decode_end.record()
                torch.cuda.synchronize(fast_logits.device)
                warp_times.append(warp_start.elapsed_time(warp_end))
                fusion_times.append(fusion_start.elapsed_time(fusion_end))
                decode_times.append(decode_start.elapsed_time(decode_end))
            else:
                start = time.perf_counter()
                warped = self._warp_anchor_to_target(
                    slow_anchor,
                    frame_ego2global[rollout_start_step],
                    frame_ego2global[target_index],
                )
                after_warp = time.perf_counter()
                fused = self.fusion(
                    warped.unsqueeze(0),
                    target_fast_features[local_index : local_index + 1],
                )
                after_fusion = time.perf_counter()
                refined = self._decode_targets(
                    fused,
                    target_fast_logits[local_index : local_index + 1],
                )
                after_decode = time.perf_counter()
                warp_times.append((after_warp - start) * 1000.0)
                fusion_times.append((after_fusion - after_warp) * 1000.0)
                decode_times.append((after_decode - after_fusion) * 1000.0)

            step_logits.append(refined[0].float())
            magnitudes.append((fused[0] - warped).abs().mean())

        warp_tensor = torch.tensor(
            warp_times,
            device=fast_logits.device,
            dtype=torch.float32,
        )
        fusion_tensor = torch.tensor(
            fusion_times,
            device=fast_logits.device,
            dtype=torch.float32,
        )
        decode_tensor = torch.tensor(
            decode_times,
            device=fast_logits.device,
            dtype=torch.float32,
        )
        magnitude = (
            torch.stack(magnitudes).mean()
            if magnitudes
            else fast_logits.new_tensor(0.0)
        )
        return {
            "step_logits": torch.stack(step_logits, dim=0),
            "step_indices": target_tensor,
            "step_time_ms": warp_tensor + fusion_tensor + decode_tensor,
            "step_warp_ms": warp_tensor,
            # 复用现有评估字段名；这里表示 direct fusion 耗时。
            "step_solver_ms": fusion_tensor,
            "step_decode_ms": decode_tensor,
            "diagnostics": {"fusion_abs_mean": magnitude.float()},
        }

    @staticmethod
    def _unsqueeze_inputs(
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if fast_logits.dim() == 5:
            fast_logits = fast_logits.unsqueeze(0)
            slow_logits = slow_logits.unsqueeze(0)
            frame_ego2global = frame_ego2global.unsqueeze(0)
            if frame_timestamps is not None:
                frame_timestamps = frame_timestamps.unsqueeze(0)
            if frame_dt is not None:
                frame_dt = frame_dt.unsqueeze(0)
        return (
            fast_logits,
            slow_logits,
            frame_ego2global,
            frame_timestamps,
            frame_dt,
        )

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
        del frame_timestamps, frame_dt
        fast_logits, slow_logits, frame_ego2global, _, _ = self._unsqueeze_inputs(
            fast_logits,
            slow_logits,
            frame_ego2global,
            None,
            None,
        )
        if mode == "default":
            return self._forward_batched_default(
                fast_logits,
                slow_logits,
                frame_ego2global,
                rollout_start_step,
            )
        if mode == "stepwise_train":
            return self._forward_batched_stepwise_train(
                fast_logits,
                slow_logits,
                frame_ego2global,
                max_step_index,
                rollout_start_step,
            )
        if mode == "stepwise_eval":
            return self._forward_batched_stepwise_eval(
                fast_logits,
                slow_logits,
                frame_ego2global,
                rollout_start_step,
            )
        raise ValueError(
            f"未知 forward mode: {mode!r}，可选 'default'/'stepwise_train'/'stepwise_eval'"
        )

    @staticmethod
    def _rollout_start(
        rollout_start_step: torch.Tensor | None,
        batch_index: int,
    ) -> int:
        if rollout_start_step is None:
            return 0
        return int(rollout_start_step[batch_index].item())

    def _forward_batched_default(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        rollout_start_step: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        aligned: list[torch.Tensor] = []
        diagnostics: list[dict[str, torch.Tensor]] = []
        for batch_index in range(fast_logits.shape[0]):
            output = self._forward_single(
                fast_logits[batch_index],
                slow_logits[batch_index],
                frame_ego2global[batch_index],
                self._rollout_start(rollout_start_step, batch_index),
            )
            aligned.append(cast(torch.Tensor, output["aligned"]))
            diagnostics.append(
                cast(dict[str, torch.Tensor], output["diagnostics"])
            )
        return {
            "aligned": torch.stack(aligned, dim=0),
            "diagnostics": diagnostics,
        }

    def _forward_batched_stepwise_train(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        max_step_index: int | None,
        rollout_start_step: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        step_logits: list[torch.Tensor] = []
        diagnostics: list[dict[str, torch.Tensor]] = []
        fast_kl: list[torch.Tensor] = []
        step_indices: torch.Tensor | None = None
        for batch_index in range(fast_logits.shape[0]):
            output = self._forward_single_stepwise_train(
                fast_logits[batch_index],
                slow_logits[batch_index],
                frame_ego2global[batch_index],
                max_step_index,
                self._rollout_start(rollout_start_step, batch_index),
            )
            sample_indices = cast(torch.Tensor, output["step_indices"])
            if step_indices is None:
                step_indices = sample_indices
            elif sample_indices.shape != step_indices.shape:
                raise ValueError("batch 内 step 数不一致，无法 stack")
            step_logits.append(cast(torch.Tensor, output["step_logits"]))
            diagnostics.append(
                cast(dict[str, torch.Tensor], output["diagnostics"])
            )
            if "fast_kl" in output:
                fast_kl.append(cast(torch.Tensor, output["fast_kl"]))
        if step_indices is None:
            step_indices = torch.zeros(
                0,
                device=fast_logits.device,
                dtype=torch.long,
            )
        result: Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "step_logits": torch.stack(step_logits, dim=0),
            "step_indices": step_indices,
            "diagnostics": diagnostics,
        }
        if fast_kl:
            result["fast_kl"] = torch.stack(fast_kl).mean()
        return result

    def _forward_batched_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        rollout_start_step: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        tensor_keys = (
            "step_logits",
            "step_time_ms",
            "step_warp_ms",
            "step_solver_ms",
            "step_decode_ms",
        )
        values: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_keys}
        diagnostics: list[dict[str, torch.Tensor]] = []
        step_indices: torch.Tensor | None = None
        for batch_index in range(fast_logits.shape[0]):
            output = self._forward_single_stepwise_eval(
                fast_logits[batch_index],
                slow_logits[batch_index],
                frame_ego2global[batch_index],
                self._rollout_start(rollout_start_step, batch_index),
            )
            sample_indices = cast(torch.Tensor, output["step_indices"])
            if step_indices is None:
                step_indices = sample_indices
            elif sample_indices.shape != step_indices.shape:
                raise ValueError("batch 内 step 数不一致，无法 stack")
            for key in tensor_keys:
                values[key].append(cast(torch.Tensor, output[key]))
            diagnostics.append(
                cast(dict[str, torch.Tensor], output["diagnostics"])
            )
        if step_indices is None:
            step_indices = torch.zeros(
                0,
                device=fast_logits.device,
                dtype=torch.long,
            )
        result: Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            key: torch.stack(items, dim=0)
            for key, items in values.items()
        }
        result["step_indices"] = step_indices
        result["diagnostics"] = diagnostics
        return result
