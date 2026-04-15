"""Online NCDE 对齐主模型——纯 Dense 架构版本。

完全抛弃稀疏体素列表重整逻辑，全程使用稠密特征进行计算。
核心步骤：
  1. Dense Encoder 编码快慢系统。
  2. 每步计算 ego-motion 并进行 Dense Trilinear Backward Warp。
  3. Warp 后直接在全图进行 Dense Conv3d 动力学更新。
  4. 最终全图特征送入 Dense Decoder 预测残差。
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from online_ncde.data.ego_warp_list import (
    backward_warp_dense_trilinear,
    build_sampling_grid,
    compute_transform_prev_to_curr,
)
from online_ncde.data.scene_delta import build_scene_delta_ctrl
from online_ncde.data.time_series import compute_segment_dt, cumulative_tau
from online_ncde.models.decoder import DenseDecoder
from online_ncde.models.encoder import DenseEncoder
from online_ncde.models.func_g import FuncG
from online_ncde.models.heads import CtrlProjector
from online_ncde.models.solver_heun import HeunSolver


class OnlineNcdeAligner(nn.Module):
    """编码 + Dense Warp + Dense Dynamics + 解码。

    状态表示：
      - Z_dense : (C, X, Y, Z)，稠密隐藏状态缓存，整个体素网格均有值

    输出形式（残差）：
      - aligned = decoder(Z_dense_T) + fast_logits[-1]
      - decoder 初始化为 0，初始先验即为 fast_logits（不扰动快系统输出）
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        hidden_dim: int,
        encoder_in_channels: int,
        free_index: int,
        pc_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        decoder_init_scale: float = 0.0,
        use_fast_residual: bool = True,
        func_g_inner_dim: int = 32,
        func_g_body_dilations: tuple[int, ...] = (1, 2, 3),
        func_g_gn_groups: int = 8,
        timestamp_scale: float = 1.0e-6,
        amp_fp16: bool = False,
    ) -> None:
        super().__init__()
        self.amp_fp16 = bool(amp_fp16)
        self.use_fast_residual = bool(use_fast_residual)
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder_in_channels = int(encoder_in_channels)
        if self.hidden_dim != self.feat_dim:
            raise ValueError(
                f"hidden_dim 必须与 feat_dim 相同，当前 hidden_dim={self.hidden_dim}, "
                f"feat_dim={self.feat_dim}。请在配置中只设置一个值并保持一致。"
            )
        self.free_index = int(free_index)

        pc_range_tuple = tuple(pc_range)
        if len(pc_range_tuple) != 6:
            raise ValueError(f"pc_range 必须长度为 6，当前: {pc_range_tuple}")
        self.pc_range = cast(Tuple[float, float, float, float, float, float], pc_range_tuple)
        self.voxel_size = tuple(voxel_size)
        self.timestamp_scale = float(timestamp_scale)

        # 默认 hidden state 采用 100x100x16：encoder 只在 XY 下采样 2 倍。
        self.voxel_size_ds = (
            self.voxel_size[0] * 2.0,
            self.voxel_size[1] * 2.0,
            self.voxel_size[2],
        )

        self.fast_encoder = DenseEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        self.slow_encoder = DenseEncoder(
            in_channels=self.encoder_in_channels,
            out_channels=feat_dim,
        )
        self.ctrl_proj = CtrlProjector(feat_dim + 1, hidden_dim)
        self.func_g = FuncG(
            in_channels=hidden_dim + feat_dim,
            hidden_channels=hidden_dim,
            inner_dim=func_g_inner_dim,
            body_dilations=func_g_body_dilations,
            gn_groups=func_g_gn_groups,
        )
        self.solver = HeunSolver(func_g=self.func_g, ctrl_proj=self.ctrl_proj)
        self.decoder = DenseDecoder(
            in_channels=hidden_dim,
            out_channels=num_classes,
            init_scale=decoder_init_scale,
        )

    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        """编码快系统序列（仅 logits 18 通道），输出 dense (T, C_f, Xd, Yd, Zd)。"""
        num_frames = fast_logits.shape[0]
        fast_feat = self.fast_encoder(fast_logits)
        return fast_feat.reshape(num_frames, fast_feat.shape[1], *fast_feat.shape[2:])

    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        """编码慢系统输入（仅 logits 18 通道），输出 dense (C_f, Xd, Yd, Zd)。"""
        return self.slow_encoder(slow_logits.unsqueeze(0))[0]

    def _downsample_logits_xy(self, logits: torch.Tensor, num_down: int) -> torch.Tensor:
        """对 logits 做 XY 下采样。支持 (C,X,Y,Z) 或 (T,C,X,Y,Z)。"""
        if num_down <= 0:
            return logits
        if logits.dim() == 4:
            zyx = logits.permute(0, 3, 2, 1).unsqueeze(0)
            for _ in range(num_down):
                zyx = F.max_pool3d(zyx, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            return zyx.squeeze(0).permute(0, 3, 2, 1).contiguous()
        if logits.dim() == 5:
            t, c = logits.shape[:2]
            zyx = logits.permute(0, 1, 4, 3, 2).reshape(
                t * c, 1, logits.shape[4], logits.shape[3], logits.shape[2]
            )
            for _ in range(num_down):
                zyx = F.max_pool3d(zyx, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            x_new = zyx.shape[-1]
            y_new = zyx.shape[-2]
            z_new = zyx.shape[-3]
            out = zyx.reshape(t, c, z_new, y_new, x_new).permute(0, 1, 4, 3, 2).contiguous()
            return out
        raise ValueError(f"logits 维度不支持: {logits.shape}")

    def _decode_dense_state(self, z_dense: torch.Tensor) -> torch.Tensor:
        """将稠密隐藏状态解码为残差 logits，输出 (C, X, Y, Z)。"""
        z_tensor = z_dense.unsqueeze(0)  # (1, C, X, Y, Z)
        z_tensor = z_tensor.permute(0, 1, 4, 3, 2).contiguous()  # (1, C, Z, Y, X)
        out_dense = self.decoder(z_tensor)
        return out_dense.permute(0, 1, 4, 3, 2).contiguous()[0]

    def _forward_single(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """处理单样本（不含 batch 维），执行纯 Dense 流程。"""
        num_frames = fast_logits.shape[0]

        # 全图 dense 编码
        fast_feat = self._encode_fast(fast_logits)
        slow_feat = self._encode_slow(slow_logits)

        spatial_shape_xyz = (
            int(fast_feat.shape[2]),
            int(fast_feat.shape[3]),
            int(fast_feat.shape[4]),
        )
        pc_range_6 = cast(Tuple[float, float, float, float, float, float], self.pc_range)
        voxel_size_3 = cast(Tuple[float, float, float], self.voxel_size_ds)

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)
        tau = cumulative_tau(dt).to(device=fast_logits.device)

        # 初始稠密隐藏状态：可选残差初始化 / 纯慢系统初始化
        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[0]
        else:
            Z_dense = slow_feat

        delta_mag_sum = torch.tensor(0.0, device=fast_logits.device)
        delta_mag_cnt = torch.tensor(0.0, device=fast_logits.device)

        for k in range(num_frames - 1):
            # 每步现构建 grid，循环内两次 warp 复用同一个 grid，避免重复构建
            transform = compute_transform_prev_to_curr(
                pose_prev_ego2global=frame_ego2global[k],
                pose_curr_ego2global=frame_ego2global[k + 1],
            )
            grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)

            # 1. 稠密 backward trilinear warp，两次复用同一个 grid
            Z_warp_dense = backward_warp_dense_trilinear(
                dense_prev_feat=Z_dense,
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_prev_adv = backward_warp_dense_trilinear(
                dense_prev_feat=fast_feat[k],
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_t = fast_feat[k + 1]

            # 2. 全图 Dense 控制增量与 Heun 求解
            delta_info = build_scene_delta_ctrl(
                fast_curr=f_t,
                fast_prev_adv=f_prev_adv,
                tau_curr=tau[k + 1],
                tau_prev=tau[k],
            )
            Z_dense, delta_scene = self.solver.step(
                h_adv=Z_warp_dense,
                f_prev_adv=f_prev_adv,
                f_t=f_t,
                delta_ctrl=delta_info["delta_ctrl"],
            )

            delta_mag_sum = delta_mag_sum + delta_scene.abs().mean()
            delta_mag_cnt = delta_mag_cnt + 1.0

        logits_delta = self._decode_dense_state(Z_dense)
        if self.use_fast_residual:
            logits = logits_delta + fast_logits[-1]
        else:
            logits = logits_delta

        diagnostics = {
            "delta_scene_abs_mean": delta_mag_sum / delta_mag_cnt.clamp_min(1.0),
        }
        return {"aligned": logits.float(), "diagnostics": {k: v.float() for k, v in diagnostics.items()}}

    def _forward_single_stepwise_train(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        max_step_index: int | None = None,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """逐步训练单样本：每步更新后解码并返回全时刻 logits，不包含计时逻辑。"""
        num_frames = fast_logits.shape[0]
        rollout_steps = num_frames - 1
        if max_step_index is not None:
            rollout_steps = min(rollout_steps, max(int(max_step_index), 0))

        fast_feat = self._encode_fast(fast_logits)
        slow_feat = self._encode_slow(slow_logits)
        spatial_shape_xyz = (
            int(fast_feat.shape[2]),
            int(fast_feat.shape[3]),
            int(fast_feat.shape[4]),
        )
        pc_range_6 = cast(Tuple[float, float, float, float, float, float], self.pc_range)
        voxel_size_3 = cast(Tuple[float, float, float], self.voxel_size_ds)

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)
        tau = cumulative_tau(dt).to(device=fast_logits.device)

        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[0]
        else:
            Z_dense = slow_feat

        delta_mag_sum = torch.tensor(0.0, device=fast_logits.device)
        delta_mag_cnt = torch.tensor(0.0, device=fast_logits.device)
        step_logits_list: list[torch.Tensor] = []

        for k in range(rollout_steps):
            transform = compute_transform_prev_to_curr(
                pose_prev_ego2global=frame_ego2global[k],
                pose_curr_ego2global=frame_ego2global[k + 1],
            )
            grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)

            Z_warp_dense = backward_warp_dense_trilinear(
                dense_prev_feat=Z_dense,
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_prev_adv = backward_warp_dense_trilinear(
                dense_prev_feat=fast_feat[k],
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_t = fast_feat[k + 1]

            delta_info = build_scene_delta_ctrl(
                fast_curr=f_t,
                fast_prev_adv=f_prev_adv,
                tau_curr=tau[k + 1],
                tau_prev=tau[k],
            )
            Z_dense, delta_scene = self.solver.step(
                h_adv=Z_warp_dense,
                f_prev_adv=f_prev_adv,
                f_t=f_t,
                delta_ctrl=delta_info["delta_ctrl"],
            )

            logits_delta = self._decode_dense_state(Z_dense)
            if self.use_fast_residual:
                logits_now = logits_delta + fast_logits[k + 1]
            else:
                logits_now = logits_delta
            step_logits_list.append(logits_now.float())

            delta_mag_sum = delta_mag_sum + delta_scene.abs().mean()
            delta_mag_cnt = delta_mag_cnt + 1.0

        if step_logits_list:
            step_logits = torch.stack(step_logits_list, dim=0)
        else:
            step_logits = fast_logits.new_zeros(
                (0, self.num_classes, fast_logits.shape[2], fast_logits.shape[3], fast_logits.shape[4])
            )
        step_indices = torch.arange(1, rollout_steps + 1, device=fast_logits.device, dtype=torch.long)
        diagnostics = {
            "delta_scene_abs_mean": delta_mag_sum / delta_mag_cnt.clamp_min(1.0),
        }
        return {
            "step_logits": step_logits,
            "step_indices": step_indices,
            "diagnostics": {k: v.float() for k, v in diagnostics.items()},
        }

    def _forward_single_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """逐步评估单样本：每步更新后都做解码并和当前 fast logits 残差融合。"""
        num_frames = fast_logits.shape[0]

        fast_feat = self._encode_fast(fast_logits)
        slow_feat = self._encode_slow(slow_logits)
        spatial_shape_xyz = (
            int(fast_feat.shape[2]),
            int(fast_feat.shape[3]),
            int(fast_feat.shape[4]),
        )
        pc_range_6 = cast(Tuple[float, float, float, float, float, float], self.pc_range)
        voxel_size_3 = cast(Tuple[float, float, float], self.voxel_size_ds)

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps,
            frame_dt=frame_dt,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=fast_logits.device)
        tau = cumulative_tau(dt).to(device=fast_logits.device)

        if self.use_fast_residual:
            Z_dense = slow_feat - fast_feat[0]
        else:
            Z_dense = slow_feat

        delta_mag_sum = torch.tensor(0.0, device=fast_logits.device)
        delta_mag_cnt = torch.tensor(0.0, device=fast_logits.device)

        step_logits_list: list[torch.Tensor] = []
        step_time_ms_values: list[float] = []
        step_time_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        use_cuda_timing = fast_logits.is_cuda

        for k in range(num_frames - 1):
            if use_cuda_timing:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()

            transform = compute_transform_prev_to_curr(
                pose_prev_ego2global=frame_ego2global[k],
                pose_curr_ego2global=frame_ego2global[k + 1],
            )
            grid = build_sampling_grid(transform, spatial_shape_xyz, pc_range_6, voxel_size_3)

            Z_warp_dense = backward_warp_dense_trilinear(
                dense_prev_feat=Z_dense,
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_prev_adv = backward_warp_dense_trilinear(
                dense_prev_feat=fast_feat[k],
                transform_prev_to_curr=None,
                spatial_shape_xyz=spatial_shape_xyz,
                pc_range=pc_range_6,
                voxel_size=voxel_size_3,
                padding_mode="border",
                prebuilt_grid=grid,
            )
            f_t = fast_feat[k + 1]

            delta_info = build_scene_delta_ctrl(
                fast_curr=f_t,
                fast_prev_adv=f_prev_adv,
                tau_curr=tau[k + 1],
                tau_prev=tau[k],
            )
            Z_dense, delta_scene = self.solver.step(
                h_adv=Z_warp_dense,
                f_prev_adv=f_prev_adv,
                f_t=f_t,
                delta_ctrl=delta_info["delta_ctrl"],
            )

            logits_delta = self._decode_dense_state(Z_dense)
            if self.use_fast_residual:
                logits_now = logits_delta + fast_logits[k + 1]
            else:
                logits_now = logits_delta
            step_logits_list.append(logits_now.float())

            delta_mag_sum = delta_mag_sum + delta_scene.abs().mean()
            delta_mag_cnt = delta_mag_cnt + 1.0

            if use_cuda_timing:
                end_event.record()
                step_time_events.append((start_event, end_event))
            else:
                step_time_ms_values.append((time.perf_counter() - start_time) * 1000.0)

        if use_cuda_timing:
            torch.cuda.synchronize(device=fast_logits.device)
            step_time_ms_values = [s.elapsed_time(e) for s, e in step_time_events]

        if step_logits_list:
            step_logits = torch.stack(step_logits_list, dim=0)
        else:
            step_logits = fast_logits.new_zeros(
                (0, self.num_classes, fast_logits.shape[2], fast_logits.shape[3], fast_logits.shape[4])
            )

        step_indices = torch.arange(1, num_frames, device=fast_logits.device, dtype=torch.long)
        step_time_ms = torch.tensor(step_time_ms_values, device=fast_logits.device, dtype=torch.float32)
        diagnostics = {
            "delta_scene_abs_mean": delta_mag_sum / delta_mag_cnt.clamp_min(1.0),
        }
        return {
            "step_logits": step_logits,
            "step_time_ms": step_time_ms,
            "step_indices": step_indices,
            "diagnostics": {k: v.float() for k, v in diagnostics.items()},
        }

    def forward(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if fast_logits.dim() == 5:
            fast_logits = fast_logits.unsqueeze(0)
            slow_logits = slow_logits.unsqueeze(0)
            frame_ego2global = frame_ego2global.unsqueeze(0)
            if frame_timestamps is not None:
                frame_timestamps = frame_timestamps.unsqueeze(0)
            if frame_dt is not None:
                frame_dt = frame_dt.unsqueeze(0)

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if self.amp_fp16 else nullcontext()
        aligned_list = []
        diag_list = []
        with amp_ctx:
            for b in range(fast_logits.shape[0]):
                out = self._forward_single(
                    fast_logits=fast_logits[b],
                    slow_logits=slow_logits[b],
                    frame_ego2global=frame_ego2global[b],
                    frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                    frame_dt=frame_dt[b] if frame_dt is not None else None,
                )
                aligned_list.append(out["aligned"])
                diag_list.append(out["diagnostics"])
        aligned = torch.stack(aligned_list, dim=0)
        return {"aligned": aligned, "diagnostics": diag_list}

    def forward_stepwise_eval(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """评估接口：返回每个循环步的 logits 与耗时。"""
        if fast_logits.dim() == 5:
            fast_logits = fast_logits.unsqueeze(0)
            slow_logits = slow_logits.unsqueeze(0)
            frame_ego2global = frame_ego2global.unsqueeze(0)
            if frame_timestamps is not None:
                frame_timestamps = frame_timestamps.unsqueeze(0)
            if frame_dt is not None:
                frame_dt = frame_dt.unsqueeze(0)

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if self.amp_fp16 else nullcontext()
        step_logits_list: list[torch.Tensor] = []
        step_time_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        step_indices: torch.Tensor | None = None
        with amp_ctx:
            for b in range(fast_logits.shape[0]):
                out = self._forward_single_stepwise_eval(
                    fast_logits=fast_logits[b],
                    slow_logits=slow_logits[b],
                    frame_ego2global=frame_ego2global[b],
                    frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                    frame_dt=frame_dt[b] if frame_dt is not None else None,
                )
                sample_step_logits = cast(torch.Tensor, out["step_logits"])
                sample_step_time = cast(torch.Tensor, out["step_time_ms"])
                sample_step_indices = cast(torch.Tensor, out["step_indices"])
                if step_indices is None:
                    step_indices = sample_step_indices
                elif sample_step_indices.shape != step_indices.shape:
                    raise ValueError(
                        f"batch 内 step 数不一致: {sample_step_indices.shape} vs {step_indices.shape}"
                    )

                step_logits_list.append(sample_step_logits)
                step_time_list.append(sample_step_time)
                diag_list.append(cast(dict[str, torch.Tensor], out["diagnostics"]))

        if step_indices is None:
            step_indices = torch.zeros((0,), dtype=torch.long, device=fast_logits.device)
        step_logits = torch.stack(step_logits_list, dim=0)
        step_time_ms = torch.stack(step_time_list, dim=0)
        return {
            "step_logits": step_logits,
            "step_time_ms": step_time_ms,
            "step_indices": step_indices,
            "diagnostics": diag_list,
        }

    def forward_stepwise_train(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        frame_ego2global: torch.Tensor,
        frame_timestamps: torch.Tensor | None,
        frame_dt: torch.Tensor | None,
        max_step_index: int | None = None,
    ) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """训练接口：返回逐步 logits（完整 BPTT），不做计时。"""
        if fast_logits.dim() == 5:
            fast_logits = fast_logits.unsqueeze(0)
            slow_logits = slow_logits.unsqueeze(0)
            frame_ego2global = frame_ego2global.unsqueeze(0)
            if frame_timestamps is not None:
                frame_timestamps = frame_timestamps.unsqueeze(0)
            if frame_dt is not None:
                frame_dt = frame_dt.unsqueeze(0)

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if self.amp_fp16 else nullcontext()
        step_logits_list: list[torch.Tensor] = []
        diag_list: list[dict[str, torch.Tensor]] = []
        step_indices: torch.Tensor | None = None
        with amp_ctx:
            for b in range(fast_logits.shape[0]):
                out = self._forward_single_stepwise_train(
                    fast_logits=fast_logits[b],
                    slow_logits=slow_logits[b],
                    frame_ego2global=frame_ego2global[b],
                    frame_timestamps=frame_timestamps[b] if frame_timestamps is not None else None,
                    frame_dt=frame_dt[b] if frame_dt is not None else None,
                    max_step_index=max_step_index,
                )
                sample_step_logits = cast(torch.Tensor, out["step_logits"])
                sample_step_indices = cast(torch.Tensor, out["step_indices"])
                if step_indices is None:
                    step_indices = sample_step_indices
                elif sample_step_indices.shape != step_indices.shape:
                    raise ValueError(
                        f"batch 内 step 数不一致: {sample_step_indices.shape} vs {step_indices.shape}"
                    )

                step_logits_list.append(sample_step_logits)
                diag_list.append(cast(dict[str, torch.Tensor], out["diagnostics"]))

        if step_indices is None:
            step_indices = torch.zeros((0,), dtype=torch.long, device=fast_logits.device)
        step_logits = torch.stack(step_logits_list, dim=0)
        return {
            "step_logits": step_logits,
            "step_indices": step_indices,
            "diagnostics": diag_list,
        }
