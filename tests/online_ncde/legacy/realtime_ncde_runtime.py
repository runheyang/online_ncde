#!/usr/bin/env python3
"""实时 benchmark 里 Online NCDE 的状态化推理包装。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base, resolve_path  # noqa: E402
from online_ncde.data.ego_warp_list import (  # noqa: E402
    backward_warp_dense_trilinear,
    build_sampling_grid,
    compute_transform_prev_to_curr,
)
from online_ncde.data.logits_io import (  # noqa: E402
    decode_single_frame_sparse_full,
    decode_single_frame_sparse_topk,
    load_logits_npz,
)
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.data.scene_delta import build_scene_delta_ctrl  # noqa: E402
from online_ncde.data.time_series import compute_segment_dt, cumulative_tau  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner  # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint  # noqa: E402


@dataclass
class RuntimeState:
    """缓存单个样本的在线状态。"""

    slow_logits_dense: torch.Tensor | None = None
    z_dense: torch.Tensor | None = None
    prev_fast_logits_dense: torch.Tensor | None = None
    prev_fast_feat: torch.Tensor | None = None
    prev_pose: torch.Tensor | None = None
    prev_timestamp: torch.Tensor | None = None
    tau_values: torch.Tensor | None = None
    frame_ego2global: torch.Tensor | None = None
    initialized: bool = False


class RealtimeNcdeRuntime:
    """把离线 stepwise 逻辑拆成 step0 初始化 + step>=1 单步演化。"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
    ) -> None:
        self.config_path = os.path.abspath(config_path)
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.cfg = load_config_with_base(self.config_path)
        data_cfg = self.cfg["data"]
        model_cfg = self.cfg["model"]
        eval_cfg = self.cfg["eval"]

        self.root_path = self.cfg["root_path"]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = int(data_cfg["num_classes"])
        self.free_index = int(data_cfg["free_index"])
        self.grid_size = tuple(int(v) for v in data_cfg["grid_size"])
        self.other_fill_value = float(data_cfg.get("topk_other_fill_value", -5.0))
        self.free_fill_value = float(data_cfg.get("topk_free_fill_value", 5.0))
        self.timestamp_scale = float(data_cfg.get("timestamp_scale", 1.0e-6))
        self.slow_logit_root = str(data_cfg.get("slow_logit_root", ""))
        self.slow_logit_variant = str(data_cfg.get("slow_logit_variant", "topk"))
        self.path_resolver = Occ3DOnlineNcdeDataset.__new__(Occ3DOnlineNcdeDataset)
        self.path_resolver.root_path = self.root_path

        self.model = OnlineNcdeAligner(
            num_classes=self.num_classes,
            feat_dim=int(model_cfg["feat_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            encoder_in_channels=int(model_cfg["encoder_in_channels"]),
            free_index=self.free_index,
            pc_range=tuple(data_cfg["pc_range"]),
            voxel_size=tuple(data_cfg["voxel_size"]),
            decoder_init_scale=float(model_cfg.get("decoder_init_scale", 1.0e-3)),
            use_fast_residual=bool(model_cfg.get("use_fast_residual", True)),
            func_g_inner_dim=int(model_cfg.get("func_g_inner_dim", 32)),
            func_g_body_dilations=tuple(model_cfg.get("func_g_body_dilations", [1, 2, 3])),
            func_g_gn_groups=int(model_cfg.get("func_g_gn_groups", 8)),
            timestamp_scale=self.timestamp_scale,
            amp_fp16=bool(eval_cfg.get("amp_fp16", False)),
        ).to(self.device)
        load_checkpoint(self.checkpoint_path, model=self.model, strict=False)
        self.model.eval()

        self.spatial_shape_xyz = (
            self.grid_size[0] // 2,
            self.grid_size[1] // 2,
            self.grid_size[2],
        )
        self.pc_range = cast(tuple[float, float, float, float, float, float], self.model.pc_range)
        self.voxel_size_ds = cast(tuple[float, float, float], self.model.voxel_size_ds)
        self.state = RuntimeState()

    def _encode_single_fast(self, fast_logits_dense: torch.Tensor) -> torch.Tensor:
        fast_seq = fast_logits_dense.unsqueeze(0).float()
        return self.model._encode_fast(fast_seq)[0].float()

    def _resolve_slow_logits_path(self, info: dict[str, Any]) -> str:
        return Occ3DOnlineNcdeDataset._resolve_logits_path(
            self.path_resolver,
            root_rel=self.slow_logit_root,
            info=info,
            info_key="slow_logit_path",
            variant=self.slow_logit_variant,
            default_name="slow_logit_full.npz" if self.slow_logit_variant == "full" else "slow_logit.npz",
        )

    def _load_slow_logits_dense(self, info: dict[str, Any]) -> torch.Tensor:
        slow_logits_path = self._resolve_slow_logits_path(info)
        slow_npz = load_logits_npz(slow_logits_path)
        if self.slow_logit_variant == "full":
            dense = decode_single_frame_sparse_full(
                sparse_coords=slow_npz["sparse_coords"],
                sparse_values=slow_npz["sparse_values"],
                grid_size=self.grid_size,
                num_classes=self.num_classes,
                free_index=self.free_index,
                other_fill_value=self.other_fill_value,
                free_fill_value=self.free_fill_value,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            dense = decode_single_frame_sparse_topk(
                sparse_coords=slow_npz["sparse_coords"],
                sparse_topk_values=slow_npz["sparse_topk_values"],
                sparse_topk_indices=slow_npz["sparse_topk_indices"],
                grid_size=self.grid_size,
                num_classes=self.num_classes,
                free_index=self.free_index,
                other_fill_value=self.other_fill_value,
                free_fill_value=self.free_fill_value,
                device=self.device,
                dtype=torch.float32,
            )
        return dense.float()

    def begin_sample(self, info: dict[str, Any]) -> int:
        """在样本开始时只重置状态和时间轴，不提前做 slow decode。"""
        frame_ego2global = torch.as_tensor(info["frame_ego2global"], dtype=torch.float32, device=self.device)
        num_frames = int(frame_ego2global.shape[0])

        frame_timestamps = info.get("frame_timestamps", None)
        frame_timestamps_t = None
        if frame_timestamps is not None:
            frame_timestamps_t = torch.as_tensor(frame_timestamps, dtype=torch.long, device=self.device)

        frame_dt = info.get("frame_dt", None)
        frame_dt_t = None
        if frame_dt is not None:
            frame_dt_t = torch.as_tensor(frame_dt, dtype=torch.float32, device=self.device)

        dt = compute_segment_dt(
            frame_timestamps=frame_timestamps_t,
            frame_dt=frame_dt_t,
            num_frames=num_frames,
            timestamp_scale=self.timestamp_scale,
        ).to(device=self.device)
        tau_values = cumulative_tau(dt).to(device=self.device)

        self.state = RuntimeState(
            tau_values=tau_values,
            frame_ego2global=frame_ego2global,
            initialized=False,
        )
        return num_frames

    def step(self, step_idx: int, fast_logits_dense: torch.Tensor, info: dict[str, Any]) -> torch.Tensor:
        """执行一步在线推理，并返回当前时刻的 dense logits。"""
        fast_logits_dense = fast_logits_dense.to(device=self.device, dtype=torch.float32)
        curr_pose = cast(torch.Tensor, self.state.frame_ego2global)[step_idx]
        curr_tau = cast(torch.Tensor, self.state.tau_values)[step_idx]

        if step_idx == 0:
            if self.state.initialized:
                raise RuntimeError("step0 只能在样本初始化状态下调用一次。")

            if self.state.slow_logits_dense is None:
                # 设计要求 slow decode 计入 step0，而不是样本级别的预处理时间。
                self.state.slow_logits_dense = self._load_slow_logits_dense(info)

            fast_feat0 = self._encode_single_fast(fast_logits_dense)
            slow_feat = self.model._encode_slow(cast(torch.Tensor, self.state.slow_logits_dense))
            if self.model.use_fast_residual:
                z_dense = (slow_feat - fast_feat0).float()
            else:
                z_dense = slow_feat.float()

            logits_delta = self.model._decode_dense_state(z_dense)
            if self.model.use_fast_residual:
                aligned = logits_delta.float() + fast_logits_dense.float()
            else:
                aligned = logits_delta.float()

            self.state.z_dense = z_dense.float()
            self.state.prev_fast_logits_dense = fast_logits_dense.float()
            self.state.prev_fast_feat = fast_feat0.float()
            self.state.prev_pose = curr_pose.float()
            self.state.prev_timestamp = curr_tau.float()
            self.state.initialized = True
            return aligned.float()

        if not self.state.initialized:
            raise RuntimeError("step>=1 之前必须先执行 step0。")

        prev_pose = cast(torch.Tensor, self.state.prev_pose)
        prev_fast_feat = cast(torch.Tensor, self.state.prev_fast_feat)
        z_dense_prev = cast(torch.Tensor, self.state.z_dense)
        prev_tau = cast(torch.Tensor, self.state.prev_timestamp)

        fast_feat_t = self._encode_single_fast(fast_logits_dense)
        transform = compute_transform_prev_to_curr(
            pose_prev_ego2global=prev_pose,
            pose_curr_ego2global=curr_pose,
        )
        grid = build_sampling_grid(
            transform,
            self.spatial_shape_xyz,
            self.pc_range,
            self.voxel_size_ds,
        )

        z_warp_dense = backward_warp_dense_trilinear(
            dense_prev_feat=z_dense_prev,
            transform_prev_to_curr=None,
            spatial_shape_xyz=self.spatial_shape_xyz,
            pc_range=self.pc_range,
            voxel_size=self.voxel_size_ds,
            padding_mode="border",
            prebuilt_grid=grid,
        )
        f_prev_adv = backward_warp_dense_trilinear(
            dense_prev_feat=prev_fast_feat,
            transform_prev_to_curr=None,
            spatial_shape_xyz=self.spatial_shape_xyz,
            pc_range=self.pc_range,
            voxel_size=self.voxel_size_ds,
            padding_mode="border",
            prebuilt_grid=grid,
        )
        delta_info = build_scene_delta_ctrl(
            fast_curr=fast_feat_t,
            fast_prev_adv=f_prev_adv,
            tau_curr=curr_tau,
            tau_prev=prev_tau,
        )
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if self.model.amp_fp16
            else torch.amp.autocast("cuda", enabled=False)
        )
        with amp_ctx:
            z_dense_amp, _ = self.model.solver.step(
                h_adv=z_warp_dense,
                f_prev_adv=f_prev_adv,
                f_t=fast_feat_t,
                delta_ctrl=delta_info["delta_ctrl"],
            )
        z_dense = z_dense_amp.float()

        logits_delta = self.model._decode_dense_state(z_dense)
        if self.model.use_fast_residual:
            aligned = logits_delta.float() + fast_logits_dense.float()
        else:
            aligned = logits_delta.float()

        self.state.z_dense = z_dense
        self.state.prev_fast_logits_dense = fast_logits_dense.float()
        self.state.prev_fast_feat = fast_feat_t.float()
        self.state.prev_pose = curr_pose.float()
        self.state.prev_timestamp = curr_tau.float()
        self.state.initialized = True
        return aligned.float()
