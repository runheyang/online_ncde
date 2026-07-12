"""StreamingFlow baseline 的逐 keyframe streaming adapter."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from evoocc.baselines import StreamingFlowBEVOdeAligner
from evoocc.config import load_config, load_config_with_base, merge_dict
from evoocc.utils.checkpoints import load_checkpoint_for_eval


def load_config_with_streamingflow_overlay(config_path: str, overlay_path: str) -> dict:
    """加载普通 Occ3D 配置，并叠加 StreamingFlow baseline 覆盖层."""
    cfg = load_config_with_base(config_path)
    overlay = load_config(overlay_path)
    return merge_dict(cfg, overlay)


def build_streamingflow_model(
    config_path: str,
    checkpoint_path: str,
    overlay_path: str,
    device: torch.device,
) -> Tuple[StreamingFlowBEVOdeAligner, dict]:
    """构建 StreamingFlowBEVOdeAligner 并加载权重."""
    cfg = load_config_with_streamingflow_overlay(config_path, overlay_path)
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    model = StreamingFlowBEVOdeAligner(
        num_classes=int(data_cfg["num_classes"]),
        feat_dim=int(model_cfg.get("feat_dim", 192)),
        hidden_dim=int(model_cfg.get("hidden_dim", 192)),
        encoder_in_channels=int(model_cfg.get("encoder_in_channels", 18)),
        free_index=int(data_cfg["free_index"]),
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=model_cfg.get("decoder_init_scale", None),
        timestamp_scale=float(data_cfg.get("timestamp_scale", 1.0e-6)),
        streamingflow_cfg=dict(model_cfg.get("streamingflow", {}) or {}),
    ).to(device)
    load_checkpoint_for_eval(checkpoint_path, model=model, strict=False)
    model.eval()
    return model, data_cfg


class StreamingFlowStreamAligner:
    """把 StreamingFlowBEVOdeAligner 拆成逐 keyframe reset/evolve 调用.

    StreamingFlow baseline 不使用 ego pose；参数保留是为了对齐现有 streaming loop。
    """

    def __init__(self, model: StreamingFlowBEVOdeAligner) -> None:
        self.m = model
        self.m.eval()
        self.hidden: Optional[torch.Tensor] = None
        self.input_feat: Optional[torch.Tensor] = None
        self.prev_t_us: Optional[int] = None

    def reset_scene(self) -> None:
        self.hidden = None
        self.input_feat = None
        self.prev_t_us = None

    @torch.no_grad()
    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        return self.m._encode_fast(fast_logits.unsqueeze(0))[0].unsqueeze(0)

    @torch.no_grad()
    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        return self.m._encode_slow(slow_logits).unsqueeze(0)

    @torch.no_grad()
    def _decode_current(self, state: torch.Tensor) -> torch.Tensor:
        # sequence_refiner 期望 5D sequence；单步 streaming 用长度 1 的序列。
        seq = state.unsqueeze(1)
        bev = self.m.sequence_refiner(seq)
        bev = self.m.small_decoder(bev)
        logits = self.m.bev_to_3d(bev)
        return logits[0, 0].float()

    @torch.no_grad()
    def reset_with_slow(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        del ego2global
        fast_bev = self._encode_fast(fast_logits)
        slow_bev = self._encode_slow(slow_logits)
        obs0 = self.m.same_time_fusion(slow_bev, fast_bev)
        self.hidden = self.m.core.obs_cell(obs0, torch.zeros_like(obs0))
        self.input_feat = self.m.core.infer_state(self.hidden)
        self.prev_t_us = int(t_us)
        return slow_logits.float()

    @torch.no_grad()
    def evolve(
        self,
        fast_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        del ego2global
        if self.hidden is None or self.input_feat is None or self.prev_t_us is None:
            raise RuntimeError("evolve called before reset_with_slow; first keyframe must be a slow injection")

        dt = float(t_us - self.prev_t_us) * float(self.m.timestamp_scale)
        self.hidden, self.input_feat = self.m.core.ode_step(
            self.hidden,
            self.input_feat,
            dt,
        )
        fast_bev = self._encode_fast(fast_logits)
        self.hidden = self.m.core.obs_cell(fast_bev, self.hidden)
        self.input_feat = self.m.core.infer_state(self.hidden)
        self.prev_t_us = int(t_us)
        return self._decode_current(self.hidden)
