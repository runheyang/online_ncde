"""No-warp attention baseline 的 streaming 适配器."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from evoocc.baselines import NoWarpMotionBiasAttnAligner
from evoocc.config import load_config_with_base
from evoocc.data.ego_warp_list import compute_transform_prev_to_curr
from evoocc.utils.checkpoints import load_checkpoint_for_eval


class NoWarpAttnStreamAligner:
    """把 NoWarpMotionBiasAttnAligner 拆成逐 keyframe 调用接口."""

    def __init__(self, base_aligner: NoWarpMotionBiasAttnAligner) -> None:
        self.m = base_aligner
        self.m.eval()
        self.hidden: Optional[torch.Tensor] = None
        self.prev_ego: Optional[torch.Tensor] = None
        self.prev_t_us: Optional[int] = None
        self._spatial_shape = None
        self._ts_scale = float(self.m.timestamp_scale)

    def reset_scene(self) -> None:
        self.hidden = None
        self.prev_ego = None
        self.prev_t_us = None
        self._spatial_shape = None

    @torch.no_grad()
    def _encode_fast(self, fast_logits: torch.Tensor) -> torch.Tensor:
        return self.m._encode_fast(fast_logits.unsqueeze(0))[0]

    @torch.no_grad()
    def _encode_slow(self, slow_logits: torch.Tensor) -> torch.Tensor:
        return self.m._encode_slow(slow_logits)

    @torch.no_grad()
    def reset_with_slow(
        self,
        fast_logits: torch.Tensor,
        slow_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        fast_feat = self._encode_fast(fast_logits)
        self.hidden = self._encode_slow(slow_logits)
        self.prev_ego = ego2global
        self.prev_t_us = int(t_us)
        self._spatial_shape = (
            int(fast_feat.shape[1]),
            int(fast_feat.shape[2]),
            int(fast_feat.shape[3]),
        )
        return slow_logits

    @torch.no_grad()
    def evolve(
        self,
        fast_logits: torch.Tensor,
        ego2global: torch.Tensor,
        t_us: int,
    ) -> torch.Tensor:
        if self.hidden is None or self.prev_ego is None or self.prev_t_us is None:
            raise RuntimeError("evolve called before reset_with_slow; first keyframe must be a slow injection")

        fast_feat = self._encode_fast(fast_logits)
        if self._spatial_shape is None:
            self._spatial_shape = (
                int(fast_feat.shape[1]),
                int(fast_feat.shape[2]),
                int(fast_feat.shape[3]),
            )

        transform = compute_transform_prev_to_curr(
            pose_prev_ego2global=self.prev_ego,
            pose_curr_ego2global=ego2global,
        )
        dt = torch.tensor(
            float(t_us - self.prev_t_us) * self._ts_scale,
            device=fast_feat.device,
            dtype=fast_feat.dtype,
        )
        h_new = self.m._advance_no_warp(
            h_dense=self.hidden,
            fast_curr=fast_feat,
            dt_value=dt,
            transform_prev_to_curr=transform,
            spatial_shape_xyz=self._spatial_shape,
        )
        logits_delta = self.m._decode_dense_state(h_new)
        aligned = logits_delta + fast_logits if self.m.use_fast_residual else logits_delta

        self.hidden = h_new
        self.prev_ego = ego2global
        self.prev_t_us = int(t_us)
        return aligned


def build_no_warp_aligner(
    aligner_cfg: str,
    aligner_ckpt: str,
    device: torch.device,
    use_fast_residual: bool = False,
) -> Tuple[NoWarpMotionBiasAttnAligner, dict]:
    """构建 no-warp attention baseline aligner."""
    cfg = load_config_with_base(aligner_cfg)
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    inner_dim = int(model_cfg.get("no_warp_inner_dim", model_cfg.get("func_g_inner_dim", 24)))
    num_heads = int(model_cfg.get("no_warp_attn_num_heads", 3))
    if inner_dim % num_heads != 0:
        raise ValueError(
            f"no-warp attention inner_dim={inner_dim} 必须能被 num_heads={num_heads} 整除"
        )

    decoder_init_scale = model_cfg.get("decoder_init_scale", 1.0e-3) if use_fast_residual else None
    aligner = NoWarpMotionBiasAttnAligner(
        num_classes=data_cfg["num_classes"],
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        encoder_in_channels=model_cfg["encoder_in_channels"],
        free_index=data_cfg["free_index"],
        pc_range=tuple(data_cfg["pc_range"]),
        voxel_size=tuple(data_cfg["voxel_size"]),
        decoder_init_scale=decoder_init_scale,
        use_fast_residual=use_fast_residual,
        fusion_inner_dim=inner_dim,
        fusion_attn_num_heads=num_heads,
        fusion_attn_window_size=tuple(model_cfg.get("no_warp_attn_window_size", [8, 8, 4])),
        fusion_attn_head_dilations=tuple(model_cfg.get("no_warp_attn_head_dilations", [1, 2])),
        fusion_gn_groups=int(model_cfg.get("no_warp_gn_groups", model_cfg.get("func_g_gn_groups", 8))),
        fusion_attn_mlp_ratio=float(model_cfg.get("no_warp_attn_mlp_ratio", 2.0)),
        timestamp_scale=data_cfg.get("timestamp_scale", 1.0e-6),
    ).to(device)
    load_checkpoint_for_eval(aligner_ckpt, model=aligner, strict=False)
    aligner.eval()
    return aligner, data_cfg
