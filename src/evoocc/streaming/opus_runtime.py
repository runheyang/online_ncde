"""OPUS streaming 入口共用的路径和 runner 构建工具。"""
from __future__ import annotations

import os
from typing import Optional

from evoocc.streaming.opusv1_fast_runner import OpusV1FastRunner


DEFAULT_OPUS_ROOT = "third_party/OPUS"
DEFAULT_OPUSV1_CONFIG = (
    "configs/opusv1_nusc-occ3d/opusv1-t_r50_704x256_8f_nusc-occ3d_100e.py"
)
DEFAULT_OPUSV1_CKPT = "checkpoints/opusv1-t_r50_704x256_8f_nusc-occ3d_100e.pth"
# OPUS 目录仅提供同架构的 50e 配置；100e checkpoint 仍使用该模型定义。
DEFAULT_OPUSV2_CONFIG = (
    "configs/opusv2_nusc-occ3d/opusv2-l_r50_704x256_8f_nusc-occ3d_50e.py"
)
DEFAULT_OPUSV2_CKPT = "checkpoints/opusv2-l_r50_704x256_8f_nusc-occ3d_100e.pth"
DEFAULT_META_PKL = "data/nuscenes/nuscenes_infos_val_sweep.pkl"
DEFAULT_GT_ROOT = "data/nuscenes/gts"
DEFAULT_SWEEP_PKL = DEFAULT_META_PKL


def resolve_opus_path(path: str, opus_root: str, repo_root: Optional[str] = None) -> str:
    """解析绝对路径、仓库相对路径或 OPUS 根目录相对路径。"""
    if os.path.isabs(path):
        return os.path.normpath(path)

    opus_candidate = os.path.abspath(os.path.join(opus_root, path))
    if repo_root is not None:
        repo_candidate = os.path.abspath(os.path.join(repo_root, path))
        if os.path.exists(repo_candidate):
            return repo_candidate
    return opus_candidate


def build_opus_runner(
    data_cfg: dict,
    *,
    opus_root: str,
    config_path: str,
    ckpt_path: str,
    role: str,
    repo_root: Optional[str] = None,
    device: str = "cuda:0",
) -> OpusV1FastRunner:
    """构建 OPUSv1 fast 或 OPUSv2 slow runner。"""
    config_path = resolve_opus_path(config_path, opus_root, repo_root)
    ckpt_path = resolve_opus_path(ckpt_path, opus_root, repo_root)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"OPUS {role} config 不存在: {config_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"OPUS {role} checkpoint 不存在: {ckpt_path}")

    runner = OpusV1FastRunner(
        opus_root=opus_root,
        config_path=config_path,
        ckpt_path=ckpt_path,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        grid_size=tuple(data_cfg["grid_size"]),
        other_fill_value=float(data_cfg.get("opus_other_fill_value", -5.0)),
        free_fill_value=float(data_cfg.get("opus_free_fill_value", 5.0)),
        topk_k=int(data_cfg.get("opus_full_topk_k", 3)),
        clamp_min=float(data_cfg.get("opus_clamp_min", -5.0)),
        device=device,
    )
    print(f"  {role} config={config_path}")
    print(f"  {role} ckpt={ckpt_path}")
    runner.build()
    return runner
