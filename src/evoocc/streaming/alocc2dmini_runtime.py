"""ALOcc2DMini streaming 入口共用构建工具。"""
from __future__ import annotations

import os
from typing import Optional, Tuple

from evoocc.streaming.fast_runner import FastRunner


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OCCSTUDIO_ROOT = os.path.join(_REPO_ROOT, "third_party", "OccStudio")
DEFAULT_BDV2_PKL = "/root/autodl-tmp/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl"

_FAST_PATHS_BY_VARIANT = {
    "occ3d": (
        "configs/alocc/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.py",
        "ckpts/alocc_2d_mini_r50_256x704_bevdet_preatrain_16f_wo_mask.pth",
    ),
    "surroundocc": (
        "configs/alocc/alocc_2d_mini_r50_256x704_bevdet_preatrain_surroundocc.py",
        "ckpts/alocc_2d_mini_r50_256x704_bevdet_preatrain_surroundocc.pth",
    ),
}

_SLOW_PATHS_BY_VARIANT = {
    "occ3d": (
        "configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f_wo_mask.py",
        "ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f_wo_mask.pth",
    ),
    "surroundocc": (
        "configs/alocc/alocc_3d_r50_900x1600_bevdet_preatrain_surroundocc.py",
        "ckpts/alocc_3d_r50_900x1600_bevdet_preatrain_surroundocc.pth",
    ),
}


def dataset_variant(data_cfg: dict) -> str:
    """读取数据集变体，默认保持 Occ3D 旧行为。"""
    return str(data_cfg.get("dataset_variant", "occ3d")).strip().lower()


def resolve_repo_path(path: Optional[str], repo_root: str) -> Optional[str]:
    """把相对仓库根目录路径解析为绝对路径。"""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def resolve_cfg_path(data_cfg: dict, key: str, repo_root: str, override: Optional[str] = None) -> Optional[str]:
    """命令行 override 优先，否则读取 data_cfg[key]。"""
    value = override if override else data_cfg.get(key, None)
    return resolve_repo_path(value, repo_root) if value else None


def select_fast_paths(
    data_cfg: dict,
    occ_config: Optional[str] = None,
    occ_ckpt: Optional[str] = None,
) -> Tuple[str, str]:
    """根据 dataset_variant 选择 ALOcc2DMini fast config/ckpt。"""
    variant = dataset_variant(data_cfg)
    if variant not in _FAST_PATHS_BY_VARIANT:
        raise ValueError(f"alocc2dmini streaming 暂不支持 dataset_variant={variant!r}")
    default_config, default_ckpt = _FAST_PATHS_BY_VARIANT[variant]
    return occ_config or default_config, occ_ckpt or default_ckpt


def select_slow_paths(
    data_cfg: dict,
    occ_config: Optional[str] = None,
    occ_ckpt: Optional[str] = None,
) -> Tuple[str, str]:
    """根据 dataset_variant 选择 ALOcc3D slow config/ckpt。"""
    variant = dataset_variant(data_cfg)
    if variant not in _SLOW_PATHS_BY_VARIANT:
        raise ValueError(f"alocc3d streaming 暂不支持 dataset_variant={variant!r}")
    default_config, default_ckpt = _SLOW_PATHS_BY_VARIANT[variant]
    return occ_config or default_config, occ_ckpt or default_ckpt


def _build_alocc_runner(
    data_cfg: dict,
    *,
    occstudio_root: str,
    config_path: str,
    ckpt_path: str,
    role: str,
    device: str,
) -> FastRunner:
    drop_others = dataset_variant(data_cfg) == "surroundocc"
    runner = FastRunner(
        occstudio_root=occstudio_root,
        config_path=config_path,
        ckpt_path=ckpt_path,
        num_classes=data_cfg["num_classes"],
        free_index=data_cfg["free_index"],
        topk_k=int(data_cfg.get("alocc_topk_k", 3)),
        clamp_min=float(data_cfg.get("alocc_clamp_min", -5.0)),
        fill_value=float(data_cfg.get("alocc_fill_value", -5.0)),
        max_centering=bool(data_cfg.get("alocc_max_centering", False)),
        label_id_offset=int(data_cfg.get("alocc_label_id_offset", 0)),
        drop_others_label=drop_others,
        device=device,
    )
    print(f"  {role} config={config_path}")
    print(f"  {role} ckpt={ckpt_path}")
    print(f"  dataset_variant={dataset_variant(data_cfg)}, drop_others_label={drop_others}")
    runner.build()
    return runner


def build_alocc2dmini_fast_runner(
    data_cfg: dict,
    *,
    occstudio_root: str = DEFAULT_OCCSTUDIO_ROOT,
    occ_config: Optional[str] = None,
    occ_ckpt: Optional[str] = None,
    device: str = "cuda:0",
) -> FastRunner:
    """构建与 data_cfg label-space 对齐的 ALOcc2DMini FastRunner。"""
    config_path, ckpt_path = select_fast_paths(
        data_cfg,
        occ_config=occ_config,
        occ_ckpt=occ_ckpt,
    )
    return _build_alocc_runner(
        data_cfg,
        occstudio_root=occstudio_root,
        config_path=config_path,
        ckpt_path=ckpt_path,
        role="fast",
        device=device,
    )


def build_alocc3d_slow_runner(
    data_cfg: dict,
    *,
    occstudio_root: str = DEFAULT_OCCSTUDIO_ROOT,
    occ_config: Optional[str] = None,
    occ_ckpt: Optional[str] = None,
    device: str = "cuda:0",
) -> FastRunner:
    """构建与 data_cfg label-space 对齐的 ALOcc3D slow runner。"""
    config_path, ckpt_path = select_slow_paths(
        data_cfg,
        occ_config=occ_config,
        occ_ckpt=occ_ckpt,
    )
    return _build_alocc_runner(
        data_cfg,
        occstudio_root=occstudio_root,
        config_path=config_path,
        ckpt_path=ckpt_path,
        role="slow",
        device=device,
    )
