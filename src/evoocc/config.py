"""evoocc 配置加载与路径解析工具。"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def merge_dict(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置字典。"""
    merged = dict(base)
    for key, value in (extra or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _find_repo_root(start: str) -> str:
    """从 start 向上查找包含 .git 的目录作为 repo 根目录。"""
    current = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    # fallback: 从本文件推导 src/../..
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(root_path: str, path: str) -> str:
    """将相对路径解析为绝对路径。"""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(root_path, path)


def config_output_subdir(config_path: str, configs_root: str | None = None) -> str:
    """根据配置文件路径生成稳定输出子目录。"""
    config_abs = os.path.abspath(os.path.expanduser(config_path))
    rel_path = None

    if configs_root:
        configs_abs = os.path.abspath(os.path.expanduser(configs_root))
        try:
            if os.path.commonpath([config_abs, configs_abs]) == configs_abs:
                rel_path = os.path.relpath(config_abs, configs_abs)
        except ValueError:
            rel_path = None

    if rel_path is None:
        rel_path = os.path.basename(config_abs)

    rel_no_ext = os.path.splitext(rel_path)[0]
    return rel_no_ext or os.path.splitext(os.path.basename(config_abs))[0]


def load_config_with_base(path: str) -> Dict[str, Any]:
    """支持 base_config 的递归配置加载。自动注入 root_path。"""
    cfg = _load_config_recursive(path)
    if "root_path" not in cfg:
        cfg["root_path"] = _find_repo_root(os.path.dirname(os.path.abspath(path)))
    return cfg


def _load_config_recursive(path: str) -> Dict[str, Any]:
    """递归加载 base_config 链。"""
    cfg = load_config(path)
    base_path = cfg.pop("base_config", None)
    if base_path:
        base_abs = os.path.join(os.path.dirname(os.path.abspath(path)), base_path)
        base_cfg = _load_config_recursive(base_abs)
        return merge_dict(base_cfg, cfg)
    return cfg
