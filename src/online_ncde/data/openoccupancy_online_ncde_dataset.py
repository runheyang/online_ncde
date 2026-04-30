"""OpenOccupancy-nuScenes online_ncde 数据集。

复用现有 canonical_infos pkl（时序 / pose / token 字段与 benchmark 无关，
fast_logits / slow_logit 路径模板 OPUS 在两个 benchmark 共用），
只 override GT 加载钩子：

  - GT 路径：`<gt_root>/scene_<scene_token>/occupancy/<lidar_sd_token>.npy`
  - GT 内容：稀疏 (N, 4) `[z, y, x, raw_label]`，raw_label 0='noise' 1..16=barrier..vegetation
  - 类映射：raw 1..16 → new 0..15；未存 voxel → new 16 (free)；raw 0 (noise) → mask out
  - mask：raw_label != 0 形成的 noise mask（dataset 内部自构，覆盖 yaml 的 gt_mask_key）

sample_token → lidar_sd_token 映射来自 `scripts/build_oo_token_map.py` 生成的小 pkl。
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch

from online_ncde.config import resolve_path
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset

# raw label → new class：noise=0 → 16(占位，会被 mask 排除)，1..16 → 0..15，empty(17) → 16(free)
_OO_RAW_TO_NEW = np.array(
    [16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    dtype=np.int64,
)
# 未存 voxel 的占位 raw 值（高于所有合法 raw 类，便于一次映射搞定）
_OO_EMPTY_RAW = 17


def _decode_oo_gt_npy(
    npy_path: str,
    grid_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """OO 稀疏 .npy → (semantics: int64 (X,Y,Z), mask: float32 (X,Y,Z))。

    npy_path 文件结构：(N, 4) int32，前三列 (z, y, x)，第四列 raw_label。
    grid_size 是 dataset 配置的 (X, Y, Z)。
    """
    occ = np.load(npy_path)  # (N, 4) int32
    if occ.ndim != 2 or occ.shape[1] != 4:
        raise ValueError(f"OO GT 形状异常: {occ.shape} @ {npy_path}")
    coors_zyx = occ[:, :3].astype(np.int64)
    raw_labels = occ[:, 3].astype(np.int64)

    X, Y, Z = grid_size
    # 全部初始化为 empty (17)，未存的位置就保持 17（→ 映射为 free=16）
    raw_dense_zyx = np.full((Z, Y, X), _OO_EMPTY_RAW, dtype=np.int64)
    z_idx = coors_zyx[:, 0]
    y_idx = coors_zyx[:, 1]
    x_idx = coors_zyx[:, 2]
    raw_dense_zyx[z_idx, y_idx, x_idx] = raw_labels

    # noise mask：raw == 0 是 ignore，其它（含 empty）都是有效统计区
    mask_zyx = (raw_dense_zyx != 0).astype(np.float32)
    # 类映射：noise 也走表（落到 16）但会被 mask 排除
    semantics_zyx = _OO_RAW_TO_NEW[raw_dense_zyx]

    # (Z, Y, X) → (X, Y, Z)
    semantics = np.transpose(semantics_zyx, (2, 1, 0)).copy()
    mask = np.transpose(mask_zyx, (2, 1, 0)).copy()
    return semantics, mask


class OpenOccupancyOnlineNcdeDataset(Occ3DOnlineNcdeDataset):
    """OpenOccupancy-nuScenes 数据集，与 Occ3D 共享时序/IO 主体，仅替换 GT 钩子。"""

    def __init__(
        self,
        *args,
        oo_token_map_path: str = "configs/online_ncde/oo_token_map.pkl",
        **kwargs,
    ) -> None:
        # 父类构造时不会触发 GT 加载（lazy in __getitem__），先调父类再加载 token map
        super().__init__(*args, **kwargs)
        token_map_full = resolve_path(self.root_path, oo_token_map_path)
        if not os.path.exists(token_map_full):
            raise FileNotFoundError(
                f"OO sample→lidar token 映射不存在：{token_map_full}\n"
                "请先运行：conda run -n neural_ode python scripts/build_oo_token_map.py"
            )
        with open(token_map_full, "rb") as f:
            self._oo_token_map: Dict[str, Dict[str, str]] = pickle.load(f)
        # 缓存 dataset 内 sample 数对应的 token 命中数，便于 __init__ 时早报错
        missing = [
            info.get("token", "")
            for info in self.infos
            if info.get("token", "") and info["token"] not in self._oo_token_map
        ]
        if missing:
            raise KeyError(
                f"OO token 映射缺失 {len(missing)} 个 sample_token，"
                f"示例：{missing[:3]}。请重建 oo_token_map.pkl。"
            )

    # ---------- 工具：定位 OO GT 路径 ---------- #
    def _resolve_oo_gt_path(self, sample_token: str, info: Dict[str, Any]) -> str:
        """根据 sample_token 拼 OO GT 绝对路径。"""
        entry = self._oo_token_map[sample_token]
        scene_token = entry["scene_token"] or info.get("scene_token", "")
        lidar_token = entry["lidar_token"]
        return os.path.join(
            self.gt_root,
            f"scene_{scene_token}",
            "occupancy",
            f"{lidar_token}.npy",
        )

    # ---------- override：当前帧 GT ---------- #
    def _load_curr_gt(
        self,
        info: Dict[str, Any],
        load_npz_cached: Callable[[str], Dict[str, np.ndarray]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token = info.get("token", "")
        gt_path = self._resolve_oo_gt_path(token, info)
        sem_np, mask_np = _decode_oo_gt_npy(gt_path, self.grid_size)
        return torch.from_numpy(sem_np), torch.from_numpy(mask_np)

    # ---------- override：supervision 帧 GT ---------- #
    def _load_sup_gt(
        self,
        info: Dict[str, Any],
        sup_index: int,
        gt_rel: str,  # 父类传入的 occ3d 风格 rel path，OO 不用
        load_npz_cached: Callable[[str], Dict[str, np.ndarray]],
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        sup_tokens = info.get("supervision_gt_tokens", [])
        if sup_index >= len(sup_tokens):
            return None
        sup_token = str(sup_tokens[sup_index])
        if not sup_token or sup_token not in self._oo_token_map:
            return None
        gt_path = self._resolve_oo_gt_path(sup_token, info)
        if not os.path.exists(gt_path):
            return None
        sem_np, mask_np = _decode_oo_gt_npy(gt_path, self.grid_size)
        return torch.from_numpy(sem_np), torch.from_numpy(mask_np)
