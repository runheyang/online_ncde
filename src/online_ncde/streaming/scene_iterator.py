"""按 scene 分组迭代 OccStudio bevdetv2 dataset.

OccStudio val dataset 已经按 scene 连续排列(每 scene ~40 keyframe), 我们只需
检测 scene_name 切换作为边界, 同时把 nuscenes pkl 里的 ego2global / timestamp /
frame_token / GT 路径 / slow logits 路径打包成 KeyframeMeta, 供下游使用.

注意: 不打乱 dataset 顺序 -- alocc 模型内部 do_history=True 的特性依赖 dataset
顺序连续.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
from pyquaternion import Quaternion


@dataclass
class KeyframeMeta:
    sample_token: str         # nuscenes sample_token, 也是 alocc3d/GT 的 token
    frame_token: str          # cams.CAM_FRONT.sample_data_token, alocc2d_mini 离线 token
    scene_name: str           # scene-XXXX
    scene_token: str
    timestamp_us: int         # 微秒
    ego2global: np.ndarray    # (4, 4) float64
    slow_logit_path: str      # 绝对路径, alocc3d_wo_mask 下的 logits.npz
    gt_label_path: str        # 绝对路径, gts/<scene>/<token>/labels.npz


def _ego2global_matrix(translation, rotation_wxyz) -> np.ndarray:
    """nuscenes ego2global 的 [tx,ty,tz] + [w,x,y,z] -> 4x4."""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    M[:3, 3] = np.asarray(translation, dtype=np.float64)
    return M


def build_sample_meta_index(
    bevdetv2_pkl_path: str,
    slow_logit_root: str,
    gt_root: str,
) -> dict:
    """sample_token -> KeyframeMeta(slow/gt 路径已 absolute).

    Args:
        bevdetv2_pkl_path: data/nuscenes/bevdetv2-nuscenes_infos_val.pkl 绝对路径.
        slow_logit_root:   data/alocc3d_wo_mask 绝对路径.
        gt_root:           data/nuscenes/gts 绝对路径.
    """
    with open(bevdetv2_pkl_path, "rb") as f:
        bdv2 = pickle.load(f)
    out = {}
    for e in bdv2["infos"]:
        sample_token = e["token"]
        frame_token = e["cams"]["CAM_FRONT"]["sample_data_token"]
        scene_name = e["scene_name"]
        ego = _ego2global_matrix(e["ego2global_translation"], e["ego2global_rotation"])
        out[sample_token] = KeyframeMeta(
            sample_token=sample_token,
            frame_token=frame_token,
            scene_name=scene_name,
            scene_token=e["scene_token"],
            timestamp_us=int(e["timestamp"]),
            ego2global=ego,
            slow_logit_path=os.path.join(slow_logit_root, scene_name, sample_token, "logits.npz"),
            gt_label_path=os.path.join(gt_root, scene_name, sample_token, "labels.npz"),
        )
    return out


def build_opus_sample_meta_index(
    meta_pkl_path: str,
    slow_logit_root: str,
    gt_root: str,
) -> dict:
    """OPUS sweep pkl -> sample_token 到 KeyframeMeta 的索引.

    OPUS 的 nuscenes_infos_val_sweep.pkl 里 CAM_FRONT 可能没有 sample_data_token，
    streaming 只依赖 sample_token/scene/timestamp/ego/slow/gt，因此 frame_token 用
    data_path basename 兜底。
    """
    with open(meta_pkl_path, "rb") as f:
        data = pickle.load(f)
    out = {}
    for e in data["infos"]:
        sample_token = e["token"]
        cam_front = e.get("cams", {}).get("CAM_FRONT", {})
        frame_token = (
            cam_front.get("sample_data_token")
            or cam_front.get("token")
            or os.path.splitext(os.path.basename(cam_front.get("data_path", "")))[0]
            or sample_token
        )
        scene_name = e["scene_name"]
        ego = _ego2global_matrix(e["ego2global_translation"], e["ego2global_rotation"])
        out[sample_token] = KeyframeMeta(
            sample_token=sample_token,
            frame_token=frame_token,
            scene_name=scene_name,
            scene_token=e["scene_token"],
            timestamp_us=int(e["timestamp"]),
            ego2global=ego,
            slow_logit_path=os.path.join(slow_logit_root, scene_name, sample_token, "logits.npz"),
            gt_label_path=os.path.join(gt_root, scene_name, sample_token, "labels.npz"),
        )
    return out


def iter_scenes(
    occ_dataset,
    sample_meta_index: dict,
    limit_scenes: int | None = None,
) -> Iterator[Tuple[str, List[Tuple[int, KeyframeMeta]]]]:
    """按 scene 分组迭代 OccStudio val dataset.

    Yields:
        (scene_name, [(global_dataset_idx, meta), ...])
        global_dataset_idx 是 sample 在 OccStudio dataset 里的 index, 调用方负责
        用 dataset[idx] 取实际样本. 这样避免预读全部 sample 占内存.
    """
    cur_scene = None
    cur_buffer: List[Tuple[int, KeyframeMeta]] = []
    n_scenes_yielded = 0
    for i in range(len(occ_dataset)):
        # OccStudio dataset.__getitem__ 较重 (要走 pipeline), 先用 data_infos 直接取 token
        info = occ_dataset.data_infos[i]
        sample_token = info["token"]
        meta = sample_meta_index[sample_token]
        if meta.scene_name != cur_scene:
            if cur_buffer:
                yield cur_scene, cur_buffer
                n_scenes_yielded += 1
                if limit_scenes is not None and n_scenes_yielded >= limit_scenes:
                    return
                cur_buffer = []
            cur_scene = meta.scene_name
        cur_buffer.append((i, meta))
    if cur_buffer and (limit_scenes is None or n_scenes_yielded < limit_scenes):
        yield cur_scene, cur_buffer
