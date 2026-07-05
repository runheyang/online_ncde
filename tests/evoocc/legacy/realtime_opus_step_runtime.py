#!/usr/bin/env python3
"""实时 benchmark 里 OPUS 单步推理的胶水封装。"""

from __future__ import annotations

import copy
import importlib
import logging
import os
import queue
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import mmcv  # type: ignore
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope
from mmdet3d.registry import DATASETS, MODELS
from nuscenes import NuScenes
from nuscenes.eval.common.utils import Quaternion
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
OPUS_ROOT = ROOT / "third_party" / "OPUS"
if str(OPUS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPUS_ROOT))

from loaders.utils import compose_ego2img  # noqa: E402


mmcv.use_backend("turbojpeg")


CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@dataclass
class PreparedSample:
    """缓存单个 info 在 OPUS 侧需要的只读上下文。"""

    info: dict[str, Any]
    dataset_idx: int
    scene_name: str
    token: str
    base_input: dict[str, Any]
    frames: list[dict[str, Any]]
    step_source_indices: list[int]

    @property
    def num_steps(self) -> int:
        return len(self.step_source_indices)


class OpusPreparedStepDataset(Dataset):
    """把逐 step 的 pipeline 预处理前移到 DataLoader worker。"""

    def __init__(self, dataset: Any, samples: Sequence[PreparedSample]) -> None:
        self.dataset = dataset
        self.samples = list(samples)
        self.flat_indices: list[tuple[int, int]] = []
        for sample_idx, sample in enumerate(self.samples):
            for step_idx in range(sample.num_steps):
                self.flat_indices.append((sample_idx, step_idx))

    def __len__(self) -> int:
        return len(self.flat_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_idx, step_idx = self.flat_indices[index]
        sample = self.samples[sample_idx]
        current_idx = int(sample.step_source_indices[step_idx])
        try:
            step_input = build_virtual_input_dict(
                base_input=sample.base_input,
                scene_name=sample.scene_name,
                base_sample_token=sample.token,
                frames=sample.frames,
                current_idx=current_idx,
            )
            processed = self.dataset.pipeline(step_input)
            if processed is None:
                raise RuntimeError("OPUS pipeline 返回了 None。")
            return {
                "status": "ok",
                "sample_idx": sample_idx,
                "step_idx": step_idx,
                "num_steps": sample.num_steps,
                "token": sample.token,
                "processed": processed,
            }
        except Exception as exc:
            # worker 里不要直接抛异常，否则整个 DataLoader 会中断；
            # benchmark 主循环收到错误标记后再按整条 rollout 跳过。
            return {
                "status": "error",
                "sample_idx": sample_idx,
                "step_idx": step_idx,
                "num_steps": sample.num_steps,
                "token": sample.token,
                "error_reason": "worker_pipeline_failed",
                "error_message": repr(exc),
            }


def _register_opus_modules() -> None:
    """绑定 registry，保持与 OPUS 原始入口一致。"""
    DefaultScope.get_instance("mmdet3d", scope_name="mmdet3d")

    from mmdet.registry import MODELS as MMDET_MODELS  # noqa: E402
    from mmdet3d.registry import MODELS as MMDET3D_MODELS  # noqa: E402

    MMDET_MODELS.import_from_location()
    if MMDET3D_MODELS.parent is not MMDET_MODELS:
        MMDET3D_MODELS.parent = MMDET_MODELS

    importlib.import_module("models")
    importlib.import_module("loaders")


def _resolve_data_root(data_root: str) -> str:
    """优先相对当前仓库解析，其次保持绝对路径。"""
    if os.path.isabs(data_root):
        return data_root
    cwd_candidate = os.path.abspath(os.path.join(str(ROOT), data_root))
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(data_root)


def _to_nusc_filename(data_path: str, data_root_abs: str) -> str:
    """把路径规范成 NuScenes sample_data['filename'] 形态。"""
    path = os.path.normpath(str(data_path)).replace("\\", "/")
    root = os.path.normpath(str(data_root_abs)).replace("\\", "/")
    if os.path.isabs(path) and path.startswith(root + "/"):
        path = path[len(root) + 1 :]
    parts = path.split("/")
    for marker in ("samples", "sweeps"):
        if marker in parts:
            idx = parts.index(marker)
            return "/".join(parts[idx:])
    return path.lstrip("./")


def build_cam_front_ego_pose_lookup(
    data_root_abs: str,
    nusc_version: str,
) -> dict[str, dict[str, np.ndarray]]:
    """给历史 camera sweeps 还原 ego pose，用于构造正确的 step 输入。"""
    nusc = NuScenes(version=nusc_version, dataroot=data_root_abs, verbose=False)
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for rec in nusc.sample_data:
        if rec.get("sensor_modality") != "camera":
            continue
        if rec.get("channel") != "CAM_FRONT":
            continue
        filename = str(rec.get("filename", ""))
        if not filename:
            continue
        pose_rec = nusc.get("ego_pose", rec["ego_pose_token"])
        lookup[filename] = {
            "ego2global_translation": np.array(pose_rec["translation"], dtype=np.float32),
            "ego2global_rotation": Quaternion(pose_rec["rotation"]).rotation_matrix.astype(np.float32),
        }
    return lookup


def build_frame_list(
    info: dict[str, Any],
    cam_sweeps_prev: list[dict[str, dict[str, Any]]],
    pose_lookup: dict[str, dict[str, np.ndarray]] | None = None,
    data_root_abs: str | None = None,
) -> list[dict[str, Any]]:
    """复用 gen_fast_logits_new 的时序约定：frames[0] 是当前关键帧。"""

    def _pose_from_cam_front(cam_front: dict[str, Any]) -> dict[str, np.ndarray] | None:
        if pose_lookup is None:
            return None
        path = str(cam_front.get("data_path", ""))
        if not path:
            return None
        filename = _to_nusc_filename(path, data_root_abs or "")
        pose = pose_lookup.get(filename, None)
        if pose is None:
            return None
        return {
            "ego2global_translation": np.array(pose["ego2global_translation"], dtype=np.float32),
            "ego2global_rotation": np.array(pose["ego2global_rotation"], dtype=np.float32),
        }

    frames: list[dict[str, Any]] = []
    cam_front = cast(dict[str, Any], info["cams"].get("CAM_FRONT", {}))
    ego_pose = _pose_from_cam_front(cam_front)
    if ego_pose is None:
        ego_rot = np.array(info["ego2global_rotation"])
        if ego_rot.shape == (3, 3):
            ego_rot_mat = ego_rot
        else:
            ego_rot_mat = Quaternion(ego_rot).rotation_matrix
        ego_pose = {
            "ego2global_translation": np.array(info["ego2global_translation"], dtype=np.float32),
            "ego2global_rotation": np.array(ego_rot_mat, dtype=np.float32),
        }

    frames.append(
        {
            "cams": info["cams"],
            "ego_pose": ego_pose,
            "timestamp": info.get("timestamp", 0),
        }
    )

    for sweep in cam_sweeps_prev:
        sweep_cam_front = cast(dict[str, Any], sweep.get("CAM_FRONT", {}))
        sweep_pose = _pose_from_cam_front(sweep_cam_front)
        if sweep_pose is None:
            # pose 查不到时退回当前帧，会让几何略差，但至少不让 benchmark 中断。
            sweep_pose = {
                "ego2global_translation": np.array(ego_pose["ego2global_translation"], dtype=np.float32),
                "ego2global_rotation": np.array(ego_pose["ego2global_rotation"], dtype=np.float32),
            }
        frames.append(
            {
                "cams": sweep,
                "ego_pose": sweep_pose,
                "timestamp": sweep_cam_front.get("timestamp", 0),
            }
        )
    return frames


def build_virtual_input_dict(
    base_input: dict[str, Any],
    scene_name: str,
    base_sample_token: str,
    frames: list[dict[str, Any]],
    current_idx: int,
) -> dict[str, Any]:
    """构造与 OPUS dataset.get_data_info + test_pipeline 对齐的 step 输入。"""
    current_frame = frames[current_idx]
    current_cams = cast(dict[str, dict[str, Any]], current_frame["cams"])
    current_pose = cast(dict[str, np.ndarray], current_frame["ego_pose"])
    ego_t = np.array(current_pose["ego2global_translation"], dtype=np.float32)
    ego_r = np.array(current_pose["ego2global_rotation"], dtype=np.float32)

    img_filename: list[str] = []
    img_timestamp: list[float] = []
    ego2img: list[np.ndarray] = []
    for cam in CAM_ORDER:
        cam_info = current_cams[cam]
        img_filename.append(os.path.relpath(str(cam_info["data_path"])))
        img_timestamp.append(float(cam_info["timestamp"]) / 1e6)
        ego2img.append(
            compose_ego2img(
                ego_t,
                ego_r,
                np.array(cam_info["sensor2global_translation"], dtype=np.float32),
                np.array(cam_info["sensor2global_rotation"], dtype=np.float32).T,
                np.array(cam_info["cam_intrinsic"], dtype=np.float32),
            )
        )

    # 关键点：当前 step 的历史序列必须相对“该 step 的当前帧”回溯。
    cam_sweeps_prev = [frames[j]["cams"] for j in range(current_idx + 1, len(frames))]

    out = copy.deepcopy(base_input)
    out["sample_token"] = base_sample_token
    out["scene_name"] = scene_name
    out["timestamp"] = float(current_frame.get("timestamp", 0)) / 1e6
    out["ego2global_translation"] = ego_t
    out["ego2global_rotation"] = ego_r
    out["img_filename"] = img_filename
    out["img_timestamp"] = img_timestamp
    out["ego2img"] = ego2img
    out["cam_sweeps"] = {"prev": cam_sweeps_prev, "next": []}
    return out


def force_offline_sweeps_for_pipeline(pipeline: Any) -> int:
    """递归打开 force_offline，避免 online shortcut 低估 IO 耗时。"""
    if pipeline is None:
        return 0

    queue: list[Any] = [pipeline]
    visited: set[int] = set()
    nodes: list[Any] = []
    while queue:
        node = queue.pop()
        if node is None:
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        if hasattr(node, "force_offline"):
            nodes.append(node)

        for attr in ("transforms", "pipeline"):
            child = getattr(node, attr, None)
            if child is None:
                continue
            if isinstance(child, (list, tuple)):
                queue.extend(list(child))
            else:
                queue.append(child)
    updated = 0
    for node in nodes:
        try:
            if not bool(getattr(node, "force_offline")):
                setattr(node, "force_offline", True)
                updated += 1
        except Exception:
            pass
    return updated


@contextmanager
def temporary_force_offline_sweeps_for_pipeline(pipeline: Any, enabled: bool):
    """临时切换 pipeline 里的 force_offline，退出时恢复原值。"""
    if pipeline is None:
        yield
        return

    queue: list[Any] = [pipeline]
    visited: set[int] = set()
    previous_states: list[tuple[Any, bool]] = []
    while queue:
        node = queue.pop()
        if node is None:
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        if hasattr(node, "force_offline"):
            try:
                previous_states.append((node, bool(getattr(node, "force_offline"))))
                setattr(node, "force_offline", bool(enabled))
            except Exception:
                pass

        for attr in ("transforms", "pipeline"):
            child = getattr(node, attr, None)
            if child is None:
                continue
            if isinstance(child, (list, tuple)):
                queue.extend(list(child))
            else:
                queue.append(child)
    try:
        yield
    finally:
        for node, previous in reversed(previous_states):
            try:
                setattr(node, "force_offline", previous)
            except Exception:
                pass


def opus_collate_fn(batch: list[Any]) -> dict[str, Any]:
    """沿用 OPUS val.py 的测试 collate 契约。"""
    while batch and isinstance(batch[0], list):
        batch = [sample[0] if isinstance(sample, list) and len(sample) > 0 else sample for sample in batch]
    if not batch or not isinstance(batch[0], dict):
        raise TypeError(f"Expected batch to be list of dicts, got {type(batch[0]) if batch else 'empty'}")

    batch = cast(list[dict[str, Any]], batch)
    data: dict[str, Any] = {}
    for key in batch[0].keys():
        if key == "img_metas":
            data[key] = [[sample[key]] for sample in batch]
        elif key == "img":
            data[key] = [
                sample[key].unsqueeze(0) if isinstance(sample[key], torch.Tensor) else sample[key]
                for sample in batch
            ]
        else:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                data[key] = torch.stack(values, dim=0)
            else:
                data[key] = values
    return data


def opus_move_to_device(obj: Any, device: torch.device) -> Any:
    """递归把 pipeline 输出搬到指定设备。"""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: opus_move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [opus_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(opus_move_to_device(v, device) for v in obj)
    return obj


def opus_prefetch_collate_fn(batch: list[Any]) -> dict[str, Any]:
    """batch_size=1 时直接解包，保留 worker 产出的 step 结构。"""
    if not batch:
        raise RuntimeError("OPUS 预取 DataLoader 返回了空 batch。")
    if len(batch) != 1:
        raise RuntimeError(f"当前 benchmark 只支持 batch_size=1，实际拿到 {len(batch)}。")
    return cast(dict[str, Any], batch[0])


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return cast(torch.nn.Module, model.module if hasattr(model, "module") else model)


def _infer_num_output_frames(info: dict[str, Any]) -> int:
    if int(info.get("num_output_frames", 0)) > 0:
        return int(info["num_output_frames"])
    for key in ("frame_ego2global", "frame_timestamps", "frame_dt", "frame_tokens"):
        value = info.get(key, None)
        if value is None:
            continue
        if hasattr(value, "shape") and len(getattr(value, "shape")) > 0:
            return int(getattr(value, "shape")[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("无法从 info 推断输出帧数。")


def build_step_source_indices(info: dict[str, Any]) -> list[int]:
    """把 oldest -> newest 的逻辑 step 映射回 build_frame_list 的索引。"""
    num_output_frames = _infer_num_output_frames(info)
    output_stride = int(info.get("output_stride", 2))
    return [
        (num_output_frames - 1 - step_idx) * output_stride
        for step_idx in range(num_output_frames)
    ]


class OpusRealtimeStepRunner:
    """管理 OPUS dataset / model 初始化，以及单步真实推理。"""

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        device: str = "cuda:0",
        amp_enabled: bool = True,
        nusc_version: str = "v1.0-trainval",
        force_offline_sweeps: bool = True,
    ) -> None:
        _register_opus_modules()
        for name in ("mmengine", "mmcv"):
            logging.getLogger(name).setLevel(logging.WARNING)

        if not torch.cuda.is_available():
            raise RuntimeError("实时 benchmark 需要 CUDA 环境。")

        self.config_path = os.path.abspath(config_path)
        self.weights_path = os.path.abspath(weights_path)
        self.cfg = Config.fromfile(self.config_path)
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.amp_enabled = bool(amp_enabled)

        dataset_cfg = copy.deepcopy(self.cfg.data.val)
        self.dataset = DATASETS.build(dataset_cfg)
        self.force_offline_sweeps = bool(force_offline_sweeps)
        self.force_offline_count = (
            force_offline_sweeps_for_pipeline(getattr(self.dataset, "pipeline", None))
            if self.force_offline_sweeps
            else 0
        )

        data_infos = getattr(self.dataset, "data_infos", None)
        if data_infos is None:
            data_infos = getattr(self.dataset, "data_list", None)
        if data_infos is None:
            raise AttributeError("OPUS dataset 既没有 data_infos，也没有 data_list。")
        self.data_infos = list(data_infos)
        self.token_to_dataset_idx = {
            str(info.get("token", "")): idx for idx, info in enumerate(self.data_infos) if str(info.get("token", ""))
        }

        self.model = MODELS.build(self.cfg.model).to(self.device)
        self.model.eval()
        self.model.fp16_enabled = True

        checkpoint = torch.load(self.weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)

        self.model_impl = _unwrap_model(self.model)
        head = getattr(self.model_impl, "pts_bbox_head", None)
        if head is None:
            raise AttributeError("OPUS 模型缺少 pts_bbox_head，无法开启 return_logits。")
        head.return_logits = True
        if hasattr(head, "test_cfg") and getattr(head, "test_cfg") is not None:
            test_cfg = head.test_cfg
            if hasattr(test_cfg, "get") and test_cfg.get("score_thr", None) is not None:
                test_cfg["score_thr"] = 0.0

        self.num_input_frames = int(self.cfg.model.pts_bbox_head.transformer.num_frames)
        self.data_root_abs = _resolve_data_root(str(dataset_cfg.data_root))
        self.pose_lookup = build_cam_front_ego_pose_lookup(
            data_root_abs=self.data_root_abs,
            nusc_version=nusc_version,
        )
        self.loader_num_workers = int(getattr(self.cfg.data, "workers_per_gpu", 4))

    def reset_model_cache(self) -> None:
        """重置 OPUS online 模式的逐帧特征缓存。"""
        if hasattr(self.model_impl, "memory"):
            self.model_impl.memory = {}
        if hasattr(self.model_impl, "queue"):
            self.model_impl.queue = queue.Queue()

    def get_dataset_index(self, token: str) -> int | None:
        return self.token_to_dataset_idx.get(str(token), None)

    def prepare_sample(self, info: dict[str, Any]) -> PreparedSample:
        token = str(info.get("token", ""))
        dataset_idx = self.get_dataset_index(token)
        if dataset_idx is None:
            raise KeyError(f"token={token} 不在 OPUS val dataset 中。")

        data_info = cast(dict[str, Any], self.data_infos[dataset_idx])
        scene_name = str(info.get("scene_name", data_info.get("scene_name", "")))
        base_input = cast(dict[str, Any], self.dataset.get_data_info(dataset_idx))
        step_source_indices = build_step_source_indices(info)
        max_output_idx = max(step_source_indices) if step_source_indices else 0
        requested_history_len = max_output_idx + int(info.get("history_interval", 6)) * (self.num_input_frames - 1) + 1

        cam_sweeps_prev, _ = self.dataset.collect_cam_sweeps(dataset_idx, into_past=requested_history_len)
        frames = build_frame_list(
            info=data_info,
            cam_sweeps_prev=cam_sweeps_prev,
            pose_lookup=self.pose_lookup,
            data_root_abs=self.data_root_abs,
        )
        # 这里只要求 oldest step 的“当前帧”存在即可。
        # 对于 oldest step 再往前缺失的 past 7 帧，不在这里直接判失败，
        # 而是交给 OPUS 的 LoadMultiViewImageFromMultiSweeps.load_offline()
        # 按作者原生逻辑用当前帧/最老可用 sweep 自动补齐。
        if len(frames) <= max_output_idx:
            raise RuntimeError(
                "OPUS 共享缓冲连 oldest step 的当前帧都不存在，无法构造这条 rollout: "
                f"token={token} len(frames)={len(frames)} max_output_idx={max_output_idx}"
            )

        return PreparedSample(
            info=info,
            dataset_idx=dataset_idx,
            scene_name=scene_name,
            token=token,
            base_input=base_input,
            frames=frames,
            step_source_indices=step_source_indices,
        )

    def build_prefetch_dataloader(
        self,
        samples: Sequence[PreparedSample],
        *,
        num_workers: int | None = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ) -> DataLoader:
        dataset = OpusPreparedStepDataset(dataset=self.dataset, samples=samples)
        worker_count = self.loader_num_workers if num_workers is None else int(num_workers)
        worker_count = max(worker_count, 0)

        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "num_workers": worker_count,
            "collate_fn": opus_prefetch_collate_fn,
            "pin_memory": bool(pin_memory),
        }
        if worker_count > 0:
            loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
            loader_kwargs["persistent_workers"] = bool(persistent_workers)

        return DataLoader(**loader_kwargs)

    def run_processed_step_result(self, processed: dict[str, Any]) -> dict[str, Any]:
        data = opus_collate_fn([processed])
        data = cast(dict[str, Any], opus_move_to_device(data, self.device))
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                result = self.model(return_loss=False, rescale=True, **data)
        if isinstance(result, list):
            if not result:
                raise RuntimeError("OPUS 推理结果为空。")
            return cast(dict[str, Any], result[0])
        return cast(dict[str, Any], result)

    def run_step_result(self, sample: PreparedSample, step_idx: int) -> dict[str, Any]:
        current_idx = int(sample.step_source_indices[step_idx])
        return self.run_source_index_result(sample=sample, current_idx=current_idx)

    def build_processed_source_index(
        self,
        sample: PreparedSample,
        current_idx: int,
        *,
        force_offline_override: bool | None = None,
    ) -> Any:
        step_input = build_virtual_input_dict(
            base_input=sample.base_input,
            scene_name=sample.scene_name,
            base_sample_token=sample.token,
            frames=sample.frames,
            current_idx=current_idx,
        )
        if force_offline_override is None:
            processed = self.dataset.pipeline(step_input)
        else:
            with temporary_force_offline_sweeps_for_pipeline(
                getattr(self.dataset, "pipeline", None),
                bool(force_offline_override),
            ):
                processed = self.dataset.pipeline(step_input)
        if processed is None:
            raise RuntimeError(
                f"token={sample.token} current_idx={current_idx} 的 OPUS pipeline 返回了 None。"
            )
        return processed

    def run_source_index_result(
        self,
        sample: PreparedSample,
        current_idx: int,
        *,
        force_offline_override: bool | None = None,
    ) -> dict[str, Any]:
        processed = self.build_processed_source_index(
            sample=sample,
            current_idx=current_idx,
            force_offline_override=force_offline_override,
        )
        return self.run_processed_step_result(processed)

    def opus_result_to_fast_logits(
        self,
        result_dict: dict[str, Any],
        *,
        num_classes: int,
        free_index: int,
        grid_size: Sequence[int],
        other_fill_value: float,
        free_fill_value: float,
        topk: int | None = 3,
    ) -> torch.Tensor:
        """在内存中复现离线 fast logits 的 dense 重建，支持 top-k 或全量语义通道。"""
        logits_value = result_dict["logits"]
        occ_loc_value = result_dict["occ_loc"]

        if torch.is_tensor(logits_value):
            logits_t = logits_value.to(device=self.device, dtype=torch.float32, non_blocking=True)
        else:
            logits_t = torch.from_numpy(np.asarray(logits_value)).to(device=self.device, dtype=torch.float32)

        if torch.is_tensor(occ_loc_value):
            occ_loc_t = occ_loc_value.to(device=self.device, dtype=torch.long, non_blocking=True).reshape(-1, 3)
        else:
            occ_loc_t = torch.from_numpy(np.asarray(occ_loc_value, dtype=np.int64)).to(
                device=self.device,
                dtype=torch.long,
            ).reshape(-1, 3)

        x_size, y_size, z_size = [int(v) for v in grid_size]
        dense = torch.full(
            (int(num_classes), x_size, y_size, z_size),
            fill_value=float(other_fill_value),
            dtype=torch.float32,
            device=self.device,
        )
        if occ_loc_t.shape[0] == 0:
            dense[int(free_index)].fill_(float(free_fill_value))
            return dense

        if logits_t.ndim == 5 and logits_t.shape[0] == 1:
            logits_t = logits_t[0]

        if logits_t.ndim == 4:
            hit_logits = logits_t[occ_loc_t[:, 0], occ_loc_t[:, 1], occ_loc_t[:, 2]]
        elif logits_t.ndim == 2:
            if logits_t.shape[0] != occ_loc_t.shape[0]:
                raise RuntimeError(
                    "稀疏 logits 的体素数与 occ_loc 不一致: "
                    f"logits={tuple(logits_t.shape)} occ_loc={tuple(occ_loc_t.shape)}"
                )
            hit_logits = logits_t
        else:
            raise RuntimeError(f"不支持的 logits 形状: {tuple(logits_t.shape)}")

        sem_class_indices = torch.cat(
            [
                torch.arange(0, int(free_index), device=self.device, dtype=torch.long),
                torch.arange(int(free_index) + 1, int(num_classes), device=self.device, dtype=torch.long),
            ],
            dim=0,
        )

        if hit_logits.shape[-1] == int(num_classes):
            sem_logits = torch.cat(
                [hit_logits[:, : int(free_index)], hit_logits[:, int(free_index) + 1 :]],
                dim=-1,
            )
            free_logits = hit_logits[:, int(free_index)]
            # dense logits 输入里 free 通道可能仍然存在，先过滤掉被判成 free 的体素。
            not_free_mask = sem_logits.max(dim=-1).values > free_logits
            occ_loc_t = occ_loc_t[not_free_mask]
            sem_logits = sem_logits[not_free_mask]
        elif hit_logits.shape[-1] == int(num_classes) - 1:
            # 稀疏返回路径只包含语义通道，occ_loc 自身就代表占据体素，
            # 因此无需再和 free 通道比较。
            sem_logits = hit_logits
        else:
            raise RuntimeError(
                "logits 通道数与 num_classes 不匹配: "
                f"logits_last_dim={hit_logits.shape[-1]} num_classes={num_classes}"
            )

        if occ_loc_t.shape[0] == 0:
            dense[int(free_index)].fill_(float(free_fill_value))
            return dense

        hit_mask = torch.zeros((x_size, y_size, z_size), dtype=torch.bool, device=self.device)
        x_idx = occ_loc_t[:, 0]
        y_idx = occ_loc_t[:, 1]
        z_idx = occ_loc_t[:, 2]
        hit_mask[x_idx, y_idx, z_idx] = True

        dense[int(free_index), ~hit_mask] = float(free_fill_value)
        if topk is None or int(topk) <= 0 or int(topk) >= int(sem_logits.shape[-1]):
            dense[
                sem_class_indices[:, None],
                x_idx[None, :],
                y_idx[None, :],
                z_idx[None, :],
            ] = sem_logits.transpose(0, 1).contiguous()
            return dense

        topk_k = min(int(topk), int(sem_logits.shape[-1]))
        values_t, topk_pos = torch.topk(sem_logits, k=topk_k, dim=-1)
        indices_t = sem_class_indices[topk_pos]
        x_rep = x_idx.repeat_interleave(topk_k)
        y_rep = y_idx.repeat_interleave(topk_k)
        z_rep = z_idx.repeat_interleave(topk_k)
        c_rep = indices_t.reshape(-1)
        v_rep = values_t.reshape(-1)
        dense[c_rep, x_rep, y_rep, z_rep] = v_rep
        return dense
