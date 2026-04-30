"""Occ3D-nuScenes online_ncde 数据集。"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from online_ncde.config import resolve_path
from online_ncde.data.labels_io import load_labels_npz
from online_ncde.data.logits_loader import LogitsLoader
from online_ncde.data.ray_sidecar import RaySidecar

# 进程级 flag：缺 token warning 每个 worker 进程只打一次
_RAY_MISSING_WARNED = False


class Occ3DOnlineNcdeDataset(Dataset):
    """读取 online_ncde 训练/评估样本。"""

    def __init__(
        self,
        info_path: str,
        root_path: str,
        gt_root: str,
        num_classes: int,
        free_index: int,
        grid_size: Tuple[int, int, int],
        logits_loader: LogitsLoader,
        gt_mask_key: str = "mask_camera",
        ray_sidecar_dir: str | None = None,
        ray_sidecar_split: str | None = None,
        fast_frame_stride: int = 1,
        min_history_completeness: int | None = None,
        eval_only_mode: bool = False,
    ) -> None:
        self.root_path = root_path
        self.info_path = resolve_path(root_path, info_path)
        self.gt_root = resolve_path(root_path, gt_root)
        self.num_classes = int(num_classes)
        self.free_index = int(free_index)
        self.grid_size = tuple(grid_size)
        self.gt_mask_key = gt_mask_key
        if logits_loader is None:
            raise ValueError("logits_loader 为必填项，需通过 build_logits_loader 构造。")
        self.logits_loader = logits_loader
        self.fast_frame_stride = int(fast_frame_stride)
        if self.fast_frame_stride < 1:
            raise ValueError(
                f"fast_frame_stride 必须 >= 1，当前为 {fast_frame_stride!r}"
            )
        self.supervision_labels = ["t-1.5", "t-1.0", "t-0.5", "t"]
        self.supervision_by_token: Dict[str, Dict[str, Any]] | None = None
        # ray sidecar 延后到 supervision_labels 定版后再构造，方便做顺序校验
        self.ray_sidecar: RaySidecar | None = None
        self._ray_sidecar_dir = ray_sidecar_dir
        self._ray_sidecar_split = ray_sidecar_split

        with open(self.info_path, "rb") as f:
            payload = pickle.load(f)
        # 拒绝 evolve_infos schema（start-anchored 评估专用），除非显式 eval_only_mode=True。
        # 防止被误指给 train/eval_online_ncde 等入口：那些入口默认按 token=current keyframe 取
        # GT/sup，对 start-anchored sample（token=start keyframe）会算到错位 GT。
        _md = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        schema_ver = str(_md.get("schema_version", ""))
        if schema_ver == "online_ncde_evolve_infos_v1" and not eval_only_mode:
            raise ValueError(
                f"info_path={info_path} 是 start-anchored evolve_infos schema，"
                "只能用 scripts/eval_online_ncde_evolution_times.py（传 eval_only_mode=True）评估。"
                "其他入口（train / eval_online_ncde / stepwise）请使用 canonical_infos pkl，"
                "否则会用 start keyframe 当 current GT 算出错位的 loss/mIoU。"
            )
        self._is_evolve_schema = schema_ver == "online_ncde_evolve_infos_v1"
        infos = payload["infos"] if isinstance(payload, dict) else payload
        self.infos: List[Dict[str, Any]] = [info for info in infos if info.get("valid", True)]
        # 训练默认只吃完整 2s 历史的样本，评估可以显式传 0 覆盖全集。
        # 若 pkl 不含 history_completeness 字段（旧格式），视为 history_keyframes（兼容）。
        if min_history_completeness is not None:
            thr = int(min_history_completeness)
            self.infos = [
                info for info in self.infos
                if int(info.get("history_completeness", info.get("history_keyframes", 0))) >= thr
            ]
        self.min_history_completeness = min_history_completeness

        if self.infos and "supervision_mask" in self.infos[0]:
            # canonical pkl 直接包含多帧监督字段
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            labels = metadata.get("supervision_labels", self.supervision_labels)
            if isinstance(labels, list) and labels:
                self.supervision_labels = [str(x) for x in labels]
            self.supervision_by_token = {}
            for info in self.infos:
                token = str(info.get("token", ""))
                if token:
                    self.supervision_by_token[token] = info

        # supervision_labels 已定版，构造 ray sidecar 并校验时刻顺序
        if self._ray_sidecar_dir:
            split = self._ray_sidecar_split or "train"
            self.ray_sidecar = RaySidecar(
                sidecar_dir=resolve_path(root_path, self._ray_sidecar_dir),
                split=split,
            )
            ray_labels = list(self.ray_sidecar.supervision_labels)
            if ray_labels and ray_labels != self.supervision_labels:
                raise ValueError(
                    "ray sidecar 与 dataset 的 supervision_labels 不一致，"
                    "会把 ray GT 监督到错误时刻：\n"
                    f"  dataset: {self.supervision_labels}\n"
                    f"  ray    : {ray_labels}"
                )

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.infos[idx]
        _ctx_scene = str(info.get("scene_name", ""))
        _ctx_token = str(info.get("token", ""))
        device = torch.device("cpu")
        label_cache: Dict[str, Dict[str, Any]] = {}

        def _load_label_cached(path: str) -> Dict[str, Any]:
            if path not in label_cache:
                label_cache[path] = load_labels_npz(path)
            return label_cache[path]

        # stride>1 时抽帧：用切片而非 fancy index，numpy 数组零拷贝（view）。
        # 要求 (num_frames-1) 是 stride 的倍数，保证 supervision_step_indices 整除对齐。
        stride = self.fast_frame_stride
        if stride > 1:
            frame_paths_src = info.get("frame_rel_paths", None)
            if not frame_paths_src:
                raise KeyError(
                    "fast_frame_stride > 1 需要 info 里有 frame_rel_paths，"
                    "当前 info 不含该字段（走的是旧格式？）"
                )
            num_full_frames = len(frame_paths_src)
            if num_full_frames < 2 or (num_full_frames - 1) % stride != 0:
                raise ValueError(
                    f"fast_frame_stride={stride} 与 num_frames={num_full_frames} 不整除，"
                    "无法对齐 supervision_step_indices。"
                    f" [scene={_ctx_scene} token={_ctx_token}]"
                )
            info_view: Dict[str, Any] = dict(info)
            info_view["frame_rel_paths"] = frame_paths_src[::stride]
            for key in ("frame_ego2global", "frame_timestamps", "frame_dt"):
                if key in info_view:
                    info_view[key] = info[key][::stride]
            # frame_tokens 是 list[str]，也要按 stride 抽帧，让 meta 里的 token
            # 索引与 aligner 的 step_indices（抽帧后坐标）保持一致。
            if "frame_tokens" in info_view:
                info_view["frame_tokens"] = list(info["frame_tokens"])[::stride]
            # rollout_start_step 在抽帧后也要重映射，要求与 stride 整除
            if "rollout_start_step" in info_view:
                rss = int(info_view["rollout_start_step"])
                if rss % stride != 0:
                    raise ValueError(
                        f"rollout_start_step={rss} 不是 fast_frame_stride={stride} 的整数倍，"
                        "无法对齐到抽帧后的 rollout step。"
                        f" [scene={_ctx_scene} token={_ctx_token}]"
                    )
                info_view["rollout_start_step"] = rss // stride
            # evolve_infos schema：抽帧后同步重映射 step / num_real_frames，并切 mask
            if "frame_valid_mask" in info_view:
                info_view["frame_valid_mask"] = info["frame_valid_mask"][::stride]
            if "num_real_frames" in info_view:
                num_real = int(info_view["num_real_frames"])
                if (num_real - 1) % stride != 0:
                    raise ValueError(
                        f"num_real_frames-1={num_real-1} 不是 fast_frame_stride={stride} 的整数倍。"
                        f" [scene={_ctx_scene} token={_ctx_token}]"
                    )
                info_view["num_real_frames"] = (num_real - 1) // stride + 1
            if "evolve_keyframe_step_indices" in info_view:
                ek_steps = list(info["evolve_keyframe_step_indices"])
                remapped: List[int] = []
                for s in ek_steps:
                    s_int = int(s)
                    if s_int % stride != 0:
                        raise ValueError(
                            f"evolve_keyframe_step_indices={s_int} 不是 stride={stride} 的整数倍。"
                            f" [scene={_ctx_scene} token={_ctx_token}]"
                        )
                    remapped.append(s_int // stride)
                info_view["evolve_keyframe_step_indices"] = remapped
            info = info_view

        fast_logits = self.logits_loader.load_fast_logits(info, device)
        slow_logits = self.logits_loader.load_slow_logits(info, device)
        num_frames = fast_logits.shape[0]

        frame_ego2global = torch.from_numpy(info["frame_ego2global"]).float()

        scene_name = info.get("scene_name", "")
        token = info.get("token", "")
        gt_labels, gt_mask = self._load_curr_gt(info, _load_label_cached)

        sup_labels = None
        sup_masks = None
        sup_step_indices = None
        sup_valid_mask = None
        if self.supervision_by_token is not None:
            sup_entry = self.supervision_by_token.get(token, None)
            if sup_entry is None:
                raise KeyError(f"canonical pkl 缺少 token={token} 的监督条目。")
            num_sup = len(self.supervision_labels)
            sup_labels = torch.zeros((num_sup, *self.grid_size), dtype=torch.long)
            sup_masks = torch.zeros((num_sup, *self.grid_size), dtype=torch.float32)
            sup_step_indices = torch.full((num_sup,), -1, dtype=torch.long)
            sup_valid_mask = torch.zeros((num_sup,), dtype=torch.float32)

            sup_mask_list = sup_entry.get("supervision_mask", [0] * num_sup)
            sup_steps = sup_entry.get("supervision_step_indices", [-1] * num_sup)
            sup_rel_paths = sup_entry.get("supervision_gt_rel_paths", [""] * num_sup)

            for sup_i in range(num_sup):
                valid = int(sup_mask_list[sup_i]) if sup_i < len(sup_mask_list) else 0
                if valid <= 0:
                    continue
                step_idx = int(sup_steps[sup_i]) if sup_i < len(sup_steps) else -1
                gt_rel = str(sup_rel_paths[sup_i]) if sup_i < len(sup_rel_paths) else ""
                if step_idx < 0 or not gt_rel:
                    continue
                if self.fast_frame_stride > 1:
                    # 抽帧后 aligner 的 step_indices 为 arange(1, T_sub)，
                    # 原始 [3,6,9,12] 需要按 stride 重映射到 [1,2,3,4]。
                    if step_idx % self.fast_frame_stride != 0:
                        raise ValueError(
                            f"supervision_step_indices={step_idx} 不是 "
                            f"fast_frame_stride={self.fast_frame_stride} 的整数倍，"
                            "无法对齐到抽帧后的 rollout step。"
                        )
                    step_idx = step_idx // self.fast_frame_stride
                loaded = self._load_sup_gt(info, sup_i, gt_rel, _load_label_cached)
                if loaded is None:
                    continue
                sup_sem, sup_m = loaded

                sup_labels[sup_i] = sup_sem
                sup_masks[sup_i] = sup_m
                sup_step_indices[sup_i] = step_idx
                sup_valid_mask[sup_i] = 1.0

        frame_timestamps = info.get("frame_timestamps", None)
        if frame_timestamps is not None:
            frame_timestamps = torch.from_numpy(frame_timestamps).long()
        frame_dt = info.get("frame_dt", None)
        if frame_dt is not None:
            frame_dt = torch.from_numpy(frame_dt).float()

        ray_gt_dist = None
        ray_origin = None
        ray_sup_valid = None
        ray_origin_mask = None
        if self.ray_sidecar is not None:
            hit = self.ray_sidecar.query(token)
            num_sup = len(self.supervision_labels)
            num_rays = self.ray_sidecar.num_rays
            num_origins = self.ray_sidecar.num_origins
            if hit is None:
                # token 缺失：整条样本跳过 ray loss。每个 worker 进程首次遇到时
                # 打印一次 warning（DataLoader 多 worker 下无法做到全局唯一）。
                global _RAY_MISSING_WARNED
                if not _RAY_MISSING_WARNED:
                    pid = os.getpid()
                    print(
                        f"[ray_sidecar pid={pid}] WARN: token={token} 缺失，"
                        f"该样本将跳过 ray loss；本 worker 后续缺失不再提示。"
                    )
                    _RAY_MISSING_WARNED = True
                ray_gt_dist = torch.full(
                    (num_sup, num_origins, num_rays), float("nan"), dtype=torch.float32
                )
                ray_origin = torch.zeros((num_sup, num_origins, 3), dtype=torch.float32)
                ray_origin_mask = torch.zeros((num_sup, num_origins), dtype=torch.float32)
                ray_sup_valid = torch.zeros((num_sup,), dtype=torch.float32)
            else:
                dist_np, origin_np, sup_mask_np, origin_mask_np = hit
                ray_gt_dist = torch.from_numpy(dist_np)                    # (sup, K, R)
                ray_origin = torch.from_numpy(origin_np)                   # (sup, K, 3)
                ray_sup_valid = torch.from_numpy(sup_mask_np.astype("float32"))  # (sup,)
                ray_origin_mask = torch.from_numpy(origin_mask_np.astype("float32"))  # (sup, K)

        # rollout 起点：旧格式 pkl 无此字段，默认 0（完整历史）
        rollout_start_step = int(info.get("rollout_start_step", 0))
        history_completeness = int(
            info.get("history_completeness", info.get("history_keyframes", 0))
        )

        return {
            "fast_logits": fast_logits,
            "slow_logits": slow_logits,
            "frame_ego2global": frame_ego2global,
            "frame_timestamps": frame_timestamps,
            "frame_dt": frame_dt,
            "gt_labels": gt_labels,
            "gt_mask": gt_mask,
            "sup_labels": sup_labels,
            "sup_masks": sup_masks,
            "sup_step_indices": sup_step_indices,
            "sup_valid_mask": sup_valid_mask,
            "ray_gt_dist": ray_gt_dist,
            "ray_origin": ray_origin,
            "ray_origin_mask": ray_origin_mask,
            "ray_sup_valid": ray_sup_valid,
            "rollout_start_step": torch.tensor(rollout_start_step, dtype=torch.long),
            "history_completeness": torch.tensor(history_completeness, dtype=torch.long),
            "meta": {
                "scene_name": scene_name,
                "token": token,
                "frame_tokens": info.get("frame_tokens", []),
                "logits_path": info.get("logits_path", ""),
                "slow_logit_path": info.get("slow_logit_path", ""),
                "supervision_labels": self.supervision_labels,
                "rollout_start_step": rollout_start_step,
                "history_completeness": history_completeness,
                # evolve_infos schema：start-anchored 评估专用字段。旧 schema 下这些字段缺失，
                # 调用方读 .get(...) 即可（评估脚本只在新 schema 下用）。
                "num_real_frames": int(info.get("num_real_frames", num_frames)),
                "max_evolve_keyframes": int(info.get("max_evolve_keyframes", -1)),
                "evolve_keyframe_step_indices": list(
                    info.get("evolve_keyframe_step_indices", [])
                ),
                "evolve_keyframe_sample_tokens": list(
                    info.get("evolve_keyframe_sample_tokens", [])
                ),
                "evolve_keyframe_gt_exists": list(
                    info.get("evolve_keyframe_gt_exists", [])
                ),
                "start_sample_token": str(info.get("start_sample_token", "")),
                "start_keyframe_local_idx": int(info.get("start_keyframe_local_idx", -1)),
            },
        }

    # ------- GT 加载钩子（子类可 override 以支持其它 GT 格式，如 OpenOccupancy） ------- #
    def _load_curr_gt(
        self,
        info: Dict[str, Any],
        load_npz_cached,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取当前帧 GT，返回 (semantics: long (X,Y,Z), mask: float (X,Y,Z))。

        Occ3D 默认实现：从 `<gt_root>/<scene_name>/<token>/labels.npz` 读
        `semantics` 与 `gt_mask_key` 字段（缺 mask 时全 1）。
        """
        scene_name = info.get("scene_name", "")
        token = info.get("token", "")
        curr_gt_path = os.path.join(self.gt_root, scene_name, token, "labels.npz")
        curr_npz = load_npz_cached(curr_gt_path)
        gt_labels = torch.from_numpy(curr_npz["semantics"].astype("int64"))
        gt_mask_np = curr_npz.get(self.gt_mask_key, None)
        if gt_mask_np is None:
            gt_mask_np = torch.ones(self.grid_size, dtype=torch.float32).numpy()
        gt_mask = torch.from_numpy(gt_mask_np.astype("float32"))
        return gt_labels, gt_mask

    def _load_sup_gt(
        self,
        info: Dict[str, Any],
        sup_index: int,
        gt_rel: str,
        load_npz_cached,
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        """读取指定 supervision 帧 GT，返回 (semantics, mask)，找不到返回 None。

        Occ3D 默认实现：以 pkl 中的 `supervision_gt_rel_paths[sup_index]` 拼路径。
        """
        gt_path = gt_rel if os.path.isabs(gt_rel) else os.path.join(self.gt_root, gt_rel)
        if not os.path.exists(gt_path):
            return None
        sup_npz = load_npz_cached(gt_path)
        sup_sem = torch.from_numpy(sup_npz["semantics"].astype("int64"))
        sup_mask_np = sup_npz.get(self.gt_mask_key, None)
        if sup_mask_np is None:
            sup_mask_np = torch.ones(self.grid_size, dtype=torch.float32).numpy()
        sup_m = torch.from_numpy(sup_mask_np.astype("float32"))
        return sup_sem, sup_m
