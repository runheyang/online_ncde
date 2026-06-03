#!/usr/bin/env python3
"""生成 Online NCDE 的 start-anchored evolve info pkl。

设计目标：
- 每个 sample 以某个 keyframe K_j 为 slow 锚（"推演起点"）。
- 推演终点 = K_{j + max_evolve}，其中
  max_evolve = min(history_keyframes, scene_keyframe_total - 1 - j)。
- 这样一次完整 forward 就天然覆盖了 0.5/1.0/.../max_evolve*0.5 秒所有"推演长度"。
- scene 末尾 sample 自然只贡献到短桶（推不到 5s 就只推 4.5/4.0/.../0.5s）。

变长处理：所有 sample 的 num_output_frames 统一 pad 到固定长度
(history_keyframes * steps_per_interval + 1)，末尾 pad；pad 段：
- frame_rel_paths = ""（loader 跳过 IO，返回零张量占位）
- frame_ego2global = 复制最后一真实帧（pad 段 transform 为单位阵）
- frame_dt = 0（pad 段 ODE 不积分）
- frame_valid_mask = 0
评估脚本只取 step <= num_real_frames-1 的输出。
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from pyquaternion import Quaternion
from nuscenes import NuScenes  # type: ignore[import-not-found]

try:
    import progressbar
except Exception:  # pragma: no cover
    progressbar = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_CAMERA = "CAM_FRONT"

SCHEMA_VERSION = "online_ncde_evolve_infos_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成 online_ncde start-anchored evolve infos。")
    p.add_argument("--src-info", default="data/nuscenes/nuscenes_infos_val_sweep.pkl",
                   help="输入 sweep pkl，用于确定 split 与 keyframe sample 集合。")
    p.add_argument("--output", default="",
                   help="输出 pkl 路径；空则默认 configs/online_ncde/evolve_infos_<split>.pkl")
    p.add_argument("--data-root", default="data/nuscenes", help="NuScenes dataroot")
    p.add_argument("--version", default="v1.0-trainval", help="NuScenes 版本")
    p.add_argument("--token-camera", default=DEFAULT_TOKEN_CAMERA,
                   help="用于定义 fast frame token 的相机通道")
    p.add_argument("--history-keyframes", type=int, default=10,
                   help="单个 sample 的最长推演 keyframe 数（默认 10 = 5s）")
    p.add_argument("--steps-per-interval", type=int, default=3,
                   help="每个 0.5s keyframe interval 重采样为多少个快系统 step（默认 3 = 6Hz）")
    p.add_argument("--gt-root", default="data/nuscenes/gts",
                   help="GT labels 根目录（相对项目根）")
    p.add_argument("--max-scenes", type=int, default=0,
                   help="可选：仅处理前 N 个 scene，0 表示全量")
    p.add_argument("--min-evolve-keyframes", type=int, default=1,
                   help="最小 max_evolve_keyframes，<该值的样本被跳过（默认 1，即至少能推 0.5s）")
    p.add_argument("--validate-fast-logits-root", default="",
                   help="可选：fast logits 根目录（相对项目根）。传了则预检每个非 pad "
                        "frame_rel_paths 文件存在，缺失就跳过该 sample 并计数。")
    p.add_argument("--validate-slow-logits-root", default="",
                   help="可选：slow logits 根目录。传了则预检 slow_logit_path。")
    return p.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def infer_split_name(path: Path) -> str:
    stem = path.stem.lower()
    if "train" in stem:
        return "train"
    if "val" in stem:
        return "val"
    if "test" in stem:
        return "test"
    return "unknown"


def build_default_output_path(src_info_path: Path) -> Path:
    split = infer_split_name(src_info_path)
    name = "evolve_infos.pkl" if split == "unknown" else f"evolve_infos_{split}.pkl"
    return (ROOT / "configs" / "online_ncde" / name).resolve()


def load_source_infos(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "infos" in payload:
        infos = payload["infos"]
        metadata = payload.get("metadata", {})
    else:
        infos = payload
        metadata = {}
    if not isinstance(infos, list):
        raise TypeError(f"infos 类型异常: {type(infos)}")
    return infos, metadata


def coerce_valid_flag(flag: Any) -> bool:
    if isinstance(flag, np.ndarray):
        if flag.ndim == 0 or flag.size == 1:
            return bool(flag.reshape(()))
        return True
    if isinstance(flag, (list, tuple)):
        if len(flag) == 1:
            return coerce_valid_flag(flag[0])
        return True
    return bool(flag)


def build_pose_matrix(translation: list[float], rotation: list[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Quaternion(rotation).rotation_matrix.astype(np.float32)
    pose[:3, 3] = np.asarray(translation, dtype=np.float32)
    return pose


def group_infos_by_scene(infos: list[dict[str, Any]]) -> "OrderedDict[str, dict[str, Any]]":
    grouped: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for src_index, info in enumerate(sorted(infos, key=lambda item: item["timestamp"])):
        scene_token = str(info.get("scene_token", ""))
        scene_name = str(info.get("scene_name", ""))
        if not scene_token or not scene_name:
            raise KeyError("源 sweep pkl 缺少 scene_token 或 scene_name。")
        if scene_token not in grouped:
            grouped[scene_token] = {"scene_name": scene_name, "infos": []}
        info_copy = dict(info)
        info_copy["_source_index"] = int(src_index)
        grouped[scene_token]["infos"].append(info_copy)
    return grouped


def collect_interval_camera_tokens(
    start_sample_token: str,
    end_sample_token: str,
    token_camera: str,
    scene_token: str,
    sample_by_token: dict[str, dict[str, Any]],
    sample_data_by_token: dict[str, dict[str, Any]],
) -> tuple[list[str], str]:
    """复用 canonical 脚本的逻辑：沿 token_camera 链收集 [start, end) 的 sample_data。"""
    start_sample = sample_by_token[start_sample_token]
    end_sample = sample_by_token[end_sample_token]
    start_sd_token = str(start_sample["data"][token_camera])
    end_sd_token = str(end_sample["data"][token_camera])

    tokens: list[str] = []
    cursor = start_sd_token
    while cursor and cursor != end_sd_token:
        sd_rec = sample_data_by_token[cursor]
        if str(sd_rec.get("channel", "")) != token_camera:
            raise ValueError(f"期望通道 {token_camera}，实际为 {sd_rec.get('channel')}")
        sample_token = str(sd_rec.get("sample_token", ""))
        if str(sample_by_token[sample_token].get("scene_token", "")) != scene_token:
            raise ValueError(
                f"{token_camera} 链跨 scene 了：{start_sample_token} -> {end_sample_token}"
            )
        tokens.append(cursor)
        cursor = str(sd_rec.get("next", ""))

    if cursor != end_sd_token:
        raise ValueError(
            f"未能沿 {token_camera} 链到达下一个 keyframe：{start_sample_token} -> {end_sample_token}"
        )
    if not tokens:
        raise ValueError(
            f"{token_camera} interval 为空：{start_sample_token} -> {end_sample_token}"
        )
    return tokens, end_sd_token


def select_interval_local_indices(
    interval_tokens: list[str],
    end_token: str,
    sample_data_by_token: dict[str, dict[str, Any]],
    steps_per_interval: int,
) -> list[int]:
    """复用 canonical 脚本的最近邻选帧规则。"""
    if steps_per_interval <= 0:
        raise ValueError("steps_per_interval 必须为正。")

    timestamps = np.asarray(
        [int(sample_data_by_token[token]["timestamp"]) for token in interval_tokens],
        dtype=np.int64,
    )
    start_ts = int(timestamps[0])
    end_ts = int(sample_data_by_token[end_token]["timestamp"])
    duration = end_ts - start_ts

    selected: list[int] = []
    min_idx = 0
    for phase_idx in range(steps_per_interval):
        frac = phase_idx / steps_per_interval
        target_ts = start_ts + duration * frac
        candidate_ts = timestamps[min_idx:]
        rel_idx = int(np.argmin(np.abs(candidate_ts.astype(np.float64) - float(target_ts))))
        idx = min_idx + rel_idx
        selected.append(idx)
        min_idx = idx
    return selected


def make_logit_rel_path(scene_name: str, token: str) -> str:
    return str(Path(scene_name) / token / "logits.npz")


def make_gt_rel_path(scene_name: str, sample_token: str) -> str:
    return str(Path(scene_name) / sample_token / "labels.npz")


def main() -> None:
    args = parse_args()
    src_info_path = resolve_repo_path(args.src_info)
    output_path = (
        resolve_repo_path(args.output) if args.output else build_default_output_path(src_info_path)
    )
    data_root = resolve_repo_path(args.data_root)
    split_name = infer_split_name(src_info_path)

    if args.history_keyframes <= 0:
        raise ValueError("--history-keyframes 必须为正。")
    if args.steps_per_interval <= 0:
        raise ValueError("--steps-per-interval 必须为正。")
    if args.min_evolve_keyframes < 1 or args.min_evolve_keyframes > args.history_keyframes:
        raise ValueError(
            f"--min-evolve-keyframes 须在 [1, {args.history_keyframes}]，"
            f"当前 {args.min_evolve_keyframes}"
        )

    gt_root_abs = str(resolve_repo_path(args.gt_root))

    source_infos, source_metadata = load_source_infos(src_info_path)
    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=False)

    sample_by_token = {rec["token"]: rec for rec in nusc.sample}
    sample_data_by_token = {rec["token"]: rec for rec in nusc.sample_data}
    ego_pose_by_token = {rec["token"]: rec for rec in nusc.ego_pose}

    grouped = group_infos_by_scene(source_infos)
    scene_items = list(grouped.items())
    if args.max_scenes > 0:
        scene_items = scene_items[: args.max_scenes]

    pose_cache: dict[str, np.ndarray] = {}

    def get_pose_matrix_from_frame_token(frame_token: str) -> np.ndarray:
        if frame_token not in pose_cache:
            sd_rec = sample_data_by_token[frame_token]
            pose_rec = ego_pose_by_token[str(sd_rec["ego_pose_token"])]
            pose_cache[frame_token] = build_pose_matrix(
                translation=pose_rec["translation"],
                rotation=pose_rec["rotation"],
            )
        return pose_cache[frame_token]

    history_keyframes = int(args.history_keyframes)
    steps_per_interval = int(args.steps_per_interval)
    fixed_num_output_frames = history_keyframes * steps_per_interval + 1

    infos_out: list[dict[str, Any]] = []
    index_by_token: dict[str, int] = {}

    interval_len_hist: Counter[int] = Counter()
    max_evolve_hist: Counter[int] = Counter()
    skipped_short = 0
    skipped_no_gt = 0
    bad_lidar_meta_count = 0
    skipped_invalid_source = 0
    skipped_no_fast_logits = 0
    skipped_no_slow_logits = 0

    fast_root_abs = (
        str(resolve_repo_path(args.validate_fast_logits_root))
        if args.validate_fast_logits_root else ""
    )
    slow_root_abs = (
        str(resolve_repo_path(args.validate_slow_logits_root))
        if args.validate_slow_logits_root else ""
    )
    if fast_root_abs:
        print(f"[validate] fast logits root: {fast_root_abs}")
    if slow_root_abs:
        print(f"[validate] slow logits root: {slow_root_abs}")
    valid_count = 0

    iterator = (
        progressbar.progressbar(scene_items, prefix="[gen evolve infos] ")
        if progressbar is not None else scene_items
    )
    for scene_token, scene_group in iterator:
        scene_infos = scene_group["infos"]
        scene_name = scene_group["scene_name"]
        n_kf = len(scene_infos)

        for j, start_info in enumerate(scene_infos):
            # max_evolve = 该 start 实际能推演的 keyframe 数（受 scene 末尾限制）
            max_evolve = min(history_keyframes, n_kf - 1 - j)
            if max_evolve < args.min_evolve_keyframes:
                skipped_short += 1
                continue

            start_sample_token = str(start_info["token"])
            # start keyframe 必须有 GT 才能作为评估的"0s 锚点"参考；
            # 不过 0 step 我们不评估，主要是为了 dataset 加载 curr_gt_rel_path 不报错。
            start_gt_rel = make_gt_rel_path(scene_name, start_sample_token)
            start_gt_abs = str(Path(gt_root_abs) / start_gt_rel)
            if not Path(start_gt_abs).exists():
                skipped_no_gt += 1
                continue

            # source sweep pkl 的 valid_flag 一般用于过滤异常 sample；这里只剔除明显损坏的
            if not coerce_valid_flag(start_info.get("valid_flag", True)):
                skipped_invalid_source += 1
                continue

            # 构造 [K_j, K_{j+1}, ..., K_{j+max_evolve}] 间的 fast frame 序列
            evolve_keyframe_infos = scene_infos[j : j + max_evolve + 1]
            evolve_keyframe_sample_tokens = [str(info["token"]) for info in evolve_keyframe_infos]

            real_frame_tokens: list[str] = []
            real_frame_interval_indices: list[int] = []
            real_frame_phase_indices: list[int] = []
            interval_frame_counts: list[int] = []
            try:
                for itv_i in range(max_evolve):
                    s_tok = evolve_keyframe_sample_tokens[itv_i]
                    e_tok = evolve_keyframe_sample_tokens[itv_i + 1]
                    interval_tokens, end_frame_token = collect_interval_camera_tokens(
                        start_sample_token=s_tok,
                        end_sample_token=e_tok,
                        token_camera=args.token_camera,
                        scene_token=scene_token,
                        sample_by_token=sample_by_token,
                        sample_data_by_token=sample_data_by_token,
                    )
                    selected_local_indices = select_interval_local_indices(
                        interval_tokens=interval_tokens,
                        end_token=end_frame_token,
                        sample_data_by_token=sample_data_by_token,
                        steps_per_interval=steps_per_interval,
                    )
                    interval_frame_counts.append(len(interval_tokens))
                    interval_len_hist[len(interval_tokens)] += 1
                    for phase_idx, local_frame_idx in enumerate(selected_local_indices):
                        real_frame_tokens.append(interval_tokens[local_frame_idx])
                        real_frame_interval_indices.append(itv_i)
                        real_frame_phase_indices.append(phase_idx)
                # 末帧 = K_{j+max_evolve} 的 token_camera frame
                end_kf_sample_token = evolve_keyframe_sample_tokens[-1]
                end_kf_frame_token = str(
                    sample_by_token[end_kf_sample_token]["data"][args.token_camera]
                )
                real_frame_tokens.append(end_kf_frame_token)
                real_frame_interval_indices.append(max_evolve)
                real_frame_phase_indices.append(0)
            except Exception as e:
                # 极少数 scene 帧链异常，跳过
                print(f"[skip] scene={scene_name} j={j}: {e}")
                continue

            num_real_frames = max_evolve * steps_per_interval + 1
            assert len(real_frame_tokens) == num_real_frames

            # 加载真实帧的 ts / pose
            real_ts_list: list[int] = []
            real_pose_list: list[np.ndarray] = []
            real_is_kf_list: list[int] = []
            for tok in real_frame_tokens:
                sd_rec = sample_data_by_token[tok]
                real_ts_list.append(int(sd_rec["timestamp"]))
                real_pose_list.append(get_pose_matrix_from_frame_token(tok))
                real_is_kf_list.append(1 if bool(sd_rec["is_key_frame"]) else 0)

            # pad 到 fixed_num_output_frames：末尾 pad
            pad_steps = fixed_num_output_frames - num_real_frames
            last_ts = real_ts_list[-1]
            last_pose = real_pose_list[-1]

            frame_tokens = real_frame_tokens + [""] * pad_steps
            frame_interval_indices = real_frame_interval_indices + [-1] * pad_steps
            frame_phase_indices = real_frame_phase_indices + [-1] * pad_steps
            frame_is_kf = real_is_kf_list + [0] * pad_steps
            frame_valid_mask = [1] * num_real_frames + [0] * pad_steps
            ts_full = real_ts_list + [last_ts] * pad_steps
            pose_full = real_pose_list + [last_pose] * pad_steps

            frame_timestamps = np.asarray(ts_full, dtype=np.int64)
            # frame_dt: 真实段沿用 (ts - ts_last_real) * 1e-6；pad 段强制 0，
            # 让 ODE 在 pad 段 dt=0 不积分（不影响真实段输出）。
            real_dt = ((frame_timestamps[:num_real_frames] - last_ts).astype(np.float64) * 1.0e-6)
            pad_dt = np.zeros((pad_steps,), dtype=np.float64)
            frame_dt = np.concatenate([real_dt, pad_dt]).astype(np.float32)

            frame_ego2global = np.stack(pose_full, axis=0).astype(np.float32)
            frame_is_keyframe = np.asarray(frame_is_kf, dtype=np.uint8)
            frame_valid_mask_arr = np.asarray(frame_valid_mask, dtype=np.uint8)

            frame_rel_paths = [
                make_logit_rel_path(scene_name, tok) if tok else "" for tok in frame_tokens
            ]
            frame_sample_tokens = [
                str(sample_data_by_token[tok]["sample_token"]) if tok else "" for tok in frame_tokens
            ]

            # 可选：预检 fast/slow logits 文件存在，缺失则跳过本 sample（fail-fast）。
            slow_rel = make_logit_rel_path(scene_name, start_sample_token)
            if fast_root_abs:
                missing_fast = next(
                    (rel for rel in frame_rel_paths
                     if rel and not Path(fast_root_abs, rel).exists()),
                    None,
                )
                if missing_fast is not None:
                    skipped_no_fast_logits += 1
                    continue
            if slow_root_abs and not Path(slow_root_abs, slow_rel).exists():
                skipped_no_slow_logits += 1
                continue

            # evolve keyframe 在帧序列中的 step 位置：[0, spi, 2*spi, ..., max_evolve*spi]
            evolve_keyframe_step_indices = [
                k * steps_per_interval for k in range(max_evolve + 1)
            ]
            # GT 检查：对每个 evolve keyframe（除起点），确认 GT 文件存在；缺失的 keyframe 标空
            evolve_keyframe_gt_rel_paths: list[str] = []
            evolve_keyframe_gt_exists: list[int] = []
            for kf_token in evolve_keyframe_sample_tokens:
                rel = make_gt_rel_path(scene_name, kf_token)
                exists = Path(gt_root_abs, rel).exists()
                evolve_keyframe_gt_rel_paths.append(rel)
                evolve_keyframe_gt_exists.append(1 if exists else 0)

            # slow anchor = K_j
            slow_pose = real_pose_list[0]
            slow_ts = real_ts_list[0]

            # lidar2ego（用于 RayIoU）：source sweep info 直接给的是 K_j 的，先存下来；
            # 评估时各 evolve keyframe 的 lidar origin 还是从外部 sweep pkl 按 token 查（与现脚本一致）
            lidar2ego_translation = np.asarray(
                start_info.get("lidar2ego_translation", np.zeros(3)), dtype=np.float32
            )
            lidar2ego_rotation = np.asarray(
                start_info.get("lidar2ego_rotation", np.zeros(4)), dtype=np.float32
            )
            # lidar2ego shape 异常时记一次 metadata（不跳过样本：start keyframe 本身的
            # lidar2ego 不参与 RayIoU 计算——RayIoU 用的是 evaluate target keyframe 的 origin，
            # 评估时按 token 从 sweep_pkl 查；此字段保留只是为了 backward-compat 占位）。
            if lidar2ego_translation.shape != (3,):
                bad_lidar_meta_count += 1

            entry: dict[str, Any] = {
                # 标识
                "token": start_sample_token,           # 兼容现有 dataset：以 start 为 token
                "start_sample_token": start_sample_token,
                "start_keyframe_local_idx": int(j),
                "scene_name": scene_name,
                "scene_token": str(scene_token),
                "split": split_name,
                "source_index": int(start_info.get("_source_index", -1)),
                "valid": True,

                # 配置
                "history_keyframes": history_keyframes,
                "steps_per_interval": steps_per_interval,
                "num_output_frames": int(fixed_num_output_frames),
                "max_evolve_keyframes": int(max_evolve),
                "num_real_frames": int(num_real_frames),

                # 帧序列（pad 后定长）
                "frame_tokens": frame_tokens,
                "frame_sample_tokens": frame_sample_tokens,
                "frame_rel_paths": frame_rel_paths,
                "frame_interval_indices": np.asarray(frame_interval_indices, dtype=np.int16),
                "frame_phase_indices": np.asarray(frame_phase_indices, dtype=np.int16),
                "frame_is_keyframe": frame_is_keyframe,
                "frame_valid_mask": frame_valid_mask_arr,
                "frame_timestamps": frame_timestamps,
                "frame_dt": frame_dt,
                "frame_ego2global": frame_ego2global,
                "interval_frame_counts": np.asarray(interval_frame_counts, dtype=np.int16),

                # evolve keyframe 元信息（变长）
                "evolve_keyframe_step_indices": evolve_keyframe_step_indices,
                "evolve_keyframe_sample_tokens": evolve_keyframe_sample_tokens,
                "evolve_keyframe_gt_rel_paths": evolve_keyframe_gt_rel_paths,
                "evolve_keyframe_gt_exists": evolve_keyframe_gt_exists,

                # slow（=start）
                "slow_sample_token": start_sample_token,
                "slow_frame_token": real_frame_tokens[0],
                "slow_logit_path": make_logit_rel_path(scene_name, start_sample_token),
                "slow_timestamp": int(slow_ts),
                "slow_ego2global": slow_pose.copy(),
                # T_slow_to_curr：start-anchored 下 "curr" 等价于 end keyframe；
                # 模型 forward 不依赖该字段，留单位阵占位。
                "T_slow_to_curr": np.eye(4, dtype=np.float32),

                # rollout / history（兼容字段）
                "rollout_start_step": 0,
                "history_completeness": int(max_evolve),

                # GT 路径（兼容现有 dataset 的 curr_gt 加载，指向 start keyframe，evaluator 不使用）
                "curr_gt_rel_path": start_gt_rel,

                # RayIoU 字段（占位，评估按 token 查 sweep_pkl）
                "lidar2ego_translation": lidar2ego_translation,
                "lidar2ego_rotation": lidar2ego_rotation,

                # supervision 字段：start-anchored pkl 不参与监督训练，全置空避免 dataset / loss 报错
                "supervision_labels": ["t-1.5", "t-1.0", "t-0.5", "t"],
                "supervision_mask": [0, 0, 0, 0],
                "supervision_step_indices": [-1, -1, -1, -1],
                "supervision_gt_tokens": ["", "", "", ""],
                "supervision_frame_tokens": ["", "", "", ""],
                "supervision_gt_rel_paths": ["", "", "", ""],
                "num_supervision": 0,
                "has_all_4_supervision": False,
            }

            index_by_token[start_sample_token] = len(infos_out)
            infos_out.append(entry)
            valid_count += 1
            max_evolve_hist[max_evolve] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "description": (
                "start-anchored evolve infos：每个 sample 以一个 keyframe K_j 为 slow 锚，"
                "推演到 K_{j+max_evolve}，max_evolve = min(history_keyframes, scene_end - j)。"
                "num_output_frames 统一 pad 到 history_keyframes*steps_per_interval+1，"
                "末尾 pad；评估只取前 num_real_frames 步的输出。"
            ),
            "source_info_path": str(src_info_path),
            "source_metadata": source_metadata,
            "data_root": str(data_root),
            "version": args.version,
            "split": split_name,
            "token_camera": args.token_camera,
            "history_keyframes": history_keyframes,
            "steps_per_interval": steps_per_interval,
            "num_output_frames": int(fixed_num_output_frames),
            "min_evolve_keyframes": int(args.min_evolve_keyframes),
            "selection_rule_name": "nearest_timestamp_fractional_phase_v1",
            "fast_logit_path_template": "{scene_name}/{frame_token}/logits.npz",
            "slow_logit_path_template": "{scene_name}/{start_sample_token}/logits.npz",
            "gt_root": args.gt_root,
            "num_infos": len(infos_out),
            "num_valid_infos": int(valid_count),
            "num_skipped_short": int(skipped_short),
            "num_skipped_no_gt": int(skipped_no_gt),
            "num_skipped_invalid_source": int(skipped_invalid_source),
            "num_bad_lidar_meta": int(bad_lidar_meta_count),
            "num_skipped_no_fast_logits": int(skipped_no_fast_logits),
            "num_skipped_no_slow_logits": int(skipped_no_slow_logits),
            "interval_length_hist": {str(k): int(v) for k, v in sorted(interval_len_hist.items())},
            "max_evolve_keyframes_hist": {
                str(k): int(v) for k, v in sorted(max_evolve_hist.items())
            },
            # supervision 字段为空，但保留 supervision_labels 让 dataset 不报错
            "supervision_labels": ["t-1.5", "t-1.0", "t-0.5", "t"],
        },
        "index_by_token": index_by_token,
        "infos": infos_out,
    }

    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"[save] {output_path}")
    print(
        f"[stats] total={len(infos_out)} valid={valid_count} "
        f"skipped_short={skipped_short} skipped_no_gt={skipped_no_gt} "
        f"skipped_invalid_source={skipped_invalid_source} "
        f"skipped_no_fast_logits={skipped_no_fast_logits} "
        f"skipped_no_slow_logits={skipped_no_slow_logits} "
        f"bad_lidar_meta={bad_lidar_meta_count}"
    )
    print(f"[stats] max_evolve_keyframes_hist={dict(sorted(max_evolve_hist.items()))}")
    print(f"[stats] interval_length_hist={dict(sorted(interval_len_hist.items()))}")


if __name__ == "__main__":
    main()
