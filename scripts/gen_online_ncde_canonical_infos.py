#!/usr/bin/env python3
"""生成 Online NCDE 的通用 canonical info pkl。

目标：
1. 用统一的 13-step / 6Hz 时间轴描述每个样本前 2s 的快系统帧；
2. 不再假设快系统 logits 是单个打包的 13 帧文件；
3. 让 OPUS / ALOCC 都依赖同一份时间轴定义，再各自导出逐帧 logits。
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

# 多帧监督定义：4 个 keyframe 时间点（t-1.5 / t-1.0 / t-0.5 / t）
# 始终锚到 keyframe_sample_tokens 的末 4 个位置，即
# offsets = [H-3, H-2, H-1, H]（H = history_keyframes），
# 保证 "t" 永远对应当前帧 step = num_output_frames-1。
# H=4 时退化为旧版的 [1, 2, 3, 4]，与历史 pkl 完全兼容。
SUPERVISION_LABELS = ["t-1.5", "t-1.0", "t-0.5", "t"]
NUM_SUPERVISION = len(SUPERVISION_LABELS)


def compute_supervision_keyframe_offsets(history_keyframes: int) -> list[int]:
    """返回末 NUM_SUPERVISION 个 keyframe 的 offset（相对起点）。"""
    if history_keyframes < NUM_SUPERVISION:
        raise ValueError(
            f"--history-keyframes 需 >= {NUM_SUPERVISION}（监督点数），"
            f"当前 {history_keyframes}。"
        )
    return [history_keyframes - (NUM_SUPERVISION - 1 - i) for i in range(NUM_SUPERVISION)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 online_ncde canonical infos。")
    parser.add_argument(
        "--src-info",
        default="data/nuscenes/nuscenes_infos_train_sweep.pkl",
        help="输入 sweep pkl，用于确定 split 与 keyframe sample 集合。",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出 pkl 路径；空则默认写到 configs/online_ncde/canonical_infos_<split>.pkl。",
    )
    parser.add_argument(
        "--data-root",
        default="data/nuscenes",
        help="NuScenes dataroot。",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="NuScenes 版本。",
    )
    parser.add_argument(
        "--token-camera",
        default=DEFAULT_TOKEN_CAMERA,
        help="用于定义 fast frame token 的相机通道。",
    )
    parser.add_argument(
        "--history-keyframes",
        type=int,
        default=4,
        help="当前时刻往前追溯多少个 keyframe interval；默认 4，对应 2s 历史。",
    )
    parser.add_argument(
        "--steps-per-interval",
        type=int,
        default=3,
        help="每个 0.5s keyframe interval 重采样为多少个快系统 step；默认 3，对应 6Hz。",
    )
    parser.add_argument(
        "--gt-root",
        default="data/nuscenes/gts",
        help="GT labels 根目录（相对于项目根），用于生成多帧监督字段。",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="可选：仅处理前 N 个 scene，0 表示全量。",
    )
    parser.add_argument(
        "--allow-short-history",
        action="store_true",
        help=(
            "允许 scene 开头不足 history_keyframes 的 keyframe 也生成条目："
            "前面用 null 填充保持 13 帧定长，并记录 rollout_start_step。"
            "默认关闭，保持与旧行为兼容。"
        ),
    )
    return parser.parse_args()


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
    name = "canonical_infos.pkl" if split == "unknown" else f"canonical_infos_{split}.pkl"
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
            grouped[scene_token] = {
                "scene_name": scene_name,
                "infos": [],
            }
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


def build_supervision_fields(
    keyframe_sample_tokens: list[str],
    keyframe_frame_tokens: list[str],
    keyframe_step_indices: list[int],
    supervision_keyframe_offsets: list[int],
    scene_name: str,
    gt_root_abs: str,
) -> dict[str, Any]:
    """为一条 valid info 构建多帧监督字段。

    监督 4 个 keyframe 时间点（t-1.5 / t-1.0 / t-0.5 / t），
    锚到 keyframe_sample_tokens 的末 4 个位置，保证 "t" 对应当前帧。
    """
    sup_mask = [0] * NUM_SUPERVISION
    sup_step_indices = [-1] * NUM_SUPERVISION
    sup_gt_tokens = [""] * NUM_SUPERVISION
    sup_frame_tokens = [""] * NUM_SUPERVISION
    sup_gt_rel_paths = [""] * NUM_SUPERVISION
    sup_count = 0

    for sup_i, kf_offset in enumerate(supervision_keyframe_offsets):
        sup_sample_token = keyframe_sample_tokens[kf_offset]
        # pad 段的 keyframe 位置为空字符串，此 sup 点不可用
        if not sup_sample_token:
            continue
        gt_rel = make_gt_rel_path(scene_name, sup_sample_token)
        gt_abs = str(Path(gt_root_abs) / gt_rel)

        if Path(gt_abs).exists():
            sup_mask[sup_i] = 1
            sup_step_indices[sup_i] = keyframe_step_indices[kf_offset]
            sup_gt_tokens[sup_i] = sup_sample_token
            sup_frame_tokens[sup_i] = (
                keyframe_frame_tokens[kf_offset]
                if kf_offset < len(keyframe_frame_tokens)
                else ""
            )
            sup_gt_rel_paths[sup_i] = gt_rel
            sup_count += 1

    return {
        "supervision_labels": list(SUPERVISION_LABELS),
        "supervision_mask": sup_mask,
        "supervision_step_indices": sup_step_indices,
        "supervision_gt_tokens": sup_gt_tokens,
        "supervision_frame_tokens": sup_frame_tokens,
        "supervision_gt_rel_paths": sup_gt_rel_paths,
        "num_supervision": sup_count,
        "has_all_4_supervision": sup_count == NUM_SUPERVISION,
    }


def build_invalid_entry(
    info: dict[str, Any],
    split_name: str,
    history_keyframes: int,
    steps_per_interval: int,
    base_valid: bool,
    invalid_reasons: list[str],
) -> dict[str, Any]:
    current_token = str(info["token"])
    scene_name = str(info["scene_name"])
    num_output_frames = history_keyframes * steps_per_interval + 1
    keyframe_step_indices = [step * steps_per_interval for step in range(history_keyframes + 1)]
    return {
        "token": current_token,
        "scene_name": scene_name,
        "scene_token": str(info["scene_token"]),
        "split": split_name,
        "source_index": int(info["_source_index"]),
        "base_valid": bool(base_valid),
        "valid": False,
        "invalid_reasons": invalid_reasons,
        "history_keyframes": int(history_keyframes),
        "steps_per_interval": int(steps_per_interval),
        "num_output_frames": int(num_output_frames),
        "keyframe_step_indices": keyframe_step_indices,
        "keyframe_sample_tokens": [],
        "keyframe_frame_tokens": [],
        "interval_frame_counts": np.zeros((history_keyframes,), dtype=np.int16),
        "interval_selected_local_indices": np.full(
            (history_keyframes, steps_per_interval),
            -1,
            dtype=np.int16,
        ),
        "frame_tokens": [],
        "frame_sample_tokens": [],
        "frame_rel_paths": [],
        "frame_interval_indices": np.zeros((0,), dtype=np.int16),
        "frame_phase_indices": np.zeros((0,), dtype=np.int16),
        "frame_is_keyframe": np.zeros((0,), dtype=np.uint8),
        "frame_valid_mask": np.zeros((0,), dtype=np.uint8),
        "frame_timestamps": np.zeros((0,), dtype=np.int64),
        "frame_dt": np.zeros((0,), dtype=np.float32),
        "frame_ego2global": np.zeros((0, 4, 4), dtype=np.float32),
        "has_duplicate_fast_frames": False,
        "num_unique_fast_frames": 0,
        "rollout_start_step": int(num_output_frames - 1),
        "history_completeness": 0,
        "slow_sample_token": "",
        "slow_frame_token": "",
        "slow_logit_path": "",
        "slow_timestamp": 0,
        "slow_ego2global": np.eye(4, dtype=np.float32),
        "T_slow_to_curr": np.eye(4, dtype=np.float32),
        "curr_gt_rel_path": str(Path(scene_name) / current_token / "labels.npz"),
        # RayIoU 评估所需（invalid 样本用零值占位）
        "lidar2ego_translation": np.zeros(3, dtype=np.float32),
        "lidar2ego_rotation": np.zeros(4, dtype=np.float32),
        # 多帧监督字段（invalid 样本全为空）
        "supervision_labels": list(SUPERVISION_LABELS),
        "supervision_mask": [0] * NUM_SUPERVISION,
        "supervision_step_indices": [-1] * NUM_SUPERVISION,
        "supervision_gt_tokens": [""] * NUM_SUPERVISION,
        "supervision_frame_tokens": [""] * NUM_SUPERVISION,
        "supervision_gt_rel_paths": [""] * NUM_SUPERVISION,
        "num_supervision": 0,
        "has_all_4_supervision": False,
    }


def main() -> None:
    args = parse_args()
    src_info_path = resolve_repo_path(args.src_info)
    output_path = resolve_repo_path(args.output) if args.output else build_default_output_path(src_info_path)
    data_root = resolve_repo_path(args.data_root)
    split_name = infer_split_name(src_info_path)

    if args.history_keyframes <= 0:
        raise ValueError("--history-keyframes 必须为正。")
    if args.steps_per_interval <= 0:
        raise ValueError("--steps-per-interval 必须为正。")

    supervision_keyframe_offsets = compute_supervision_keyframe_offsets(args.history_keyframes)

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

    infos_out: list[dict[str, Any]] = []
    index_by_token: dict[str, int] = {}

    interval_len_hist: Counter[int] = Counter()
    selection_pattern_hist: Counter[tuple[int, tuple[int, ...]]] = Counter()
    history_completeness_hist: Counter[int] = Counter()
    duplicate_fast_frame_count = 0
    valid_count = 0
    base_valid_count = 0
    invalid_due_to_history = 0
    invalid_due_to_source = 0
    sup_any_count = 0
    sup_all_count = 0

    iterator = (
        progressbar.progressbar(scene_items, prefix="[gen canonical infos] ")
        if progressbar is not None
        else scene_items
    )
    for scene_token, scene_group in iterator:
        scene_infos = scene_group["infos"]
        for local_idx, curr_info in enumerate(scene_infos):
            base_valid = coerce_valid_flag(curr_info.get("valid_flag", True))
            if base_valid:
                base_valid_count += 1

            enough_history = local_idx >= args.history_keyframes
            invalid_reasons: list[str] = []
            # 短历史：是否允许取决于 --allow-short-history
            use_short_entry = (not enough_history) and bool(args.allow_short_history)
            if not enough_history and not args.allow_short_history:
                invalid_reasons.append("insufficient_keyframe_history")
                invalid_due_to_history += 1
            if not base_valid:
                invalid_reasons.append("source_valid_flag_false")
                invalid_due_to_source += 1

            if (not enough_history) and (not args.allow_short_history):
                entry = build_invalid_entry(
                    info=curr_info,
                    split_name=split_name,
                    history_keyframes=args.history_keyframes,
                    steps_per_interval=args.steps_per_interval,
                    base_valid=base_valid,
                    invalid_reasons=invalid_reasons,
                )
            else:
                # 真实历史数 h ∈ [0, history_keyframes]；pad 是前段填空的 step 数
                h = min(local_idx, args.history_keyframes)
                pad_intervals = args.history_keyframes - h
                pad_steps = pad_intervals * args.steps_per_interval

                history_infos_real = scene_infos[local_idx - h : local_idx + 1]
                keyframe_sample_tokens_real = [str(info["token"]) for info in history_infos_real]
                # 长度恒为 history_keyframes + 1，前面 pad_intervals 个位置填 ""
                keyframe_sample_tokens: list[str] = [""] * pad_intervals + keyframe_sample_tokens_real

                # 初始化按 num_output_frames 长度的占位列表
                frame_tokens: list[str] = [""] * pad_steps
                frame_interval_indices: list[int] = [-1] * pad_steps
                frame_phase_indices: list[int] = [-1] * pad_steps
                interval_frame_counts: list[int] = [0] * pad_intervals
                interval_selected_local_indices: list[list[int]] = [
                    [-1] * args.steps_per_interval for _ in range(pad_intervals)
                ]

                for real_i in range(h):
                    interval_idx = pad_intervals + real_i  # 绝对 interval 位置
                    start_sample_token = keyframe_sample_tokens_real[real_i]
                    end_sample_token = keyframe_sample_tokens_real[real_i + 1]
                    interval_tokens, end_frame_token = collect_interval_camera_tokens(
                        start_sample_token=start_sample_token,
                        end_sample_token=end_sample_token,
                        token_camera=args.token_camera,
                        scene_token=scene_token,
                        sample_by_token=sample_by_token,
                        sample_data_by_token=sample_data_by_token,
                    )
                    selected_local_indices = select_interval_local_indices(
                        interval_tokens=interval_tokens,
                        end_token=end_frame_token,
                        sample_data_by_token=sample_data_by_token,
                        steps_per_interval=args.steps_per_interval,
                    )
                    interval_frame_counts.append(len(interval_tokens))
                    interval_selected_local_indices.append(selected_local_indices)
                    interval_len_hist[len(interval_tokens)] += 1
                    selection_pattern_hist[
                        (len(interval_tokens), tuple(int(v) for v in selected_local_indices))
                    ] += 1

                    for phase_idx, local_frame_idx in enumerate(selected_local_indices):
                        frame_tokens.append(interval_tokens[local_frame_idx])
                        frame_interval_indices.append(interval_idx)
                        frame_phase_indices.append(phase_idx)

                current_frame_token = str(
                    sample_by_token[keyframe_sample_tokens_real[-1]]["data"][args.token_camera]
                )
                frame_tokens.append(current_frame_token)
                frame_interval_indices.append(args.history_keyframes)
                frame_phase_indices.append(0)

                expected_num_frames = args.history_keyframes * args.steps_per_interval + 1
                if len(frame_tokens) != expected_num_frames:
                    raise ValueError(
                        f"frame_tokens 长度异常：实际 {len(frame_tokens)}，预期 {expected_num_frames}"
                    )

                # 真实帧的 ts / ego 先算出来，pad 段用第一个真实帧的值复制
                first_real_token = frame_tokens[pad_steps]
                first_real_ts = int(sample_data_by_token[first_real_token]["timestamp"])
                first_real_pose = get_pose_matrix_from_frame_token(first_real_token)

                frame_sample_tokens: list[str] = []
                frame_ts_list: list[int] = []
                frame_pose_list: list[np.ndarray] = []
                frame_is_kf_list: list[int] = []
                for pos, token in enumerate(frame_tokens):
                    if pos < pad_steps or token == "":
                        # pad 段：sample_token 为空串，ts/ego 用第一个真实帧复制，is_keyframe=0
                        frame_sample_tokens.append("")
                        frame_ts_list.append(first_real_ts)
                        frame_pose_list.append(first_real_pose)
                        frame_is_kf_list.append(0)
                    else:
                        sd_rec = sample_data_by_token[token]
                        frame_sample_tokens.append(str(sd_rec["sample_token"]))
                        frame_ts_list.append(int(sd_rec["timestamp"]))
                        frame_pose_list.append(get_pose_matrix_from_frame_token(token))
                        frame_is_kf_list.append(1 if bool(sd_rec["is_key_frame"]) else 0)

                frame_timestamps = np.asarray(frame_ts_list, dtype=np.int64)
                frame_dt = ((frame_timestamps - frame_timestamps[-1]).astype(np.float64) * 1.0e-6).astype(
                    np.float32
                )
                frame_ego2global = np.stack(frame_pose_list, axis=0).astype(np.float32)
                frame_is_keyframe = np.asarray(frame_is_kf_list, dtype=np.uint8)
                frame_valid_mask = np.zeros((expected_num_frames,), dtype=np.uint8)
                frame_valid_mask[pad_steps:] = 1

                keyframe_step_indices = [
                    step * args.steps_per_interval for step in range(args.history_keyframes + 1)
                ]
                keyframe_frame_tokens = [frame_tokens[step] for step in keyframe_step_indices]

                # slow 锚到最老真实 keyframe：即 step = pad_steps 的帧
                slow_sample_token = keyframe_sample_tokens_real[0]
                slow_frame_token = frame_tokens[pad_steps]
                slow_ego2global = first_real_pose.copy()
                current_ego2global = frame_ego2global[-1]
                T_slow_to_curr = (np.linalg.inv(current_ego2global) @ slow_ego2global).astype(np.float32)

                # 只在真实段统计重复帧
                real_frame_tokens = frame_tokens[pad_steps:]
                has_duplicate_fast_frames = len(set(real_frame_tokens)) < len(real_frame_tokens)
                if has_duplicate_fast_frames:
                    duplicate_fast_frame_count += 1

                # frame_rel_paths：pad 段保持 ""，logits_loader 端用零张量占位
                frame_rel_paths: list[str] = []
                for token in frame_tokens:
                    if token == "":
                        frame_rel_paths.append("")
                    else:
                        frame_rel_paths.append(
                            make_logit_rel_path(str(curr_info["scene_name"]), token)
                        )

                entry = {
                    "token": keyframe_sample_tokens_real[-1],
                    "scene_name": str(curr_info["scene_name"]),
                    "scene_token": str(curr_info["scene_token"]),
                    "split": split_name,
                    "source_index": int(curr_info["_source_index"]),
                    "base_valid": bool(base_valid),
                    "valid": bool(base_valid),
                    "invalid_reasons": invalid_reasons,
                    "history_keyframes": int(args.history_keyframes),
                    "steps_per_interval": int(args.steps_per_interval),
                    "num_output_frames": int(expected_num_frames),
                    "rollout_start_step": int(pad_steps),
                    "history_completeness": int(h),
                    "keyframe_step_indices": keyframe_step_indices,
                    "keyframe_sample_tokens": keyframe_sample_tokens,
                    "keyframe_frame_tokens": keyframe_frame_tokens,
                    "interval_frame_counts": np.asarray(interval_frame_counts, dtype=np.int16),
                    "interval_selected_local_indices": np.asarray(
                        interval_selected_local_indices,
                        dtype=np.int16,
                    ),
                    "frame_tokens": frame_tokens,
                    "frame_sample_tokens": frame_sample_tokens,
                    "frame_rel_paths": frame_rel_paths,
                    "frame_interval_indices": np.asarray(frame_interval_indices, dtype=np.int16),
                    "frame_phase_indices": np.asarray(frame_phase_indices, dtype=np.int16),
                    "frame_is_keyframe": frame_is_keyframe,
                    "frame_valid_mask": frame_valid_mask,
                    "frame_timestamps": frame_timestamps,
                    "frame_dt": frame_dt,
                    "frame_ego2global": frame_ego2global,
                    "has_duplicate_fast_frames": bool(has_duplicate_fast_frames),
                    "num_unique_fast_frames": int(len(set(real_frame_tokens))),
                    "slow_sample_token": slow_sample_token,
                    "slow_frame_token": slow_frame_token,
                    "slow_logit_path": make_logit_rel_path(str(curr_info["scene_name"]), slow_sample_token),
                    "slow_timestamp": int(first_real_ts),
                    "slow_ego2global": slow_ego2global,
                    "T_slow_to_curr": T_slow_to_curr,
                    "curr_gt_rel_path": str(
                        Path(str(curr_info["scene_name"])) / keyframe_sample_tokens_real[-1] / "labels.npz"
                    ),
                    # RayIoU 评估所需的 lidar→ego 刚体变换（当前 keyframe）
                    "lidar2ego_translation": np.asarray(
                        curr_info["lidar2ego_translation"], dtype=np.float32
                    ),
                    "lidar2ego_rotation": np.asarray(
                        curr_info["lidar2ego_rotation"], dtype=np.float32
                    ),
                }
                # 添加多帧监督字段（pad 段自动置 0，见 build_supervision_fields）
                sup_fields = build_supervision_fields(
                    keyframe_sample_tokens=keyframe_sample_tokens,
                    keyframe_frame_tokens=keyframe_frame_tokens,
                    keyframe_step_indices=keyframe_step_indices,
                    supervision_keyframe_offsets=supervision_keyframe_offsets,
                    scene_name=str(curr_info["scene_name"]),
                    gt_root_abs=gt_root_abs,
                )
                entry.update(sup_fields)

            index_by_token[str(entry["token"])] = len(infos_out)
            infos_out.append(entry)
            history_completeness_hist[int(entry.get("history_completeness", 0))] += 1
            if entry["valid"]:
                valid_count += 1
                if entry["num_supervision"] > 0:
                    sup_any_count += 1
                if entry["has_all_4_supervision"]:
                    sup_all_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    selection_pattern_hist_str = {
        f"{length}:{list(indices)}": int(count)
        for (length, indices), count in sorted(selection_pattern_hist.items())
    }
    payload = {
        "metadata": {
            "schema_version": "online_ncde_canonical_infos_v1",
            "description": (
                "系统无关的 canonical 13-step / 6Hz 时间轴定义；"
                "每个 0.5s keyframe interval 通过 token_camera 的真实时间戳近邻重采样为 3 个快系统 step。"
            ),
            "source_info_path": str(src_info_path),
            "source_metadata": source_metadata,
            "data_root": str(data_root),
            "version": args.version,
            "split": split_name,
            "token_camera": args.token_camera,
            "history_keyframes": int(args.history_keyframes),
            "steps_per_interval": int(args.steps_per_interval),
            "num_output_frames": int(args.history_keyframes * args.steps_per_interval + 1),
            "keyframe_step_indices": [
                step * args.steps_per_interval for step in range(args.history_keyframes + 1)
            ],
            "selection_rule_name": "nearest_timestamp_fractional_phase_v1",
            "selection_rule_description": (
                "对每个 keyframe interval，收集 token_camera 的原始 sample_data 链（起点含、终点不含），"
                "然后对 {0, 1/3, 2/3} × interval_duration 这 3 个 canonical 时间点做最近邻选帧；"
                "为保持时间顺序，局部索引约束为单调不减，允许在短 interval 上重复 token。"
            ),
            "num_infos": len(infos_out),
            "num_base_valid_infos": int(base_valid_count),
            "num_valid_infos": int(valid_count),
            "num_invalid_infos": int(len(infos_out) - valid_count),
            "num_invalid_due_to_history": int(invalid_due_to_history),
            "num_invalid_due_to_source_valid": int(invalid_due_to_source),
            "num_infos_with_duplicate_fast_frames": int(duplicate_fast_frame_count),
            "interval_length_hist": {str(k): int(v) for k, v in sorted(interval_len_hist.items())},
            "selection_pattern_hist": selection_pattern_hist_str,
            "fast_logit_path_template": "{scene_name}/{frame_token}/logits.npz",
            "slow_logit_path_template": "{scene_name}/{slow_sample_token}/logits.npz",
            "gt_root": args.gt_root,
            "supervision_labels": list(SUPERVISION_LABELS),
            "supervision_keyframe_offsets": list(supervision_keyframe_offsets),
            "supervision_step_indices": [
                step * args.steps_per_interval
                for step in supervision_keyframe_offsets
            ],
            "num_valid_with_any_supervision": int(sup_any_count),
            "num_valid_with_all_4_supervision": int(sup_all_count),
            "allow_short_history": bool(args.allow_short_history),
            "history_completeness_hist": {
                str(k): int(v) for k, v in sorted(history_completeness_hist.items())
            },
        },
        "index_by_token": index_by_token,
        "infos": infos_out,
    }

    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"[save] {output_path}")
    print(
        "[stats] "
        f"infos={len(infos_out)} "
        f"valid={valid_count} "
        f"invalid={len(infos_out) - valid_count} "
        f"duplicate_fast={duplicate_fast_frame_count}"
    )
    print(
        "[stats] "
        f"supervision: any={sup_any_count} all_4={sup_all_count}"
    )
    print(f"[stats] interval_length_hist={dict(sorted(interval_len_hist.items()))}")
    print(
        "[stats] history_completeness_hist="
        f"{dict(sorted(history_completeness_hist.items()))} "
        f"(allow_short_history={bool(args.allow_short_history)})"
    )
    top_patterns = selection_pattern_hist.most_common(8)
    print(
        "[stats] top_selection_patterns="
        + str(
            [
                {"interval_len": length, "local_indices": list(indices), "count": count}
                for (length, indices), count in top_patterns
            ]
        )
    )


if __name__ == "__main__":
    main()
