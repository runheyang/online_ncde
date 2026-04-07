#!/usr/bin/env python3
"""Generate backward velocity GT files from UniOcc occ_flow_backward."""

from __future__ import annotations

import argparse
import os
import json
import pickle
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNI_OCC_CENTER_EGO = np.asarray([40.0, 40.0, 2.2], dtype=np.float32)

UNI_OCC_CLASS_TO_ID: dict[str, int] = {
    "others": 0,
    "car": 1,
    "bicycle": 2,
    "motorcycle": 3,
    "pedestrian": 4,
    "traffic_cone": 5,
    "vegetation": 6,
    "road": 7,
    "terrain": 8,
    "building": 9,
    "free": 10,
}

# Foreground/background split follows UniOcc 11-class encoding.
FOREGROUND_CLASS_NAMES = [
    "car",
    "bicycle",
    "motorcycle",
    "pedestrian",
]

BACKGROUND_CLASS_NAMES = [
    "others",
    "traffic_cone",
    "vegetation",
    "road",
    "terrain",
    "building",
]

FREE_ID = int(UNI_OCC_CLASS_TO_ID["free"])
FOREGROUND_IDS = np.asarray([UNI_OCC_CLASS_TO_ID[name] for name in FOREGROUND_CLASS_NAMES], dtype=np.int64)
BACKGROUND_IDS = np.asarray([UNI_OCC_CLASS_TO_ID[name] for name in BACKGROUND_CLASS_NAMES], dtype=np.int64)


@dataclass(frozen=True)
class SampleMeta:
    prev_token: str
    timestamp_us: int


@dataclass(frozen=True)
class FrameTask:
    scene_name: str
    sample_token: str
    uniocc_npz_path: Path
    out_file: Path
    ego_to_world_t: np.ndarray
    ego_to_world_t_minus_1: np.ndarray | None
    dt_seconds: float | None


@dataclass(frozen=True)
class FrameResult:
    written: int = 0
    missing_required_fields: int = 0
    invalid_flow_shape: int = 0
    invalid_pose_shape: int = 0
    error: int = 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build data/nuscenes/gts_uniocc_occ3d/<scene_name>/<frame_token>/gts.npz "
            "from UniOcc occ_flow_backward with ego-motion-compensated residual velocity."
        )
    )
    parser.add_argument("--root-path", type=str, default=str(ROOT), help="Repository root path.")
    parser.add_argument(
        "--uniocc-root",
        type=str,
        default="data/UniOcc-nuScenes-Occ3D",
        help="UniOcc root with scene folders and scene_infos_*.pkl.",
    )
    parser.add_argument(
        "--scene-info",
        type=str,
        action="append",
        default=[],
        help=(
            "Path to scene infos pkl. Can be provided multiple times. "
            "If omitted, auto-uses scene_infos_train.pkl and scene_infos_val.pkl under --uniocc-root."
        ),
    )
    parser.add_argument(
        "--nusc-meta-root",
        type=str,
        default="data/nuscenes/v1.0-trainval",
        help="nuScenes metadata directory containing sample.json.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/nuscenes/gts_uniocc_occ3d",
        help="Output root directory.",
    )
    parser.add_argument(
        "--pc-range",
        type=float,
        nargs=6,
        default=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="Retained for CLI compatibility only. Ignored by the strict UniOcc num_voxels computation path.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.4,
        help="Occ3D voxel size in meters, used to convert m/s to voxel_size/s.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output gts.npz.")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Only process first N scenes (0 means all scenes).",
    )
    parser.add_argument(
        "--max-frames-per-scene",
        type=int,
        default=0,
        help="Only process first N frames in each scene (0 means all frames).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(32, (os.cpu_count() or 8))),
        help="Thread count for per-frame processing.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N processed samples.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-frame logs.")
    return parser.parse_args()


def resolve_path(root_path: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (root_path / path).resolve()


def dedup_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        real = path.resolve()
        if real in seen:
            continue
        seen.add(real)
        out.append(real)
    return out


def resolve_scene_info_paths(root_path: Path, uniocc_root: Path, cli_paths: list[str]) -> list[Path]:
    if cli_paths:
        paths = [resolve_path(root_path, item) for item in cli_paths]
    else:
        candidates = [
            uniocc_root / "scene_infos_train.pkl",
            uniocc_root / "scene_infos_val.pkl",
            uniocc_root / "scene_infos.pkl",
        ]
        paths = [path for path in candidates if path.exists()]
    paths = dedup_paths(paths)
    if not paths:
        raise FileNotFoundError(
            "No scene-info file found. Provide --scene-info or place scene_infos_train.pkl/scene_infos_val.pkl under --uniocc-root."
        )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Scene-info file not found: {path}")
    return paths


def extract_scene_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = None
        for key in ("scene_infos", "scene_lists", "scenes", "infos"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                entries = candidate
                break
        if entries is None:
            raise ValueError("Unsupported scene-info payload format: list field not found.")
    else:
        raise ValueError(f"Unsupported scene-info payload type: {type(payload)}")

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            raise TypeError(f"scene entry #{idx} is not a dict: {type(item)}")
        out.append(item)
    return out


def load_scene_entries(scene_info_paths: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for scene_info_path in scene_info_paths:
        with scene_info_path.open("rb") as f:
            payload = pickle.load(f)
        chunk = extract_scene_entries(payload)
        entries.extend(chunk)
    return entries


def normalize_occ_rel_path(scene_name: str, raw_rel_path: str) -> str:
    rel = str(raw_rel_path).replace("\\", "/").strip().lstrip("/")
    if rel.startswith(scene_name + "/"):
        return rel
    filename = Path(rel).name
    return f"{scene_name}/{filename}"


def load_sample_meta(nusc_meta_root: Path) -> dict[str, SampleMeta]:
    sample_json = nusc_meta_root / "sample.json"
    if not sample_json.exists():
        raise FileNotFoundError(f"sample.json not found: {sample_json}")
    with sample_json.open("r", encoding="utf-8") as f:
        sample_records = json.load(f)

    sample_meta: dict[str, SampleMeta] = {}
    for idx, record in enumerate(sample_records):
        if not isinstance(record, dict):
            raise TypeError(f"sample.json record #{idx} is not a dict: {type(record)}")
        token = str(record.get("token", ""))
        if not token:
            continue
        prev_token = str(record.get("prev", ""))
        timestamp_us = int(record.get("timestamp", 0))
        sample_meta[token] = SampleMeta(prev_token=prev_token, timestamp_us=timestamp_us)
    return sample_meta


def compute_dt_seconds(sample_token: str, sample_meta: dict[str, SampleMeta]) -> float | None:
    curr = sample_meta.get(sample_token)
    if curr is None:
        return None
    if not curr.prev_token:
        return None
    prev = sample_meta.get(curr.prev_token)
    if prev is None:
        return None
    dt = float(curr.timestamp_us - prev.timestamp_us) / 1.0e6
    if dt <= 0.0:
        return None
    return dt


@lru_cache(maxsize=8)
def build_occ_coords(grid_shape: tuple[int, int, int]) -> np.ndarray:
    l, w, h = grid_shape
    xs = np.arange(l, dtype=np.float32)
    ys = np.arange(w, dtype=np.float32)
    zs = np.arange(h, dtype=np.float32)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xg, yg, zg], axis=-1)  # (L,W,H,3)


def occ_frame_to_ego(
    occ_coords: np.ndarray,
    voxel_size: float,
    center_ego: np.ndarray = UNI_OCC_CENTER_EGO,
) -> np.ndarray:
    """Match UniOcc's OccFrameToEgoFrame mapping exactly."""
    ego_coords = np.zeros_like(occ_coords, dtype=np.float32)
    ego_coords[..., 0] = occ_coords[..., 0] * float(voxel_size) - float(center_ego[0])
    ego_coords[..., 1] = occ_coords[..., 1] * float(voxel_size) - float(center_ego[1])
    ego_coords[..., 2] = occ_coords[..., 2] * float(voxel_size) - float(center_ego[2])
    return ego_coords


def build_velocity_xy_residual(
    occ_label: np.ndarray,
    occ_coords: np.ndarray,
    occ_flow_backward: np.ndarray,
    ego_to_world_t: np.ndarray,
    ego_to_world_t_minus_1: np.ndarray | None,
    dt_seconds: float | None,
    voxel_size: float,
) -> np.ndarray:
    if occ_flow_backward.ndim != 4 or occ_flow_backward.shape[-1] < 2:
        raise ValueError(
            f"occ_flow_backward shape should be (L,W,H,>=2), got {tuple(occ_flow_backward.shape)}"
        )
    if occ_label.ndim != 3:
        raise ValueError(f"occ_label should be 3D, got shape={tuple(occ_label.shape)}")
    if occ_coords.shape != occ_flow_backward.shape:
        raise ValueError(
            f"occ_coords shape should be {tuple(occ_flow_backward.shape)}, got {tuple(occ_coords.shape)}"
        )

    l, w, h = occ_label.shape
    velocity_xy = np.zeros((2, l, w, h), dtype=np.float16)
    if dt_seconds is None or ego_to_world_t_minus_1 is None:
        return velocity_xy

    labels = occ_label.astype(np.int64, copy=False)
    fg_mask = np.isin(labels, FOREGROUND_IDS)
    if not np.any(fg_mask):
        return velocity_xy

    if ego_to_world_t.shape != (4, 4) or ego_to_world_t_minus_1.shape != (4, 4):
        return velocity_xy

    occ_curr = occ_coords[fg_mask].astype(np.float32, copy=False)
    flow_bwd_vox = occ_flow_backward[fg_mask].astype(np.float32, copy=False)

    # UniOcc stores flow in num_voxels. Add it in occupancy coordinates first,
    # then map both current and previous positions to ego-frame metric coordinates.
    occ_prev_local = occ_curr + flow_bwd_vox
    xyz_curr = occ_frame_to_ego(occ_curr, voxel_size=voxel_size)
    xyz_prev_local = occ_frame_to_ego(occ_prev_local, voxel_size=voxel_size)

    ones = np.ones((xyz_prev_local.shape[0], 1), dtype=np.float32)
    prev_local_h = np.concatenate([xyz_prev_local, ones], axis=1)
    world_h = prev_local_h @ ego_to_world_t_minus_1.T.astype(np.float32, copy=False)

    ego_t_inv = np.linalg.inv(ego_to_world_t).astype(np.float32, copy=False)
    prev_to_t_h = world_h @ ego_t_inv.T

    # Disp_residual = P_{t-1->t} - x
    disp_residual = prev_to_t_h[:, :3] - xyz_curr
    vel_xy_fg = disp_residual[:, :2] / float(dt_seconds) / float(voxel_size)

    vel_xy_lwh2 = np.zeros((l, w, h, 2), dtype=np.float32)
    vel_xy_lwh2[fg_mask] = vel_xy_fg
    velocity_xy = np.moveaxis(vel_xy_lwh2, -1, 0).astype(np.float16, copy=False)
    return velocity_xy


def build_mask(occ_label: np.ndarray) -> np.ndarray:
    if occ_label.ndim != 3:
        raise ValueError(f"occ_label should be 3D, got shape={tuple(occ_label.shape)}")
    labels = occ_label.astype(np.int64, copy=False)

    # 0: free, 1: background, 2: foreground
    mask = np.ones(labels.shape, dtype=np.uint8)
    mask[labels == FREE_ID] = np.uint8(0)
    mask[np.isin(labels, FOREGROUND_IDS)] = np.uint8(2)
    return mask[None, ...]


def process_frame_task(
    task: FrameTask,
    voxel_size: float,
    verbose: bool,
) -> FrameResult:
    try:
        with np.load(task.uniocc_npz_path, allow_pickle=False) as uniocc_npz:
            if "occ_label" not in uniocc_npz.files or "occ_flow_backward" not in uniocc_npz.files:
                if verbose:
                    print(f"[warn] required fields missing in: {task.uniocc_npz_path}")
                return FrameResult(missing_required_fields=1)
            occ_label = uniocc_npz["occ_label"]
            occ_flow_backward = uniocc_npz["occ_flow_backward"]
    except Exception:
        if verbose:
            print(f"[warn] failed to load npz: {task.uniocc_npz_path}")
        return FrameResult(error=1)

    if task.ego_to_world_t.shape != (4, 4):
        if verbose:
            print(f"[warn] invalid ego_to_world shape in: {task.uniocc_npz_path}")
        return FrameResult(invalid_pose_shape=1)

    try:
        grid_shape = tuple(int(v) for v in occ_label.shape)
        occ_coords = build_occ_coords(grid_shape=grid_shape)
        velocity_xy = build_velocity_xy_residual(
            occ_label=occ_label,
            occ_coords=occ_coords,
            occ_flow_backward=occ_flow_backward,
            ego_to_world_t=task.ego_to_world_t,
            ego_to_world_t_minus_1=task.ego_to_world_t_minus_1,
            dt_seconds=task.dt_seconds,
            voxel_size=voxel_size,
        )
        mask = build_mask(occ_label)
    except ValueError:
        if verbose:
            print(f"[warn] invalid occ tensor shape in: {task.uniocc_npz_path}")
        return FrameResult(invalid_flow_shape=1)
    except Exception:
        if verbose:
            print(f"[warn] failed to compute velocity for: {task.uniocc_npz_path}")
        return FrameResult(error=1)

    try:
        task.out_file.parent.mkdir(parents=True, exist_ok=True)
        dt_scalar = np.float32(task.dt_seconds if task.dt_seconds is not None else 0.0)
        np.savez_compressed(task.out_file, velocity_xy=velocity_xy, mask=mask, dt=dt_scalar)
    except Exception:
        if verbose:
            print(f"[warn] failed to save output: {task.out_file}")
        return FrameResult(error=1)

    if verbose:
        print(
            f"[write] {task.out_file} "
            f"velocity_xy={tuple(velocity_xy.shape)}:{velocity_xy.dtype} "
            f"mask={tuple(mask.shape)}:{mask.dtype} dt={float(dt_scalar):.6f} "
            f"prev_pose_found={task.ego_to_world_t_minus_1 is not None}"
        )
    return FrameResult(written=1)


def main() -> None:
    args = parse_args()
    root_path = Path(args.root_path).expanduser().resolve()
    uniocc_root = resolve_path(root_path, args.uniocc_root)
    nusc_meta_root = resolve_path(root_path, args.nusc_meta_root)
    output_root = resolve_path(root_path, args.output_root)
    voxel_size = float(args.voxel_size)
    pc_range = tuple(float(v) for v in args.pc_range)
    num_workers = max(1, int(args.num_workers))

    if voxel_size <= 0.0:
        raise ValueError(f"--voxel-size must be > 0, got {voxel_size}")
    if not uniocc_root.exists():
        raise FileNotFoundError(f"UniOcc root not found: {uniocc_root}")
    if not nusc_meta_root.exists():
        raise FileNotFoundError(f"nuScenes meta root not found: {nusc_meta_root}")

    scene_info_paths = resolve_scene_info_paths(root_path, uniocc_root, args.scene_info)
    sample_meta = load_sample_meta(nusc_meta_root)
    scene_entries = load_scene_entries(scene_info_paths)

    if args.max_scenes > 0:
        scene_entries = scene_entries[: int(args.max_scenes)]

    output_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    written_frames = 0
    skipped_existing = 0
    missing_uniocc_file = 0
    missing_sample_token = 0
    missing_required_fields = 0
    invalid_flow_shape = 0
    invalid_pose_shape = 0
    processing_errors = 0
    dt_missing_or_invalid = 0
    missing_prev_pose = 0
    dt_valid_count = 0
    dt_sum = 0.0
    dt_min = float("inf")
    dt_max = 0.0

    print(f"[config] root_path={root_path}")
    print(f"[config] uniocc_root={uniocc_root}")
    print(f"[config] nusc_meta_root={nusc_meta_root}")
    print(f"[config] output_root={output_root}")
    print(f"[config] voxel_size={voxel_size}")
    print(f"[config] pc_range={pc_range}")
    print(f"[config] uniocc_center_ego={UNI_OCC_CENTER_EGO.tolist()} flow_unit=num_voxels")
    print(
        f"[config] scene_info_files={len(scene_info_paths)} "
        + ", ".join(str(path) for path in scene_info_paths)
    )
    print(
        "[config] class_to_id="
        + ", ".join(f"{name}:{idx}" for name, idx in UNI_OCC_CLASS_TO_ID.items())
    )
    print(
        "[config] mask_rule "
        f"free={FREE_ID}->0, background={BACKGROUND_IDS.tolist()}->1, foreground={FOREGROUND_IDS.tolist()}->2"
    )
    print(f"[config] scenes={len(scene_entries)}")
    frame_tasks: list[FrameTask] = []
    for scene_idx, scene_info in enumerate(scene_entries, start=1):
        scene_name = str(scene_info.get("scene_name", "")).strip()
        raw_paths = scene_info.get("occ_in_scene_paths")
        if not scene_name or not isinstance(raw_paths, list):
            continue

        rel_paths = [normalize_occ_rel_path(scene_name, str(item)) for item in raw_paths]
        if args.max_frames_per_scene > 0:
            rel_paths = rel_paths[: int(args.max_frames_per_scene)]

        if args.verbose:
            print(f"[scene {scene_idx}/{len(scene_entries)}] {scene_name}, frames={len(rel_paths)}")

        pose_by_token: dict[str, np.ndarray] = {}
        frame_meta: list[tuple[Path, str, np.ndarray]] = []

        # Pass-1: light metadata scan for sample_token + ego pose in current scene.
        for rel_path in rel_paths:
            total_frames += 1
            uniocc_npz_path = uniocc_root / rel_path
            if not uniocc_npz_path.exists():
                missing_uniocc_file += 1
                if args.verbose:
                    print(f"[warn] missing uniocc file: {uniocc_npz_path}")
                continue

            try:
                with np.load(uniocc_npz_path, allow_pickle=False) as uniocc_npz:
                    if "sample_token" not in uniocc_npz.files:
                        missing_sample_token += 1
                        if args.verbose:
                            print(f"[warn] sample_token missing in: {uniocc_npz_path}")
                        continue
                    if "ego_to_world_transformation" not in uniocc_npz.files:
                        missing_required_fields += 1
                        if args.verbose:
                            print(f"[warn] ego_to_world_transformation missing in: {uniocc_npz_path}")
                        continue
                    sample_token = str(uniocc_npz["sample_token"])
                    ego_to_world_t = uniocc_npz["ego_to_world_transformation"].astype(np.float32, copy=False)
            except Exception:
                processing_errors += 1
                if args.verbose:
                    print(f"[warn] failed to read metadata from: {uniocc_npz_path}")
                continue

            if ego_to_world_t.shape != (4, 4):
                invalid_pose_shape += 1
                if args.verbose:
                    print(f"[warn] invalid ego_to_world shape in: {uniocc_npz_path}")
                continue

            pose_by_token[sample_token] = ego_to_world_t
            frame_meta.append((uniocc_npz_path, sample_token, ego_to_world_t))

        # Pass-2: build per-frame compute tasks.
        for uniocc_npz_path, sample_token, ego_to_world_t in frame_meta:
            out_file = output_root / scene_name / sample_token / "gts.npz"
            if out_file.exists() and not args.overwrite:
                skipped_existing += 1
                continue

            dt_seconds = compute_dt_seconds(sample_token, sample_meta)
            prev_token = sample_meta.get(sample_token, SampleMeta(prev_token="", timestamp_us=0)).prev_token
            prev_pose = pose_by_token.get(prev_token) if prev_token else None

            if dt_seconds is None:
                dt_missing_or_invalid += 1
            if prev_token and (prev_pose is None):
                missing_prev_pose += 1
            if (dt_seconds is not None) and (prev_pose is not None):
                dt_valid_count += 1
                dt_sum += dt_seconds
                dt_min = min(dt_min, dt_seconds)
                dt_max = max(dt_max, dt_seconds)

            frame_tasks.append(
                FrameTask(
                    scene_name=scene_name,
                    sample_token=sample_token,
                    uniocc_npz_path=uniocc_npz_path,
                    out_file=out_file,
                    ego_to_world_t=ego_to_world_t,
                    ego_to_world_t_minus_1=prev_pose,
                    dt_seconds=dt_seconds,
                )
            )

    submitted_tasks = len(frame_tasks)
    progress_interval = max(1, int(args.progress_interval))
    print(
        f"[config] num_workers={num_workers} "
        f"progress_interval={progress_interval} submitted_tasks={submitted_tasks}"
    )

    if submitted_tasks > 0:
        processed_tasks = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures: list[Future[FrameResult]] = [
                executor.submit(
                    process_frame_task,
                    task,
                    voxel_size,
                    bool(args.verbose),
                )
                for task in frame_tasks
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    processing_errors += 1
                    result = FrameResult(error=1)

                processed_tasks += 1
                written_frames += int(result.written)
                missing_required_fields += int(result.missing_required_fields)
                invalid_flow_shape += int(result.invalid_flow_shape)
                invalid_pose_shape += int(result.invalid_pose_shape)
                processing_errors += int(result.error)

                if (processed_tasks % progress_interval == 0) or (processed_tasks == submitted_tasks):
                    print(
                        f"[progress] processed={processed_tasks}/{submitted_tasks} "
                        f"written={written_frames} skipped_existing={skipped_existing} "
                        f"missing_required={missing_required_fields} invalid_flow={invalid_flow_shape} "
                        f"errors={processing_errors}"
                    )

    print("[done]")
    print(
        f"[stats] total={total_frames} submitted={submitted_tasks} written={written_frames} "
        f"skipped_existing={skipped_existing} "
        f"missing_uniocc={missing_uniocc_file} missing_sample_token={missing_sample_token} "
        f"missing_required_fields={missing_required_fields} "
        f"invalid_flow_shape={invalid_flow_shape} invalid_pose_shape={invalid_pose_shape} "
        f"missing_prev_pose={missing_prev_pose} dt_missing_or_invalid={dt_missing_or_invalid} "
        f"errors={processing_errors}"
    )
    if dt_valid_count > 0:
        print(
            f"[stats] dt_seconds min={dt_min:.6f} max={dt_max:.6f} mean={dt_sum / dt_valid_count:.6f} "
            f"(valid_count={dt_valid_count})"
        )
    else:
        print("[stats] dt_seconds no valid samples found.")


if __name__ == "__main__":
    main()
