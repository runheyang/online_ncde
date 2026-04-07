#!/usr/bin/env python3
"""Generate 4-step supervision sidecar for transport_online_nde."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from nuscenes import NuScenes  # type: ignore[import-not-found]

from online_ncde.config import resolve_path  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SUPERVISION_LABELS = ["t-1.5", "t-1.0", "t-0.5", "t"]
EMPTY_STEPS = [-1, -1, -1, -1]
EMPTY_STR4 = ["", "", "", ""]
EMPTY_MASK4 = [0, 0, 0, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", default=str(ROOT), help="Repository root")
    parser.add_argument("--info-path", required=True, help="Input ncde_align_infos_*.pkl")
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output sidecar path. If empty, save to "
            "configs/transport_online_nde/<info_stem>_sup4_sidecar.pkl"
        ),
    )
    parser.add_argument(
        "--nusc-dataroot",
        default="data/nuscenes",
        help="NuScenes dataroot",
    )
    parser.add_argument(
        "--nusc-version",
        default="v1.0-trainval",
        help="NuScenes version",
    )
    parser.add_argument(
        "--occ-gt-root",
        default="data/nuscenes/gts",
        help="Occupancy GT root containing labels.npz",
    )
    parser.add_argument(
        "--motion-gt-root",
        default="data/nuscenes/gts_uniocc_occ3d",
        help="Motion GT root containing gts.npz",
    )
    return parser.parse_args()


def build_output_path(root_path: str, info_path: str, output: str) -> str:
    if output:
        return resolve_path(root_path, output)
    info = Path(info_path)
    out_dir = Path(resolve_path(root_path, "configs/transport_online_nde"))
    return str(out_dir / f"{info.stem}_sup4_sidecar{info.suffix}")


def load_infos(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "infos" in payload:
        infos = payload["infos"]
        meta = payload.get("metadata", {})
    else:
        infos = payload
        meta = {}
    if not isinstance(infos, list):
        raise TypeError(f"Unexpected infos type: {type(infos)}")
    return infos, meta


def collect_target_tokens(sample_table: dict[str, dict[str, Any]], curr_token: str) -> list[str]:
    """Return keyframe sample tokens ordered as [t-1.5, t-1.0, t-0.5, t]."""
    curr = curr_token
    chain = [curr]
    for _ in range(3):
        sample = sample_table.get(curr)
        if sample is None:
            break
        prev_token = str(sample.get("prev", ""))
        if not prev_token:
            break
        chain.append(prev_token)
        curr = prev_token
    chain = chain[::-1]
    if len(chain) < 4:
        chain = [""] * (4 - len(chain)) + chain
    elif len(chain) > 4:
        chain = chain[-4:]
    return chain


def build_gt_paths(scene_name: str, sample_token: str) -> tuple[str, str]:
    occ_rel_path = os.path.join(scene_name, sample_token, "labels.npz")
    motion_rel_path = os.path.join(scene_name, sample_token, "gts.npz")
    return occ_rel_path, motion_rel_path


def main() -> None:
    args = parse_args()
    root_path = args.root_path
    info_path = resolve_path(root_path, args.info_path)
    output_path = build_output_path(root_path=root_path, info_path=info_path, output=args.output)
    nusc_dataroot = resolve_path(root_path, args.nusc_dataroot)
    occ_gt_root = resolve_path(root_path, args.occ_gt_root)
    motion_gt_root = resolve_path(root_path, args.motion_gt_root)

    infos, info_meta = load_infos(info_path)
    nusc = NuScenes(version=args.nusc_version, dataroot=nusc_dataroot, verbose=False)

    sample_table = {rec["token"]: rec for rec in nusc.sample}
    sample_data_to_sample: dict[str, tuple[bool, str]] = {}
    for rec in nusc.sample_data:
        sample_data_to_sample[rec["token"]] = (
            bool(rec.get("is_key_frame", False)),
            str(rec.get("sample_token", "")),
        )

    entries: list[dict[str, Any]] = []
    index_by_token: dict[str, int] = {}

    valid_total = 0
    valid_with_any = 0
    valid_with_all4 = 0

    iterator = (
        tqdm(enumerate(infos), total=len(infos), desc="[gen transport sup4 sidecar]")
        if tqdm is not None
        else enumerate(infos)
    )
    for idx, info in iterator:
        token = str(info.get("token", ""))
        scene_name = str(info.get("scene_name", ""))
        is_valid = bool(info.get("valid", True))
        frame_tokens = [str(tok) for tok in info.get("frame_tokens", [])]

        supervision_steps = list(EMPTY_STEPS)
        supervision_gt_tokens = list(EMPTY_STR4)
        supervision_occ_gt_rel_paths = list(EMPTY_STR4)
        supervision_motion_gt_rel_paths = list(EMPTY_STR4)
        supervision_frame_tokens = list(EMPTY_STR4)
        supervision_mask = list(EMPTY_MASK4)

        if token:
            index_by_token[token] = idx

        if is_valid:
            valid_total += 1

        if token and scene_name and is_valid and frame_tokens:
            target_tokens = collect_target_tokens(sample_table=sample_table, curr_token=token)

            keyframe_step_map: dict[str, tuple[int, str]] = {}
            for step_idx, frame_token in enumerate(frame_tokens):
                if not frame_token:
                    continue
                pair = sample_data_to_sample.get(frame_token)
                if pair is None:
                    continue
                is_key_frame, sample_token = pair
                if not is_key_frame or not sample_token:
                    continue
                if sample_token not in keyframe_step_map:
                    keyframe_step_map[sample_token] = (step_idx, frame_token)

            for sup_i, sup_token in enumerate(target_tokens):
                if not sup_token:
                    continue
                step_pair = keyframe_step_map.get(sup_token)
                if step_pair is None:
                    continue

                occ_rel_path, motion_rel_path = build_gt_paths(scene_name=scene_name, sample_token=sup_token)
                occ_abs_path = os.path.join(occ_gt_root, occ_rel_path)
                motion_abs_path = os.path.join(motion_gt_root, motion_rel_path)
                if not (os.path.exists(occ_abs_path) and os.path.exists(motion_abs_path)):
                    continue

                step_idx, frame_token = step_pair
                supervision_steps[sup_i] = int(step_idx)
                supervision_gt_tokens[sup_i] = sup_token
                supervision_occ_gt_rel_paths[sup_i] = occ_rel_path
                supervision_motion_gt_rel_paths[sup_i] = motion_rel_path
                supervision_frame_tokens[sup_i] = frame_token
                supervision_mask[sup_i] = 1

        available = int(sum(supervision_mask))
        if is_valid and available > 0:
            valid_with_any += 1
        if is_valid and available == 4:
            valid_with_all4 += 1

        entries.append(
            {
                "idx": int(idx),
                "token": token,
                "scene_name": scene_name,
                "valid": is_valid,
                "num_output_frames": int(info.get("num_output_frames", len(frame_tokens))),
                "supervision_labels": SUPERVISION_LABELS,
                "supervision_mask": supervision_mask,
                "supervision_step_indices": supervision_steps,
                "supervision_gt_tokens": supervision_gt_tokens,
                "supervision_occ_gt_rel_paths": supervision_occ_gt_rel_paths,
                "supervision_motion_gt_rel_paths": supervision_motion_gt_rel_paths,
                "supervision_frame_tokens": supervision_frame_tokens,
                "num_supervision": available,
                "has_all_4_supervision": bool(available == 4),
            }
        )

    payload = {
        "metadata": {
            "schema_version": "transport_online_nde_sup4_sidecar_v1",
            "description": "4-step supervision index for transport_online_nde",
            "source_info_path": info_path,
            "source_info_metadata": info_meta,
            "nusc_dataroot": nusc_dataroot,
            "nusc_version": args.nusc_version,
            "occ_gt_root": occ_gt_root,
            "motion_gt_root": motion_gt_root,
            "supervision_labels": SUPERVISION_LABELS,
            "num_infos": len(infos),
            "num_valid_infos": valid_total,
            "num_valid_with_any_supervision": valid_with_any,
            "num_valid_with_all_4_supervision": valid_with_all4,
        },
        "index_by_token": index_by_token,
        "entries": entries,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[save] {output_path}")
    print(
        "[stats] "
        f"infos={len(infos)} valid={valid_total} "
        f"valid_any={valid_with_any} valid_all4={valid_with_all4}"
    )


if __name__ == "__main__":
    main()
