#!/usr/bin/env python3
"""为 Online NCDE 生成 4 时刻监督 sidecar（t-1.5, t-1.0, t-0.5, t）。"""

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
    parser.add_argument("--root-path", default=str(ROOT), help="仓库根目录")
    parser.add_argument("--info-path", required=True, help="输入 ncde_align_infos_*.pkl")
    parser.add_argument(
        "--output",
        default="",
        help="输出 sidecar 路径；空则默认写到与 info 同目录，文件名追加 _sup4_sidecar.pkl",
    )
    parser.add_argument(
        "--nusc-dataroot",
        default="data/nuscenes",
        help="NuScenes dataroot",
    )
    parser.add_argument(
        "--nusc-version",
        default="v1.0-trainval",
        help="NuScenes 版本",
    )
    parser.add_argument(
        "--gt-root",
        default="data/nuscenes/gts",
        help="GT 根目录",
    )
    return parser.parse_args()


def build_output_path(info_path: str, output: str) -> str:
    if output:
        return output
    info = Path(info_path)
    stem = info.stem + "_sup4_sidecar"
    return str(info.with_name(stem + info.suffix))


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
        raise TypeError(f"infos 类型异常: {type(infos)}")
    return infos, meta


def collect_target_tokens(sample_table: dict[str, dict[str, Any]], curr_token: str) -> list[str]:
    """返回按 [t-1.5, t-1.0, t-0.5, t] 顺序排列的 keyframe sample token。"""
    curr = curr_token
    chain = [curr]  # [t, t-0.5, t-1.0, t-1.5]
    for _ in range(3):
        sample = sample_table.get(curr, None)
        if sample is None:
            break
        prev_token = str(sample.get("prev", ""))
        if not prev_token:
            break
        chain.append(prev_token)
        curr = prev_token
    chain = chain[::-1]  # oldest -> current
    if len(chain) < 4:
        chain = [""] * (4 - len(chain)) + chain
    elif len(chain) > 4:
        chain = chain[-4:]
    return chain


def main() -> None:
    args = parse_args()
    root_path = args.root_path
    info_path = resolve_path(root_path, args.info_path)
    output_path = resolve_path(root_path, build_output_path(info_path, args.output))
    nusc_dataroot = resolve_path(root_path, args.nusc_dataroot)
    gt_root = resolve_path(root_path, args.gt_root)

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
    valid_with_all4 = 0
    valid_with_any = 0

    iterator = tqdm(enumerate(infos), total=len(infos), desc="[gen sup4 sidecar]") if tqdm is not None else enumerate(infos)
    for idx, info in iterator:
        token = str(info.get("token", ""))
        scene_name = str(info.get("scene_name", ""))
        is_valid = bool(info.get("valid", True))
        frame_tokens = [str(tok) for tok in info.get("frame_tokens", [])]

        supervision_steps = list(EMPTY_STEPS)
        supervision_gt_tokens = list(EMPTY_STR4)
        supervision_gt_rel_paths = list(EMPTY_STR4)
        supervision_frame_tokens = list(EMPTY_STR4)
        supervision_mask = list(EMPTY_MASK4)

        if token:
            index_by_token[token] = idx

        if is_valid:
            valid_total += 1

        if token and scene_name and is_valid and frame_tokens:
            target_tokens = collect_target_tokens(sample_table=sample_table, curr_token=token)

            # 在当前 13 帧序列中找到 keyframe sample token -> step_idx
            keyframe_step_map: dict[str, tuple[int, str]] = {}
            for step_idx, frame_token in enumerate(frame_tokens):
                if not frame_token:
                    continue
                pair = sample_data_to_sample.get(frame_token, None)
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
                step_pair = keyframe_step_map.get(sup_token, None)
                if step_pair is None:
                    continue

                step_idx, frame_token = step_pair
                gt_rel_path = os.path.join(scene_name, sup_token, "labels.npz")
                gt_abs_path = os.path.join(gt_root, gt_rel_path)
                if not os.path.exists(gt_abs_path):
                    continue

                supervision_steps[sup_i] = int(step_idx)
                supervision_gt_tokens[sup_i] = sup_token
                supervision_gt_rel_paths[sup_i] = gt_rel_path
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
                "supervision_mask": supervision_mask,  # 长度4，1表示该时刻可监督
                "supervision_step_indices": supervision_steps,  # 长度4，缺失为 -1
                "supervision_gt_tokens": supervision_gt_tokens,  # 长度4，缺失为空串
                "supervision_gt_rel_paths": supervision_gt_rel_paths,  # 相对 gt_root
                "supervision_frame_tokens": supervision_frame_tokens,  # 对应 step 的 sample_data token
                "num_supervision": available,
                "has_all_4_supervision": bool(available == 4),
            }
        )

    payload = {
        "metadata": {
            "schema_version": "online_ncde_sup4_sidecar_v1",
            "description": "4时刻监督索引（t-1.5/t-1.0/t-0.5/t）",
            "source_info_path": info_path,
            "source_info_metadata": info_meta,
            "nusc_dataroot": nusc_dataroot,
            "nusc_version": args.nusc_version,
            "gt_root": gt_root,
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
