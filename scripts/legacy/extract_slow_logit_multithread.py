#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np


Result = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract first-frame sparse logits from each logits.npz and save "
            "as slow_logit.npz in the same directory."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspaces/neural_ode/data/logits_opusv2l"),
        help="Root directory containing scene/frame folders.",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="logits.npz",
        help="Input npz filename to search for recursively.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="slow_logit.npz",
        help="Output filename saved beside each input file.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(32, (os.cpu_count() or 8) * 2),
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing slow_logit.npz files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N files for quick testing. 0 means all.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Progress print interval.",
    )
    return parser.parse_args()


def process_one(
    logits_path: Path,
    output_name: str,
    overwrite: bool,
) -> Result:
    out_path = logits_path.with_name(output_name)
    if out_path.exists() and not overwrite:
        return ("skipped", str(logits_path), "")

    try:
        with np.load(logits_path, allow_pickle=False) as data:
            frame_splits = data["frame_splits"]
            if frame_splits.ndim != 1 or frame_splits.shape[0] < 2:
                return (
                    "error",
                    str(logits_path),
                    f"invalid frame_splits shape: {frame_splits.shape}",
                )

            start = int(frame_splits[0])
            end = int(frame_splits[1])
            if start < 0 or end < start:
                return ("error", str(logits_path), f"invalid split [{start}, {end})")

            sparse_coords = data["sparse_coords"][start:end]
            sparse_topk_values = data["sparse_topk_values"][start:end]
            sparse_topk_indices = data["sparse_topk_indices"][start:end]

        np.savez(
            out_path,
            sparse_coords=sparse_coords,
            sparse_topk_values=sparse_topk_values,
            sparse_topk_indices=sparse_topk_indices,
        )
        return ("processed", str(logits_path), "")
    except Exception as exc:  # pragma: no cover
        return ("error", str(logits_path), repr(exc))


def collect_files(root: Path, input_name: str, limit: int) -> List[Path]:
    files = sorted(root.rglob(input_name))
    if limit > 0:
        files = files[:limit]
    return files


def main() -> int:
    args = parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 2

    files = collect_files(root=root, input_name=args.input_name, limit=args.limit)
    total = len(files)
    if total == 0:
        print(f"[INFO] no {args.input_name} found under: {root}")
        return 0

    print(f"[INFO] root={root}")
    print(f"[INFO] found files={total}, workers={args.num_workers}")
    print(f"[INFO] input={args.input_name}, output={args.output_name}")

    processed = 0
    skipped = 0
    errors: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_one, p, args.output_name, args.overwrite)
            for p in files
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            status, file_path, message = fut.result()
            if status == "processed":
                processed += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors.append((file_path, message))

            if i % max(1, args.log_every) == 0 or i == total:
                print(
                    f"[PROGRESS] {i}/{total} "
                    f"processed={processed} skipped={skipped} errors={len(errors)}"
                )

    print("[DONE]")
    print(f"[RESULT] processed={processed} skipped={skipped} errors={len(errors)}")
    if errors:
        first_path, first_err = errors[0]
        print(f"[FIRST_ERROR] {first_path} :: {first_err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
