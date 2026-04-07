#!/usr/bin/env python3
"""Scene-wise downloader for UniOcc NuScenes-via-Occ3D-2Hz train/val."""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SceneRecord:
    scene_name: str
    rel_paths: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download UniOcc NuScenes-via-Occ3D-2Hz train/val scene by scene from "
            "Hugging Face without recursively listing the repo tree."
        )
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split for auto defaults (repo subdir + scene info file).",
    )
    parser.add_argument(
        "--scene-info",
        default="",
        help=(
            "Path to scene info pkl. If empty, auto-detect from output_root "
            "(scene_infos_<split>.pkl -> scene_infos.pkl)."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data/uniocc-nuScenes-Occ3D"),
        help="Output root directory. Files are saved as <output-root>/<scene_name>/*.npz",
    )
    parser.add_argument("--repo-id", default="tasl-lab/uniocc", help="HF dataset repo id")
    parser.add_argument(
        "--repo-subdir",
        default="",
        help=(
            "Subdirectory inside repo containing scene folders. "
            "If empty, auto-set by split: train->NuScenes-via-Occ3D-2Hz-train, "
            "val->NuScenes-via-Occ3D-2Hz-val."
        ),
    )
    parser.add_argument("--revision", default="main", help="HF revision")
    parser.add_argument("--token", default="", help="HF token (optional)")
    parser.add_argument("--workers", type=int, default=16, help="Thread count within each scene")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per file")
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.5,
        help="Base sleep seconds for retries (uses linear backoff)",
    )
    parser.add_argument("--start-scene", type=int, default=0, help="Start scene index (0-based)")
    parser.add_argument("--max-scenes", type=int, default=0, help="Max number of scenes (0=all)")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split selected scenes into N shards for multi-process parallel runs",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Current shard id in [0, num_shards-1]",
    )
    parser.add_argument(
        "--max-files-per-scene",
        type=int,
        default=0,
        help="Only download first N files in each scene (0=all, useful for quick tests)",
    )
    parser.add_argument(
        "--scenes",
        default="",
        help="Comma-separated scene names to keep, e.g. scene-0001,scene-0002",
    )
    parser.add_argument("--force", action="store_true", help="Redownload files even if they exist")
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place cached files to output-root",
    )
    parser.add_argument(
        "--proxy",
        default="",
        help="Proxy URL (sets http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY)",
    )
    parser.add_argument(
        "--hf-transfer",
        action="store_true",
        help="Set HF_HUB_ENABLE_HF_TRANSFER=1 for potentially faster downloads",
    )
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Enable Xet backend if hf_xet is installed (disabled by default for stability)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print download plan")
    parser.add_argument("--verbose", action="store_true", help="Print per-file errors")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    if args.proxy:
        proxy = args.proxy.strip()
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ[name] = proxy
    if args.hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if not args.enable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if not args.verbose:
        disable_progress_bars()


def normalize_rel_path(rel_path: str, scene_name: str, repo_subdir: str) -> str:
    path = rel_path.strip().replace("\\", "/").lstrip("/")
    subdir = repo_subdir.strip().replace("\\", "/").strip("/")
    if subdir and path.startswith(subdir + "/"):
        path = path[len(subdir) + 1 :]
    if not path.startswith(scene_name + "/"):
        filename = path.split("/")[-1]
        path = f"{scene_name}/{filename}"
    return path


def load_scene_records(scene_info_path: Path, repo_subdir: str) -> list[SceneRecord]:
    with scene_info_path.open("rb") as f:
        payload: Any = pickle.load(f)

    entries: list[Any] | None = None
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for key in ("scene_lists", "scene_infos", "scenes"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                entries = candidate
                break

    if entries is None:
        raise ValueError("Unsupported scene_infos.pkl format.")

    scene_records: list[SceneRecord] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"scene entry #{index} is not a dict: {type(item)}")

        scene_name = str(item.get("scene_name", "")).strip()
        if not scene_name:
            raise ValueError(f"scene entry #{index} missing scene_name")

        raw_paths = item.get("occ_in_scene_paths")
        if not isinstance(raw_paths, list):
            raise ValueError(
                f"scene {scene_name} missing occ_in_scene_paths list, cannot map file paths."
            )

        seen: set[str] = set()
        rel_paths: list[str] = []
        for raw in raw_paths:
            path = normalize_rel_path(str(raw), scene_name=scene_name, repo_subdir=repo_subdir)
            if path not in seen:
                seen.add(path)
                rel_paths.append(path)

        scene_records.append(SceneRecord(scene_name=scene_name, rel_paths=rel_paths))

    return scene_records


def resolve_repo_subdir(repo_subdir_arg: str, split: str) -> str:
    if repo_subdir_arg.strip():
        return repo_subdir_arg.strip()
    return f"NuScenes-via-Occ3D-2Hz-{split}"


def resolve_scene_info_path(scene_info_arg: str, output_root: Path, split: str) -> Path:
    if scene_info_arg.strip():
        scene_info_path = Path(scene_info_arg).expanduser().resolve()
        if not scene_info_path.exists():
            raise FileNotFoundError(f"scene info file not found: {scene_info_path}")
        return scene_info_path

    split_name = split.strip().lower()
    candidates = [
        output_root / f"scene_infos_{split_name}.pkl",
        output_root / "scene_infos.pkl",
        ROOT / "data/uniocc-nuScenes-Occ3D" / f"scene_infos_{split_name}.pkl",
        ROOT / "data/uniocc-nuScenes-Occ3D/scene_infos.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No scene info file found. Checked: {joined}")


def filter_scene_records(records: list[SceneRecord], args: argparse.Namespace) -> list[SceneRecord]:
    selected = records
    if args.start_scene > 0:
        selected = selected[args.start_scene :]

    if args.scenes.strip():
        keep = {name.strip() for name in args.scenes.split(",") if name.strip()}
        selected = [record for record in selected if record.scene_name in keep]

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards-1]")
    if args.num_shards > 1:
        selected = [
            record for idx, record in enumerate(selected) if idx % args.num_shards == args.shard_id
        ]

    if args.max_scenes > 0:
        selected = selected[: args.max_scenes]
    return selected


def place_cached_file(cache_file: Path, target_file: Path, copy_mode: str) -> None:
    target_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = target_file.with_suffix(target_file.suffix + ".part")
    if tmp_file.exists():
        tmp_file.unlink()

    try:
        if copy_mode == "hardlink":
            try:
                os.link(cache_file, tmp_file)
            except OSError:
                shutil.copy2(cache_file, tmp_file)
        else:
            shutil.copy2(cache_file, tmp_file)
        tmp_file.replace(target_file)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def download_one_file(
    rel_path: str,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[str, str]:
    target_file = output_root / rel_path
    if target_file.exists() and not args.force:
        return "skipped", rel_path

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            cache_file = hf_hub_download(
                repo_id=args.repo_id,
                filename=rel_path,
                subfolder=args.repo_subdir or None,
                repo_type="dataset",
                revision=args.revision,
                token=args.token or None,
                force_download=args.force,
            )
            place_cached_file(Path(cache_file), target_file, copy_mode=args.copy_mode)
            return "downloaded", rel_path
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep * attempt)

    return "failed", f"{rel_path} :: {type(last_error).__name__}: {last_error}"


def main() -> None:
    args = parse_args()
    configure_environment(args)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    args.repo_subdir = resolve_repo_subdir(args.repo_subdir, args.split)
    scene_info_path = resolve_scene_info_path(args.scene_info, output_root, args.split)

    all_records = load_scene_records(scene_info_path=scene_info_path, repo_subdir=args.repo_subdir)
    records = filter_scene_records(all_records, args)
    if not records:
        print("No scenes selected.")
        return

    print(
        f"[config] scene_info={scene_info_path}\n"
        f"[config] output_root={output_root}\n"
        f"[config] split={args.split}\n"
        f"[config] repo_id={args.repo_id}, repo_subdir={args.repo_subdir}, revision={args.revision}\n"
        f"[config] scenes={len(records)}, workers={args.workers}, retries={args.retries}, "
        f"num_shards={args.num_shards}, shard_id={args.shard_id}, "
        f"copy_mode={args.copy_mode}, dry_run={args.dry_run}"
    )

    started_at = time.time()
    total_requested = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for scene_idx, scene in enumerate(records, start=1):
        rel_paths = scene.rel_paths
        if args.max_files_per_scene > 0:
            rel_paths = rel_paths[: args.max_files_per_scene]
        if not rel_paths:
            print(f"[scene {scene_idx}/{len(records)}] {scene.scene_name}: 0 files, skipped.")
            continue

        pending = rel_paths if args.force else [
            path for path in rel_paths if not (output_root / path).exists()
        ]
        pre_skipped = len(rel_paths) - len(pending)
        total_requested += len(rel_paths)
        total_skipped += pre_skipped

        print(
            f"[scene {scene_idx}/{len(records)}] {scene.scene_name}: "
            f"total={len(rel_paths)} pending={len(pending)} pre_skipped={pre_skipped}"
        )

        if args.dry_run or not pending:
            continue

        workers = max(1, min(args.workers, len(pending)))
        scene_downloaded = 0
        scene_failed = 0

        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download_one_file, path, args, output_root): path for path in pending}
            for future in as_completed(futures):
                status, payload = future.result()
                if status == "downloaded":
                    scene_downloaded += 1
                elif status == "skipped":
                    total_skipped += 1
                else:
                    scene_failed += 1
                    if len(failures) < 10:
                        failures.append(payload)

        total_downloaded += scene_downloaded
        total_failed += scene_failed
        print(
            f"[scene {scene_idx}/{len(records)}] {scene.scene_name}: "
            f"downloaded={scene_downloaded} failed={scene_failed}"
        )
        if failures and args.verbose:
            for line in failures:
                print(f"  [error] {line}")

    elapsed = time.time() - started_at
    print(
        f"[summary] requested={total_requested}, downloaded={total_downloaded}, "
        f"skipped={total_skipped}, failed={total_failed}, elapsed_sec={elapsed:.1f}"
    )
    if total_failed > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
