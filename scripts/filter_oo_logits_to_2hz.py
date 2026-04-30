"""按 2Hz canonical_infos 的 stride=3 抽帧规则，从 6Hz fast logits 目录里筛出 keyframe，
镜像到 *_2hz 目录（默认硬链接，不占空间），方便只把这 1/3 体积传到云服务器。

依据：
  Occ3DOnlineNcdeDataset.__getitem__ 在 fast_frame_stride=3 下做 frame_rel_paths[::3]，
  只读 idx ∈ {0,3,6,9,12} 的 5 个 npz，sweep（idx ∉ {0,3,6,9,12}）一定不会被读。

slow logits 导出时只对 keyframe 输出，无冗余，本脚本只处理 fast。

用法（默认硬链接）：
  conda run -n neural_ode python scripts/filter_oo_logits_to_2hz.py \
      --canonical-pkls configs/online_ncde/canonical_infos_train_full.pkl \
                       configs/online_ncde/canonical_infos_val_full.pkl \
      --src-root data/logits_opusv2t_openocc \
      --dst-root data/logits_opusv2t_openocc_2hz
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Iterable


def _collect_paths(
    pkl_paths: Iterable[Path], stride: int, include_fast: bool, include_slow: bool
) -> set[str]:
    keep: set[str] = set()
    total = 0
    for pkl in pkl_paths:
        with open(pkl, "rb") as f:
            payload = pickle.load(f)
        infos = payload["infos"] if isinstance(payload, dict) else payload
        total += len(infos)
        for info in infos:
            if include_fast:
                for rel in (info.get("frame_rel_paths", []) or [])[::stride]:
                    if rel:
                        keep.add(str(rel))
            if include_slow:
                slow_rel = info.get("slow_logit_path", "")
                if slow_rel:
                    keep.add(str(slow_rel))
    print(f"[scan] 扫描 {total} 个 sample，唯一保留路径 = {len(keep)}")
    return keep


def _apply(
    src_root: Path,
    dst_root: Path,
    rel_paths: Iterable[str],
    mode: str,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    n_ok = n_skip = n_missing = 0
    bytes_total = 0
    for rel in rel_paths:
        src = (src_root / rel).resolve()
        dst = dst_root / rel
        if not src.is_file():
            n_missing += 1
            if n_missing <= 5:
                print(f"  [missing] {src}")
            continue
        if dst.exists():
            if skip_existing:
                n_skip += 1
                continue
            if not dry_run:
                dst.unlink()
        if dry_run:
            n_ok += 1
            bytes_total += src.stat().st_size
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            try:
                os.link(src, dst)
            except OSError as e:
                print(f"  [hardlink fail, fallback to copy] {src}: {e}")
                shutil.copy2(src, dst)
        elif mode == "symlink":
            os.symlink(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        n_ok += 1
        bytes_total += src.stat().st_size

    print(
        f"[apply mode={mode} dry_run={dry_run}] ok={n_ok} skip_existing={n_skip} "
        f"missing={n_missing}  size_total≈{bytes_total/1024/1024:.1f} MB"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-pkls", nargs="+", required=True, type=Path)
    p.add_argument("--src-root", required=True, type=Path)
    p.add_argument("--dst-root", required=True, type=Path)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    fg = p.add_mutually_exclusive_group()
    fg.add_argument("--include-fast", dest="include_fast", action="store_true", default=True)
    fg.add_argument("--no-include-fast", dest="include_fast", action="store_false")
    p.add_argument("--include-slow", action="store_true")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.include_fast and not args.include_slow:
        sys.exit("--no-include-fast 与 --include-slow 至少要开一个")

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    if not src_root.exists():
        sys.exit(f"src_root 不存在: {src_root}")
    if dst_root == src_root:
        sys.exit("dst_root 不能与 src_root 相同")
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] stride={args.stride} mode={args.mode} fast={args.include_fast} slow={args.include_slow}")
    print(f"[cfg] src={src_root}\n[cfg] dst={dst_root}")
    keep = _collect_paths(args.canonical_pkls, args.stride, args.include_fast, args.include_slow)
    _apply(src_root, dst_root, sorted(keep), args.mode, args.skip_existing, args.dry_run)


if __name__ == "__main__":
    main()
