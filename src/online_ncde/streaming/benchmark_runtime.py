"""Runtime defaults for reproducible streaming benchmarks."""
from __future__ import annotations

import os


def _read_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def configure_benchmark_env(default_threads: int = 1) -> None:
    """Set CPU thread env defaults before importing torch."""
    threads = os.environ.get("TORCH_NUM_THREADS", str(default_threads))
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    os.environ.setdefault("MKL_NUM_THREADS", threads)


def configure_torch_benchmark_runtime(torch_module, default_threads: int = 1) -> None:
    """Apply torch thread defaults after importing torch."""
    threads = _read_positive_int(
        "TORCH_NUM_THREADS",
        _read_positive_int("OMP_NUM_THREADS", default_threads),
    )
    interop_threads = _read_positive_int("TORCH_NUM_INTEROP_THREADS", 1)
    torch_module.set_num_threads(threads)
    torch_module.set_num_interop_threads(interop_threads)
