"""ALOcc2DMini + learned direct fusion benchmark 入口。"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from evoocc.streaming.benchmark_runtime import configure_benchmark_env

configure_benchmark_env()

from evoocc.streaming.alocc2dmini_baseline_benchmark import main


if __name__ == "__main__":
    main("learned_direct_fusion", REPO_ROOT)
