#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
GPU_ID="${GPU_ID:-0}"
SAMPLES="${SAMPLES:-400}"
WARMUP="${WARMUP:-20}"
SLOW_INTERVAL="${SLOW_INTERVAL:-5}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/system_benchmark_fair_workers/${RUN_TAG}}"

CONFIG="${REPO_ROOT}/configs/evoocc/fast_alocc2dmini__slow_alocc3d.yaml"
CHECKPOINT="${REPO_ROOT}/ckpts/epoch_9.pth"
BENCHMARK="${REPO_ROOT}/scripts/streaming/alocc2dmini/evoocc/system_benchmark.py"

FAST_GFLOPS="176.796458628"
SLOW_GFLOPS="383.799587012"
EVOLVE_GFLOPS="60.57088"
RESET_GFLOPS="20.02944"

if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "Conda 不存在或不可执行: ${CONDA_BIN}" >&2
    exit 1
fi

mkdir -p "${OUT_ROOT}/warm_cache" "${OUT_ROOT}/official"
cd "${REPO_ROOT}"

{
    echo "run_tag=${RUN_TAG}"
    echo "gpu_id=${GPU_ID}"
    echo "samples=${SAMPLES}"
    echo "warmup=${WARMUP}"
    echo "slow_interval=${SLOW_INTERVAL}"
    echo "prefetch_factor=${PREFETCH_FACTOR}"
    echo "worker_budget=6"
    echo "fast_only_workers=6"
    echo "slow_only_workers=6"
    echo "fast_ours_workers=4+2"
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader
    lscpu | grep -E '^(Model name|CPU\\(s\\)):'
} | tee "${OUT_ROOT}/hardware_and_schedule.txt"

run_mode() {
    local phase="$1"
    local mode="$2"
    local fast_workers="$3"
    local slow_workers="$4"
    local phase_dir="${OUT_ROOT}/${phase}"
    local stem="${mode//-/_}"

    echo "[run] phase=${phase}, mode=${mode}, fast_workers=${fast_workers}, slow_workers=${slow_workers}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${REPO_ROOT}/src" \
    "${CONDA_BIN}" run --no-capture-output -n OccStudio \
        python "${BENCHMARK}" \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --mode "${mode}" \
        --slow-interval "${SLOW_INTERVAL}" \
        --warmup "${WARMUP}" \
        --samples "${SAMPLES}" \
        --num-workers 6 \
        --fast-num-workers "${fast_workers}" \
        --slow-num-workers "${slow_workers}" \
        --prefetch-factor "${PREFETCH_FACTOR}" \
        --fast-gflops "${FAST_GFLOPS}" \
        --slow-gflops "${SLOW_GFLOPS}" \
        --evolve-gflops "${EVOLVE_GFLOPS}" \
        --reset-gflops "${RESET_GFLOPS}" \
        --out-json "${phase_dir}/${stem}.json" \
        2>&1 | tee "${phase_dir}/${stem}.log"
}

echo "[phase] warm filesystem cache; results are not reported"
run_mode warm_cache fast-only 6 0
run_mode warm_cache slow-only 0 6
run_mode warm_cache fast-ours 4 2

echo "[phase] official measurements"
run_mode official fast-only 6 0
run_mode official slow-only 0 6
run_mode official fast-ours 4 2

echo "[done] ${OUT_ROOT}"
