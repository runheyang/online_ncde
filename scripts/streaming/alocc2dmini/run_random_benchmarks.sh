#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-20260725}"
SAMPLES="${SAMPLES:-200}"
WARMUP="${WARMUP:-20}"
PREHEAT_SAMPLES="${PREHEAT_SAMPLES:-10}"
PREHEAT_WARMUP="${PREHEAT_WARMUP:-5}"
SLOW_INTERVAL="${SLOW_INTERVAL:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/random_streaming_benchmark/${RUN_TAG}}"

CONFIG="${REPO_ROOT}/configs/evoocc/fast_alocc2dmini__slow_alocc3d.yaml"
RANDOM_CKPT_DIR="${OUT_ROOT}/random_checkpoints"
EVOOCC_BENCHMARK="${SCRIPT_DIR}/evoocc/benchmark.py"
ATTENTION_BENCHMARK="${SCRIPT_DIR}/learned_direct_attention/benchmark.py"
FUSION_BENCHMARK="${SCRIPT_DIR}/learned_direct_fusion/benchmark.py"
NEURAL_ODE_BENCHMARK="${SCRIPT_DIR}/neural_ode_dt_100/benchmark.py"

if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "Conda 不存在或不可执行: ${CONDA_BIN}" >&2
    exit 1
fi

mkdir -p \
    "${RANDOM_CKPT_DIR}" \
    "${OUT_ROOT}/preheat" \
    "${OUT_ROOT}/final"
cd "${REPO_ROOT}"

{
    echo "run_tag=${RUN_TAG}"
    echo "seed=${SEED}"
    echo "samples=${SAMPLES}"
    echo "warmup=${WARMUP}"
    echo "preheat_samples=${PREHEAT_SAMPLES}"
    echo "preheat_warmup=${PREHEAT_WARMUP}"
    echo "slow_interval=${SLOW_INTERVAL}"
    echo "num_workers=${NUM_WORKERS}"
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader
} | tee "${OUT_ROOT}/run_info.txt"

echo "[prepare] create random initialized checkpoints"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="${REPO_ROOT}/src" \
"${CONDA_BIN}" run --no-capture-output -n OccStudio \
    python "${SCRIPT_DIR}/create_random_benchmark_checkpoints.py" \
    --config "${CONFIG}" \
    --out-dir "${RANDOM_CKPT_DIR}" \
    --seed "${SEED}" \
    2>&1 | tee "${OUT_ROOT}/create_random_checkpoints.log"

run_evoocc() {
    local phase="$1"
    local warmup="$2"
    local samples="$3"
    echo "[run] phase=${phase}, benchmark=evoocc, mode=all"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${REPO_ROOT}/src" \
    "${CONDA_BIN}" run --no-capture-output -n OccStudio \
        python "${EVOOCC_BENCHMARK}" \
        --config "${CONFIG}" \
        --checkpoint "${RANDOM_CKPT_DIR}/evoocc_random.pth" \
        --mode all \
        --slow-interval "${SLOW_INTERVAL}" \
        --warmup "${warmup}" \
        --samples "${samples}" \
        --num-workers "${NUM_WORKERS}" \
        --prefetch-factor "${PREFETCH_FACTOR}" \
        --out-json "${OUT_ROOT}/${phase}/evoocc.json" \
        2>&1 | tee "${OUT_ROOT}/${phase}/evoocc.log"
}

run_baseline() {
    local phase="$1"
    local name="$2"
    local benchmark="$3"
    local warmup="$4"
    local samples="$5"
    echo "[run] phase=${phase}, benchmark=${name}, mode=fast-baseline"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${REPO_ROOT}/src" \
    "${CONDA_BIN}" run --no-capture-output -n OccStudio \
        python "${benchmark}" \
        --config "${CONFIG}" \
        --checkpoint "${RANDOM_CKPT_DIR}/${name}_random.pth" \
        --mode fast-baseline \
        --slow-interval "${SLOW_INTERVAL}" \
        --warmup "${warmup}" \
        --samples "${samples}" \
        --num-workers "${NUM_WORKERS}" \
        --prefetch-factor "${PREFETCH_FACTOR}" \
        --out-json "${OUT_ROOT}/${phase}/${name}.json" \
        2>&1 | tee "${OUT_ROOT}/${phase}/${name}.log"
}

echo "[phase] preheat all four benchmarks"
run_evoocc preheat "${PREHEAT_WARMUP}" "${PREHEAT_SAMPLES}"
run_baseline \
    preheat learned_direct_attention "${ATTENTION_BENCHMARK}" \
    "${PREHEAT_WARMUP}" "${PREHEAT_SAMPLES}"
run_baseline \
    preheat learned_direct_fusion "${FUSION_BENCHMARK}" \
    "${PREHEAT_WARMUP}" "${PREHEAT_SAMPLES}"
run_baseline \
    preheat neural_ode_dt_100 "${NEURAL_ODE_BENCHMARK}" \
    "${PREHEAT_WARMUP}" "${PREHEAT_SAMPLES}"

echo "[phase] final measurements"
run_evoocc final "${WARMUP}" "${SAMPLES}"
run_baseline \
    final learned_direct_attention "${ATTENTION_BENCHMARK}" \
    "${WARMUP}" "${SAMPLES}"
run_baseline \
    final learned_direct_fusion "${FUSION_BENCHMARK}" \
    "${WARMUP}" "${SAMPLES}"
run_baseline \
    final neural_ode_dt_100 "${NEURAL_ODE_BENCHMARK}" \
    "${WARMUP}" "${SAMPLES}"

echo "[summary]"
"${CONDA_BIN}" run --no-capture-output -n OccStudio \
    python - "${OUT_ROOT}" <<'PY'
import json
import os
import sys

root = sys.argv[1]
final_dir = os.path.join(root, "final")

with open(os.path.join(final_dir, "evoocc.json")) as file:
    evoocc = json.load(file)["results"]

rows = [
    ("fast-only", evoocc["fast_only"]),
    ("slow-only", evoocc["slow_only"]),
    ("fast+ours", evoocc["fast_ours"]),
]
for name in (
    "learned_direct_attention",
    "learned_direct_fusion",
    "neural_ode_dt_100",
):
    with open(os.path.join(final_dir, f"{name}.json")) as file:
        result = json.load(file)["results"][f"fast_{name}"]
    rows.append((name, result))

print(f"{'method':<28} {'mean(ms)':>10} {'median(ms)':>12} {'p95(ms)':>10} {'FPS':>9}")
for name, result in rows:
    print(
        f"{name:<28} "
        f"{result['latency_ms_mean']:>10.2f} "
        f"{result['latency_ms_median']:>12.2f} "
        f"{result['latency_ms_p95']:>10.2f} "
        f"{result['fps']:>9.2f}"
    )
PY

echo "[done] ${OUT_ROOT}"
