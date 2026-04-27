#!/usr/bin/env bash
# 批量跑 eval_online_ncde_random_interval.py 多 seed（默认 0..9），收集日志。
#
# 用法：
#   bash tests/online_ncde/run_random_interval_seeds.sh \
#       --config <cfg.yaml> --checkpoint <ckpt.pt>
#
# 可选参数：
#   --gap-choices "1,2,3"         默认 "1,2,3"
#   --target-last-step 12         默认 12
#   --seeds "0 1 2 3 4 5 6 7 8 9" 默认 0..9（空格分隔）
#   --solver euler|heun           默认 euler
#   --limit N                     默认 0（全部样本）
#   --log-dir logs/random_interval  日志目录
#   --extra "..."                 透传给 python 脚本的额外参数

set -euo pipefail

# ---- 默认值 ----
CONFIG=""
CKPT=""
GAP_CHOICES="1,2,3"
TARGET_LAST_STEP=12
SEEDS="0 1 2 3 4 5 6 7 8 9"
SOLVER="euler"
LIMIT=0
LOG_DIR=""
EXTRA=""

usage() {
    cat <<EOF
Usage: $0 --config <cfg> --checkpoint <ckpt> [options]
  --config           (required) 配置文件路径
  --checkpoint       (required) 模型权重路径
  --gap-choices      默认 "${GAP_CHOICES}"
  --target-last-step 默认 ${TARGET_LAST_STEP}
  --seeds            空格分隔，默认 "${SEEDS}"
  --solver           euler|heun，默认 ${SOLVER}
  --limit            前 N 个样本，0 全部，默认 0
  --log-dir          日志目录，默认 logs/random_interval/<ckpt名>_g<gaps>
  --extra            透传到 python 的额外参数
  -h, --help         显示帮助
EOF
}

# ---- 参数解析 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)            CONFIG="$2"; shift 2 ;;
        --checkpoint)        CKPT="$2"; shift 2 ;;
        --gap-choices)       GAP_CHOICES="$2"; shift 2 ;;
        --target-last-step)  TARGET_LAST_STEP="$2"; shift 2 ;;
        --seeds)             SEEDS="$2"; shift 2 ;;
        --solver)            SOLVER="$2"; shift 2 ;;
        --limit)             LIMIT="$2"; shift 2 ;;
        --log-dir)           LOG_DIR="$2"; shift 2 ;;
        --extra)             EXTRA="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" || -z "$CKPT" ]]; then
    echo "错误：必须指定 --config 和 --checkpoint" >&2
    usage
    exit 1
fi

# ---- 路径处理 ----
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd )"
PY_SCRIPT="${SCRIPT_DIR}/eval_online_ncde_random_interval.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "错误：找不到 ${PY_SCRIPT}" >&2
    exit 1
fi

# 默认 log_dir：logs/random_interval/<ckpt_basename>_g<gaps>
if [[ -z "$LOG_DIR" ]]; then
    CKPT_TAG=$(basename "${CKPT}" .pt)
    GAP_TAG=$(echo "${GAP_CHOICES}" | tr ',' '-')
    LOG_DIR="${REPO_ROOT}/logs/random_interval/${CKPT_TAG}_g${GAP_TAG}"
fi
mkdir -p "${LOG_DIR}"

# ---- 摘要 ----
SUMMARY="${LOG_DIR}/summary.csv"
{
    echo "# config=${CONFIG}"
    echo "# checkpoint=${CKPT}"
    echo "# gap_choices=${GAP_CHOICES}"
    echo "# target_last_step=${TARGET_LAST_STEP}"
    echo "# solver=${SOLVER}"
    echo "# limit=${LIMIT}"
    echo "# seeds=${SEEDS}"
    echo "seed,miou,miou_d,rayiou,rayiou_at1,rayiou_at2,rayiou_at4"
} > "${SUMMARY}"

echo "[run] log_dir=${LOG_DIR}"
echo "[run] seeds=${SEEDS}"
echo "[run] gap_choices=${GAP_CHOICES}  target_last_step=${TARGET_LAST_STEP}"
echo

# ---- 逐 seed 运行 ----
for seed in ${SEEDS}; do
    LOG="${LOG_DIR}/seed${seed}.log"
    echo "==================== seed=${seed} ===================="
    echo "[run] -> ${LOG}"

    CMD=(
        conda run -n neural_ode python "${PY_SCRIPT}"
        --config "${CONFIG}"
        --checkpoint "${CKPT}"
        --random
        --gap-choices "${GAP_CHOICES}"
        --target-last-step "${TARGET_LAST_STEP}"
        --seed "${seed}"
        --solver "${SOLVER}"
    )
    if [[ "${LIMIT}" -gt 0 ]]; then
        CMD+=(--limit "${LIMIT}")
    fi
    if [[ -n "${EXTRA}" ]]; then
        # 透传额外参数（用 eval 拆词以支持引号）
        eval "EXTRA_ARGS=( ${EXTRA} )"
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    "${CMD[@]}" 2>&1 | tee "${LOG}"

    # ---- 从 log 解析关键指标 ----
    miou=$(grep -m1 -oE "miou=[0-9.]+" "${LOG}" | head -1 | cut -d= -f2 || echo "")
    miou_d=$(grep -m1 -oE "miou_d=[0-9.]+" "${LOG}" | head -1 | cut -d= -f2 || echo "")
    rayiou=$(grep -m1 "RayIoU=" "${LOG}" | grep -oE "RayIoU=[0-9.]+" | cut -d= -f2 || echo "")
    rayiou1=$(grep -m1 "RayIoU@1=" "${LOG}" | grep -oE "RayIoU@1=[0-9.]+" | cut -d= -f2 || echo "")
    rayiou2=$(grep -m1 "RayIoU@2=" "${LOG}" | grep -oE "RayIoU@2=[0-9.]+" | cut -d= -f2 || echo "")
    rayiou4=$(grep -m1 "RayIoU@4=" "${LOG}" | grep -oE "RayIoU@4=[0-9.]+" | cut -d= -f2 || echo "")

    echo "${seed},${miou},${miou_d},${rayiou},${rayiou1},${rayiou2},${rayiou4}" >> "${SUMMARY}"
    echo "[run] seed=${seed} miou=${miou} miou_d=${miou_d} rayiou=${rayiou}"
    echo
done

echo "==================== 汇总 ===================="
echo "[run] summary csv: ${SUMMARY}"
column -t -s, "${SUMMARY}" || cat "${SUMMARY}"

# ---- mean ± std（用 python 计算，避免依赖 bc/awk 浮点）----
echo
echo "==================== mean ± std ===================="
conda run -n neural_ode python - <<PY
import csv, math, statistics, sys
path = "${SUMMARY}"
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("seed,"):
            continue
        rows.append(line.split(","))

if not rows:
    print("no data parsed from ${SUMMARY}")
    sys.exit(0)

cols = ["miou", "miou_d", "rayiou", "rayiou_at1", "rayiou_at2", "rayiou_at4"]
data = {c: [] for c in cols}
for r in rows:
    for i, c in enumerate(cols, start=1):
        try:
            data[c].append(float(r[i]))
        except (ValueError, IndexError):
            pass

print(f"{'metric':<12} {'n':>3} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
print("-" * 60)
for c in cols:
    vs = data[c]
    if not vs:
        print(f"{c:<12} {0:>3} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
        continue
    mean = statistics.mean(vs)
    std = statistics.stdev(vs) if len(vs) >= 2 else 0.0
    print(f"{c:<12} {len(vs):>3} {mean:>10.4f} {std:>10.4f} {min(vs):>10.4f} {max(vs):>10.4f}")
PY
