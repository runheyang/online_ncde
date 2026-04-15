#!/usr/bin/env bash
# v2 Run 8-10 串行启动脚本
# - Run 8: no-ray baseline（EMA 开启）
# - Run 9: v1 Run 5 配置复现（lambda_hit=0.3, depth=0.2）
# - Run 10: Run 9 + 加强 depth（lambda_depth=0.3, depth_asym_far=3.0）
#
# 远端执行（推荐 nohup，全程约 10-12h）：
#   cd ~/autodl-tmp/online_ncde
#   nohup bash scripts/ray_tuning_runs/v2_run_8_9_10.sh \
#     > scripts/ray_tuning_runs/v2_with_ema/logs/driver.log 2>&1 &
#
# 如需跳过已完成的 run，注释掉对应 run_one 调用

set -eo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate neural_ode

PROJECT_DIR="${PROJECT_DIR:-$HOME/autodl-tmp/online_ncde}"
cd "$PROJECT_DIR"

LOG_DIR="scripts/ray_tuning_runs/v2_with_ema/logs"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
    --config configs/online_ncde_200x200x16/fast_alocc2dmini__slow_alocc3d/train.yaml
    --train-limit 3000
    --val-scene-count 20
    --epochs 8
    --rayiou
    --save-metrics-json
)

run_one() {
    local run_id=$1
    local ray_override=$2
    local log_file="$LOG_DIR/run_${run_id}.log"
    echo "========== Run ${run_id} START $(date '+%F %T') =========="
    echo "[driver] ray_override: ${ray_override}"
    echo "[driver] log: ${log_file}"
    torchrun --nproc_per_node=2 scripts/train_online_ncde_200x200x16.py \
        "${COMMON_ARGS[@]}" \
        --ray-override "$ray_override" \
        > "$log_file" 2>&1
    echo "========== Run ${run_id} DONE  $(date '+%F %T') =========="
}

# Run 8: 完全关 ray loss，建立 EMA 下的新基线
run_one 8 '{"lambda_ray": 0.0}'

# Run 9: v1 Run 5 配置（lambda_hit=0.3, depth=0.2, empty=0）
run_one 9 '{"lambda_ray": 0.2, "lambda_hit": 0.3, "lambda_empty": 0.0, "lambda_pre_free": 0.0, "lambda_depth": 0.2, "near_weight": 2.0, "depth_asym_far": 2.0, "depth_asym_near": 1.0}'

# Run 10: Run 9 基础上加强 depth（lambda_hit=0.3 下首次测 depth 强化）
run_one 10 '{"lambda_ray": 0.2, "lambda_hit": 0.3, "lambda_empty": 0.0, "lambda_pre_free": 0.0, "lambda_depth": 0.3, "near_weight": 2.0, "depth_asym_far": 3.0, "depth_asym_near": 1.0}'

echo "[driver] All 3 runs completed at $(date '+%F %T')"
