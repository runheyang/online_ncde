# Ray Loss 调参 Loop 工作流（agent 读取的权威文档）

> 每轮 agent 必须按本文件步骤执行。方向性结论/硬约束见 `SUMMARY.md`，历史数据/搜索空间/v2_plan 见 `state.json`。三处如有冲突以最新修改为准。

## 版本上下文
- **v1 (Run 1-7, 无 EMA)**：已归档到 `v1_pre_ema/`，绝对数值弃用
- **v2 (Run 8+, EMA 默认开启)**：当前阶段。v2 结果只与 Run 8（no-ray baseline）做 delta 对比，**不要**与 v1 数字比
- 新 run log 路径：`scripts/ray_tuning_runs/v2_with_ema/logs/run_N.log`
- state.json 里 v2 run 必须带 `"ema": true`

## 目标
在 3000-sample / 8-epoch 子集上最大化 **0-10m RayIoU**，mIoU 允许 -0.5 内波动。快系统 baseline（20 scene val）是 overall 0.3829 / 0-10m 0.4284 / 0-10m signed_err 2.16m——这是真正要追平的参照。

## 每轮工作流

### 1. 读权威文件
- `SUMMARY.md`（v1 保留的方向性结论 + v2 硬约束）
- `state.json`（历史 runs、`search_space`、`v2_plan`、`reference_runs.analysis`）

### 2. 检查 GPU
```bash
ssh -p 14661 root@connect.westd.seetacloud.com 'nvidia-smi'
```
若训练在跑 → 结束本轮（ScheduleWakeup 等待）

### 3. 收集未入库结果
若远端最新 `outputs/.../metrics.json` 未写入 state.json：
```bash
ssh -p 14661 root@connect.westd.seetacloud.com 'cat ~/autodl-tmp/online_ncde/<output_dir>/metrics.json'
```
结果写到对应 run 的 `results` 字段，补充 `analysis`（证实/证伪假设）

### 4. 决定下一轮参数
- Run 8-10 按 `state.json.v2_plan` 必跑，顺序不变
- Run 11-13 自主探索，参考 SUMMARY 的硬约束和 v1 保留结论
- 每轮只改 1-2 个参数，控制变量
- `change_from_ref` 和 `hypothesis` 必填

### 5. 启动训练
```bash
ssh -p 14661 root@connect.westd.seetacloud.com \
  'bash -lc "source /root/miniconda3/etc/profile.d/conda.sh && conda activate neural_ode && \
  cd ~/autodl-tmp/online_ncde && \
  nohup torchrun --nproc_per_node=2 scripts/train_online_ncde_200x200x16.py \
    --config configs/online_ncde_200x200x16/fast_alocc2dmini__slow_alocc3d/train.yaml \
    --train-limit 3000 --val-scene-count 20 --epochs 8 --rayiou --save-metrics-json \
    --ray-override '"'"'{参数 JSON}'"'"' \
    > scripts/ray_tuning_runs/v2_with_ema/logs/run_NNN.log 2>&1 &"'
```
EMA 默认由 config 开启，无需 CLI 参数。

### 6. 写 state.json
新 run 条目：`id`、`status: "running"`、`started`、`params`、`change_from_ref`、`hypothesis`、`log`、`output_dir`、`ema: true`

### 7. 结束条件
- Run 13 完成 → 输出总结到 `v2_with_ema/SUMMARY.md` 并停止循环
- 若 Run 11-13 都无法突破 Run 9/10 → 写 `proposals.md`（结构性修改建议），不要改代码

## 硬约束（v2）

1. **ray loss 必须覆盖近场**：不 mask、不降 near_weight 到 1.0 以下、不 `lambda_empty ≥ 0.2`
2. **每轮只改 1-2 个参数**
3. **v2 结果只与 Run 8 比 delta**，不与 v1 数字比
4. **反直觉对照 / 方差估计触发式规则**（满足任一条件，下一轮必须做对照而非主线探索）：
   - 连续 3 轮沿同一参数轴单调调整 → 跑反向值或中间值证伪
   - 累计 v2 run ≥ 6 且未做过 seed 方差估计 → 换 seed 重跑当前最佳
   - 主线叙事连续 2 轮 delta < 0.003 → 停止沿该轴调整
5. 对照轮的 `change_from_ref` 标注"反直觉对照"或"方差估计"，不追求指标改善

## metrics.json 结构（远端训练产出，供 agent 解析）

```json
{
  "epoch": 8,
  "mIoU": 37.0,
  "mIoU_d": 32.0,
  "loss": 0.85,
  "ray_config": {"lambda_ray": 0.2, "lambda_hit": 0.3, ...},
  "rayiou": {"RayIoU": 0.38, "RayIoU@1": 0.32, "RayIoU@2": 0.39, "RayIoU@4": 0.43},
  "binned_ray": {
    "all":   {"RayIoU": 0.38, "mean_signed_err": 2.5, "miss_rate": 0.10, "false_hit_rate": 0.13, ...},
    "0-10m": {"RayIoU": 0.41, "mean_signed_err": 3.2, ...},
    "10-20m": {...},
    "20-40m": {...}
  }
}
```
