# Ray Loss 调参 Loop 工作流（agent 读取的权威文档）

> 每轮 agent 必须按本文件步骤执行。方向性结论/硬约束见 `SUMMARY.md`，历史数据/搜索空间/v2_with_gate_plan 见 `state.json`，快系统 baseline 见 `baseline.json`。三处如有冲突以最新修改为准。

## 阶段上下文
- **当前阶段：v2_with_gate**（门控 aligner，Run 16+）
- **预算**：4 个固定对照（Run 16-19）+ 3 轮自主探索（Run 20-22），共 7 轮
- 历史 v1_wo_gate（Run 8-15）已完成，详细数据在 `v1_wo_gate/results/run_N.json`
- 新 run 输出：log 写到 `v2_with_gate/logs/run_N.log`，详细 results 写到 `v2_with_gate/results/run_N.json`

## 目标
在 3000-sample / 8-epoch 子集上最大化 **0-10m RayIoU**，mIoU 允许 -0.5 内波动。
- 快系统 baseline（baseline.json）：0-10m RayIoU 0.4284, false_hit 0.086, signed_err 2.16m
- v1 最高 0-10m RayIoU（Run 13）：0.4135
- v2 成功线：Run 17 ≥ 0.4135 且 0-10m false_hit ≤ 0.15

---

## 每轮工作流

### 1. 读权威文件
- `SUMMARY.md`（v1 保留结论 + v2 硬约束 + 判定标准）
- `state.json`（runs_index、search_space、v2_with_gate_plan）
- `baseline.json`（快系统 baseline 完整指标，对比锚点）

### 2. 检查 GPU
```bash
ssh -p 14661 root@connect.westd.seetacloud.com 'nvidia-smi'
```
若训练在跑 → 结束本轮（ScheduleWakeup 等待）。

### 3. 收集未入库结果
若远端最新 `outputs/.../metrics.json` 未入库：
```bash
ssh -p 14661 root@connect.westd.seetacloud.com 'cat ~/autodl-tmp/online_ncde/<output_dir>/metrics.json'
```
然后：
- 把完整 JSON（含 `gate` 字段）写到 `v2_with_gate/results/run_N.json`，加 `analysis` 字段
- 在 `state.json.runs_index` 追加一条轻量索引（id/stage/use_gate/params_brief/key_metrics/tag/details）
- `key_metrics` 必须含：`mIoU`、`RayIoU`、`0-10m_RayIoU`、`0-10m_false_hit`、`0-10m_signed_err`，gate 阶段还要加 `gate_eval_mean` 和 `gate_eval_std`
- `analysis` 必填，必须包含：vs Run 8 / Run 11 / 对应 v1 ref 的 delta、gate 行为解读（mean/std 是否符合预期）

### 4. 决定下一轮参数

**优先按 `state.json.v2_with_gate_plan.fixed_runs` 顺序执行 Run 16-19**（不允许跳过）。

Run 16-19 完成后进入自主探索（Run 20-22），约束如下：
- ❌ **禁止**调整 `lambda_ray`（固定 0.2）
- ✅ 可探索：`lambda_hit / lambda_depth / lambda_empty / lambda_pre_free / near_weight`
- 每轮只改 1-2 个参数，控制变量
- `change_from_ref` 和 `hypothesis` 必填，要明确 ref 是哪个 run

### 5. 启动训练
```bash
ssh -p 14661 root@connect.westd.seetacloud.com \
  'bash -lc "source /root/miniconda3/etc/profile.d/conda.sh && conda activate neural_ode && \
  cd ~/autodl-tmp/online_ncde && \
  nohup torchrun --nproc_per_node=2 scripts/train_online_ncde_200x200x16.py \
    --config configs/online_ncde_200x200x16/fast_alocc2dmini__slow_alocc3d/train.yaml \
    --train-limit 3000 --val-scene-count 20 --epochs 8 --rayiou --save-metrics-json \
    --ray-override '"'"'{参数 JSON}'"'"' \
    > scripts/ray_tuning_runs/v2_with_gate/logs/run_N.log 2>&1 &"'
```
EMA 和 gate 默认由 config 开启，无需 CLI 参数。

### 6. 写 state.json + 占位 results 文件
新 run 在 `state.json.runs_index` 追加：
```json
{
  "id": N, "stage": "v2_with_gate", "use_gate": true, "status": "running",
  "params_brief": "...", "tag": "...",
  "details": "v2_with_gate/results/run_N.json"
}
```
完成后回填 `key_metrics` 并升级 `status: "completed"`。

### 7. 结束条件
- **Run 22 完成** → 写 `v2_with_gate/SUMMARY.md` 并停止循环
- **gate 失败**：若 Run 16-19 全部 `0-10m_RayIoU < 0.41` → 写 `proposals.md`（结构性修改建议），跳过 Run 20-22 直接结束
- **触顶**：若 Run 20-22 连续 2 轮 `0-10m_RayIoU` delta < 0.005 → 写 `proposals.md` 并结束

---

## 硬约束（v2_with_gate）

1. **lambda_ray 锁死 0.2**：v2 阶段不动，gate 已是新的 tradeoff 机制
2. **ray loss 必须覆盖近场**：不 mask、不降 near_weight 到 1.0 以下、不 `lambda_empty ≥ 0.3`（小权重 0.05-0.2 可探索）
3. **每轮只改 1-2 个参数**
4. **Run 16-19 必须按顺序跑完**才能进入自主探索
5. **每轮必须解析 `metrics.json` 的 `gate` 字段**写入 results；分析 gate_mean 是否在合理区间（0.05-0.5 是有效范围；接近 0 表示 gate 全关，接近 1 表示等同无 gate）
6. **反直觉对照触发式规则**（满足任一，下一轮必须做对照而非主线探索）：
   - 连续 3 轮沿同一参数轴单调调整 → 跑反向值或中间值证伪
   - 主线叙事连续 2 轮 0-10m_RayIoU delta < 0.003 → 停止沿该轴调整

---

## metrics.json 结构（远端训练产出，供 agent 解析）

```json
{
  "epoch": 8,
  "mIoU": 37.0,
  "mIoU_d": 32.0,
  "loss": 0.85,
  "gate": {
    "eval_mean": 0.12, "eval_std": 0.08,
    "train_mean": 0.15, "train_std": 0.10
  },
  "ray_config": {"lambda_ray": 0.2, "lambda_hit": 0.3, "...": "..."},
  "rayiou": {"RayIoU": 0.38, "RayIoU@1": 0.32, "RayIoU@2": 0.39, "RayIoU@4": 0.43},
  "binned_ray": {
    "all":   {"RayIoU": 0.38, "mean_signed_err": 2.5, "miss_rate": 0.10, "false_hit_rate": 0.13, "...": "..."},
    "0-10m": {"RayIoU": 0.41, "mean_signed_err": 3.2, "...": "..."},
    "10-20m": {"...": "..."},
    "20-40m": {"...": "..."}
  }
}
```
`gate` 字段仅在 `model.use_gate=true` 时出现。
