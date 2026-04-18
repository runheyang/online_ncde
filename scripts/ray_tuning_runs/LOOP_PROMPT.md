# Ray Loss 调参 Loop 工作流（agent 读取的权威文档）

> 每轮 agent 必须按本文件步骤执行。方向性结论/硬约束见 `SUMMARY.md`，历史数据/搜索空间见 `state.json`，快系统 baseline 见 `baseline.json`。三处如有冲突以最新修改为准。

## 阶段上下文
- **当前阶段：待重新设计**（gate 模块已于 2026-04-18 从代码删除；新阶段计划由用户后续补充）
- 历史阶段：v1_wo_gate（Run 8-15）、v2_with_gate（Run 16-25，gate 路线已终止）、v4_fastkl_pastfree（Run 26-28，past_free 证伪）均已归档于各自子目录
- 新 run 输出目录待新阶段启动时确定（log 与 results 的存放路径由新阶段 PLAN.md 指定）

## 目标
在 3000-sample / 8-epoch 子集上最大化 **0-10m RayIoU**，mIoU 允许 -0.5 内波动。
- 快系统 baseline（baseline.json）：0-10m RayIoU 0.4284, false_hit 0.086, signed_err 2.16m
- 历史最高 0-10m RayIoU（v1 Run 13）：0.4135
- 当前代码可用机制：`lambda_fast_kl`（conf-weighted KL-to-fast 正则）；其余 ray_loss 超参同 v1 搜索空间

---

## 每轮工作流

### 1. 读权威文件
- `SUMMARY.md`（历史阶段保留结论 + 当前硬约束 + 判定标准）
- `state.json`（runs_index、search_space、当前阶段 plan）
- `baseline.json`（快系统 baseline 完整指标，对比锚点）
- 当前阶段 `PLAN.md`（如已由用户创建；未创建则等待用户指令）

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
- 把完整 JSON 写到当前阶段 `<stage>/results/run_N.json`，加 `analysis` 字段
- 在 `state.json.runs_index` 追加一条轻量索引（id/stage/params_brief/key_metrics/tag/details）
- `key_metrics` 必须含：`mIoU`、`RayIoU`、`0-10m_RayIoU`、`0-10m_false_hit`、`0-10m_signed_err`；若启用 `lambda_fast_kl>0` 追加 `fast_kl_train`
- `analysis` 必填，必须包含：vs 指定 ref run 的 delta 与解读

### 4. 决定下一轮参数

按当前阶段 `PLAN.md` 顺序执行；若无 PLAN.md 则等待用户指令后再启动训练。

通用约束（可被当前阶段 PLAN.md 覆盖）：
- 可探索：`lambda_hit / lambda_depth / lambda_empty / lambda_pre_free / near_weight / lambda_fast_kl`
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
    [--lambda-fast-kl <float>] \
    > scripts/ray_tuning_runs/<stage>/logs/run_N.log 2>&1 &"'
```
EMA 默认由 config 开启，无需 CLI 参数。`--lambda-fast-kl` 可选，不传等于 0（关闭）。

### 6. 写 state.json + 占位 results 文件
新 run 在 `state.json.runs_index` 追加：
```json
{
  "id": N, "stage": "<stage>", "status": "running",
  "params_brief": "...", "tag": "...",
  "details": "<stage>/results/run_N.json"
}
```
完成后回填 `key_metrics` 并升级 `status: "completed"`。

### 7. 结束条件
由当前阶段 `PLAN.md` 指定；通用兜底：
- **预算用尽** → 写 `<stage>/SUMMARY.md` 并停止循环
- **触顶**：连续 2 轮 `0-10m_RayIoU` delta < 0.005 → 写 `proposals.md` 并结束

---

## 硬约束（通用，新阶段 PLAN.md 可覆盖/补充）

1. **ray loss 必须覆盖近场**：不 mask、不降 near_weight 到 1.0 以下、不 `lambda_empty ≥ 0.3`（小权重 0.05-0.2 可探索）
2. **每轮只改 1-2 个参数**，必填 `change_from_ref` 和 `hypothesis`
3. **反直觉对照触发式规则**（满足任一，下一轮必须做对照而非主线探索）：
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
  "fast_kl": {"train": 0.12},
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
`fast_kl` 字段仅在 `train.lambda_fast_kl > 0` 时出现。
