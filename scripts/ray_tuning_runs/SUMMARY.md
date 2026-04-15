# Ray Loss 调参总结

**目录结构**
- `v1_pre_ema/`：无 EMA 时代的历史 runs（Run 1-7）和当时的 SUMMARY，仅作方向性参考
- `v2_with_ema/`：EMA(decay=0.999) 启用后的新 runs，绝对数字以此为准
- `state.json`：所有 run 的结构化记录，每条带 `ema: true/false` 标记

## 版本说明

| 版本 | 时期 | 状态 | 绝对数值可用？ | 方向性结论可用？ |
|------|------|------|----------------|------------------|
| v1 (pre-EMA) | Run 1-7 | 归档 | ❌ 弃用（单点 val 噪声 ±1 pt） | ✅ 保留（见下） |
| v2 (with-EMA) | Run 8+ | 进行中 | ✅ 以此为准 | 待累积 |

## v1 保留的方向性结论（高置信度）

以下发现基于单调趋势或反事实证据，EMA 不会推翻：

### 1. hit loss 是 push-far 主因
- Run 1 `lambda_depth 0.2→0.4`：signed_err **反而加剧** (+0.19)
- Run 4 `depth_asym_far 2→4`：signed_err **反而加剧** (+0.14)
- Run 2→5→6 `lambda_hit 0.5→0.3→0.2`：signed_err 3.31→3.16→2.93，**单调改善**
- **结论**：push-far 必须从 hit loss 侧下手，调 depth 参数无效

### 2. `depth_asym_far` 在现有设置下无效
Run 4 将该参数从 2.0 提到 4.0，signed_err/false_hit/RayIoU 全部略变差。**反直觉但有反事实证据，不要再试**。

### 3. `lambda_empty` 的 tradeoff
- 小权重（0.1，Run 7）：false_hit 显著降（0.132→0.098），但 miss +1.8pt、signed_err +0.37m、RayIoU -0.13
- 大权重（≥0.5，v1 reference_runs）：严重伤 recall，RayIoU 下降明显
- **结论**：empty loss 是 false_hit 专用工具，默认关（=0），只在诊断 false_hit 过高时小剂量（≤0.1）使用

### 4. `near_weight` 提高无法解决 0-10m 问题
reference_runs 中 `near_weight=5` 比 `near_weight=2` 效果差。0-10m 问题是 loss 设计问题，不是权重分配问题。

### 5. Ray loss 整体有价值但有代价（EMA 前的结论）
- vs no-ray：overall RayIoU +1.26~1.38 pt（主要来自 0-10m）
- 代价：signed_err +0.6~1.0m（推远）、false_hit +2~9pt、mIoU -0.3~1.3
- **v2 需重新验证代价幅度**——EMA 可能改变 ray loss 相对收益

## v1 废弃的内容

- 所有**绝对数值**（`overall RayIoU = 0.3789` 等）——不可作为 v2 对比基线
- 所有 **RayIoU delta < 0.003** 的结论（噪声阈值内）
- **Run 5 vs Run 6 谁更优**（差 0.0004，纯噪声；v2 需重测）
- **`lambda_empty=0.1` 的真实代价幅度**（需 EMA 重测）

## v2 开局计划（Run 8-10）

| Run | 配置 | 目的 |
|-----|------|------|
| 8 | `lambda_ray=0`（no-ray） + EMA | 新 baseline，替代 v1 Run 3 |
| 9 | `lambda_hit=0.3, depth=0.2, empty=0` + EMA | 验证 v1 Run 5 最佳结论 |
| 10 | `lambda_hit=0.2, depth=0.2, empty=0` + EMA | 决出 hit loss 甜蜜点 |

跑完三轮后：
- 确定 EMA 下的 3000-sample baseline
- 验证 v1 方向性结论是否迁移
- 再决定是否继续调参 / 进入 Stage 2（10000 sample）/ 改动 loss 设计

## 工作流文件

- `LOOP_PROMPT.md`：调参 agent 每轮流程
- `state.json`：结构化 run 记录 + 搜索空间 + 基线参考
- `proposals.md`（尚未创建）：新 loss 项提案
