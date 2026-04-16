# Ray Loss 调参总结

**当前阶段：v2_with_gate（门控启用）**

**训练设置**：EMA(decay=0.999) + 3000-sample + 8 epoch + 20 val scenes + DDP 双卡

**目录结构**
- `baseline.json`：快系统 baseline（无对齐器）的完整 20 scene 指标，所有对照锚点
- `state.json`：精简索引（runs_index、search_space、v2_with_gate_plan），不放详细数据
- `v1_wo_gate/`：Run 8-15（门控关闭阶段，已完成）
  - `logs/run_N.log`、`results/run_N.json`
- `v2_with_gate/`：Run 16+（门控启用阶段，进行中）
  - `logs/run_N.log`、`results/run_N.json`
- `LOOP_PROMPT.md`：调参 agent 每轮工作流（权威）
- `proposals.md`（按需创建）：调到天花板时写结构性修改提案

---

## v2_with_gate 阶段（当前）

### 必跑 4 轮对照（按顺序）
| Run | 配置来源 | 关键参数 | 看什么 |
|-----|----------|----------|--------|
| 16  | v1 Run 8（no-ray）| `lambda_ray=0` | gate 在 ray loss 关闭时是否伤害基线（理想 gate_mean 自学到很小） |
| 17  | v1 Run 13         | `ray=0.2,hit=0.3,depth=0.3,near=5` | v1 0-10m RayIoU 最高的配置 + gate，看 false_hit 0.194 是否被压下来 |
| 18  | v1 Run 10         | `ray=0.2,hit=0.3,depth=0.3,asym_far=3,near=2` | v1 false_hit 0.245 暴涨配置，看 gate 能否救 |
| 19  | v1 Run 14         | `ray=0.2,hit=0.3,empty=0.2,depth=0.3,near=2` | gate + empty loss 联合是否能压 false_hit 同时保 recall |

### 自主探索 3 轮（Run 20-22）
**禁止**调整 `lambda_ray`（固定 0.2）。可探索轴：`lambda_hit / lambda_depth / lambda_empty / lambda_pre_free`（每轮只改 1-2 个）。

### v2 判定标准
- **成功**：Run 17 的 0-10m RayIoU ≥ 0.4135（≥ v1 Run 13）且 0-10m false_hit ≤ 0.15
- **失败**：Run 16-19 全部 0-10m RayIoU < 0.41 → gate 设计失败 → 写 `proposals.md` 后停止 free explore

### gate 诊断
- `metrics.json` 中 `gate.eval_mean / eval_std / train_mean / train_std`（agent 直接读，不用 grep 日志）
- 关注：
  - `eval_mean` 接近 0 → gate 几乎全关，模型行为接近纯快系统
  - `eval_mean` 接近 1 → gate 全开，等价于无 gate
  - `eval_std` 大 → gate 有空间分化（在不同体素上做不同决策），是期望状态

---

## v1_wo_gate 阶段保留结论（方向性，下面这些是事实，不要再走老路）

### ❌ 误区 1：near_weight 提高无法改善 0-10m RayIoU
- **反证**：Run 13 干净对照 `near_weight 2→5`（其他参数不变），0-10m RayIoU **+0.42pt**
- **正确理解**：near_weight 高能改善 0-10m RayIoU，但 false_hit 同时上升——真实 tradeoff

### ❌ 误区 2：depth loss 是核心有效项、应加强
- **反证 1**：v1 Run 1 `lambda_depth 0.2→0.4`，signed_err +0.19、false_hit +3pt
- **反证 2**：Run 10 `lambda_depth 0.2→0.3, depth_asym_far 2→3`，false_hit **暴涨到 0.245**
- **正确理解**：depth loss 是辅助项，加强它会"拉回"预测制造更多 false_hit

### ❌ 误区 3：ray loss 降低 miss_rate
- **反证**：Run 8 miss = 0.072 → Run 9 miss = 0.106（加 ray loss 后 miss 反而涨 3.4pt）
- **正确理解**：ray loss 让模型"更激进地预测占用"，miss 和 false_hit **同时上升**

### ❌ 误区 4：ray loss 净提升 0-10m RayIoU
- aligner 本身（Run 8, no-ray）就让 0-10m RayIoU 从快系统的 0.4284 跌到 0.4001（-2.83pt）
- ray loss 只是**部分挽回 aligner 的回撤**（Run 13 最佳 0.4135 = 挽回 +1.34pt，但净值仍 -1.49pt 于快系统）
- **正确理解**：ray loss 在修补 aligner 对近场几何的破坏，不是"提升"近场 RayIoU
- **gate 的目标**：彻底解决这个结构性回撤

### ❌ 误区 5：降 lambda_ray 是 v2 应该探索的方向
- v1 Run 11 (`lambda_ray=0.1`) 已是 v1 最佳 tradeoff，但 0-10m RayIoU 仍 -2pt 于快系统
- **v2_with_gate 锁定 `lambda_ray=0.2`**——gate 已是新的"稀释机制"，无需再用低 lambda_ray 稀释

---

## v1 已确认的发现（仍适用 v2）

1. **aligner 的结构性近场损失**：无 ray loss 的 aligner 相比快系统 0-10m 差 -2.83pt——gate 就是为这个而设计
2. **aligner 的 miss ↓ / false_hit ↑ 模式**：mIoU 涨/RayIoU 压的根本原因
3. **near_weight 方向有效但触 Pareto 前沿**：Run 13 已是 v1 调参极限
4. **纯 hit+depth 配置优于加 empty/pre_free 大权重**：所有 v1 最佳保持 `lambda_empty=0, lambda_pre_free=0`
5. **ray loss 整体越强越 push-far**：调权重无法消除，需结构改动（gate 来了）
6. **seed 噪声**：RayIoU 测量噪声约 ±0.003-0.004，false_hit 是更稳定指标（差异 < 0.001）

---

## v1 Run 概览

| Run | 关键参数 | mIoU | 0-10m RayIoU | 0-10m false_hit | 备注 |
|-----|---------|------|--------------|-----------------|------|
| 8   | `lambda_ray=0`                          | 37.72 | 0.4001 | 0.098 | no-ray baseline |
| 9   | `ray=0.2,hit=0.3,depth=0.2,near=2`      | 36.91 | 0.4065 | 0.146 | 基础 ray-loss |
| 10  | `ray=0.2,hit=0.3,depth=0.3,asym_far=3`  | 36.69 | 0.4093 | **0.245** | depth 加强 → false_hit 暴涨 |
| 11 ★| `ray=0.1,hit=0.3,depth=0.2,near=2`      | 37.24 | 0.408  | 0.117 | v1 最佳 tradeoff |
| 12  | `ray=0.05,hit=0.3,depth=0.2,near=2`     | 37.53 | 0.4064 | 0.110 | 与 Run 11 噪声内持平 |
| 13  | `ray=0.2,hit=0.3,depth=0.3,near=5`      | 36.84 | **0.4135** | 0.194 | v1 0-10m 最高 |
| 14  | `ray=0.2,hit=0.3,empty=0.2,depth=0.3`   | 36.48 | 0.4119 | **0.096** | empty 压 false_hit，但 recall 崩 |
| 15  | Run 11 配置 seed=42                     | 37.33 | 0.4043 | 0.117 | 方差估计 |

详见 `v1_wo_gate/results/run_N.json`。

---

## 备注：gate 设计

- 模块：`src/online_ncde_200x200x16/gate.py`（GatedResidualHead）
- 输入：`fast logits` 置信度 + `f_t - f_prev_adv` 时序变化 + residual 幅度 + hidden 投影
- 输出：逐体素 alpha ∈ [0,1]，`hat_O = O_fast + alpha * delta_O`
- 初始化：`gate_init_bias=-3.0` → 起步 alpha ≈ 0.047，`decoder_init_scale=1.0`（避免双重 damping）
- 期望：模型在 fast 已强的近场区域学到 alpha→0，保住 fast 的近场强项
