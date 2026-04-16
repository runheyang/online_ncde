# Ray Loss 调参总结

**目录结构**
- `v1_pre_ema/`：无 EMA 时代的历史 runs、SUMMARY 和数据归档，**仅作追溯，数字和多数结论已废弃**
- `v2_with_ema/`：EMA(decay=0.999) 启用后的新 runs，当前主数据
- `state.json`：v2 only，所有 v2 run 的结构化记录 + 不变参考（快系统 baseline）

## 版本策略

- **v1 数据全面弃用**：last-epoch + no-EMA 单点 val 噪声 ±0.003 量级，且多条方向性结论已被 v2 证伪（见下）
- **v2 是当前唯一可信数据源**
- 引用数字一律从 v2 state.json 或 baseline_20scene 取

---

## v1 已被 v2 证伪的错误结论（历史记录，不要再当事实用）

### ❌ 错误 1：near_weight 提升无法改善 0-10m
- v1 推论来自 `with_empty_and_pre_free (near_w=5)` 对照 → **混淆变量**（同时叠加 empty + pre_free，差是 pre_free 造成的）
- **v2 Run 13 反证**：干净对照 `near_weight 2→5`（其他参数不变），0-10m RayIoU **+0.55pt**（v2 单参数调整最大提升）
- **正确理解**：near_weight 提高能显著改善 0-10m RayIoU，但 false_hit 同时上升

### ❌ 错误 2：depth loss 是核心有效项
- v1 基于表面观察推测 "lambda_depth 应提到 0.3-0.5 作主力"
- **v1 Run 1 反证**：`lambda_depth 0.2→0.4`，signed_err 反而 +0.19，false_hit +3pt
- **v2 Run 10 反证**：`lambda_depth 0.2→0.3, depth_asym_far 2→3`，false_hit **暴涨到 0.245**（最差）
- **正确理解**：depth loss 是辅助项，加强它会"拉回"预测制造更多 false_hit

### ❌ 错误 3：ray loss 降低 miss_rate
- v1 自己的数据就不支持：Run 3 (no-ray) 0-10m miss = 0.077 → Run 5 (ray) miss = 0.109（反而涨）
- **v2 同样反证**：Run 8 miss = 0.072 → Run 9 miss = 0.106（涨）
- **正确理解**：ray loss 让模型"更激进地预测占用"，miss 和 false_hit **同时上升**

### ❌ 错误 4：ray loss 提升 0-10m RayIoU
- v1 叙事："Run 5 相比 Run 3 RayIoU +1.38pt"
- **被遗漏的参照系**：aligner 本身（Run 3/Run 8）就让 0-10m RayIoU 跌了 2-3pt；ray loss 只是**部分挽回 aligner 的回撤**，净值仍低于快系统
- **正确理解**：ray loss 在修补 aligner 对近场几何的破坏，不是"提升"近场 RayIoU

### ❌ 错误 5：降低 lambda_ray 是正确方向
- v1/早期 v2 推断 Run 11 (lambda_ray=0.1) 是"最佳 tradeoff"
- **实际**：降 lambda_ray 等于变相削弱 ray loss，Run 12（0.05）已经在往 no-ray 退化
- **正确方向**：保持 `lambda_ray=0.2` 或更高，增强 ray loss 的实际作用（如 `near_weight=5`）

---

## v1 仍可用的结论（有限置信度）

### [部分成立] ray loss 整体越强越 push-far
- `lambda_hit` 降低（0.5→0.3→0.2）或 `lambda_ray` 降低（0.2→0.1）都能减小 signed_err
- **含义**：push-far 是 ray loss 的系统性副作用，不是单一参数问题
- **注意**：不要基于此继续"降权重"；应通过结构（门控）或针对性 loss 改动解决

### [未证实] lambda_empty 的 false_hit ↔ recall tradeoff
- 大权重（≥0.5）严重伤 recall（v1 reference_runs 和 loss 机制直觉支持）
- 小权重（~0.1）是否值得用：**v2 未测**，不要当事实
- 默认保持 `lambda_empty=0`

### [已 v2 确认] empty + pre_free 大权重严重有害
- 保持 `lambda_empty=0, lambda_pre_free=0` 作为默认

### [已 v2 确认] 纯 hit+depth 配置优于加 empty/pre_free
- v2 所有最佳配置保持 `lambda_empty=0, lambda_pre_free=0`

---

## v2 确认的新发现

### 1. aligner 结构性近场损失
- Run 8（no-ray, aligner）vs 快系统：0-10m RayIoU 差约 -2.8pt，10-20m 和 20-40m 反而略优
- **含义**：aligner 天然破坏快系统的近场强项；ray loss 只能部分挽回，完整修复需结构改动（门控）

### 2. near_weight 方向被证实有效
- Run 13（near_weight=2→5）：0-10m RayIoU **+0.55pt**（v2 单参数调整最大提升）
- 但 false_hit 同时 +7.7pt（代价大）
- **含义**：参数调整已到 Pareto 前沿，靠近 fast baseline 需结构改动

### 3. aligner 的 miss ↓ / false_hit ↑ 模式
- aligner 大幅降低 miss_rate（更积极预测占用，mIoU 涨 3-4pt 的原因）
- aligner 大幅抬高 false_hit_rate（代价：RayIoU 受压）
- ray loss 进一步放大 false_hit

### 4. ray loss 参数调整已触天花板
- Run 9 / 10 / 11 之间 RayIoU 差异 < 0.002（噪声内）
- Run 13 (near_w=5) 是目前 0-10m RayIoU 最高（0.4135），但仍 -1.49pt 低于快系统

---

## v2 Run 概览

| Run | 关键参数 | 定位 | 状态 |
|-----|---------|------|------|
| 8 | `lambda_ray=0` | v2 baseline | 参考 |
| 9 | `lambda_hit=0.3, depth=0.2, near_w=2` | 复现 v1 Run 5 | 参考 |
| 10 | 加强 depth（`depth=0.3, asym_far=3`） | false_hit 爆炸 | **废弃** |
| 11 | `lambda_ray=0.1`（降整体权重） | 方向错误（削弱 ray loss） | **废弃** |
| 12 | `lambda_ray=0.05` | 同 Run 11 方向错误 | **废弃** |
| 13 ⭐ | `near_weight=5, ray=0.2, depth=0.3` | v2 0-10m RayIoU 最佳 | **当前最佳**（但 false_hit 高） |

具体数值查 `state.json` 的 runs 字段。

---

## 下一步

**参数调整已到 Pareto 前沿，开始门控 aligner 结构改动**

- Ray loss 配置锁定在 Run 13 附近（`lambda_ray=0.2, lambda_hit=0.3, lambda_depth=0.3, near_weight=5`）
- 门控设计初版：**fast encoder 输出 → 1×1 conv → sigmoid → per-voxel gate，bias=-2.0 偏向保持 fast**
- 预期：门控让模型在 fast 已强的近场区域自动"不动"，保住强 ray loss 的 RayIoU 收益同时压下 false_hit

---

## 工作流文件

- `LOOP_PROMPT.md`：调参 agent 每轮流程
- `state.json`：v2 结构化 run 记录 + 搜索空间 + 快系统 baseline 锚点
- `v1_pre_ema/`：v1 归档（数字和多数结论作废，文档保留作追溯）
- `proposals.md`（尚未创建）：新 loss 项 / 结构改动提案
