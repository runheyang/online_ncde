# Ray Loss 调参总结

**配置**: 3000 train samples, 20 val scenes, 8 epochs, DDP 双卡
**完成 runs**: 7 组（含 3000-sample no-ray baseline）

## 最佳配置（推荐）

**Run 5** — `lambda_hit=0.3, lambda_depth=0.2, empty=0, pre_free=0`

| 指标 | 值 |
|------|-----|
| overall RayIoU | 0.3789 |
| 0-10m RayIoU | 0.4138 |
| 0-10m signed_err | +3.16 m |
| 0-10m miss_rate | 0.109 |
| 0-10m false_hit | 0.132 |
| mIoU | 37.02 |

**vs 3000-sample no-ray baseline (Run 3)**: overall RayIoU +1.38, 0-10m RayIoU +1.70；代价 signed_err +0.62、false_hit +3.7pt、mIoU -0.46。

## 全部结果对比（3000-sample 对照）

| Run | lambda_hit | lambda_depth | empty | depth_asym_far | overall | 0-10m | signed | miss | false_hit | mIoU |
|-----|-----------|--------------|-------|----------------|---------|-------|--------|------|-----------|------|
| 3 (no-ray) | - | - | - | - | 0.3651 | 0.3968 | 2.54 | 0.077 | 0.095 | 37.48 |
| 2 | 0.5 | 0.2 | 0 | 2.0 | 0.3777 | 0.4086 | 3.31 | 0.107 | 0.151 | 36.79 |
| 1 | 0.5 | 0.4 | 0 | 2.0 | 0.3796 | 0.4099 | 3.50 | 0.117 | 0.181 | 36.23 |
| 4 | 0.5 | 0.2 | 0 | **4.0** | 0.3764 | 0.4062 | 3.45 | 0.113 | 0.162 | 36.17 |
| **5** ⭐ | **0.3** | 0.2 | 0 | 2.0 | **0.3789** | **0.4138** | 3.16 | 0.109 | 0.132 | 37.02 |
| 6 | **0.2** | 0.2 | 0 | 2.0 | 0.3785 | 0.4132 | 2.93 | 0.095 | 0.150 | 37.11 |
| 7 | 0.3 | 0.2 | **0.1** | 2.0 | 0.3776 | 0.4071 | 3.53 | 0.127 | 0.098 | 36.91 |

## 核心发现

### 1. Ray loss 有价值但有代价
- vs no-ray: RayIoU +1.26~1.38 pt（主要来源 0-10m 的 +1.18~1.70）
- 代价: signed_err +0.6~1.0m（推远），false_hit +2~9pt，mIoU -0.3~1.3

### 2. hit loss 是 push-far 主因（非 depth loss 弱）
- Run 1 lambda_depth 0.2→0.4: signed_err **反而加剧** (+0.19)
- Run 4 depth_asym_far 2→4: signed_err **反而加剧** (+0.14)
- Run 5 lambda_hit 0.5→0.3: signed_err **显著改善** (-0.15)
- 结论：hit loss (window-based NLL) 的梯度方向在 0-10m 带系统性偏差，改 depth 参数无效，必须降 hit

### 3. hit loss 存在甜蜜点
| lambda_hit | signed_err | false_hit | RayIoU |
|-----------|-----------|-----------|--------|
| 0.5 | 3.31 | 0.151 | 0.3777 |
| 0.3 ⭐ | 3.16 | **0.132** | 0.3789 |
| 0.2 | **2.93** | 0.150 | 0.3785 |

lambda_hit=0.3 对 false_hit 最优；继续压到 0.2 虽改善 signed_err/miss 但 false_hit 反弹，RayIoU 不增。

### 4. empty loss：false_hit 杀器，但伤害 recall
Run 7 (+lambda_empty=0.1): false_hit 0.132→0.098（接近 no-ray），但 signed_err 3.16→3.53，miss +1.8pt, RayIoU -0.13。

## 调参策略建议（未来）

1. **起点**：Run 5 配置作为新基线
2. **如果优化目标是 false_hit**：用 Run 7 配置（lambda_empty=0.1）
3. **未探索**：
   - lambda_ray 整体权重（0.1/0.3/0.4/0.5）的影响
   - depth_asym_near 降到 0.5（鼓励 pred<GT）
   - combined：Run 5 + depth_asym_far 3.0（反向未测）

## 工具/工作流改进（本次修复）

1. **进度条日志污染** — `make_pbar()` 检测非 tty 时返回 None，trainer 走 fallback 间隔打印。日志体积从 12000+ 行/run 降到 1200 行/run。
2. **baseline_20scene** 已加入 state.json，为 tuning runs 提供 15k-train 对照（mIoU=33.72, 0-10m RayIoU=0.4284）。

## 未发现 loss 根本缺陷

现有 ray loss 设计可用但不完美：
- hit loss 的 window-based NLL 在 0-10m 有结构性 push-far bias（通过降权缓解，非消除）
- depth loss 在 3000-sample 设置下梯度信号偏弱，调 lambda/asym 效果有限
- empty loss 是明确的 false_hit 克星，但需小心调权（>=0.5 时严重伤 recall）

proposals.md 暂无新增提案——当前 loss 组合已能通过调参取得可控改进，未达到必须重设计的程度。
