# Ray Loss 调参 Loop 工作流

## 使用方法

在项目根目录下运行：

```
/loop 以下是调参 loop 的一次迭代。

## 任务
你是 Online NCDE ray loss 调参 agent。目标：优化 ray loss 参数使 0-10m RayIoU 提升，同时保持整体 RayIoU 和 mIoU 不退化。

## 背景
- 对齐器在语义 mIoU 上提升显著（+4.89），但 0-10m RayIoU 略有下降（-0.006）
- 根因：对齐器系统性地把体素推远（0-10m mean signed error 从 1.99 增到 2.56）
- 当前 L_empty 和 L_pre_free 加上后效果可能反而变差

## 工作流
1. 读取 scripts/ray_tuning_runs/state.json 查看历史 runs 和搜索空间
2. 检查是否有正在运行的训练（通过 SSH 检查 GPU 占用）：
   ```bash
   ssh -p 14661 root@connect.westd.seetacloud.com 'nvidia-smi'
   ```
3. 如果有未收集结果的 run（远端 output_dir 下有 metrics.json），通过 SSH 读取结果并更新 state.json：
   ```bash
   ssh -p 14661 root@connect.westd.seetacloud.com 'cat ~/autodl-tmp/online_ncde/<output_dir>/metrics.json'
   ```
4. 分析所有历史结果，重点关注：
   - 0-10m RayIoU（最重要，baseline 0.4555）
   - overall RayIoU（baseline 0.3986）
   - mIoU 是否保持（baseline 38.49 for aligner）
   - mean_signed_err 正值越大=越系统性推远
   - miss_rate 和 false_hit_rate 的变化
5. 基于分析推理，决定下一组参数（每次只改 1-2 个参数，控制变量）
6. 启动训练（通过 SSH 在远端后台运行，双卡 DDP）：
   ```bash
   ssh -p 14661 root@connect.westd.seetacloud.com \
     'bash -lc "source /root/miniconda3/etc/profile.d/conda.sh && conda activate neural_ode && \
     cd ~/autodl-tmp/online_ncde && \
     nohup torchrun --nproc_per_node=2 scripts/train_online_ncde_200x200x16.py \
       --config configs/online_ncde_200x200x16/fast_alocc2dmini__slow_alocc3d/train.yaml \
       --train-limit 3000 --val-scene-count 20 --epochs 8 --rayiou --save-metrics-json \
       --ray-override '"'"'{"lambda_ray": 0.2, ...}'"'"' \
       > scripts/ray_tuning_runs/run_NNN.log 2>&1 &"'
   ```
7. 将 run 信息（参数、output_dir、启动时间）写入 state.json
8. 等待下次迭代（约 3.5 小时后）

## 决策指南（不是硬规则，需结合实际数据判断）
- 第一步建议：关闭 L_empty 和 L_pre_free（设为 0），看纯 hit+depth loss 的效果
- 如果 signed_err 高（推远）：增大 depth_asym_far 或 near_weight
- 如果 false_hit 多：可能需要 L_empty（但先确认）
- 如果 miss_rate 高：增大 lambda_hit
- 注意子集上的绝对指标不等于全量，关注 delta（相对于子集 baseline）
- 参数之间有交互：lambda_ray 放大所有 ray loss 的效果

## 反直觉对照与方差估计（防止过拟合主线叙事）
低预算 expert-guided 调参的主要风险是"把噪声当信号"或"沿一个方向调到撞墙"。当满足以下任一条件时，下一轮必须改做对照实验而不是继续沿主线探索：
- **连续 3 轮沿同一参数轴单调调整**（如 lambda_hit 0.5→0.3→0.2）→ 跑一次反向值或回到中间值，证伪当前假设
- **累计 run 数 ≥ 6 且从未做过方差估计** → 用当前最佳配置换一个 seed 重跑（仅改 --seed，其他参数不动），估计噪声下界
- **主线叙事连续 2 轮边际改善 < 0.003**（接近噪声阈值）→ 停止沿该轴调整，检查是否已触底

对照轮的 `change_from_ref` 要明确标注"反直觉对照"或"方差估计"，`hypothesis` 写清想证伪什么，`analysis` 里给出证伪/证实的结论。这些轮不追求改善指标，是 meta-实验。

## 新 loss 项提案
如果你在分析数据后认为现有 ray loss 项存在根本性缺陷，或需要新的 loss 项才能解决问题：
- 不要自动修改 ray_loss.py 代码
- 将提案写入 scripts/ray_tuning_runs/proposals.md，包括：
  - 问题诊断（什么指标异常、为什么现有 loss 无法解决）
  - 提议的新 loss 项公式和直觉
  - 预期效果和可能的副作用
- 继续用现有 loss 项做参数调优，不要等提案被采纳

## 重要
- 每次只调 1-2 个参数
- 分析时要看完整指标表再做判断，不要只看单一指标
- 如果连续 3 次结果无改善，考虑改变策略（如调不同参数维度）
- 跑完 8-10 组后输出总结报告，推荐最佳参数组合
```

## 输出示例

metrics.json 内容示例：
```json
{
  "epoch": 5,
  "mIoU": 37.5,
  "mIoU_d": 32.1,
  "loss": 0.85,
  "ray_config": {
    "lambda_ray": 0.2,
    "lambda_hit": 0.5,
    "lambda_empty": 0.0,
    "lambda_pre_free": 0.0,
    "lambda_depth": 0.1,
    "near_weight": 4.0,
    "depth_asym_far": 2.0,
    "depth_asym_near": 1.0
  },
  "rayiou": {
    "RayIoU": 0.41,
    "RayIoU@1": 0.35,
    "RayIoU@2": 0.42,
    "RayIoU@4": 0.46
  },
  "binned_ray": {
    "all": {"RayIoU": 0.41, "mean_signed_err": 1.8, ...},
    "0-10m": {"RayIoU": 0.46, "mean_signed_err": 2.1, ...},
    "10-20m": {"RayIoU": 0.38, ...},
    "20-40m": {"RayIoU": 0.25, ...}
  }
}
```
