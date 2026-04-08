# Online NCDE 200x200x16 全分辨率变体

## 概述

本变体在 200x200x16 全分辨率空间中进行 ODE 演化，不做空间下采样。
与原始 `online_ncde`（100x100x16 隐藏状态）相比，精度更高但显存/计算开销约 4 倍。

## 与原版的区别

| 组件 | 原版 (online_ncde) | 本变体 (online_ncde_200x200x16) |
|------|--------------------|---------------------------------|
| Encoder | Conv3d stride=(1,2,2)，200→100 | Conv3d stride=(1,1,1)，保持 200 |
| 隐藏状态 | 100x100x16 | 200x200x16 |
| FuncG | 在 100x100x16 演化 | 在 200x200x16 演化（同结构） |
| Warp voxel_size | (0.8, 0.8, 0.4) | (0.4, 0.4, 0.4) 原始值 |
| Decoder | grouped conv + interpolate 2x + depthwise refine + 1x1 | 3x3x3 conv + 1x1x1 conv（无上采样） |

## 复用关系

以下组件直接从 `online_ncde` 包导入，无复制：
- `FuncG`、`HeunSolver`、`CtrlProjector`
- 所有 data 模块（ego_warp、scene_delta、time_series、dataset）
- losses、metrics、trainer、config、utils

## 使用方式

### 训练

```bash
python scripts/train_online_ncde_200x200x16.py \
    --config configs/online_ncde_200x200x16/fast_opusv1t__slow_opusv2l/train.yaml
```

### 配置

配置文件通过 `base_config` 继承原版配置，仅覆盖输出目录。
如需调整模型参数（如降低 `func_g_inner_dim` 节省显存），在配置中添加 `model` 段即可。

## 显存估算

| 项目 | 100x100x16 | 200x200x16 |
|------|-----------|-------------|
| 单张特征图 (32ch, fp32) | ~20 MB | ~82 MB |
| ODE 每步激活 | ~120-160 MB | ~490-650 MB |

建议使用 96GB 显存 GPU（如 Pro6000）运行。如需在较小显存 GPU 上运行，可：
- 降低 `func_g_inner_dim`（如 16）
- 减少 `func_g_body_dilations`（如 `[1, 2]`）
- 启用 AMP fp16
