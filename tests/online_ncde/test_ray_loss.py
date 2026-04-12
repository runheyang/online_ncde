"""RayLoss 单元测试。

覆盖：
  1. generate_lidar_rays 的 shape / 方向归一
  2. grid_sample 坐标对齐：在已知 occupied voxel 中心查询 p_occ ≈ 1
  3. first-hit 分布 q 的解析值
  4. 无 hit 时 hit_loss 很大；完美对齐时 hit_loss 很小
  5. 梯度方向正确：对 free logit 的梯度 > 0（应当被压下）
  6. Adam 优化能把 hit_loss 压下去
  7. 非对称 depth loss：同样 |err|，pred 更远的 loss ≈ 2× pred 更近
"""

from __future__ import annotations

import math
import pickle
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from online_ncde.data.ray_sidecar import RaySidecar, _SCHEMA_V3  # noqa: E402
from online_ncde.ray_loss import RayLoss, generate_lidar_rays  # noqa: E402


PC_RANGE = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
VOXEL_SIZE = 0.4
FREE_IDX = 17
NUM_CLASSES = 18
GRID = (200, 200, 16)


def _voxel_center(ix: int, iy: int, iz: int) -> tuple[float, float, float]:
    return (
        PC_RANGE[0] + (ix + 0.5) * VOXEL_SIZE,
        PC_RANGE[1] + (iy + 0.5) * VOXEL_SIZE,
        PC_RANGE[2] + (iz + 0.5) * VOXEL_SIZE,
    )


def _free_everywhere_logits(device: str = "cpu") -> torch.Tensor:
    """所有体素都是高置信 free。"""
    logits = torch.zeros((1, NUM_CLASSES, *GRID), device=device)
    logits[:, FREE_IDX] = 10.0
    return logits


def _occupied_at(
    ix: int, iy: int, iz: int, free_val: float = 10.0, device: str = "cpu"
) -> torch.Tensor:
    """只有 (ix, iy, iz) 这个 voxel 不是 free，其他都是 free。"""
    logits = torch.zeros((1, NUM_CLASSES, *GRID), device=device)
    logits[:, FREE_IDX] = free_val
    logits[0, FREE_IDX, ix, iy, iz] = -free_val  # 这一格 p_free ≈ 0
    return logits


# ---------------------------------------------------------------------------
# 1. generate_lidar_rays 基本性质
# ---------------------------------------------------------------------------


def test_generate_lidar_rays_unit_norm():
    rays = generate_lidar_rays()
    assert rays.dtype == torch.float32
    assert rays.dim() == 2 and rays.shape[1] == 3
    norms = rays.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # 39 pitches × 360 azimuths = 14040
    assert rays.shape[0] == 14040


# ---------------------------------------------------------------------------
# 2. grid_sample 坐标对齐
# ---------------------------------------------------------------------------


def _make_ray_loss(**overrides) -> RayLoss:
    """测试里共用的 RayLoss 构造器：pc_range / free_index 已预置。

    默认 gt_dist_bias_m=0.0，因为测试用的 GT 都是人造的 voxel 中心距离，
    不带 DVR 出射偏置。需要默认行为的用例显式传 gt_dist_bias_m=None。
    """
    params = dict(
        pc_range=PC_RANGE, free_index=FREE_IDX, gt_dist_bias_m=0.0
    )
    params.update(overrides)
    return RayLoss(**params)


def _check_grid_sample_at_voxel_center(ix: int, iy: int, iz: int) -> None:
    rl = _make_ray_loss()
    logits = _occupied_at(ix, iy, iz)
    cx, cy, cz = _voxel_center(ix, iy, iz)

    xyz = torch.tensor([[[[cx, cy, cz]]]])  # (B=1, R=1, N=1, 3)
    grid = rl._world_to_grid(xyz)
    probs = F.softmax(logits, dim=1)
    p_free_vol = probs[:, FREE_IDX : FREE_IDX + 1]
    p_free = F.grid_sample(
        p_free_vol,
        grid.view(1, 1, 1, 1, 3),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).item()
    p_occ = 1.0 - p_free
    assert p_occ > 0.9, (
        f"体素 ({ix},{iy},{iz}) 中心查询 p_occ={p_occ:.4f}，"
        f"grid 坐标或 stack 顺序可能有问题"
    )


def test_world_to_grid_and_sample_at_voxel_center():
    """把一个体素置为 p_occ ≈ 1，用 grid_sample 查该体素中心应得到 ≈ 1。"""
    for ix, iy, iz in [(0, 0, 0), (199, 199, 15), (100, 100, 8), (120, 50, 3)]:
        _check_grid_sample_at_voxel_center(ix, iy, iz)


def test_grid_sample_axis_ordering_is_correct():
    """显式验证 X/Y/Z 没搞混：只在 X 方向移动的点对应 X voxel 变化。"""
    rl = _make_ray_loss()
    # 占据两个不同 X 的 voxel，其他相同
    logits = _occupied_at(50, 100, 8)  # voxel A
    probs = F.softmax(logits, dim=1)
    p_free_vol = probs[:, FREE_IDX : FREE_IDX + 1]

    # 查 voxel A 中心 → p_occ ≈ 1
    cA = _voxel_center(50, 100, 8)
    # 查 voxel (100, 100, 8) 中心（X 方向挪开 20 格） → p_occ ≈ 0
    cB = _voxel_center(100, 100, 8)
    # 查 voxel (50, 50, 8) 中心（Y 方向挪开） → p_occ ≈ 0
    cC = _voxel_center(50, 50, 8)
    # 查 voxel (50, 100, 0) 中心（Z 方向挪开） → p_occ ≈ 0
    cD = _voxel_center(50, 100, 0)

    xyz = torch.tensor([[cA, cB, cC, cD]]).view(1, 4, 1, 3)  # (B=1,R=4,N=1,3)
    grid = rl._world_to_grid(xyz).view(1, 4, 1, 1, 3)
    p_free = F.grid_sample(
        p_free_vol,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).view(4)
    p_occ = 1.0 - p_free
    assert p_occ[0] > 0.9, f"A 应命中，p_occ={p_occ[0]:.3f}"
    assert p_occ[1] < 0.1, f"B 在 X 远处，不应命中，p_occ={p_occ[1]:.3f}"
    assert p_occ[2] < 0.1, f"C 在 Y 远处，不应命中，p_occ={p_occ[2]:.3f}"
    assert p_occ[3] < 0.1, f"D 在 Z 远处，不应命中，p_occ={p_occ[3]:.3f}"


# ---------------------------------------------------------------------------
# 3. first-hit 分布 q 的解析检查
# ---------------------------------------------------------------------------


def test_first_hit_distribution_analytical():
    """用一个手写 p 序列验证 q_i = p_i * Π_{j<i}(1 - p_j) 的实现。"""
    # 直接调 forward 比较麻烦，因为 grid_sample 会插值；这里直接跑内部逻辑的等价实现
    p = torch.tensor([[[0.1, 0.2, 0.9, 0.5, 0.0]]])  # (B=1, R=1, N=5)
    log_one_minus_p = torch.log((1.0 - p).clamp(min=1e-6))
    cum = torch.cumsum(log_one_minus_p, dim=-1)
    log_trans = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1)
    trans = torch.exp(log_trans)
    q = (p * trans).squeeze()

    # 手算:
    # q0 = 0.1
    # q1 = 0.2 * 0.9 = 0.18
    # q2 = 0.9 * 0.9 * 0.8 = 0.648
    # q3 = 0.5 * 0.9 * 0.8 * 0.1 = 0.036
    # q4 = 0.0 * ... = 0
    expected = torch.tensor([0.1, 0.18, 0.648, 0.036, 0.0])
    assert torch.allclose(q, expected, atol=1e-5), f"{q} vs {expected}"

    # 完全 hit 概率 (1 - trans_end) 应该接近 Σq
    trans_end = torch.exp(cum[..., -1]).squeeze()
    hit_prob = q.sum()
    assert abs(hit_prob.item() + trans_end.item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 4. hit loss 数值范围
# ---------------------------------------------------------------------------


def _straight_x_ray() -> tuple[torch.Tensor, torch.Tensor]:
    """沿 +X 方向的单条 ray，K=1。

    origin 选在 (0, 0.2, 0.4)：这样 ray 正好穿过 iy=100, iz=3 这一行的体素中心，
    并且等间距采样 (0.2, 0.6, ..., 19.8m) 能与 ix=100,101,...149 的体素中心精确对齐，
    避免 bilinear 插值精度问题。
    """
    origin = torch.tensor([[[0.0, 0.2, 0.4]]], dtype=torch.float32)  # (B=1,K=1,3)
    direction = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)  # (B=1,R=1,3)
    return origin, direction


def _gt(gt_value: float) -> torch.Tensor:
    """构造 (B=1,K=1,R=1) 的 GT 深度。"""
    return torch.tensor([[[gt_value]]], dtype=torch.float32)


# 参考 GT voxel：ix=120, iy=100, iz=3，ray 经过该体素中心，d_star=8.2m
GT_VOXEL = (120, 100, 3)
GT_DIST = PC_RANGE[0] + (GT_VOXEL[0] + 0.5) * VOXEL_SIZE  # 8.2
FALSE_HIT_VOXEL = (102, 100, 3)


def test_hit_loss_large_when_all_free():
    """全空 volume：窗口内 q ≈ 0，NLL 应该很大。"""
    origin, direction = _straight_x_ray()
    rl = _make_ray_loss(lambda_hit=1.0, lambda_depth=0.0)
    logits = _free_everywhere_logits()
    gt_dist = _gt(GT_DIST)
    out = rl(logits, origin, direction, gt_dist)
    assert out["valid_rays"].item() == 1
    assert out["hit_raw"].item() > 5.0, (
        f"全空 volume 的 hit_raw 应很大，实际 {out['hit_raw'].item():.3f}"
    )


def test_hit_loss_small_when_voxel_hit():
    """目标 voxel occupied：窗口内 q ≈ 1，NLL 应该很小。"""
    origin, direction = _straight_x_ray()
    rl = _make_ray_loss(lambda_hit=1.0, lambda_depth=0.0)
    logits = _occupied_at(*GT_VOXEL)
    gt_dist = _gt(GT_DIST)
    out = rl(logits, origin, direction, gt_dist)
    assert out["valid_rays"].item() == 1
    assert out["hit_raw"].item() < 1.0, (
        f"完美对齐时 hit_raw 应很小，实际 {out['hit_raw'].item():.3f}"
    )


def test_invalid_ray_skipped():
    """valid_rays=0 时应返回可反传零损失。"""
    rl = _make_ray_loss()
    logits = _free_everywhere_logits().requires_grad_(True)
    origin = torch.zeros((1, 1, 3))
    direction = torch.tensor([[1.0, 0.0, 0.0]]).view(1, 1, 3)
    gt_dist = _gt(float("nan"))
    out = rl(logits, origin, direction, gt_dist)
    assert out["valid_rays"].item() == 0
    # 零损失仍应可反传
    out["total"].backward()


def test_zero_early_return_contains_new_fields():
    """supervised_rays=0 的 early return 应返回完整字段。"""
    rl = _make_ray_loss(lambda_hit=1.0, lambda_empty=1.0, lambda_depth=1.0)
    logits = _free_everywhere_logits().requires_grad_(True)
    origin = torch.zeros((1, 1, 3))
    direction = torch.tensor([[1.0, 0.0, 0.0]]).view(1, 1, 3)
    gt_dist = _gt(float("nan"))
    out = rl(logits, origin, direction, gt_dist)
    for key in (
        "empty",
        "empty_raw",
        "hit_rays",
        "empty_rays",
        "supervised_rays",
        "valid_rays",
    ):
        assert key in out, f"缺少字段 {key}"
    assert int(out["supervised_rays"].item()) == 0
    assert int(out["empty_rays"].item()) == 0
    out["total"].backward()


def test_empty_ray_loss_small_when_all_free():
    """GT=no-hit 且 volume 全 free 时，empty loss 应很小。"""
    origin, direction = _straight_x_ray()
    rl = _make_ray_loss(lambda_hit=0.0, lambda_empty=1.0, lambda_depth=0.0)
    logits = _free_everywhere_logits()
    out = rl(logits, origin, direction, _gt(float("inf")))
    assert int(out["hit_rays"].item()) == 0
    assert int(out["empty_rays"].item()) == 1
    assert int(out["valid_rays"].item()) == 1
    assert out["hit_raw"].item() == 0.0
    assert out["depth_raw"].item() == 0.0
    assert out["empty_raw"].item() < 0.1, (
        f"全 free 时 no-hit loss 应很小，实际 {out['empty_raw'].item():.4f}"
    )
    assert torch.allclose(out["total"], out["empty"])


def test_empty_ray_loss_large_when_false_hit_exists():
    """GT=no-hit，但 ray 前方有 occupied 时，empty loss 应明显增大。"""
    origin, direction = _straight_x_ray()
    rl = _make_ray_loss(lambda_hit=0.0, lambda_empty=1.0, lambda_depth=0.0)
    out_free = rl(_free_everywhere_logits(), origin, direction, _gt(float("inf")))
    out_false = rl(
        _occupied_at(*FALSE_HIT_VOXEL, free_val=20.0),
        origin,
        direction,
        _gt(float("inf")),
    )
    assert int(out_false["hit_rays"].item()) == 0
    assert int(out_false["empty_rays"].item()) == 1
    assert out_false["empty_raw"].item() > out_free["empty_raw"].item() + 0.05, (
        f"前方假 hit 应显著降低 no-hit 概率："
        f"free={out_free['empty_raw'].item():.4f}, "
        f"false={out_false['empty_raw'].item():.4f}"
    )


def test_far_hit_is_treated_as_empty_within_horizon():
    """GT 有限但超出监督 horizon 时，应按 empty ray 处理。"""
    origin, direction = _straight_x_ray()
    rl = _make_ray_loss(
        lambda_hit=1.0, lambda_empty=1.0, lambda_depth=1.0, mid_max_m=20.0
    )
    out = rl(_free_everywhere_logits(), origin, direction, _gt(25.0))
    assert int(out["hit_rays"].item()) == 0
    assert int(out["empty_rays"].item()) == 1
    assert int(out["supervised_rays"].item()) == 1
    assert out["hit_raw"].item() == 0.0
    assert out["depth_raw"].item() == 0.0
    assert torch.allclose(out["total"], out["empty"])


def test_sidecar_v3_inf_round_trip():
    """v3 sidecar 中的 inf/no-hit 读回后仍应保持 inf。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dist = np.array([[[[1.0, np.inf, np.nan]]]], dtype=np.float16)
        origin = np.zeros((1, 1, 1, 3), dtype=np.float32)
        sup_mask = np.ones((1, 1), dtype=np.uint8)
        origin_mask = np.ones((1, 1, 1), dtype=np.uint8)
        meta = {
            "schema_version": _SCHEMA_V3,
            "token_to_idx": {"tok": 0},
            "supervision_labels": ["t"],
            "num_rays": 3,
            "dist_semantics": "finite=hit_dist_m, inf=no_hit, nan=ignore",
            "ray_horizon_m": 20.0,
        }
        np.save(Path(tmpdir) / "train_dist.npy", dist)
        np.save(Path(tmpdir) / "train_origin.npy", origin)
        np.save(Path(tmpdir) / "train_sup_mask.npy", sup_mask)
        np.save(Path(tmpdir) / "train_origin_mask.npy", origin_mask)
        with open(Path(tmpdir) / "train_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        sidecar = RaySidecar(tmpdir, "train")
        hit = sidecar.query("tok")
        assert hit is not None
        dist_q, _, _, _ = hit
        assert np.isfinite(dist_q[0, 0, 0])
        assert np.isinf(dist_q[0, 0, 1])
        assert np.isnan(dist_q[0, 0, 2])


def test_gt_dist_bias_compensates_dvr_exit_distance():
    """DVR 返回 voxel 出射距离（≈center + 0.5 voxel）。

    构造：目标 voxel 在 ix=120，center 距离 8.2m，DVR-like exit 距离 8.4m。
    - bias=0.0 + GT=8.2 (center)：depth err 应接近 0
    - bias=0.2 + GT=8.4 (exit)：应产生与上面等价的行为（内部 gt_eff=8.2）
    - bias=0.0 + GT=8.4：err ≈ +0.2，depth loss 明显 > 0
    三者对比说明 bias 参数确实把 DVR exit 语义正确校正到了 d_hat 的 center 语义。
    """
    origin, direction = _straight_x_ray()
    logits = _occupied_at(*GT_VOXEL, free_val=20.0)

    def _depth_raw(bias: float, gt: float) -> float:
        rl = _make_ray_loss(
            lambda_hit=0.0, lambda_depth=1.0, window_voxels=1,
            depth_asym_far=2.0, depth_asym_near=1.0,
            gt_dist_bias_m=bias,
        )
        out = rl(logits, origin, direction, _gt(gt))
        return out["depth_raw"].item()

    loss_center_nobias = _depth_raw(0.0, 8.2)
    loss_exit_withbias = _depth_raw(0.2, 8.4)
    loss_exit_nobias = _depth_raw(0.0, 8.4)

    # bias 能把 DVR exit GT 校正回 center 语义
    assert abs(loss_center_nobias - loss_exit_withbias) < 1.0e-4, (
        f"bias 校正后应与 center GT 等价: "
        f"center_nobias={loss_center_nobias:.4f}, exit_withbias={loss_exit_withbias:.4f}"
    )
    # 不校正时，同一个 exit GT 会产生显著更大的 loss
    # (err=-0.2, neg 分支权重 1.0, SmoothL1(0.2, β=1)=0.02)
    assert loss_exit_nobias > loss_exit_withbias + 0.01, (
        f"未补偿 DVR exit 偏置会产生系统性 depth err: "
        f"exit_nobias={loss_exit_nobias:.4f}, exit_withbias={loss_exit_withbias:.4f}"
    )


def test_out_of_bound_samples_do_not_produce_false_hits():
    """越界采样点应被强制当作 free，不能产生假 first-hit 质量。

    构造：origin 放在 pc_range 的正 X 边界外（x=50m > x_max=40m），
    ray 朝 +X 射出，所有采样点都在体积外。体素被全部设成 occupied
    （非 free 类 logit=10，free 类 logit=-10）。

    坏实现下（'border' 不加 sample_valid，或 'zeros' 不加 sample_valid），
    越界 sample 分别会沿用边界 p_free≈0 或得到 p_free=0，两者都会让
    最早的 sample q[..., 0] ≈ 1。为了能真正抓到回归，这里 gt_dist 放
    在最早 sample 处（0.2m），让窗口覆盖坏实现会堆积假 hit 的位置。

    正确实现下，越界点 p_free 被强制成 1 → p_occ=0 → q≈0 → hit_raw 很大。
    """
    rl = _make_ray_loss(
        lambda_hit=1.0, lambda_depth=0.0, window_voxels=1,
    )
    # 所有体素都是高置信 occupied（非 free）
    logits = torch.zeros((1, NUM_CLASSES, *GRID))
    logits[:, 0] = 10.0
    logits[:, FREE_IDX] = -10.0

    # origin 在 +X 边界外 10m，ray 沿 +X 射出，50 个采样点全在场外
    origin = torch.tensor([[[50.0, 0.0, 2.0]]])
    direction = torch.tensor([[[1.0, 0.0, 0.0]]])
    # GT 放在 d[0] = 0.2m，窗口 [0, 0.6] 覆盖最早几个 sample
    gt_dist = _gt(0.2)

    out = rl(logits, origin, direction, gt_dist)
    assert out["valid_rays"].item() == 1

    # 坏实现：q[..., 0] ≈ 1，窗口内 sum ≈ 1，NLL ≈ 0
    # 正确实现：q ≈ 0 → 窗口内 sum ≈ 0，NLL ≈ -log(eps) ≈ 13.8
    assert out["hit_raw"].item() > 5.0, (
        f"越界采样点应被强制成 free，窗口内不应有 first-hit 质量；"
        f"实际 hit_raw={out['hit_raw'].item():.3f}（疑似越界采样产生了假 hit）"
    )


# ---------------------------------------------------------------------------
# 5. 梯度方向正确性
# ---------------------------------------------------------------------------


def test_hit_loss_gradient_pushes_free_down():
    """在 GT 附近 voxel 处，对 free logit 的梯度应该是正的（梯度下降会减小 free）。"""
    origin, direction = _straight_x_ray()
    rl = _make_ray_loss(
        lambda_hit=1.0, lambda_depth=0.0,
        window_voxels=2,  # 放宽窗口以覆盖最近几个采样点
    )
    logits = _free_everywhere_logits().clone().requires_grad_(True)
    gt_dist = _gt(GT_DIST)
    out = rl(logits, origin, direction, gt_dist)
    out["total"].backward()

    ix, iy, iz = GT_VOXEL
    grad_free = logits.grad[0, FREE_IDX]

    # 在目标 voxel 一个 3×3×2 邻域里找最大正梯度
    nbr = grad_free[
        max(ix - 2, 0) : ix + 3,
        max(iy - 1, 0) : iy + 2,
        max(iz - 1, 0) : iz + 2,
    ]
    assert nbr.max().item() > 0, (
        f"目标体素附近 free 通道应有正梯度（梯度下降会降 free），"
        f"实际最大梯度 {nbr.max().item()}"
    )

    # 相应地，非 free 类别的梯度应为负
    other_logit = logits.grad[0, 0]  # class 0
    nbr_other = other_logit[
        max(ix - 2, 0) : ix + 3,
        max(iy - 1, 0) : iy + 2,
        max(iz - 1, 0) : iz + 2,
    ]
    assert nbr_other.min().item() < 0, (
        f"非 free 类应得到负梯度（即被推高），实际最小梯度 {nbr_other.min().item()}"
    )


def test_hit_loss_optimization_converges():
    """从全空开始 Adam 优化 logits，hit_loss 应该显著下降。"""
    torch.manual_seed(0)
    origin, direction = _straight_x_ray()

    rl = _make_ray_loss(
        lambda_hit=1.0, lambda_depth=0.0,
        window_voxels=2,
    )
    logits = _free_everywhere_logits().clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.5)
    gt_dist = _gt(GT_DIST)

    losses = []
    for _ in range(200):
        opt.zero_grad()
        out = rl(logits, origin, direction, gt_dist)
        out["total"].backward()
        opt.step()
        losses.append(out["hit_raw"].item())

    assert losses[-1] < losses[0] * 0.3, (
        f"Adam 优化应显著降低 hit_loss，初值 {losses[0]:.3f}, 末值 {losses[-1]:.3f}"
    )
    assert losses[-1] < 2.0, f"末值应接近收敛，实际 {losses[-1]:.3f}"


# ---------------------------------------------------------------------------
# 6. 非对称 depth loss
# ---------------------------------------------------------------------------


def test_depth_loss_asymmetric_far_heavier_than_near():
    """同样 |err|，pred 更远的 depth loss 应该 ≈ 2× pred 更近的。

    origin (0, 0.2, 0.4) + dir (1,0,0) 下，采样 i=19,20,21 恰好落在
    voxel (119,100,3)、(120,100,3)、(121,100,3) 的中心。
    GT 放在 ix=120；场景 A 占据 ix=121（err=+0.4），场景 B 占据 ix=119（err=-0.4）。
    """
    origin, direction = _straight_x_ray()
    gt_dist = _gt(GT_DIST)  # 8.2m

    rl = _make_ray_loss(
        lambda_hit=0.0, lambda_depth=1.0, window_voxels=1,
        depth_asym_far=2.0, depth_asym_near=1.0,
    )

    ix, iy, iz = GT_VOXEL
    # 场景 A：占据 (ix+1)，pred 更远 1 voxel
    logits_far = _occupied_at(ix + 1, iy, iz, free_val=20.0)
    # 场景 B：占据 (ix-1)，pred 更近 1 voxel
    logits_near = _occupied_at(ix - 1, iy, iz, free_val=20.0)

    out_far = rl(logits_far, origin, direction, gt_dist)
    out_near = rl(logits_near, origin, direction, gt_dist)

    loss_far = out_far["depth_raw"].item()
    loss_near = out_near["depth_raw"].item()
    assert loss_far > loss_near * 1.5, (
        f"pred 更远的 depth loss 应显著大于 pred 更近："
        f"far={loss_far:.4f}, near={loss_near:.4f}"
    )
    ratio = loss_far / max(loss_near, 1e-6)
    assert 1.5 < ratio < 2.5, f"非对称比例应接近 2:1，实际 {ratio:.2f}"


# ---------------------------------------------------------------------------
# 7. 多原点接口：K>1、origin_mask
# ---------------------------------------------------------------------------


def test_multi_origin_two_identical_origins_match_single():
    """两份完全相同的原点 → hit_raw/depth_raw 与单原点等价（加权平均的重复）。"""
    origin, direction = _straight_x_ray()             # origin:(1,1,3)
    rl = _make_ray_loss(lambda_hit=1.0, lambda_depth=1.0)
    logits = _occupied_at(*GT_VOXEL, free_val=20.0)
    gt_dist = _gt(GT_DIST)                            # (1,1,1)

    out_ref = rl(logits, origin, direction, gt_dist)

    origin_k2 = origin.repeat(1, 2, 1).contiguous()   # (1,2,3)
    gt_dist_k2 = gt_dist.repeat(1, 2, 1).contiguous() # (1,2,1)

    out_k2 = rl(logits, origin_k2, direction, gt_dist_k2)

    for key in ("hit_raw", "depth_raw"):
        diff = (out_ref[key] - out_k2[key]).abs().item()
        assert diff < 1e-6, (
            f"两份相同原点的加权平均应等于单原点，{key} 差异 {diff:.2e}"
        )
    assert int(out_k2["valid_rays"].item()) == 2 * int(out_ref["valid_rays"].item())


def test_multi_origin_mask_excludes_padded_origin():
    """origin_mask=0 的 pad 原点对 loss 完全不贡献。

    K=2：第 0 个原点是真正的 GT 对齐原点，第 1 个原点被放到一个不相干的位置（而且
    GT 也乱填），但 origin_mask 标 0。结果应与单原点等价。
    """
    origin, direction = _straight_x_ray()             # origin:(1,1,3)
    rl = _make_ray_loss(lambda_hit=1.0, lambda_depth=1.0)
    logits = _occupied_at(*GT_VOXEL, free_val=20.0)
    gt_dist = _gt(GT_DIST)

    out_ref = rl(logits, origin, direction, gt_dist)

    # 第 1 个原点放在 pc_range 边界，GT 也乱填（5.0m），正常参与会产生很大 hit loss
    pad_origin = torch.tensor([[[30.0, 30.0, 2.0]]], dtype=torch.float32)
    origin_k2 = torch.cat([origin, pad_origin], dim=1).contiguous()       # (1,2,3)
    gt_dist_k2 = torch.tensor([[[GT_DIST], [5.0]]], dtype=torch.float32)   # (1,2,1)
    origin_mask = torch.tensor([[True, False]])

    out = rl(
        logits, origin_k2, direction, gt_dist_k2, origin_mask=origin_mask
    )

    for key in ("hit_raw", "depth_raw"):
        diff = (out_ref[key] - out[key]).abs().item()
        assert diff < 1e-6, (
            f"pad 原点 mask=0 应不贡献，{key} 差异 {diff:.2e}"
        )
    assert int(out["valid_rays"].item()) == int(out_ref["valid_rays"].item())


# ---------------------------------------------------------------------------
# 运行入口：不依赖 pytest，直接 python 跑
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    passed = 0
    failed: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            tb = traceback.format_exc()
            print(f"  FAIL  {name}\n{tb}")
            failed.append((name, tb))
    print(f"\n{passed}/{len(tests)} passed, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all())
