"""RecurrentWarpFusionAligner sanity 测试。

不依赖真实数据，直接用随机张量验证：
  1. default forward 输出 shape 与 dtype 正确；
  2. stepwise_train forward 输出 step_logits 形状对，且 loss 可反向；
  3. stepwise_eval forward 返回完整计时键；
  4. h=0 退化（rollout_start_step == num_frames-1）走 slow 直出分支。

直接运行：conda run -n neural_ode python tests/test_recurrent_warp_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch  # noqa: E402

from online_ncde.baselines import RecurrentWarpFusionAligner  # noqa: E402


def _make_dummy_inputs(
    num_classes: int = 18,
    num_frames: int = 4,
    spatial: tuple[int, int, int] = (40, 40, 8),
    device: torch.device = torch.device("cpu"),
):
    """造一个小空间体积的 dummy 样本，便于 CPU 上跑 sanity。"""
    X, Y, Z = spatial
    fast_logits = torch.randn(num_frames, num_classes, X, Y, Z, device=device) * 0.1
    slow_logits = torch.randn(num_classes, X, Y, Z, device=device) * 0.1

    # ego2global：每帧加一点小平移，模拟 ego 运动
    poses = []
    for t in range(num_frames):
        T = torch.eye(4, device=device)
        T[0, 3] = 0.2 * t
        T[1, 3] = 0.1 * t
        poses.append(T)
    frame_ego2global = torch.stack(poses, dim=0)  # (T, 4, 4)

    # 累积 dt（按 0.5s 间隔）
    frame_dt = torch.tensor([0.5 * t for t in range(num_frames)], device=device)

    return fast_logits, slow_logits, frame_ego2global, frame_dt


def _make_model(device: torch.device, fusion_kind: str = "conv"):
    return RecurrentWarpFusionAligner(
        num_classes=18,
        feat_dim=24,
        hidden_dim=24,
        encoder_in_channels=18,
        free_index=17,
        # 用真实 pc_range / voxel_size 的子集，与 spatial=(40,40,8) 对应一个缩放区域
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        voxel_size=(2.0, 2.0, 0.8),
        decoder_init_scale=1.0e-3,
        use_fast_residual=True,
        fusion_kind=fusion_kind,
        fusion_inner_dim=32,
        fusion_body_dilations=(1, 2, 3),
        fusion_gn_groups=8,
        fusion_attn_num_heads=4,
        fusion_attn_window_size=(8, 8, 4),
        fusion_attn_head_dilations=(1, 2),
        fusion_attn_mlp_ratio=2.0,
    ).to(device)


def _check_forward_default(fusion_kind: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(device, fusion_kind=fusion_kind)
    fast, slow, pose, dt = _make_dummy_inputs(device=device)
    out = model(
        fast_logits=fast,
        slow_logits=slow,
        frame_ego2global=pose,
        frame_timestamps=None,
        frame_dt=dt,
        mode="default",
    )
    aligned = out["aligned"]
    assert aligned.shape == (1, 18, 40, 40, 8), f"aligned shape={aligned.shape}"
    assert aligned.dtype == torch.float32
    print(f"[default/{fusion_kind}] OK aligned={tuple(aligned.shape)}")


def _check_forward_stepwise_train_backward(fusion_kind: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(device, fusion_kind=fusion_kind)
    fast, slow, pose, dt = _make_dummy_inputs(device=device)
    out = model(
        fast_logits=fast,
        slow_logits=slow,
        frame_ego2global=pose,
        frame_timestamps=None,
        frame_dt=dt,
        mode="stepwise_train",
    )
    step_logits = out["step_logits"]
    step_indices = out["step_indices"]
    assert step_logits.shape == (1, 3, 18, 40, 40, 8), f"step_logits shape={step_logits.shape}"
    assert step_indices.tolist() == [1, 2, 3]
    loss = step_logits.float().mean()
    loss.backward()
    grad_ok = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert grad_ok, "no parameter received gradient"
    print(f"[stepwise_train/{fusion_kind}] OK step_logits={tuple(step_logits.shape)} loss={loss.item():.4f}")


def _check_forward_stepwise_eval_keys(fusion_kind: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(device, fusion_kind=fusion_kind)
    model.eval()
    fast, slow, pose, dt = _make_dummy_inputs(device=device)
    with torch.inference_mode():
        out = model.forward_stepwise_eval(
            fast_logits=fast,
            slow_logits=slow,
            frame_ego2global=pose,
            frame_timestamps=None,
            frame_dt=dt,
        )
    for key in ["step_logits", "step_indices", "step_time_ms", "step_warp_ms",
                "step_solver_ms", "step_decode_ms"]:
        assert key in out, f"missing key {key}"
    step_logits = out["step_logits"]
    assert step_logits.shape == (1, 3, 18, 40, 40, 8)
    assert out["step_warp_ms"].shape == (1, 3)
    print(f"[stepwise_eval/{fusion_kind}] OK timing keys={list(out.keys())}")


def _check_h0_degenerate(fusion_kind: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(device, fusion_kind=fusion_kind)
    model.eval()
    fast, slow, pose, dt = _make_dummy_inputs(device=device, num_frames=4)
    rss = torch.tensor([3], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.forward_stepwise_eval(
            fast_logits=fast,
            slow_logits=slow,
            frame_ego2global=pose,
            frame_timestamps=None,
            frame_dt=dt,
            rollout_start_step=rss,
        )
    step_logits = out["step_logits"]
    assert step_logits.shape == (1, 1, 18, 40, 40, 8), f"shape={step_logits.shape}"
    diff = (step_logits[0, 0] - slow.float()).abs().max().item()
    assert diff < 1e-5, f"degenerate branch diff={diff}"
    print(f"[h=0 degenerate/{fusion_kind}] OK slow 直出 diff={diff:.2e}")


def test_forward_default():
    _check_forward_default("conv")
    _check_forward_default("attn")


def test_forward_stepwise_train_backward():
    _check_forward_stepwise_train_backward("conv")
    _check_forward_stepwise_train_backward("attn")


def test_forward_stepwise_eval_keys():
    _check_forward_stepwise_eval_keys("conv")
    _check_forward_stepwise_eval_keys("attn")


def test_h0_degenerate_returns_slow():
    _check_h0_degenerate("conv")
    _check_h0_degenerate("attn")


def test_fast_kl_protocol():
    """与 NCDE 协议一致：_fast_kl_active=True 时 stepwise_train 输出含 fast_kl 标量。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for kind in ["conv", "attn"]:
        model = _make_model(device, fusion_kind=kind)
        # 默认关闭：不应输出 fast_kl
        fast, slow, pose, dt = _make_dummy_inputs(device=device)
        out_off = model(
            fast_logits=fast, slow_logits=slow, frame_ego2global=pose,
            frame_timestamps=None, frame_dt=dt, mode="stepwise_train",
        )
        assert "fast_kl" not in out_off, f"[{kind}] _fast_kl_active=False 时不应返回 fast_kl"

        # 打开后应输出 fast_kl 且可反向
        model._fast_kl_active = True
        out_on = model(
            fast_logits=fast, slow_logits=slow, frame_ego2global=pose,
            frame_timestamps=None, frame_dt=dt, mode="stepwise_train",
        )
        assert "fast_kl" in out_on, f"[{kind}] _fast_kl_active=True 时应返回 fast_kl"
        kl = out_on["fast_kl"]
        assert kl.dim() == 0, f"[{kind}] fast_kl 应为 0-dim 标量，shape={kl.shape}"
        kl.backward(retain_graph=False)
        print(f"[fast_kl/{kind}] OK kl={float(kl):.6f}")


def test_param_count_matches_ncde_close():
    """RWFA / RWFA-Attn 参数量应与 NCDE 主体量级相当（±10% 内）。"""
    from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner

    device = torch.device("cpu")
    rwfa_conv = _make_model(device, fusion_kind="conv")
    rwfa_attn = _make_model(device, fusion_kind="attn")
    ncde = OnlineNcdeAligner(
        num_classes=18,
        feat_dim=24,
        hidden_dim=24,
        encoder_in_channels=18,
        free_index=17,
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        voxel_size=(2.0, 2.0, 0.8),
        decoder_init_scale=1.0e-3,
        use_fast_residual=True,
        func_g_inner_dim=32,
        func_g_body_dilations=(1, 2, 3),
        func_g_gn_groups=8,
        solver_variant="euler",
    )
    p_conv = sum(p.numel() for p in rwfa_conv.parameters())
    p_attn = sum(p.numel() for p in rwfa_attn.parameters())
    p_ncde = sum(p.numel() for p in ncde.parameters())
    print(f"[param-count] RWFA-conv={p_conv}  RWFA-attn={p_attn}  NCDE={p_ncde}")
    print(f"[param-count] ratio conv/ncde={p_conv/p_ncde:.3f}  attn/ncde={p_attn/p_ncde:.3f}")
    assert 0.85 <= p_conv / p_ncde <= 1.15, f"conv ratio out of band: {p_conv/p_ncde}"
    assert 0.85 <= p_attn / p_ncde <= 1.15, f"attn ratio out of band: {p_attn/p_ncde}"


if __name__ == "__main__":
    test_forward_default()
    test_forward_stepwise_train_backward()
    test_forward_stepwise_eval_keys()
    test_h0_degenerate_returns_slow()
    test_fast_kl_protocol()
    test_param_count_matches_ncde_close()
    print("\nAll sanity tests passed.")
