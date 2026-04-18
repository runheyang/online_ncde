"""Fast-KL 冒烟测试：构造小体积 fake 输入，验证 forward + 反传无错。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from online_ncde_200x200x16 import OnlineNcdeAligner200


def main() -> None:
    common = dict(
        num_classes=18, feat_dim=32, hidden_dim=32, encoder_in_channels=18, free_index=17,
        pc_range=(-40., -40., -1., 40., 40., 5.4), voxel_size=(0.4, 0.4, 0.4),
        decoder_init_scale=1e-6, use_fast_residual=True,
        func_g_inner_dim=24, func_g_body_dilations=(1, 3, 5), func_g_gn_groups=8,
        timestamp_scale=1e-6,
    )
    model = OnlineNcdeAligner200(**common).cuda()
    print("n params:", sum(p.numel() for p in model.parameters()))

    T, C, X, Y, Z = 4, 18, 50, 50, 8
    fast_logits = torch.randn(T, C, X, Y, Z, device="cuda")
    slow_logits = torch.randn(C, X, Y, Z, device="cuda")
    frame_ego2global = torch.eye(4, device="cuda").unsqueeze(0).repeat(T, 1, 1)
    frame_ego2global[:, :3, 3] = torch.randn(T, 3, device="cuda") * 0.5
    frame_timestamps = torch.linspace(0, 3.0, T, device="cuda")

    # Fast-KL 关闭：不应返回 fast_kl
    model.train()
    model._fast_kl_active = False
    out_off = model(
        fast_logits=fast_logits.unsqueeze(0),
        slow_logits=slow_logits.unsqueeze(0),
        frame_ego2global=frame_ego2global.unsqueeze(0),
        frame_timestamps=frame_timestamps.unsqueeze(0),
        frame_dt=None,
        mode="stepwise_train",
    )
    assert "fast_kl" not in out_off, "关闭时不应返回 fast_kl"
    print("fast_kl 关闭路径 OK，step_logits:", out_off["step_logits"].shape)

    # Fast-KL 打开：应返回 fast_kl 且可反传
    model._fast_kl_active = True
    out_on = model(
        fast_logits=fast_logits.unsqueeze(0),
        slow_logits=slow_logits.unsqueeze(0),
        frame_ego2global=frame_ego2global.unsqueeze(0),
        frame_timestamps=frame_timestamps.unsqueeze(0),
        frame_dt=None,
        mode="stepwise_train",
    )
    assert "fast_kl" in out_on, "开启后应返回 fast_kl"
    fkl = out_on["fast_kl"]
    print(f"fast_kl: {float(fkl):.6f}, requires_grad={fkl.requires_grad}")
    assert fkl.requires_grad, "fast_kl 应可反传"

    loss = out_on["step_logits"].pow(2).mean() + 0.1 * out_on["fast_kl"]
    loss.backward()
    any_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert any_grad, "反传后模型参数应有梯度"
    print("OK")


if __name__ == "__main__":
    main()
