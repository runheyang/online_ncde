"""OpenOccupancy 分支前向 smoke test（不依赖真实数据）。

覆盖：
  1. OpusSparseTopkLoader._decode_frame: uint16 coords 解码 → 命中体素 top-3 logits 正确，
     未命中体素 free 通道为 free_fill_value、其余为 other_fill_value。
  2. OnlineNcdeAlignerDS forward 三种 mode（default / stepwise_train / stepwise_eval），
     断言 encoder 下采样 + decoder 上采样后输出形状 = 输入空间维度，类别维度 = num_classes。
  3. h=0 退化：rollout_start_step == num_frames-1 时直接返回 slow_logits。

直接运行：
  conda run -n neural_ode python tests/test_openocc_aligner_forward.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.data.logits_loader import OpusSparseTopkLoader  # noqa: E402
from online_ncde.models.online_ncde_aligner_ds import OnlineNcdeAlignerDS  # noqa: E402

NUM_CLASSES = 17
FREE_INDEX = 16
OTHER_FILL = -5.0
FREE_FILL = 5.0


def test_loader_uint16_coords_decode() -> None:
    """构造 OO 风格 npz（uint16 coords + top-3 logits），过 loader 后检查 dense 张量。"""
    # 使用 OO 真实网格 512×512×40：让 coords 必走 uint16 dtype 路径。
    grid_x, grid_y, grid_z = 512, 512, 40
    # 三个命中体素：坐标都 > 255，逼着走 uint16
    coords = np.array(
        [
            [300, 100, 4],
            [400, 256, 2],
            [10, 500, 7],
        ],
        dtype=np.uint16,
    )
    # 每个体素 top-3 logits（按 sparse_topk 约定，已经降序排好）
    topk_values = np.array(
        [
            [3.0, 1.5, -2.0],     # 类 0 / 5 / 9（其中 -2 > -5 不被 clamp）
            [4.5, 2.0, -10.0],    # 类 3 / 7 / 12（-10 应被 clamp 到 -5）
            [1.0, 0.0, -1.0],     # 类 1 / 6 / 15
        ],
        dtype=np.float16,
    )
    topk_indices = np.array(
        [
            [0, 5, 9],
            [3, 7, 12],
            [1, 6, 15],
        ],
        dtype=np.uint8,
    )

    # 构造一个临时 logits root 与 npz
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        fast_root = tmp_root / "fast"
        slow_root = tmp_root / "slow"
        fast_root.mkdir()
        slow_root.mkdir()

        sample_npz = fast_root / "frame.npz"
        np.savez_compressed(
            sample_npz,
            sparse_coords=coords,
            sparse_topk_values=topk_values,
            sparse_topk_indices=topk_indices,
        )

        # 必须把同一个 npz 也作为 slow 路径，loader 会去读
        slow_npz = slow_root / "slow.npz"
        np.savez_compressed(
            slow_npz,
            sparse_coords=coords,
            sparse_topk_values=topk_values,
            sparse_topk_indices=topk_indices,
        )

        loader = OpusSparseTopkLoader(
            root_path=str(tmp_root),
            fast_logits_root="fast",
            slow_logit_root="slow",
            num_classes=NUM_CLASSES,
            free_index=FREE_INDEX,
            grid_size=(grid_x, grid_y, grid_z),
            other_fill_value=OTHER_FILL,
            free_fill_value=FREE_FILL,
        )

        info = {
            "frame_rel_paths": ["frame.npz", "frame.npz", ""],  # 第三帧 pad
            "slow_logit_path": "slow.npz",
        }
        device = torch.device("cpu")
        fast = loader.load_fast_logits(info, device)
        slow = loader.load_slow_logits(info, device)

    # 断言 shape
    assert fast.shape == (3, NUM_CLASSES, grid_x, grid_y, grid_z), fast.shape
    assert slow.shape == (NUM_CLASSES, grid_x, grid_y, grid_z), slow.shape

    # 命中 voxel #0：coords=(300,100,4)，类 0=3.0 / 类 5=1.5 / 类 9=-2.0；其余非 free 类 = -5；free 类 = -5（被命中）
    v0 = fast[0, :, 300, 100, 4]
    assert v0[0].item() == 3.0
    assert v0[5].item() == 1.5
    assert v0[9].item() == -2.0
    # free 通道：命中体素的 free 不写入，留在 other_fill_value=-5
    assert v0[FREE_INDEX].item() == OTHER_FILL
    # 没出现的类：必须是 other_fill
    assert v0[2].item() == OTHER_FILL

    # 命中 voxel #1：第三个 logit -10 应被 clamp_min(-5)
    v1 = fast[0, :, 400, 256, 2]
    assert v1[3].item() == 4.5
    assert v1[7].item() == 2.0
    assert v1[12].item() == OTHER_FILL  # -10 clamp_min(-5) = -5

    # 未命中 voxel：free 通道 = free_fill；其余 = other_fill
    unhit = fast[0, :, 0, 0, 0]
    assert unhit[FREE_INDEX].item() == FREE_FILL
    for c in (1, 5, 10, 15):
        assert unhit[c].item() == OTHER_FILL

    # pad 帧：free 通道 = free_fill，其余 = other_fill
    pad = fast[2, :, 12, 13, 5]
    assert pad[FREE_INDEX].item() == FREE_FILL
    assert pad[0].item() == OTHER_FILL
    print("[PASS] loader uint16 coords decode")


def _make_dummy_inputs(
    num_frames: int = 5,
    spatial: tuple[int, int, int] = (32, 32, 8),
    device: torch.device = torch.device("cpu"),
):
    """造小尺寸 dummy 样本用于 CPU 前向。要求 spatial 各维能被 stride=2 整除。"""
    X, Y, Z = spatial
    fast_logits = torch.randn(num_frames, NUM_CLASSES, X, Y, Z, device=device) * 0.1
    slow_logits = torch.randn(NUM_CLASSES, X, Y, Z, device=device) * 0.1

    poses = []
    for t in range(num_frames):
        T = torch.eye(4, device=device)
        T[0, 3] = 0.2 * t
        T[1, 3] = 0.1 * t
        poses.append(T)
    frame_ego2global = torch.stack(poses, dim=0)
    frame_dt = torch.tensor([0.5 * t for t in range(num_frames)], device=device)
    return fast_logits, slow_logits, frame_ego2global, frame_dt


def _build_aligner(device: torch.device) -> OnlineNcdeAlignerDS:
    """与 base_2hz.yaml 对齐的小尺寸构造。pc_range 缩到 [-3.2, 3.2]，与 32×32×8 + voxel=0.2 一致。"""
    return OnlineNcdeAlignerDS(
        num_classes=NUM_CLASSES,
        feat_dim=8,
        hidden_dim=8,
        encoder_in_channels=NUM_CLASSES,
        free_index=FREE_INDEX,
        pc_range=(-3.2, -3.2, -0.8, 3.2, 3.2, 0.8),
        voxel_size=(0.2, 0.2, 0.2),
        encoder_downsample_stride=(2, 2, 2),
        decoder_init_scale=1.0e-6,
        use_fast_residual=True,
        func_g_inner_dim=8,
        func_g_body_dilations=(1, 3, 5),
        func_g_gn_groups=4,
        timestamp_scale=1.0,
        solver_variant="heun",
    ).to(device)


def test_aligner_forward_default() -> None:
    device = torch.device("cpu")
    fast_logits, slow_logits, ego2global, dt = _make_dummy_inputs(device=device)
    model = _build_aligner(device).eval()

    with torch.no_grad():
        out = model(
            fast_logits=fast_logits,
            slow_logits=slow_logits,
            frame_ego2global=ego2global,
            frame_timestamps=None,
            frame_dt=dt,
            mode="default",
        )

    aligned = out["aligned"]
    # default 不显式补 batch；模型内部 _unsqueeze_inputs 自动补到 (1, ...)
    assert aligned.dim() == 5, aligned.shape
    assert aligned.shape[1] == NUM_CLASSES
    assert aligned.shape[2:] == fast_logits.shape[2:]
    assert torch.isfinite(aligned).all().item()
    print(f"[PASS] default forward: aligned.shape={tuple(aligned.shape)}")


def test_aligner_forward_stepwise_train() -> None:
    device = torch.device("cpu")
    fast_logits, slow_logits, ego2global, dt = _make_dummy_inputs(device=device)
    model = _build_aligner(device).train()

    out = model(
        fast_logits=fast_logits,
        slow_logits=slow_logits,
        frame_ego2global=ego2global,
        frame_timestamps=None,
        frame_dt=dt,
        mode="stepwise_train",
    )

    step_logits = out["step_logits"]
    assert step_logits.shape == (
        1,
        fast_logits.shape[0] - 1,
        NUM_CLASSES,
        *fast_logits.shape[2:],
    ), step_logits.shape
    # 反传一下确认 graph 通畅
    loss = step_logits.float().sum()
    loss.backward()
    print(f"[PASS] stepwise_train forward + backward: step_logits.shape={tuple(step_logits.shape)}")


def test_aligner_forward_stepwise_eval() -> None:
    device = torch.device("cpu")
    fast_logits, slow_logits, ego2global, dt = _make_dummy_inputs(device=device)
    model = _build_aligner(device).eval()

    with torch.no_grad():
        out = model(
            fast_logits=fast_logits,
            slow_logits=slow_logits,
            frame_ego2global=ego2global,
            frame_timestamps=None,
            frame_dt=dt,
            mode="stepwise_eval",
        )

    for key in ("step_logits", "step_time_ms", "step_warp_ms", "step_solver_ms", "step_decode_ms"):
        assert key in out, f"missing {key}"
    assert out["step_logits"].shape[1] == fast_logits.shape[0] - 1
    print("[PASS] stepwise_eval forward: 全部计时键齐备")


def test_aligner_h0_degenerate() -> None:
    """rollout_start_step == num_frames-1：直接返回 slow_logits，不过 encoder/decoder。"""
    device = torch.device("cpu")
    fast_logits, slow_logits, ego2global, dt = _make_dummy_inputs(device=device)
    model = _build_aligner(device).eval()

    rss = torch.tensor([fast_logits.shape[0] - 1], dtype=torch.long)
    with torch.no_grad():
        out = model(
            fast_logits=fast_logits,
            slow_logits=slow_logits,
            frame_ego2global=ego2global,
            frame_timestamps=None,
            frame_dt=dt,
            mode="default",
            rollout_start_step=rss,
        )

    aligned = out["aligned"][0]
    assert torch.allclose(aligned, slow_logits.float(), atol=1.0e-6)
    print("[PASS] h=0 退化：aligned == slow_logits")


def main() -> None:
    test_loader_uint16_coords_decode()
    test_aligner_forward_default()
    test_aligner_forward_stepwise_train()
    test_aligner_forward_stepwise_eval()
    test_aligner_h0_degenerate()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
