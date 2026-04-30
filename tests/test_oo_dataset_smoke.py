"""OpenOccupancy dataset 端到端 smoke test。

需要本机可见的真实数据：
  - data/nuscenes/oo_token_map.pkl
  - data/logits_opusv2t_openocc_2hz/<scene>/<token>/logits.npz
  - data/logits_opusv2l_openocc/<scene>/<token>/logits.npz
  - data/nuscenes/occupancy/scene_<scene_token>/occupancy/<lidar_token>.npy
  - configs/online_ncde/canonical_infos_val_full.pkl

覆盖：
  1. dataset.__getitem__ 跑通：GT shape (512, 512, 40)，类范围 [0, 16]，noise mask 含 0/1 两值。
  2. supervision GT 至少有一帧有效。
  3. fast_logits / slow_logits 形状对得上。
  4. 拿 1 个样本喂 OnlineNcdeAlignerDS 跑一次 forward，断言 aligned shape。

CPU 跑（512×512×40 fp32 单步推理本机可负担，模型缩到 feat_dim=8 减体积）。

直接运行：
  conda run -n neural_ode python tests/test_oo_dataset_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
os.chdir(ROOT)

from online_ncde.data.build_dataset import build_online_ncde_dataset  # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader  # noqa: E402
from online_ncde.models.online_ncde_aligner_ds import OnlineNcdeAlignerDS  # noqa: E402

DATA_CFG = {
    "dataset_variant": "openoccupancy",
    "oo_token_map_path": "configs/online_ncde/oo_token_map.pkl",
    "logits_format": "opus_sparse_topk",
    "fast_logits_root": "data/logits_opusv2t_openocc_2hz",
    "slow_logit_root": "data/logits_opusv2l_openocc",
    "gt_root": "data/nuscenes/occupancy",
    "gt_mask_key": "noise_mask",
    "num_classes": 17,
    "free_index": 16,
    "grid_size": [512, 512, 40],
    "fast_frame_stride": 3,
    "opus_other_fill_value": -5.0,
    "opus_free_fill_value": 5.0,
}


def test_oo_dataset_getitem() -> None:
    logits_loader = build_logits_loader(DATA_CFG, root_path=str(ROOT))
    dataset = build_online_ncde_dataset(
        DATA_CFG,
        info_path="configs/online_ncde/canonical_infos_val_full.pkl",
        root_path=str(ROOT),
        logits_loader=logits_loader,
        fast_frame_stride=3,
        min_history_completeness=4,
    )
    print(f"[dataset] len = {len(dataset)}")
    assert len(dataset) > 0

    sample = dataset[0]
    fast = sample["fast_logits"]
    slow = sample["slow_logits"]
    gt = sample["gt_labels"]
    mask = sample["gt_mask"]

    assert fast.shape == (5, 17, 512, 512, 40), f"fast.shape={fast.shape}"
    assert slow.shape == (17, 512, 512, 40), f"slow.shape={slow.shape}"
    assert gt.shape == (512, 512, 40), f"gt.shape={gt.shape}"
    assert mask.shape == (512, 512, 40), f"mask.shape={mask.shape}"
    assert gt.dtype == torch.long, gt.dtype
    assert mask.dtype == torch.float32, mask.dtype

    uniq = gt.unique().tolist()
    assert min(uniq) >= 0 and max(uniq) <= 16, f"GT 类越界: {uniq}"
    # 至少包含 free（16）这一全局兜底类
    assert 16 in uniq, "GT 应该至少有 free 类（未存 voxel → 16）"
    # noise mask 取值只有 0 / 1
    mask_vals = set(mask.unique().tolist())
    assert mask_vals.issubset({0.0, 1.0}), f"mask 取值异常: {mask_vals}"

    sup_valid = sample["sup_valid_mask"]
    sup_labels = sample["sup_labels"]
    sup_masks = sample["sup_masks"]
    print(f"[sample0] sup_valid = {sup_valid.tolist()}")
    assert sup_valid.sum().item() >= 1, "至少应该有 1 帧 supervision 有效"
    # 抽一帧 supervision 校验类映射一致
    valid_idx = int((sup_valid > 0).nonzero()[0].item())
    sup_uniq = sup_labels[valid_idx].unique().tolist()
    assert min(sup_uniq) >= 0 and max(sup_uniq) <= 16, f"sup GT 类越界: {sup_uniq}"
    sup_mask_vals = set(sup_masks[valid_idx].unique().tolist())
    assert sup_mask_vals.issubset({0.0, 1.0}), f"sup mask 取值异常: {sup_mask_vals}"

    print(
        f"[PASS] OO dataset getitem: fast={tuple(fast.shape)} slow={tuple(slow.shape)} "
        f"gt unique={len(uniq)} mask_ratio={mask.mean().item():.3f}"
    )
    return sample


def test_oo_dataset_forward(sample: dict) -> None:
    """用 OO dataset 输出的 1 个样本喂下采样 aligner 跑一次 forward（CPU，feat_dim=8 缩量）。"""
    model = OnlineNcdeAlignerDS(
        num_classes=17,
        feat_dim=8,
        hidden_dim=8,
        encoder_in_channels=17,
        free_index=16,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        voxel_size=(0.2, 0.2, 0.2),
        encoder_downsample_stride=(2, 2, 2),
        decoder_init_scale=1.0e-6,
        use_fast_residual=True,
        func_g_inner_dim=8,
        func_g_body_dilations=(1, 3, 5),
        func_g_gn_groups=4,
        timestamp_scale=1.0e-6,
        solver_variant="heun",
    ).eval()

    fast = sample["fast_logits"]
    slow = sample["slow_logits"]
    ego = sample["frame_ego2global"]
    ts = sample.get("frame_timestamps")
    dt = sample.get("frame_dt")

    print("[forward] 跑一次 default forward（CPU，512×512×40，可能耗时数十秒）...")
    with torch.no_grad():
        out = model(
            fast_logits=fast,
            slow_logits=slow,
            frame_ego2global=ego,
            frame_timestamps=ts,
            frame_dt=dt,
            mode="default",
        )
    aligned = out["aligned"]
    assert aligned.shape == (1, 17, 512, 512, 40), aligned.shape
    assert torch.isfinite(aligned).all().item()
    print(f"[PASS] OO dataset → aligner forward: aligned={tuple(aligned.shape)}")


def main() -> None:
    sample = test_oo_dataset_getitem()
    test_oo_dataset_forward(sample)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
