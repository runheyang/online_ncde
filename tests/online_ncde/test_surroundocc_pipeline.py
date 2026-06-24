from pathlib import Path

import numpy as np
import pytest
import torch

from online_ncde.data.logits_loader import AloccDenseTopkLoader
from online_ncde.data.surroundocc_online_ncde_dataset import SurroundOccOnlineNcdeDataset
from online_ncde.evaluation import attach_dense_occ_targets, make_dense_occ_prediction


def test_alocc_dense_topk_sample_token_offset(tmp_path: Path) -> None:
    root = tmp_path
    logits_dir = root / "logits" / "scene-0001" / "sample-token"
    logits_dir.mkdir(parents=True)
    np.savez_compressed(
        logits_dir / "logits.npz",
        topk_values=np.array([[[[1.0, 0.5, -1.0]]]], dtype=np.float16),
        topk_indices=np.array([[[[1, 2, 17]]]], dtype=np.uint8),
    )

    loader = AloccDenseTopkLoader(
        root_path=str(root),
        fast_logits_root="logits",
        slow_logit_root="logits",
        num_classes=17,
        grid_size=(1, 1, 1),
        fill_value=-5.0,
        clamp_min=-5.0,
        topk_k=3,
        max_centering=False,
        label_id_offset=-1,
        path_token_type="sample_token",
    )
    info = {
        "scene_name": "scene-0001",
        "frame_sample_tokens": ["sample-token"],
        "slow_sample_token": "sample-token",
    }

    dense = loader.load_fast_logits(info, torch.device("cpu"))
    assert tuple(dense.shape) == (1, 17, 1, 1, 1)
    assert dense[0, 0, 0, 0, 0].item() == pytest.approx(1.0)
    assert dense[0, 1, 0, 0, 0].item() == pytest.approx(0.5)
    assert dense[0, 16, 0, 0, 0].item() == pytest.approx(-1.0)


def test_surroundocc_gt_loader_real_data() -> None:
    root = Path(__file__).resolve().parents[2]
    required = [
        root / "configs/online_ncde/canonical_infos_val_full.pkl",
        root / "data/nuscenes/v1.0-trainval/sample_data.json",
        root / "data/nuscenes/gts_surroundocc/samples",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("本地缺少 SurroundOcc smoke 数据")

    dummy_loader = AloccDenseTopkLoader(
        root_path=str(root),
        fast_logits_root="data/alocc2d_mini_surroundocc_logits",
        slow_logit_root="data/alocc3d_surroundocc_logits",
        num_classes=17,
        grid_size=(200, 200, 16),
        topk_k=3,
        label_id_offset=-1,
        path_token_type="sample_token",
    )
    dataset = SurroundOccOnlineNcdeDataset(
        info_path="configs/online_ncde/canonical_infos_val_full.pkl",
        root_path=str(root),
        gt_root="data/nuscenes/gts_surroundocc",
        num_classes=17,
        free_index=16,
        grid_size=(200, 200, 16),
        logits_loader=dummy_loader,
        min_history_completeness=4,
    )
    labels, mask = dataset._load_curr_gt(dataset.infos[0], lambda _: None)
    assert tuple(labels.shape) == (200, 200, 16)
    assert tuple(mask.shape) == (200, 200, 16)
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 16
    assert torch.all(mask == 1)


def test_attach_surroundocc_targets_real_data() -> None:
    root = Path(__file__).resolve().parents[2]
    required = [
        root / "configs/online_ncde/canonical_infos_val_full.pkl",
        root / "data/nuscenes/v1.0-trainval/sample_data.json",
        root / "data/nuscenes/gts_surroundocc/samples",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("本地缺少 SurroundOcc smoke 数据")

    import pickle

    with open(root / "configs/online_ncde/canonical_infos_val_full.pkl", "rb") as f:
        payload = pickle.load(f)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    token = str(infos[0]["token"])
    pred = np.full((200, 200, 16), 16, dtype=np.uint8)
    records = [
        make_dense_occ_prediction(
            pred=pred,
            token=token,
            scene_name=str(infos[0].get("scene_name", "")),
        )
    ]

    attached, missing = attach_dense_occ_targets(
        records,
        dataset_variant="surroundocc",
        gt_root=str(root / "data/nuscenes/gts_surroundocc"),
        nuscenes_root=str(root / "data/nuscenes"),
        nuscenes_version="v1.0-trainval",
        grid_size=(200, 200, 16),
    )
    assert missing == 0
    assert len(attached) == 1
    assert attached[0]["gt"].shape == (200, 200, 16)
    assert attached[0]["mask_camera"].shape == (200, 200, 16)
    assert np.all(attached[0]["mask_camera"] == 1)
