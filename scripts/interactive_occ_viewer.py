#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("ETS_TOOLKIT", "qt")
os.environ.setdefault("QT_API", "pyqt5")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from evoocc.config import load_config_with_base                      # noqa: E402
from evoocc.data.build_logits_loader import build_logits_loader      # noqa: E402
from evoocc.data.occ3d_evoocc_dataset import Occ3DEvoOccDataset  # noqa: E402
from evoocc.models.evoocc_aligner import EvoOccAligner       # noqa: E402
from evoocc.utils.checkpoints import load_checkpoint_for_eval         # noqa: E402
from evoocc.visualization.occ_renderer import (                       # noqa: E402
    OCC3D_CLASS_NAMES,
    OCC3D_COLORS,
    clear_figure,
    render_voxel_into_figure,
)

from PyQt5 import QtCore, QtGui, QtWidgets                                # noqa: E402
from traits.api import HasTraits, Instance                                # noqa: E402
from traitsui.api import View, Item                                       # noqa: E402
from mayavi.tools.mlab_scene_model import MlabSceneModel                  # noqa: E402
from tvtk.pyface.scene_editor import SceneEditor                          # noqa: E402
from mayavi.core.ui.mayavi_scene import MayaviScene                       # noqa: E402

PANEL_KEYS = ["gt", "aligned", "fast", "slow_m2", "slow_m1", "slow_curr"]
PANEL_NAMES = [
    "GT (curr)",
    "EvoOcc Aligned (curr)",
    "Fast (curr frame)",
    "Slow (-2s keyframe)",
    "Slow (-1s keyframe)",
    "Slow (curr keyframe)",
]
SLOW_HIST_KEYS = ("slow_m2", "slow_m1")
# indices into evolve_keyframe_sample_tokens (oldest first): tokens[0] = -2s, tokens[1] = -1s
SLOW_HIST_KF_OFFSETS = {"slow_m2": 0, "slow_m1": 1}


def warp_labels_to_ego(
    labels_src: np.ndarray,
    T_src_to_dst: np.ndarray,
    pc_range: tuple,
    voxel_size: tuple,
    free_index: int,
) -> np.ndarray:
    """Nearest-neighbor resample of labels_src (X, Y, Z) from src ego to dst ego via T_src_to_dst (4, 4); out-of-range -> free_index."""
    X, Y, Z = labels_src.shape
    vx, vy, vz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    ii, jj, kk = np.meshgrid(
        np.arange(X), np.arange(Y), np.arange(Z), indexing="ij",
    )
    px = x_min + (ii + 0.5) * vx
    py = y_min + (jj + 0.5) * vy
    pz = z_min + (kk + 0.5) * vz

    T_dst_to_src = np.linalg.inv(T_src_to_dst).astype(np.float64)
    p = np.stack([px, py, pz, np.ones_like(px)], axis=-1).astype(np.float64)
    p_src = p @ T_dst_to_src.T

    si = np.floor((p_src[..., 0] - x_min) / vx).astype(np.int64)
    sj = np.floor((p_src[..., 1] - y_min) / vy).astype(np.int64)
    sk = np.floor((p_src[..., 2] - z_min) / vz).astype(np.int64)
    in_range = (
        (si >= 0) & (si < X)
        & (sj >= 0) & (sj < Y)
        & (sk >= 0) & (sk < Z)
    )

    out = np.full_like(labels_src, free_index)
    out[in_range] = labels_src[si[in_range], sj[in_range], sk[in_range]]
    return out


def load_idx_list(path: str) -> list[int]:
    p = Path(path)
    if not p.is_absolute():
        p = (Path(__file__).resolve().parents[1] / path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"idx-list 文件不存在: {p}")

    suffix = p.suffix.lower()
    raw_idx: list[int] = []
    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, int):
                    raw_idx.append(item)
                elif isinstance(item, dict) and "idx" in item:
                    raw_idx.append(int(item["idx"]))
                else:
                    raise ValueError(f"json list 元素不支持: {item!r}")
        elif isinstance(payload, dict):
            if "indices" in payload:
                raw_idx = [int(x) for x in payload["indices"]]
            elif "samples" in payload or "items" in payload:
                arr = payload.get("samples", payload.get("items"))
                raw_idx = [int(x["idx"]) for x in arr]
            else:
                raise ValueError(
                    "json dict 必须含 'indices' 或 'samples'/'items' 字段"
                )
        else:
            raise ValueError(f"不支持的 json 顶层类型: {type(payload).__name__}")
    elif suffix == ".txt" or suffix == "":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                first = line.split()[0]
                try:
                    raw_idx.append(int(first))
                except ValueError:
                    raise ValueError(
                        f"txt 第一列必须是 int，行内容: {line!r}"
                    )
    else:
        raise ValueError(f"不支持的后缀: {suffix}（仅支持 .txt / .json）")

    seen: set[int] = set()
    out: list[int] = []
    for i in raw_idx:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


class SavedFastPredLoader:
    def __init__(
        self,
        pred_root: str,
        grid_size: tuple[int, int, int],
        filename: str = "pred.npz",
        key: str = "semantics",
        strict: bool = False,
    ) -> None:
        root = Path(pred_root)
        self.pred_root = root if root.is_absolute() else (ROOT / root).resolve()
        self.grid_size = tuple(int(x) for x in grid_size)
        self.filename = filename
        self.key = key
        self.strict = bool(strict)
        self._warned_missing = False

    def _pick_array(self, data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
        if self.key:
            if self.key not in data.files:
                raise KeyError(f"{path} 缺少 key={self.key!r}, 当前 keys={data.files}")
            return data[self.key]

        for key in ("semantics", "pred", "labels", "prediction"):
            if key in data.files:
                return data[key]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise KeyError(f"{path} 未指定 key 且无法自动判断，当前 keys={data.files}")

    def load(self, scene: str, token: str) -> np.ndarray | None:
        path = self.pred_root / scene / token / self.filename
        if not path.exists():
            if self.strict:
                raise FileNotFoundError(path)
            if not self._warned_missing:
                print(
                    f"[viewer] WARN: saved fast pred 缺失，回退 raw logits argmax: {path}"
                )
                self._warned_missing = True
            return None

        with np.load(path, allow_pickle=False) as data:
            pred = np.asarray(self._pick_array(data, path))

        if tuple(pred.shape) != self.grid_size:
            if pred.size == int(np.prod(self.grid_size)):
                pred = pred.reshape(self.grid_size)
            else:
                raise ValueError(
                    f"{path} shape={pred.shape} 与 grid_size={self.grid_size} 不一致"
                )
        return pred.astype(np.int32, copy=False)


@dataclass
class SampleData:
    gt: np.ndarray
    gt_mask: np.ndarray
    fast: np.ndarray
    slow_curr: np.ndarray
    slow_m1: np.ndarray | None
    slow_m2: np.ndarray | None
    # SE(3) (4x4) from -1s/-2s keyframe ego to curr ego; None if missing or not warped
    slow_m1_T_kf_to_curr: np.ndarray | None
    slow_m2_T_kf_to_curr: np.ndarray | None
    slow_m1_title: str
    slow_m2_title: str
    scene_name: str
    token: str
    fast_source: str
    rollout_start_step: int
    evolve_keyframe_sample_tokens: list[str]


class Backend:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        solver: str = "euler",
        val_info_path_override: str | None = None,
        fast_pred_root: str | None = None,
        fast_pred_filename: str = "pred.npz",
        fast_pred_key: str = "semantics",
        strict_fast_pred: bool = False,
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.solver = solver

        self.cfg = load_config_with_base(config_path)
        data_cfg = self.cfg["data"]
        if val_info_path_override:
            data_cfg["val_info_path"] = val_info_path_override
            print(f"[viewer] override val_info_path = {val_info_path_override}")
        self.data_cfg = data_cfg
        self.model_cfg = self.cfg["model"]
        self.eval_cfg = self.cfg.get("eval", {})

        self.free_index = int(data_cfg["free_index"])
        self.num_classes = int(data_cfg["num_classes"])
        self.pc_range = tuple(data_cfg["pc_range"])
        self.voxel_size_xyz = tuple(data_cfg["voxel_size"])
        self.voxel_size_iso = float(self.voxel_size_xyz[0])
        self.saved_fast_pred_loader: SavedFastPredLoader | None = None
        if fast_pred_root:
            self.saved_fast_pred_loader = SavedFastPredLoader(
                pred_root=fast_pred_root,
                grid_size=tuple(data_cfg["grid_size"]),
                filename=fast_pred_filename,
                key=fast_pred_key,
                strict=strict_fast_pred,
            )
            print(
                "[viewer] Fast panel uses saved pred: "
                f"root={fast_pred_root}, filename={fast_pred_filename}, key={fast_pred_key}"
            )

        self.logits_loader = build_logits_loader(data_cfg, self.cfg["root_path"])
        logits_loader = self.logits_loader
        self.dataset = Occ3DEvoOccDataset(
            info_path=data_cfg.get("val_info_path", data_cfg["info_path"]),
            root_path=self.cfg["root_path"],
            gt_root=data_cfg["gt_root"],
            num_classes=self.num_classes,
            free_index=self.free_index,
            grid_size=tuple(data_cfg["grid_size"]),
            gt_mask_key=data_cfg["gt_mask_key"],
            logits_loader=logits_loader,
            ray_sidecar_dir=data_cfg.get("ray_sidecar_dir", None),
            ray_sidecar_split="val",
            fast_frame_stride=int(data_cfg.get("fast_frame_stride", 1)),
            min_history_completeness=0,
            eval_only_mode=True,
        )

        self.device = torch.device(
            self.eval_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        self._model: EvoOccAligner | None = None
        self._raw_sample = None

    def __len__(self) -> int:
        return len(self.dataset)

    def load_sample(self, idx: int) -> SampleData:
        sample = self.dataset[idx]
        self._raw_sample = sample

        gt = sample["gt_labels"].cpu().numpy().astype(np.int32)
        gt_mask = sample["gt_mask"].cpu().numpy().astype(np.float32)
        fast_logits = sample["fast_logits"]
        fast = fast_logits[-1].argmax(0).cpu().numpy().astype(np.int32)

        meta = sample.get("meta", {})
        scene_name = str(meta.get("scene_name", ""))
        token = str(meta.get("token", ""))
        fast_source = "raw logits argmax"
        if self.saved_fast_pred_loader is not None:
            saved_fast = self.saved_fast_pred_loader.load(scene_name, token)
            if saved_fast is not None:
                fast = saved_fast
                fast_source = "saved official postprocess"

        info = self.dataset.infos[idx]
        ek_tokens = [str(t) for t in info.get("evolve_keyframe_sample_tokens", [])]
        ek_step_indices: list[int] = []
        if ek_tokens:
            ek_step_indices = [int(s) for s in meta.get("evolve_keyframe_step_indices", [])]
        if not ek_tokens:
            ek_tokens = [str(t) for t in info.get("keyframe_sample_tokens", [])]

        frame_ego2global = sample["frame_ego2global"].cpu().numpy()
        frame_timestamps = sample.get("frame_timestamps", None)
        frame_timestamps_np = (
            frame_timestamps.cpu().numpy()
            if isinstance(frame_timestamps, torch.Tensor)
            else None
        )
        if not ek_step_indices and len(ek_tokens) == frame_ego2global.shape[0]:
            ek_step_indices = list(range(len(ek_tokens)))

        def _load_slow_for(kf_token: str) -> np.ndarray:
            rel = f"{scene_name}/{kf_token}/logits.npz"
            logits = self.logits_loader.load_slow_logits(
                {"slow_logit_path": rel}, torch.device("cpu"),
            )
            return logits.argmax(0).numpy().astype(np.int32)

        def _pick_pos(pos: int) -> tuple[str | None, int | None]:
            if pos < 0 or pos >= len(ek_tokens):
                return None, None
            tok = ek_tokens[pos]
            if not tok:
                return None, None
            step = ek_step_indices[pos] if pos < len(ek_step_indices) else None
            return tok, step

        def _pick_history(
            steps_back: int,
            fallback_to_latest_available: bool = False,
        ) -> tuple[str | None, int | None, int | None, bool]:
            curr_pos = len(ek_tokens) - 1
            target_pos = curr_pos - int(steps_back)
            tok, step = _pick_pos(target_pos)
            if tok is not None:
                return tok, step, target_pos, False
            if not fallback_to_latest_available:
                return None, None, None, False
            for pos in range(curr_pos - 1, -1, -1):
                tok, step = _pick_pos(pos)
                if tok is not None:
                    return tok, step, pos, True
            return None, None, None, False

        def _history_title(default_title: str, src_step: int | None, fallback: bool) -> str:
            if src_step is None or curr_step is None:
                return f"{default_title} (fallback)" if fallback else default_title
            dt_s: float | None = None
            if frame_timestamps_np is not None:
                try:
                    dt_s = abs(float(frame_timestamps_np[curr_step] - frame_timestamps_np[src_step]) / 1.0e6)
                except Exception:
                    dt_s = None
            if dt_s is None:
                dt_s = 0.5 * abs(int(curr_step) - int(src_step))
            shown = f"Slow (-{dt_s:.1f}s keyframe)"
            return f"{shown} fallback" if fallback else shown

        curr_kf, curr_step = _pick_pos(len(ek_tokens) - 1)
        m1_kf, m1_step, _, m1_fallback = _pick_history(
            steps_back=2, fallback_to_latest_available=True,
        )
        m2_kf, m2_step, _, m2_fallback = _pick_history(
            steps_back=4, fallback_to_latest_available=False,
        )
        slow_m1_title = _history_title("Slow (-1s keyframe)", m1_step, m1_fallback)
        slow_m2_title = _history_title("Slow (-2s keyframe)", m2_step, m2_fallback)

        slow_curr = _load_slow_for(curr_kf) if curr_kf else _load_slow_for(token)

        def _load_and_warp(
            kf_token: str | None, src_step: int | None,
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
            if kf_token is None:
                return None, None
            labels_src = _load_slow_for(kf_token)
            if src_step is None or curr_step is None:
                return labels_src, None
            T_src_global = frame_ego2global[src_step]
            T_curr_global = frame_ego2global[curr_step]
            T_src_to_curr = np.linalg.inv(T_curr_global) @ T_src_global
            warped = warp_labels_to_ego(
                labels_src, T_src_to_curr,
                pc_range=self.pc_range,
                voxel_size=self.voxel_size_xyz,
                free_index=self.free_index,
            )
            return warped, T_src_to_curr.astype(np.float32)

        slow_m1, slow_m1_T = _load_and_warp(m1_kf, m1_step)
        slow_m2, slow_m2_T = _load_and_warp(m2_kf, m2_step)

        return SampleData(
            gt=gt, gt_mask=gt_mask, fast=fast,
            slow_curr=slow_curr, slow_m1=slow_m1, slow_m2=slow_m2,
            slow_m1_T_kf_to_curr=slow_m1_T, slow_m2_T_kf_to_curr=slow_m2_T,
            slow_m1_title=slow_m1_title, slow_m2_title=slow_m2_title,
            scene_name=scene_name, token=token,
            fast_source=fast_source,
            rollout_start_step=int(sample["rollout_start_step"].item()),
            evolve_keyframe_sample_tokens=ek_tokens,
        )

    def ensure_model(self) -> EvoOccAligner:
        if self._model is not None:
            return self._model
        m = EvoOccAligner(
            num_classes=self.num_classes,
            feat_dim=self.model_cfg["feat_dim"],
            hidden_dim=self.model_cfg["hidden_dim"],
            encoder_in_channels=self.model_cfg["encoder_in_channels"],
            free_index=self.free_index,
            pc_range=self.pc_range,
            voxel_size=self.voxel_size_xyz,
            decoder_init_scale=self.model_cfg.get("decoder_init_scale", 1.0e-3),
            use_fast_residual=bool(self.model_cfg.get("use_fast_residual", True)),
            func_g_inner_dim=self.model_cfg.get("func_g_inner_dim", 32),
            func_g_body_dilations=tuple(self.model_cfg.get("func_g_body_dilations", [1, 2, 3])),
            func_g_gn_groups=int(self.model_cfg.get("func_g_gn_groups", 8)),
            timestamp_scale=self.data_cfg.get("timestamp_scale", 1.0e-6),
            solver_variant=self.solver,
        ).to(self.device)
        load_checkpoint_for_eval(self.checkpoint_path, model=m, strict=False)
        m.eval()
        self._model = m
        return m

    @torch.no_grad()
    def run_aligner(self) -> np.ndarray:
        if self._raw_sample is None:
            raise RuntimeError("先调用 load_sample 再 run_aligner")
        m = self.ensure_model()
        s = self._raw_sample
        fast = s["fast_logits"].to(self.device).unsqueeze(0)
        slow = s["slow_logits"].to(self.device).unsqueeze(0)
        ego2g = s["frame_ego2global"].to(self.device).unsqueeze(0)
        ts = s["frame_timestamps"]
        if ts is not None:
            ts = ts.to(self.device).unsqueeze(0)
        dt = s["frame_dt"]
        if dt is not None:
            dt = dt.to(self.device).unsqueeze(0)
        rss = s["rollout_start_step"].to(self.device).unsqueeze(0)
        out = m.forward(
            fast_logits=fast,
            slow_logits=slow,
            frame_ego2global=ego2g,
            frame_timestamps=ts,
            frame_dt=dt,
            mode="default",
            rollout_start_step=rss,
        )
        aligned = out["aligned"][0]
        return aligned.argmax(0).cpu().numpy().astype(np.int32)


class _SceneHolder(HasTraits):
    scene = Instance(MlabSceneModel, ())
    view = View(
        Item("scene", editor=SceneEditor(scene_class=MayaviScene),
             height=300, width=400, show_label=False),
        resizable=True,
    )


class MayaviPanel(QtWidgets.QWidget):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(self.title_label)

        self.holder = _SceneHolder()
        self.ui = self.holder.edit_traits(parent=self, kind="subpanel").control
        layout.addWidget(self.ui)

    @property
    def scene(self) -> MlabSceneModel:
        return self.holder.scene

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)


class OccViewer(QtWidgets.QMainWindow):
    # default paper-style view; forward_m shifts the focal point along +x so the ego sits in the lower half
    DEFAULT_AZ = -60.0
    DEFAULT_EL = 35.0
    DEFAULT_DIST_RATIO = 2.2
    DEFAULT_FP_FORWARD_M = 10.0

    def __init__(
        self,
        backend: Backend,
        start_idx: int = 0,
        idx_list: list[int] | None = None,
        view_az: float | None = None,
        view_el: float | None = None,
        view_dist_ratio: float | None = None,
        view_forward_m: float | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.current_idx = 0
        self.current_sample: SampleData | None = None
        self._syncing_camera = False
        self._panels: dict[str, MayaviPanel] = {}
        self._ego_actors: dict[str, list] = {}

        if idx_list is None:
            self._idx_list: list[int] = list(range(len(backend)))
            self._has_subset = False
        else:
            valid = [int(i) for i in idx_list if 0 <= int(i) < len(backend)]
            dropped = len(idx_list) - len(valid)
            if not valid:
                raise RuntimeError("idx_list 全部越界或为空")
            if dropped:
                print(f"[viewer] idx_list 中 {dropped} 个 idx 越界已跳过")
            self._idx_list = valid
            self._has_subset = True
        self._list_pos = max(0, min(start_idx, len(self._idx_list) - 1))

        self._view_az = self.DEFAULT_AZ if view_az is None else float(view_az)
        self._view_el = self.DEFAULT_EL if view_el is None else float(view_el)
        self._view_dist_ratio = (
            self.DEFAULT_DIST_RATIO if view_dist_ratio is None else float(view_dist_ratio)
        )
        self._view_forward_m = (
            self.DEFAULT_FP_FORWARD_M if view_forward_m is None else float(view_forward_m)
        )

        self.setWindowTitle("EvoOcc Occupancy Viewer")
        self.resize(1800, 1000)

        self._build_toolbar()
        self._build_panels()
        self._build_statusbar()

        QtCore.QTimer.singleShot(50, self._init_after_show)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("main")
        tb.setMovable(False)

        prefix = "list#" if self._has_subset else "idx"
        tb.addWidget(QtWidgets.QLabel(f" {prefix} "))

        self.prev_btn = QtWidgets.QPushButton("◀")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        tb.addWidget(self.prev_btn)

        self.idx_box = QtWidgets.QSpinBox()
        self.idx_box.setRange(0, len(self._idx_list) - 1)
        self.idx_box.editingFinished.connect(self._on_idx_box_changed)
        tb.addWidget(self.idx_box)

        suffix = "  [from list]" if self._has_subset else ""
        self.total_label = QtWidgets.QLabel(
            f" / {len(self._idx_list) - 1}{suffix}"
        )
        tb.addWidget(self.total_label)

        self.next_btn = QtWidgets.QPushButton("▶")
        self.next_btn.clicked.connect(lambda: self._step(+1))
        tb.addWidget(self.next_btn)

        tb.addSeparator()

        self.run_btn = QtWidgets.QPushButton("▶ 运行 EvoOcc 对齐")
        self.run_btn.clicked.connect(self._on_run_aligner)
        tb.addWidget(self.run_btn)

        self.reset_view_btn = QtWidgets.QPushButton("↻ 重置视角")
        self.reset_view_btn.clicked.connect(self._reset_views)
        tb.addWidget(self.reset_view_btn)

        self.ego_toggle = QtWidgets.QCheckBox("显示自车")
        self.ego_toggle.setChecked(True)
        self.ego_toggle.toggled.connect(self._on_toggle_ego_visible)
        tb.addWidget(self.ego_toggle)

        self.save_btn = QtWidgets.QPushButton("💾 保存大图")
        self.save_btn.clicked.connect(self._on_save_composite)
        tb.addWidget(self.save_btn)

        tb.addSeparator()
        self.meta_label = QtWidgets.QLabel("")
        tb.addWidget(self.meta_label)

    def _build_panels(self) -> None:
        central = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(central)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for key, name, pos in zip(PANEL_KEYS, PANEL_NAMES, positions):
            panel = MayaviPanel(name)
            self._panels[key] = panel
            grid.addWidget(panel, *pos)
        for r in (0, 1):
            grid.setRowStretch(r, 1)
        for c in (0, 1, 2):
            grid.setColumnStretch(c, 1)
        self.setCentralWidget(central)

    def _build_statusbar(self) -> None:
        self.status = self.statusBar()
        self.status.showMessage("ready")

    def _init_after_show(self) -> None:
        for key in PANEL_KEYS:
            scene_obj = self._panels[key].scene
            scene_obj.background = (1.0, 1.0, 1.0)
            self._install_camera_sync(key, scene_obj)
        self._load_by_list_pos(self._list_pos)

    def _install_camera_sync(self, src_key: str, scene_obj: MlabSceneModel) -> None:
        try:
            iren = scene_obj.scene.interactor
        except Exception:
            return
        if iren is None:
            return
        iren.add_observer("EndInteractionEvent",
                          lambda obj, evt, k=src_key: self._sync_cameras_from(k))

    def _sync_cameras_from(self, src_key: str) -> None:
        if self._syncing_camera:
            return
        self._syncing_camera = True
        try:
            src_cam = self._panels[src_key].scene.scene.camera
            pos = tuple(src_cam.position)
            fp = tuple(src_cam.focal_point)
            up = tuple(src_cam.view_up)
            ps = float(src_cam.parallel_scale)
            va = float(src_cam.view_angle)
            cr = tuple(src_cam.clipping_range)
            for k in PANEL_KEYS:
                if k == src_key:
                    continue
                cam = self._panels[k].scene.scene.camera
                cam.position = pos
                cam.focal_point = fp
                cam.view_up = up
                cam.parallel_scale = ps
                cam.view_angle = va
                cam.clipping_range = cr
                self._panels[k].scene.scene.render()
        finally:
            self._syncing_camera = False

    def _reset_views(self) -> None:
        if self.current_sample is None:
            return
        x_min, y_min, z_min, x_max, y_max, z_max = self.backend.pc_range
        fp = (
            (x_min + x_max) * 0.5 + self._view_forward_m,
            (y_min + y_max) * 0.5,
            (z_min + z_max) * 0.5,
        )
        extent = max(x_max - x_min, y_max - y_min)
        distance = float(extent) * self._view_dist_ratio
        self._syncing_camera = True
        try:
            for k in PANEL_KEYS:
                from mayavi import mlab
                mlab.view(
                    azimuth=self._view_az,
                    elevation=self._view_el,
                    distance=distance,
                    focalpoint=fp,
                    figure=self._panels[k].scene.mayavi_scene,
                )
        finally:
            self._syncing_camera = False

    def _step(self, delta: int) -> None:
        new_pos = max(0, min(len(self._idx_list) - 1, self._list_pos + delta))
        if new_pos == self._list_pos:
            return
        self._load_by_list_pos(new_pos)

    def _on_idx_box_changed(self) -> None:
        new_pos = int(self.idx_box.value())
        if new_pos == self._list_pos:
            return
        self._load_by_list_pos(new_pos)

    def _load_by_list_pos(self, list_pos: int) -> None:
        list_pos = max(0, min(len(self._idx_list) - 1, list_pos))
        global_idx = self._idx_list[list_pos]
        self._list_pos = list_pos
        self._load_sample(global_idx)

    def _load_sample(self, idx: int) -> None:
        self.status.showMessage(f"loading sample idx={idx} ...")
        QtWidgets.QApplication.processEvents()
        try:
            sample = self.backend.load_sample(idx)
        except Exception as e:
            self.status.showMessage(f"load failed: {e}")
            return

        self.current_idx = idx
        self.current_sample = sample
        self.idx_box.blockSignals(True)
        self.idx_box.setValue(self._list_pos)
        self.idx_box.blockSignals(False)

        if self._has_subset:
            pos_info = (f"  list_pos={self._list_pos}/{len(self._idx_list)-1}"
                        f"  global_idx={idx}")
        else:
            pos_info = ""
        self.meta_label.setText(
            f"{pos_info}  scene={sample.scene_name}  "
            f"token={sample.token[:12]}…  rss={sample.rollout_start_step}"
        )

        self._render_panel("gt", sample.gt)
        self._render_panel("fast", sample.fast)
        fast_title = (
            "Fast (official postprocess)"
            if sample.fast_source == "saved official postprocess"
            else "Fast (raw logits argmax)"
        )
        self._panels["fast"].set_title(fast_title)
        self._render_panel("slow_curr", sample.slow_curr)
        self._panels["slow_curr"].set_title("Slow (curr keyframe)")

        for key in SLOW_HIST_KEYS:
            voxel = sample.slow_m2 if key == "slow_m2" else sample.slow_m1
            base_title = sample.slow_m2_title if key == "slow_m2" else sample.slow_m1_title
            if voxel is None:
                self._clear_panel(key)
                self._panels[key].set_title(f"{base_title} (N/A)")
            else:
                self._render_panel(key, voxel)
                self._panels[key].set_title(base_title)

        self._clear_panel("aligned")
        self._panels["aligned"].set_title("EvoOcc Aligned (未运行)")

        self._reset_views()
        self.status.showMessage(f"loaded sample {idx}")

    def _render_panel(self, key: str, voxel: np.ndarray,
                      mask: np.ndarray | None = None) -> None:
        scene_obj = self._panels[key].scene
        fig = scene_obj.mayavi_scene
        clear_figure(fig)
        render_voxel_into_figure(
            fig, voxel,
            voxel_size=self.backend.voxel_size_iso,
            pc_range=self.backend.pc_range,
            free_index=self.backend.free_index,
            apply_mask=mask,
        )
        self._draw_ego_marker(key)

    def _clear_panel(self, key: str) -> None:
        clear_figure(self._panels[key].scene.mayavi_scene)
        self._ego_actors.pop(key, None)

    def _draw_ego_marker(self, key: str) -> None:
        from mayavi import mlab
        fig = self._panels[key].scene.mayavi_scene

        # marker lifted 1.5 m in z to avoid occlusion by road voxels
        px, py, pz = 0.0, 0.0, 1.5
        fx, fy, fz = 1.0, 0.0, 0.0

        sample = self.current_sample
        if sample is not None and key in ("slow_m1", "slow_m2"):
            T = (sample.slow_m1_T_kf_to_curr if key == "slow_m1"
                 else sample.slow_m2_T_kf_to_curr)
            if T is not None:
                px = float(T[0, 3])
                py = float(T[1, 3])
                pz = float(T[2, 3]) + 1.5
                fx = float(T[0, 0])
                fy = float(T[1, 0])
                fz = float(T[2, 0])

        sphere = mlab.points3d(
            [px], [py], [pz],
            color=(1.0, 0.1, 0.1),
            mode="sphere",
            scale_factor=2.5,
            figure=fig,
        )
        arrow = mlab.quiver3d(
            [px], [py], [pz],
            [fx], [fy], [fz],
            color=(1.0, 0.1, 0.1),
            mode="arrow",
            scale_factor=5.0,
            figure=fig,
        )
        self._ego_actors[key] = [sphere, arrow]
        if not self._is_ego_visible():
            for a in self._ego_actors[key]:
                try:
                    a.visible = False
                except Exception:
                    pass

    def _is_ego_visible(self) -> bool:
        toggle = getattr(self, "ego_toggle", None)
        return True if toggle is None else bool(toggle.isChecked())

    def _set_ego_visible(self, visible: bool) -> None:
        for actors in self._ego_actors.values():
            for a in actors:
                try:
                    a.visible = bool(visible)
                except Exception:
                    pass

    def _on_toggle_ego_visible(self, checked: bool) -> None:
        self._set_ego_visible(checked)
        for k in PANEL_KEYS:
            try:
                self._panels[k].scene.scene.render()
            except Exception:
                pass

    def _on_run_aligner(self) -> None:
        if self.current_sample is None:
            return
        self.run_btn.setEnabled(False)
        self.status.showMessage("running EvoOcc aligner ...")
        QtWidgets.QApplication.processEvents()
        try:
            aligned = self.backend.run_aligner()
            self._render_panel("aligned", aligned)
            self._panels["aligned"].set_title("EvoOcc Aligned")
            self.status.showMessage("EvoOcc aligner done")
        except Exception as e:
            self.status.showMessage(f"aligner failed: {e}")
            QtWidgets.QMessageBox.critical(self, "EvoOcc 对齐失败", str(e))
        finally:
            self.run_btn.setEnabled(True)

    def _on_save_composite(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存四面板拼图",
            f"occ_viewer_idx{self.current_idx:05d}.png",
            "PNG (*.png)",
        )
        if not path:
            return
        try:
            self._save_composite_to(path)
            self.status.showMessage(f"saved → {path}")
        except Exception as e:
            self.status.showMessage(f"save failed: {e}")
            QtWidgets.QMessageBox.critical(self, "保存失败", str(e))

    def _save_composite_to(self, path: str) -> None:
        from mayavi import mlab
        from PIL import Image, ImageDraw, ImageFont

        self._set_ego_visible(False)
        try:
            arrs: list[np.ndarray] = []
            for k in PANEL_KEYS:
                scene_obj = self._panels[k].scene
                scene_obj.scene.render()
                QtWidgets.QApplication.processEvents()
                arr = mlab.screenshot(
                    figure=scene_obj.mayavi_scene, mode="rgb", antialiased=True,
                )
                arrs.append(np.ascontiguousarray(arr))
        finally:
            self._set_ego_visible(True)
            for k in PANEL_KEYS:
                self._panels[k].scene.scene.render()

        ph = max(a.shape[0] for a in arrs)
        pw = max(a.shape[1] for a in arrs)

        font_dir = "/usr/share/fonts/truetype/dejavu"
        try:
            font_title = ImageFont.truetype(f"{font_dir}/DejaVuSans-Bold.ttf", 22)
            font_header = ImageFont.truetype(f"{font_dir}/DejaVuSansMono.ttf", 16)
            font_footer = ImageFont.truetype(f"{font_dir}/DejaVuSans.ttf", 14)
        except OSError:
            font_title = ImageFont.load_default()
            font_header = ImageFont.load_default()
            font_footer = ImageFont.load_default()

        title_h = 36
        header_h = 30
        footer_h = 28
        n_rows, n_cols = 2, 3
        row_h = title_h + ph
        total_w = pw * n_cols
        total_h = header_h + row_h * n_rows + footer_h

        canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        sample = self.current_sample
        header_text = (
            f"idx={self.current_idx}/{len(self.backend) - 1}   "
            f"scene={sample.scene_name if sample else ''}   "
            f"token={(sample.token[:16] + '...') if (sample and sample.token) else ''}   "
            f"rss={sample.rollout_start_step if sample else ''}"
        )
        draw.rectangle([0, 0, total_w, header_h], fill=(40, 40, 40))
        draw.text((10, 6), header_text, font=font_header, fill=(230, 230, 230))

        def _measure(text: str, font) -> tuple[int, int]:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                return draw.textsize(text, font=font)

        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for a, key, (r, c) in zip(arrs, PANEL_KEYS, positions):
            x0_panel = c * pw
            y0_row = header_h + r * row_h
            y0_img = y0_row + title_h

            title_text = self._panels[key].title_label.text()
            draw.rectangle(
                [x0_panel, y0_row, x0_panel + pw, y0_row + title_h],
                fill=(235, 235, 235),
            )
            tw, th = _measure(title_text, font_title)
            tx = x0_panel + (pw - tw) // 2
            ty = y0_row + (title_h - th) // 2 - 2
            draw.text((tx, ty), title_text, font=font_title, fill=(20, 20, 20))

            ah, aw, _ = a.shape
            ix = x0_panel + (pw - aw) // 2
            iy = y0_img + (ph - ah) // 2
            canvas.paste(Image.fromarray(a), (ix, iy))

        footer_text = (
            "All 6 panels share curr/end ego frame; "
            "Slow(-1s)/Slow(-2s) resampled (nearest); out-of-range -> free."
        )
        draw.rectangle(
            [0, total_h - footer_h, total_w, total_h],
            fill=(245, 245, 245),
        )
        draw.text((10, total_h - footer_h + 6), footer_text,
                  font=font_footer, fill=(80, 80, 80))

        canvas.save(path, "PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoOcc 交互式可视化")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--solver", choices=["heun", "euler"], default="euler")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="启动时跳转到的位置——传 --idx-list 时为 list 内位置 (0-based)，"
                             "否则为全集 idx")
    parser.add_argument("--val-info-path", default=None,
                        help="覆盖 config 里的 data.val_info_path，"
                             "比如指向 evolve_infos pkl 以启用 -1s/-2s slow 面板")
    parser.add_argument("--idx-list", default=None,
                        help="可选 idx 列表文件 (.txt 或 .json)，"
                             "▶/◀/spinbox 在该列表里走，按列表顺序看样本；"
                             ".txt 即 tests/evoocc/find_top_*.py 的输出格式；"
                             ".json 接受 [int...] / {indices: [...]} / {samples: [{idx,...}]}")
    parser.add_argument("--fast-pred-root", default=None,
                        help="可选：Fast 面板改为读取离散预测 root，例如 "
                             "data/preds_opusv1t_occ3d；不影响 EvoOcc 输入")
    parser.add_argument("--fast-pred-filename", default="pred.npz",
                        help="fast-pred-root 下每个 token 目录中的文件名")
    parser.add_argument("--fast-pred-key", default="semantics",
                        help="pred.npz 中的数组 key；传空字符串则自动猜测")
    parser.add_argument("--strict-fast-pred", action="store_true",
                        help="开启后 saved fast pred 缺失直接报错；默认缺失时回退 raw fast")
    parser.add_argument("--view-az", type=float, default=None,
                        help=f"相机方位角 azimuth (默认 {OccViewer.DEFAULT_AZ}°)；"
                             "数字越大越往左侧绕")
    parser.add_argument("--view-el", type=float, default=None,
                        help=f"相机俯角 elevation (默认 {OccViewer.DEFAULT_EL}°)；"
                             "0=平视，90=正俯视；越小越像第三人称跟随")
    parser.add_argument("--view-dist-ratio", type=float, default=None,
                        help=f"相机距离 = pc_range 边长 × ratio (默认 {OccViewer.DEFAULT_DIST_RATIO})；"
                             "调大则场景在画面中变小、留白更多")
    parser.add_argument("--view-forward-m", type=float, default=None,
                        help=f"focal point 沿 +x 方向偏移 m (默认 {OccViewer.DEFAULT_FP_FORWARD_M})；"
                             "调大让自车从画面正中下沉到画面底部")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = QtWidgets.QApplication(sys.argv)

    backend = Backend(
        args.config, args.checkpoint, solver=args.solver,
        val_info_path_override=args.val_info_path,
        fast_pred_root=args.fast_pred_root,
        fast_pred_filename=args.fast_pred_filename,
        fast_pred_key=args.fast_pred_key,
        strict_fast_pred=args.strict_fast_pred,
    )
    if len(backend) == 0:
        raise RuntimeError("dataset 为空，检查 config 的 val_info_path 是否正确。")
    print(f"[viewer] dataset size = {len(backend)}, device = {backend.device}")

    idx_list: list[int] | None = None
    if args.idx_list:
        idx_list = load_idx_list(args.idx_list)
        print(f"[viewer] idx-list loaded: {len(idx_list)} samples from {args.idx_list}")

    viewer = OccViewer(
        backend, start_idx=args.start_idx, idx_list=idx_list,
        view_az=args.view_az, view_el=args.view_el,
        view_dist_ratio=args.view_dist_ratio, view_forward_m=args.view_forward_m,
    )
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
