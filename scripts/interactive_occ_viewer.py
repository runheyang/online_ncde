#!/usr/bin/env python3
"""Online NCDE 交互式 Occupancy 可视化工具。

四面板对比（全部对应当前 token）：
  - GT (curr, camera-masked) : 当前 keyframe 的 ground truth
  - NCDE Aligned (curr)      : aligner 演化到当前帧的输出
  - Fast (curr frame)        : fast_logits[-1]，当前帧 fast 系统预测
  - Slow (curr keyframe)     : 当前 keyframe 的 slow logits（额外加载，
                                与 dataset 默认的 "anchor slow" 不同——
                                后者是 2s 前最老 keyframe，仅供 aligner 输入）

用法:
    python scripts/interactive_occ_viewer.py \
        --config configs/online_ncde/fast_alocc2dmini__slow_alocc3d/base.yaml \
        --checkpoint ckpts/.../epoch_9.pth

操作:
    - 顶部 [◀ N/total ▶] 切样本，会立即刷新 GT/Fast/Slow（不跑模型）
    - [运行 NCDE 对齐] 触发一次 forward，刷新 Aligned 面板
    - 4 个视图相机自动同步（鼠标松开后）
    - [保存大图] 把 4 个面板拼成一张 PNG 存盘
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Mayavi / Qt 后端环境（必须在 import mayavi 之前）
os.environ.setdefault("ETS_TOOLKIT", "qt")
os.environ.setdefault("QT_API", "pyqt5")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from online_ncde.config import load_config_with_base                      # noqa: E402
from online_ncde.data.build_logits_loader import build_logits_loader      # noqa: E402
from online_ncde.data.occ3d_online_ncde_dataset import Occ3DOnlineNcdeDataset  # noqa: E402
from online_ncde.models.online_ncde_aligner import OnlineNcdeAligner       # noqa: E402
from online_ncde.utils.checkpoints import load_checkpoint_for_eval         # noqa: E402
from online_ncde.visualization.occ_renderer import (                       # noqa: E402
    OCC3D_CLASS_NAMES,
    OCC3D_COLORS,
    clear_figure,
    render_voxel_into_figure,
)

# Qt / Mayavi
from PyQt5 import QtCore, QtGui, QtWidgets                                # noqa: E402
from traits.api import HasTraits, Instance                                # noqa: E402
from traitsui.api import View, Item                                       # noqa: E402
from mayavi.tools.mlab_scene_model import MlabSceneModel                  # noqa: E402
from tvtk.pyface.scene_editor import SceneEditor                          # noqa: E402
from mayavi.core.ui.mayavi_scene import MayaviScene                       # noqa: E402

PANEL_NAMES = [
    "GT (curr, camera-masked)",
    "NCDE Aligned (curr)",
    "Fast (curr frame)",
    "Slow (curr keyframe)",
]
PANEL_KEYS = ["gt", "aligned", "fast", "slow"]


# ────────────────────────── Backend：模型 + 数据 ──────────────────────────


@dataclass
class SampleData:
    """单个样本预处理后供渲染用的体素 + 元信息。"""
    gt: np.ndarray
    gt_mask: np.ndarray
    fast: np.ndarray
    slow: np.ndarray
    scene_name: str
    token: str
    rollout_start_step: int


class Backend:
    """封装 dataset / 模型加载 / 单样本前向。"""

    def __init__(self, config_path: str, checkpoint_path: str, solver: str = "euler") -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.solver = solver

        self.cfg = load_config_with_base(config_path)
        data_cfg = self.cfg["data"]
        self.data_cfg = data_cfg
        self.model_cfg = self.cfg["model"]
        self.eval_cfg = self.cfg.get("eval", {})

        self.free_index = int(data_cfg["free_index"])
        self.num_classes = int(data_cfg["num_classes"])
        self.pc_range = tuple(data_cfg["pc_range"])
        self.voxel_size_xyz = tuple(data_cfg["voxel_size"])
        self.voxel_size_iso = float(self.voxel_size_xyz[0])  # 渲染用各向同性

        # dataset：跟 eval_online_ncde.py 默认 min_history_completeness=0
        # logits_loader 暴露给后续用：dataset 默认取的 slow 是 anchor（最老 keyframe），
        # 我们额外用它加载当前 keyframe 的 slow 给 GUI 显示
        self.logits_loader = build_logits_loader(data_cfg, self.cfg["root_path"])
        logits_loader = self.logits_loader
        self.dataset = Occ3DOnlineNcdeDataset(
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
        )

        self.device = torch.device(
            self.eval_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        # 模型懒加载
        self._model: OnlineNcdeAligner | None = None
        self._raw_sample = None  # 缓存当前样本（含 logits / pose / timestamps）

    # ---------- dataset ----------
    def __len__(self) -> int:
        return len(self.dataset)

    def load_sample(self, idx: int) -> SampleData:
        """读 dataset[idx]，缓存原始 logits 供 aligner 使用，
        并 argmax 出 GT / Fast / 当前 keyframe Slow 体素。"""
        sample = self.dataset[idx]
        self._raw_sample = sample

        gt = sample["gt_labels"].cpu().numpy().astype(np.int32)
        gt_mask = sample["gt_mask"].cpu().numpy().astype(np.float32)
        fast_logits = sample["fast_logits"]   # (T, C, X, Y, Z)
        fast = fast_logits[-1].argmax(0).cpu().numpy().astype(np.int32)

        meta = sample.get("meta", {})
        scene_name = str(meta.get("scene_name", ""))
        token = str(meta.get("token", ""))

        # 当前 keyframe 的 slow logits：dataset 默认给的是 anchor (oldest keyframe)
        # 这里 fake 一个 info dict 调 loader，路径模板见 gen_online_ncde_canonical_infos.py
        # 走 CPU：loader 内部预计算索引在 CPU，不上 GPU 避免 mixed device
        slow_curr_rel = f"{scene_name}/{token}/logits.npz"
        slow_curr_logits = self.logits_loader.load_slow_logits(
            {"slow_logit_path": slow_curr_rel}, torch.device("cpu"),
        )
        slow = slow_curr_logits.argmax(0).numpy().astype(np.int32)

        return SampleData(
            gt=gt, gt_mask=gt_mask, fast=fast, slow=slow,
            scene_name=scene_name, token=token,
            rollout_start_step=int(sample["rollout_start_step"].item()),
        )

    # ---------- model ----------
    def ensure_model(self) -> OnlineNcdeAligner:
        if self._model is not None:
            return self._model
        m = OnlineNcdeAligner(
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
        """对当前缓存样本跑一次 forward，返回 aligned 体素 (X, Y, Z) int。"""
        if self._raw_sample is None:
            raise RuntimeError("先调用 load_sample 再 run_aligner")
        m = self.ensure_model()
        s = self._raw_sample
        fast = s["fast_logits"].to(self.device).unsqueeze(0)      # (1, T, C, X, Y, Z)
        slow = s["slow_logits"].to(self.device).unsqueeze(0)      # (1, C, X, Y, Z)
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
        aligned = out["aligned"][0]  # (C, X, Y, Z)
        return aligned.argmax(0).cpu().numpy().astype(np.int32)


# ────────────────────────── Mayavi widget 包装 ──────────────────────────


class _SceneHolder(HasTraits):
    """traits 容器，每个面板一个，提供给 SceneEditor 渲染。"""
    scene = Instance(MlabSceneModel, ())
    view = View(
        Item("scene", editor=SceneEditor(scene_class=MayaviScene),
             height=300, width=400, show_label=False),
        resizable=True,
    )


class MayaviPanel(QtWidgets.QWidget):
    """带标题的 Mayavi widget。"""

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


# ────────────────────────── 主窗口 ──────────────────────────


class OccViewer(QtWidgets.QMainWindow):
    DEFAULT_AZ = -75.0
    DEFAULT_EL = 55.0

    def __init__(self, backend: Backend, start_idx: int = 0) -> None:
        super().__init__()
        self.backend = backend
        self.current_idx = 0
        self.current_sample: SampleData | None = None
        self._syncing_camera = False  # 防止相机同步死循环
        self._panels: dict[str, MayaviPanel] = {}
        self._start_idx = max(0, min(start_idx, len(backend) - 1))

        self.setWindowTitle("Online NCDE Occupancy Viewer")
        self.resize(1600, 1000)

        self._build_toolbar()
        self._build_panels()
        self._build_statusbar()

        # 配置完控件后再首次加载样本
        QtCore.QTimer.singleShot(50, self._init_after_show)

    # ---------- UI 构建 ----------
    def _build_toolbar(self) -> None:
        tb = self.addToolBar("main")
        tb.setMovable(False)

        self.prev_btn = QtWidgets.QPushButton("◀")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        tb.addWidget(self.prev_btn)

        self.idx_box = QtWidgets.QSpinBox()
        self.idx_box.setRange(0, max(0, len(self.backend) - 1))
        self.idx_box.editingFinished.connect(self._on_idx_box_changed)
        tb.addWidget(self.idx_box)

        self.total_label = QtWidgets.QLabel(f" / {len(self.backend) - 1}")
        tb.addWidget(self.total_label)

        self.next_btn = QtWidgets.QPushButton("▶")
        self.next_btn.clicked.connect(lambda: self._step(+1))
        tb.addWidget(self.next_btn)

        tb.addSeparator()

        self.run_btn = QtWidgets.QPushButton("▶ 运行 NCDE 对齐")
        self.run_btn.clicked.connect(self._on_run_aligner)
        tb.addWidget(self.run_btn)

        self.reset_view_btn = QtWidgets.QPushButton("↻ 重置视角")
        self.reset_view_btn.clicked.connect(self._reset_views)
        tb.addWidget(self.reset_view_btn)

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
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for key, name, pos in zip(PANEL_KEYS, PANEL_NAMES, positions):
            panel = MayaviPanel(name)
            self._panels[key] = panel
            grid.addWidget(panel, *pos)
        for r in (0, 1):
            grid.setRowStretch(r, 1)
        for c in (0, 1):
            grid.setColumnStretch(c, 1)
        self.setCentralWidget(central)

    def _build_statusbar(self) -> None:
        self.status = self.statusBar()
        self.status.showMessage("ready")

    def _init_after_show(self) -> None:
        # scene 真正初始化后再设置背景 + 注册同步 + 加载首样本
        for key in PANEL_KEYS:
            scene_obj = self._panels[key].scene
            scene_obj.background = (1.0, 1.0, 1.0)
            self._install_camera_sync(key, scene_obj)
        self._load_sample(self._start_idx)

    # ---------- 相机同步 ----------
    def _install_camera_sync(self, src_key: str, scene_obj: MlabSceneModel) -> None:
        """在每个 scene 的 interactor 上注册 EndInteractionEvent 回调。"""
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
        # 用 GT 体素估场景中心和距离，再设到所有 4 个 scene
        v = self.current_sample.gt
        x_min, y_min, z_min, x_max, y_max, z_max = self.backend.pc_range
        fp = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)
        extent = max(x_max - x_min, y_max - y_min)
        distance = float(extent) * 1.6
        self._syncing_camera = True
        try:
            for k in PANEL_KEYS:
                from mayavi import mlab
                mlab.view(
                    azimuth=self.DEFAULT_AZ,
                    elevation=self.DEFAULT_EL,
                    distance=distance,
                    focalpoint=fp,
                    figure=self._panels[k].scene.mayavi_scene,
                )
        finally:
            self._syncing_camera = False

    # ---------- 数据加载 + 渲染 ----------
    def _step(self, delta: int) -> None:
        new_idx = max(0, min(len(self.backend) - 1, self.current_idx + delta))
        if new_idx == self.current_idx:
            return
        self._load_sample(new_idx)

    def _on_idx_box_changed(self) -> None:
        new_idx = int(self.idx_box.value())
        if new_idx == self.current_idx:
            return
        self._load_sample(new_idx)

    def _load_sample(self, idx: int) -> None:
        self.status.showMessage(f"loading sample {idx} ...")
        QtWidgets.QApplication.processEvents()
        try:
            sample = self.backend.load_sample(idx)
        except Exception as e:  # 数据集偶发损坏样本
            self.status.showMessage(f"load failed: {e}")
            return

        self.current_idx = idx
        self.current_sample = sample
        self.idx_box.blockSignals(True)
        self.idx_box.setValue(idx)
        self.idx_box.blockSignals(False)

        self.meta_label.setText(
            f"  scene={sample.scene_name}  token={sample.token[:12]}…  "
            f"rss={sample.rollout_start_step}"
        )

        # 渲染 GT / Fast / Slow；Aligned 清空，等用户按按钮
        self._render_panel("gt", sample.gt, mask=sample.gt_mask)
        self._render_panel("fast", sample.fast)
        self._render_panel("slow", sample.slow)
        self._clear_panel("aligned")
        self._panels["aligned"].set_title("NCDE Aligned (未运行)")

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

    def _clear_panel(self, key: str) -> None:
        clear_figure(self._panels[key].scene.mayavi_scene)

    # ---------- 按钮：运行 aligner ----------
    def _on_run_aligner(self) -> None:
        if self.current_sample is None:
            return
        self.run_btn.setEnabled(False)
        self.status.showMessage("running NCDE aligner ...")
        QtWidgets.QApplication.processEvents()
        try:
            aligned = self.backend.run_aligner()
            self._render_panel("aligned", aligned)
            self._panels["aligned"].set_title("NCDE Aligned")
            self.status.showMessage("NCDE aligner done")
        except Exception as e:
            self.status.showMessage(f"aligner failed: {e}")
            QtWidgets.QMessageBox.critical(self, "NCDE 对齐失败", str(e))
        finally:
            self.run_btn.setEnabled(True)

    # ---------- 按钮：保存大图 ----------
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
        """用 mlab.screenshot 抓 VTK render window，比 QWidget.grab 在
        Wayland/OpenGL 下稳得多。
        """
        from mayavi import mlab

        arrs: list[np.ndarray] = []
        for k in PANEL_KEYS:
            scene_obj = self._panels[k].scene
            scene_obj.scene.render()
            QtWidgets.QApplication.processEvents()
            arr = mlab.screenshot(
                figure=scene_obj.mayavi_scene, mode="rgb", antialiased=True,
            )
            arrs.append(np.ascontiguousarray(arr))

        h = max(a.shape[0] for a in arrs)
        w = max(a.shape[1] for a in arrs)
        big = np.full((h * 2, w * 2, 3), 255, dtype=np.uint8)
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # GT / Aligned / Fast / Slow
        for a, (r, c) in zip(arrs, positions):
            ah, aw, _ = a.shape
            y0 = r * h + (h - ah) // 2
            x0 = c * w + (w - aw) // 2
            big[y0:y0 + ah, x0:x0 + aw] = a

        big = np.ascontiguousarray(big)
        qimg = QtGui.QImage(
            big.data, big.shape[1], big.shape[0], big.shape[1] * 3,
            QtGui.QImage.Format_RGB888,
        ).copy()  # copy 让 QImage 不再引用 big.data
        if not qimg.save(path, "PNG"):
            raise RuntimeError(f"QImage.save 失败：{path}")


# ────────────────────────── CLI ──────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online NCDE 交互式可视化")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--solver", choices=["heun", "euler"], default="euler")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="启动时跳转到的样本索引")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # PyQt5 在 Wayland 下默认走 XWayland，正常情况无需干预；
    # 万一弹不出窗口，可在外面 export QT_QPA_PLATFORM=xcb
    app = QtWidgets.QApplication(sys.argv)

    backend = Backend(args.config, args.checkpoint, solver=args.solver)
    if len(backend) == 0:
        raise RuntimeError("dataset 为空，检查 config 的 val_info_path 是否正确。")
    print(f"[viewer] dataset size = {len(backend)}, device = {backend.device}")

    viewer = OccViewer(backend, start_idx=args.start_idx)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
