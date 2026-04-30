#!/usr/bin/env python3
"""Online NCDE 交互式 Occupancy 可视化工具。

六面板对比（2x3 布局）。约定演化 2s 的 evolve_infos pkl，
evolve_keyframe_sample_tokens 长度 = max_evolve+1，从老到新：
  index 0 = -2s（即 dataset 默认 "anchor slow" 对应的 keyframe）
  index 1 = -1s
  index -1 = curr（与 sample.token 一致）

第 1 行（curr 相关）：
  - GT (curr)                : 当前 keyframe ground truth（不应用 camera mask，完整显示）
  - NCDE Aligned (curr)      : aligner 演化到当前帧的输出
  - Fast (curr frame)        : fast_logits[-1]，当前帧 fast 预测
第 2 行（slow 历史，按时间从老到新；统一到 curr ego frame）：
  - Slow (-2s keyframe)      : 倒数第 5 个 keyframe（2s 前），warp 到 curr ego
  - Slow (-1s keyframe)      : 倒数第 3 个 keyframe（1s 前），warp 到 curr ego
  - Slow (curr keyframe)     : 倒数第 1 个 keyframe，本身在 curr ego，无需 warp

所有 6 个面板共享同一坐标系：同一 voxel 索引 (i,j,k) 对应同一世界点，
直接像素级对比 fast 漏检 → aligned 是否补出 / 过期 slow 是否被擦除。
warp 越界（旧视角的某点跑出 curr pc_range）取 free_index。

用法:
    python scripts/interactive_occ_viewer.py \
        --config configs/online_ncde/fast_alocc2dmini__slow_alocc3d/base.yaml \
        --checkpoint ckpts/.../epoch_9.pth

操作:
    - 顶部 [◀ N/total ▶] 切样本，立即刷新 GT/Fast/3 路 Slow（不跑模型）
    - [运行 NCDE 对齐] 触发一次 forward，刷新 Aligned 面板
    - 6 个视图相机自动同步（鼠标松开后）
    - [保存大图] 把 6 个面板拼成 2x3 PNG 存盘
"""
from __future__ import annotations

import argparse
import json
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

# 2x3 布局，按行展开顺序
PANEL_KEYS = ["gt", "aligned", "fast", "slow_m2", "slow_m1", "slow_curr"]
PANEL_NAMES = [
    "GT (curr)",
    "NCDE Aligned (curr)",
    "Fast (curr frame)",
    "Slow (-2s keyframe)",
    "Slow (-1s keyframe)",
    "Slow (curr keyframe)",
]
# 对应 evolve_keyframe_sample_tokens 索引（slow_curr 走 sample.token）
SLOW_HIST_KEYS = ("slow_m2", "slow_m1")
SLOW_HIST_KF_OFFSETS = {"slow_m2": 0, "slow_m1": 1}  # tokens[0] = -2s, tokens[1] = -1s


# ────────────────────────── 工具：voxel 坐标系 warp ──────────────────────────


def warp_labels_to_ego(
    labels_src: np.ndarray,
    T_src_to_dst: np.ndarray,
    pc_range: tuple,
    voxel_size: tuple,
    free_index: int,
) -> np.ndarray:
    """把 src ego frame 下的 voxel 类别图重采样到 dst ego frame。

    - labels_src : (X, Y, Z) int 类别（未占据 = free_index）
    - T_src_to_dst : (4,4) src_ego → dst_ego 的 SE(3) 变换
    - pc_range : (x_min,y_min,z_min,x_max,y_max,z_max)，src/dst 共用
    - voxel_size : (vx,vy,vz)
    - 越界点取 free_index；nearest neighbor，避免给离散类插出虚假类别。
    """
    X, Y, Z = labels_src.shape
    vx, vy, vz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    # 在 dst grid 里枚举每个 voxel 中心
    ii, jj, kk = np.meshgrid(
        np.arange(X), np.arange(Y), np.arange(Z), indexing="ij",
    )
    px = x_min + (ii + 0.5) * vx
    py = y_min + (jj + 0.5) * vy
    pz = z_min + (kk + 0.5) * vz

    # 反向变换：dst voxel center → src ego 坐标
    T_dst_to_src = np.linalg.inv(T_src_to_dst).astype(np.float64)
    p = np.stack([px, py, pz, np.ones_like(px)], axis=-1).astype(np.float64)  # (X,Y,Z,4)
    p_src = p @ T_dst_to_src.T

    # src ego 坐标 → src grid 索引
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


# ────────────────────────── 工具：idx 列表加载 ──────────────────────────


def load_idx_list(path: str) -> list[int]:
    """按文件后缀加载 idx 列表，返回 list[int]（保持文件顺序，去重保留先后）。

    支持的格式：
      - .txt  每行非 # 开头、空白分隔，第一列必须是 int idx；
              （兼容 tests/online_ncde/find_top_*.py 输出的格式：
               idx \\t fast_miou \\t aligned_miou \\t diff \\t scene \\t end_token）
      - .json 顶层接受三种 schema：
          1. [120, 543, 22, ...]
          2. {"indices": [120, 543, 22, ...], ...其他元信息可有可无...}
          3. {"samples": [{"idx": 120, ...}, {"idx": 543, ...}], ...}
              （字段名也接受 "items" 替代 "samples"）
    """
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

    # 去重：保留首次出现顺序
    seen: set[int] = set()
    out: list[int] = []
    for i in raw_idx:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


# ────────────────────────── Backend：模型 + 数据 ──────────────────────────


@dataclass
class SampleData:
    """单个样本预处理后供渲染用的体素 + 元信息。"""
    gt: np.ndarray
    gt_mask: np.ndarray
    fast: np.ndarray
    slow_curr: np.ndarray              # 当前 keyframe slow
    slow_m1: np.ndarray | None         # -1s keyframe slow（缺失时 None）
    slow_m2: np.ndarray | None         # -2s keyframe slow（缺失时 None）
    # -1s/-2s keyframe → curr 的 SE(3) 变换（4x4），用于在 curr ego frame 下
    # 标出过去时刻自车的位置/朝向。缺失或没做 warp 时为 None。
    slow_m1_T_kf_to_curr: np.ndarray | None
    slow_m2_T_kf_to_curr: np.ndarray | None
    scene_name: str
    token: str
    rollout_start_step: int
    evolve_keyframe_sample_tokens: list[str]


class Backend:
    """封装 dataset / 模型加载 / 单样本前向。"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        solver: str = "euler",
        val_info_path_override: str | None = None,
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
            # viewer 是纯可视化，不算 loss/mIoU；放行 evolve_infos schema
            eval_only_mode=True,
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
        并 argmax 出 GT / Fast / 三路 Slow 体素。

        keyframe 列表来源（兼容两种 pkl schema）：
          - evolve_infos pkl: 用 info["evolve_keyframe_sample_tokens"]
              [0]=K_j (start anchor)、[-1]=K_{j+max_evolve} (end, aligner 输出帧)
          - canonical pkl  : 用 info["keyframe_sample_tokens"]
              [0]=最老历史、[-1]=curr (当前 keyframe)

        两种 schema 下都按"末尾倒数"取 slow keyframe：
          - 末尾[-1] = curr/end             → "Slow (curr keyframe)"
          - 末尾[-3] = 倒退 2 个 keyframe    → "Slow (-1s keyframe)"  （keyframe ~0.5s）
          - 末尾[-5] = 倒退 4 个 keyframe    → "Slow (-2s keyframe)"

        坐标系：所有 6 个面板都对齐到 curr/end keyframe 的 ego frame。
        - GT / Fast / Aligned / Slow(curr) 本身就在 curr ego；
        - Slow(-1s)/Slow(-2s) 通过 frame_ego2global 反向重采样到 curr ego。
          越界点取 free_index，与"自车在 -1s 时这块 curr 视野超出 src pc_range"语义一致。
        evolve pkl 里 evolve_keyframe_step_indices 给出每个 keyframe 在帧序列中的
        step，frame_ego2global 同步抽帧后用同一索引就能拿到 ego2global。
        """
        sample = self.dataset[idx]
        self._raw_sample = sample

        gt = sample["gt_labels"].cpu().numpy().astype(np.int32)
        gt_mask = sample["gt_mask"].cpu().numpy().astype(np.float32)
        fast_logits = sample["fast_logits"]   # (T, C, X, Y, Z)
        fast = fast_logits[-1].argmax(0).cpu().numpy().astype(np.int32)

        meta = sample.get("meta", {})
        scene_name = str(meta.get("scene_name", ""))
        token = str(meta.get("token", ""))

        # 直接从原始 info 读 keyframe token 列表，兼容 canonical / evolve schema
        info = self.dataset.infos[idx]
        ek_tokens = [str(t) for t in info.get("evolve_keyframe_sample_tokens", [])]
        ek_step_indices: list[int] = []
        if ek_tokens:
            ek_step_indices = [int(s) for s in meta.get("evolve_keyframe_step_indices", [])]
        if not ek_tokens:
            ek_tokens = [str(t) for t in info.get("keyframe_sample_tokens", [])]

        # frame_ego2global 已按 fast_frame_stride 抽帧；ek_step_indices 也同步重映射
        frame_ego2global = sample["frame_ego2global"].cpu().numpy()  # (T_sub, 4, 4)

        def _load_slow_for(kf_token: str) -> np.ndarray:
            rel = f"{scene_name}/{kf_token}/logits.npz"
            logits = self.logits_loader.load_slow_logits(
                {"slow_logit_path": rel}, torch.device("cpu"),
            )
            return logits.argmax(0).numpy().astype(np.int32)

        # 取 keyframe token + 同位置的 step（用于查 ego2global）。
        # ek_step_indices 缺失（canonical pkl 没这个字段）时，对应的 warp 跳过。
        def _pick(neg_idx: int) -> tuple[str | None, int | None]:
            if len(ek_tokens) < -neg_idx:
                return None, None
            tok = ek_tokens[neg_idx]
            if not tok:
                return None, None
            step = ek_step_indices[neg_idx] if len(ek_step_indices) >= -neg_idx else None
            return tok, step

        curr_kf, curr_step = _pick(-1)
        m1_kf, m1_step = _pick(-3)
        m2_kf, m2_step = _pick(-5)

        # curr 不需要 warp（它就是 dst ego frame）
        slow_curr = _load_slow_for(curr_kf) if curr_kf else _load_slow_for(token)

        def _load_and_warp(
            kf_token: str | None, src_step: int | None,
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
            """加载 src keyframe 的 slow 并 warp 到 curr ego frame。
            返回 (warped_labels, T_src_to_curr)；缺少 step（如 canonical pkl）
            时退化为不 warp 直接返回 labels 和 None 矩阵。"""
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
            scene_name=scene_name, token=token,
            rollout_start_step=int(sample["rollout_start_step"].item()),
            evolve_keyframe_sample_tokens=ek_tokens,
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
    # 默认视角调成论文风：
    #  - elevation 35° 偏侧（旧 55° 太俯视，会"压平"场景）
    #  - azimuth -60°（旧 -75° 太侧，朝向不够斜对角）
    #  - dist_ratio 2.2（旧 1.6 太近，场景塞满 panel；论文里留白多）
    #  - forward_m 10m：focal point 沿 +x（车头方向）推 10m，让自车出现在
    #    画面下半部，前方道路占大头，更像 SurroundOcc / OccFormer 论文图
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
        self.current_idx = 0  # 当前加载样本的全集 idx
        self.current_sample: SampleData | None = None
        self._syncing_camera = False
        self._panels: dict[str, MayaviPanel] = {}
        self._ego_actors: dict[str, list] = {}

        # idx 列表模式：spinbox / ◀▶ 都按 list 位置走
        # 不传则退化为全集顺序 [0..N-1]
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

        # 视角参数（CLI 覆盖默认值）
        self._view_az = self.DEFAULT_AZ if view_az is None else float(view_az)
        self._view_el = self.DEFAULT_EL if view_el is None else float(view_el)
        self._view_dist_ratio = (
            self.DEFAULT_DIST_RATIO if view_dist_ratio is None else float(view_dist_ratio)
        )
        self._view_forward_m = (
            self.DEFAULT_FP_FORWARD_M if view_forward_m is None else float(view_forward_m)
        )

        self.setWindowTitle("Online NCDE Occupancy Viewer")
        self.resize(1800, 1000)

        self._build_toolbar()
        self._build_panels()
        self._build_statusbar()

        # 配置完控件后再首次加载样本
        QtCore.QTimer.singleShot(50, self._init_after_show)

    # ---------- UI 构建 ----------
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

        self.run_btn = QtWidgets.QPushButton("▶ 运行 NCDE 对齐")
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
        # 2 行 3 列：第 1 行 = curr 三联，第 2 行 = slow 历史三联
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
        # scene 真正初始化后再设置背景 + 注册同步 + 加载首样本
        for key in PANEL_KEYS:
            scene_obj = self._panels[key].scene
            scene_obj.background = (1.0, 1.0, 1.0)
            self._install_camera_sync(key, scene_obj)
        self._load_by_list_pos(self._list_pos)

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
        x_min, y_min, z_min, x_max, y_max, z_max = self.backend.pc_range
        # focal point 沿 +x（车头方向）偏移 view_forward_m，让自车出现在画面
        # 下半部、前方道路占主区域，构图更像论文图
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

    # ---------- 数据加载 + 渲染 ----------
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

        # 渲染 GT / Fast / 3 路 Slow；Aligned 清空，等用户按按钮
        # GT 不应用 camera mask，完整显示（其他面板沿用其各自的 mask 或不 mask）
        self._render_panel("gt", sample.gt)
        self._render_panel("fast", sample.fast)
        self._render_panel("slow_curr", sample.slow_curr)
        self._panels["slow_curr"].set_title("Slow (curr keyframe)")

        for key in SLOW_HIST_KEYS:
            voxel = sample.slow_m2 if key == "slow_m2" else sample.slow_m1
            base_title = "Slow (-2s keyframe)" if key == "slow_m2" else "Slow (-1s keyframe)"
            if voxel is None:
                self._clear_panel(key)
                self._panels[key].set_title(f"{base_title} (N/A)")
            else:
                self._render_panel(key, voxel)
                self._panels[key].set_title(base_title)

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
        self._draw_ego_marker(key)

    def _clear_panel(self, key: str) -> None:
        clear_figure(self._panels[key].scene.mayavi_scene)
        # clear_figure 已经清掉 ego marker，丢弃悬挂引用
        self._ego_actors.pop(key, None)

    def _draw_ego_marker(self, key: str) -> None:
        """画红色球 + 朝车头方向的红色箭头，标该 panel 对应时刻的自车。

        - GT/Aligned/Fast/Slow(curr): 自车 = curr ego 原点 (0, 0)
        - Slow(-1s)/Slow(-2s): 自车 = 那一刻自车原点 warp 到 curr ego frame
          位置 = T_kf_to_curr[:3, 3]
          朝向 = T_kf_to_curr[:3, :3] @ +x = T_kf_to_curr[:3, 0]
        marker 在 z 上抬高 1.5m 避免被路面 voxel 遮挡。"""
        from mayavi import mlab
        fig = self._panels[key].scene.mayavi_scene

        # 默认：当前 panel 的对应时刻 = curr，自车在 ego 原点 + 朝 +x
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
        # 新画的 marker 也要遵循当前 toggle 状态（切样本时不会突然又显示出来）
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
        """用 mlab.screenshot 抓每个 panel 的 VTK 渲染图，
        再用 PIL 加文字注释拼成 2x3 大图。

        布局（自上而下）：
          [ 顶部 header: 样本元信息 ]
          [ 行 1: GT | NCDE Aligned | Fast ]
          [ 行 2: Slow(-2s) | Slow(-1s) | Slow(curr) ]
          [ 底部 footer: 坐标系说明 ]
        每个 panel 上方有粗体标题条，标题取 self._panels[k].title_label
        的实时文本（自动反映 "(N/A)" / "未运行" 等状态）。
        """
        from mayavi import mlab
        from PIL import Image, ImageDraw, ImageFont

        # 1) 抓 6 张 panel 截图（保存时隐藏 ego marker，截完恢复）
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

        # 2) 字体（系统找不到 DejaVu 时退化到 PIL 默认 bitmap font）
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

        # 3) 顶部 header：样本元信息
        sample = self.current_sample
        header_text = (
            f"idx={self.current_idx}/{len(self.backend) - 1}   "
            f"scene={sample.scene_name if sample else ''}   "
            f"token={(sample.token[:16] + '...') if (sample and sample.token) else ''}   "
            f"rss={sample.rollout_start_step if sample else ''}"
        )
        draw.rectangle([0, 0, total_w, header_h], fill=(40, 40, 40))
        draw.text((10, 6), header_text, font=font_header, fill=(230, 230, 230))

        # 4) 6 个 panel：标题条 + 截图
        def _measure(text: str, font) -> tuple[int, int]:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:  # Pillow < 9.2
                return draw.textsize(text, font=font)

        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for a, key, (r, c) in zip(arrs, PANEL_KEYS, positions):
            x0_panel = c * pw
            y0_row = header_h + r * row_h
            y0_img = y0_row + title_h

            # 标题条（浅灰底 + 深色字）
            title_text = self._panels[key].title_label.text()
            draw.rectangle(
                [x0_panel, y0_row, x0_panel + pw, y0_row + title_h],
                fill=(235, 235, 235),
            )
            tw, th = _measure(title_text, font_title)
            tx = x0_panel + (pw - tw) // 2
            ty = y0_row + (title_h - th) // 2 - 2
            draw.text((tx, ty), title_text, font=font_title, fill=(20, 20, 20))

            # 贴截图（在 pw x ph 槽内居中）
            ah, aw, _ = a.shape
            ix = x0_panel + (pw - aw) // 2
            iy = y0_img + (ph - ah) // 2
            canvas.paste(Image.fromarray(a), (ix, iy))

        # 5) 底部 footer：坐标系说明
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


# ────────────────────────── CLI ──────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online NCDE 交互式可视化")
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
                             ".txt 即 tests/online_ncde/find_top_*.py 的输出格式；"
                             ".json 接受 [int...] / {indices: [...]} / {samples: [{idx,...}]}")
    # 视角微调（不传走默认值；改完后点 [↻ 重置视角] 立即生效）
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

    # PyQt5 在 Wayland 下默认走 XWayland，正常情况无需干预；
    # 万一弹不出窗口，可在外面 export QT_QPA_PLATFORM=xcb
    app = QtWidgets.QApplication(sys.argv)

    backend = Backend(
        args.config, args.checkpoint, solver=args.solver,
        val_info_path_override=args.val_info_path,
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
