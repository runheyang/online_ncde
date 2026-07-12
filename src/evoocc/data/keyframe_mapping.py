"""key_frame 解析工具。"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from nuscenes import NuScenes


def load_sample_token_set_from_sweep(sweep_info_path: str) -> set[str]:
    """从 sweep pkl 读取当前 split 的 keyframe sample token 集合。"""
    path = Path(sweep_info_path)
    if not path.exists():
        raise FileNotFoundError(f"sweep pkl 不存在: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    infos = payload["infos"] if isinstance(payload, dict) and "infos" in payload else payload
    sample_tokens = {str(info.get("token", "")) for info in infos}
    sample_tokens.discard("")
    return sample_tokens


class NuScenesKeyFrameResolver:
    """把帧级 sample_data token 映射为 key_frame 的 sample token。"""

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-trainval",
        sweep_info_path: str | None = None,
    ) -> None:
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self._cache: dict[str, tuple[bool, str]] = {}
        self._valid_sample_tokens: set[str] | None = None
        if sweep_info_path:
            self._valid_sample_tokens = load_sample_token_set_from_sweep(sweep_info_path)

    def resolve_keyframe_sample_token(self, frame_token: str) -> str | None:
        """返回 frame_token 对应的 key_frame sample_token；若非 key_frame 则返回 None。"""
        if not frame_token:
            return None
        if frame_token not in self._cache:
            try:
                sample_data = self.nusc.get("sample_data", frame_token)
            except Exception:
                self._cache[frame_token] = (False, "")
                sample_data = None
            is_key_frame = bool(sample_data.get("is_key_frame", False)) if sample_data else False
            sample_token = str(sample_data.get("sample_token", "")) if sample_data else ""
            self._cache[frame_token] = (is_key_frame, sample_token)

        is_key_frame, sample_token = self._cache[frame_token]
        if not is_key_frame or not sample_token:
            return None
        if self._valid_sample_tokens is not None and sample_token not in self._valid_sample_tokens:
            return None
        return sample_token

    def resolve_keyframe_steps(self, frame_tokens: Iterable[str]) -> dict[int, str]:
        """输入整段 frame_tokens，输出 step 索引到 GT sample_token 的映射。"""
        step_to_sample_token: dict[int, str] = {}
        for step_idx, frame_token in enumerate(frame_tokens):
            sample_token = self.resolve_keyframe_sample_token(str(frame_token))
            if sample_token is not None:
                step_to_sample_token[step_idx] = sample_token
        return step_to_sample_token
