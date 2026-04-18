"""online_ncde 训练与评估流程。"""

from __future__ import annotations

from typing import Any, Dict, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from online_ncde.losses import resize_labels_and_mask_to_logits
from online_ncde.metrics import MetricMiouOcc3D, apply_free_threshold
from online_ncde.utils.checkpoints import save_checkpoint as _save_checkpoint
from online_ncde.utils.ema import ModelEMA

try:
    import progressbar
    from online_ncde.utils.progress import make_pbar
except Exception:  # pragma: no cover
    progressbar = None
    make_pbar = None


def _stack_or_none(batch: list, key: str) -> torch.Tensor | None:
    vals = [item.get(key, None) for item in batch]
    if any(v is None for v in vals):
        return None
    return torch.stack(cast(list[torch.Tensor], vals), dim=0)


def online_ncde_collate(batch):
    """组 batch，并处理可选时序字段。"""
    fast_logits = torch.stack([item["fast_logits"] for item in batch], dim=0)
    slow_logits = torch.stack([item["slow_logits"] for item in batch], dim=0)
    frame_ego2global = torch.stack([item["frame_ego2global"] for item in batch], dim=0)
    gt_labels = torch.stack([item["gt_labels"] for item in batch], dim=0)
    gt_mask = torch.stack([item["gt_mask"] for item in batch], dim=0)

    return {
        "fast_logits": fast_logits,
        "slow_logits": slow_logits,
        "frame_ego2global": frame_ego2global,
        "frame_timestamps": _stack_or_none(batch, "frame_timestamps"),
        "frame_dt": _stack_or_none(batch, "frame_dt"),
        "gt_labels": gt_labels,
        "gt_mask": gt_mask,
        "sup_labels": _stack_or_none(batch, "sup_labels"),
        "sup_masks": _stack_or_none(batch, "sup_masks"),
        "sup_step_indices": _stack_or_none(batch, "sup_step_indices"),
        "sup_valid_mask": _stack_or_none(batch, "sup_valid_mask"),
        "ray_gt_dist": _stack_or_none(batch, "ray_gt_dist"),
        "ray_origin": _stack_or_none(batch, "ray_origin"),
        "ray_origin_mask": _stack_or_none(batch, "ray_origin_mask"),
        "ray_sup_valid": _stack_or_none(batch, "ray_sup_valid"),
        "meta": [item.get("meta", {}) for item in batch],
    }


def move_to_device(sample: Dict, device: torch.device) -> Dict:
    """递归搬运 batch 到目标设备。"""

    def _move(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, dict):
            return {k: _move(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_move(v) for v in x]
        return x

    return cast(Dict[str, Any], _move(sample))


class Trainer:
    """封装 train/eval。"""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        num_classes: int,
        free_index: int,
        free_conf_thresh: float | None = None,
        log_interval: int = 10,
        clip_norm: float = 1.0,
        supervision_labels: list[str] | None = None,
        supervision_weights: list[float] | None = None,
        supervision_weight_normalize: bool = True,
        log_multistep_losses: bool = True,
        rollout_mode: str = "full",
        primary_supervision_label: str = "t-1.0",
        stepwise_max_step_index: int | None = None,
        is_main: bool = True,
        ema: ModelEMA | None = None,
        lambda_fast_kl: float = 0.0,
    ) -> None:
        self.model = model
        self.is_main = is_main
        self.ema = ema
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.num_classes = num_classes
        self.free_index = free_index
        self.free_conf_thresh = free_conf_thresh
        self.log_interval = log_interval
        self.clip_norm = clip_norm
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.supervision_labels = supervision_labels or ["t-1.5", "t-1.0", "t-0.5", "t"]
        if supervision_weights is None:
            supervision_weights = [1.0 / len(self.supervision_labels)] * len(self.supervision_labels)
        if len(supervision_weights) != len(self.supervision_labels):
            raise ValueError(
                "supervision_weights 长度必须与 supervision_labels 一致，"
                f"当前 {len(supervision_weights)} vs {len(self.supervision_labels)}"
            )
        weights = [float(w) for w in supervision_weights]
        if supervision_weight_normalize:
            weight_sum = sum(weights)
            if weight_sum <= 0:
                raise ValueError("supervision_weights 求和必须 > 0。")
            weights = [w / weight_sum for w in weights]
        self.supervision_weights = weights
        self.log_multistep_losses = bool(log_multistep_losses)
        self.rollout_mode = str(rollout_mode).strip().lower()
        if self.rollout_mode not in {"full", "one_second_tminus1"}:
            raise ValueError(
                "rollout_mode 仅支持 {'full', 'one_second_tminus1'}，"
                f"当前为 {self.rollout_mode}"
            )
        self.primary_supervision_label = str(primary_supervision_label)
        resolved_step_max = stepwise_max_step_index
        if resolved_step_max is None and self.rollout_mode == "one_second_tminus1":
            resolved_step_max = 6
        if resolved_step_max is not None and int(resolved_step_max) < 0:
            raise ValueError(f"stepwise_max_step_index 必须 >= 0，当前 {resolved_step_max}")
        self.stepwise_max_step_index = (
            None if resolved_step_max is None else int(resolved_step_max)
        )
        self.lambda_fast_kl = float(lambda_fast_kl)
        _underlying = getattr(self.model, "module", self.model)
        if hasattr(_underlying, "_fast_kl_active"):
            _underlying._fast_kl_active = self.lambda_fast_kl > 0.0

    @staticmethod
    def _require_multistep_supervision(sample: Dict[str, Any], context: str) -> None:
        """多帧监督现在是唯一路径，缺字段时立刻报错，提示去补 sidecar。"""
        missing = [
            k for k in ("sup_labels", "sup_masks", "sup_step_indices", "sup_valid_mask")
            if sample.get(k, None) is None
        ]
        if not missing:
            return
        raise RuntimeError(
            f"[{context}] sample 缺少多帧监督字段 {missing}；"
            "Trainer 现已强制走多帧路径，请在 data 配置里补上 "
            "train_supervision_sidecar_path / val_supervision_sidecar_path，"
            "或使用 canonical 带 sup_* 的 info pkl。"
        )

    def _pack_diag(self, diag_list: list[dict[str, torch.Tensor]]) -> Dict[str, float]:
        if not diag_list:
            return {}
        merged: Dict[str, float] = {}
        keys = diag_list[0].keys()
        for key in keys:
            values = [float(item[key].detach().float().item()) for item in diag_list if key in item]
            if values:
                merged[key] = sum(values) / len(values)
        return merged

    @staticmethod
    def _ddp_enabled() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _ddp_stats_device(self) -> torch.device:
        """统计 all_reduce 的设备选择。

        默认双卡训练使用 gloo，统计张量需放在 CPU；若切回 nccl，则放回当前
        rank 的 CUDA 设备。
        """
        if not self._ddp_enabled():
            return self.device
        backend = dist.get_backend()
        if backend == "nccl":
            return self.device
        return torch.device("cpu")

    def _all_reduce_sums(self, values: list[float]) -> list[float]:
        """对一组标量做一次性 sum all_reduce。"""
        if not self._ddp_enabled():
            return values
        stats = torch.tensor(values, dtype=torch.float64, device=self._ddp_stats_device())
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return [float(v) for v in stats.cpu().tolist()]

    def _compute_multistep_loss(
        self,
        step_logits: torch.Tensor,
        step_indices: torch.Tensor,
        sup_labels: torch.Tensor,
        sup_masks: torch.Tensor,
        sup_step_indices: torch.Tensor,
        sup_valid_mask: torch.Tensor,
        ray_gt_dist: torch.Tensor | None = None,
        ray_origin: torch.Tensor | None = None,
        ray_origin_mask: torch.Tensor | None = None,
        ray_sup_valid: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, int]]:
        """按 sidecar 指定 step 做 4 时刻联合监督（不做 detach）。

        ray_* 字段齐全时对每个 sup 额外把该 sup 的 origin/gt_dist/origin_mask 以
        kwargs 形式传给 loss_fn，由 SegAndRayLoss 内部决定是否启用 ray loss 分支。
        数据布局约定：ray_origin (B,sup,K,3)、ray_gt_dist (B,sup,K,R)、
        ray_origin_mask (B,sup,K)。
        """
        step_map = {int(v): i for i, v in enumerate(step_indices.detach().cpu().tolist())}
        num_sup = len(self.supervision_labels)
        if sup_labels.shape[1] != num_sup:
            raise ValueError(
                f"监督时刻数不一致: labels.shape[1]={sup_labels.shape[1]} vs expected={num_sup}"
            )

        # 既要 sidecar 数据齐全，又要 loss_fn 明确声明接受 ray_* kwargs；
        # 否则把 kwargs 透传给普通 seg loss 会直接 TypeError。
        has_ray = (
            ray_gt_dist is not None
            and ray_origin is not None
            and ray_sup_valid is not None
            and getattr(self.loss_fn, "accepts_ray_kwargs", False)
        )

        total = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        total_focal = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        total_aux = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        total_ray = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        total_ray_hit = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        total_ray_empty = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        total_ray_pre_free = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        total_ray_depth = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        ray_sup_count = 0
        per_step_loss: dict[str, float] = {}
        per_step_count: dict[str, int] = {}
        active_any = False

        if has_ray:
            if ray_origin.dim() != 4:
                raise ValueError(
                    f"ray_origin 必须是 (B,sup,K,3)，实际 {tuple(ray_origin.shape)}"
                )
            if ray_origin_mask is None:
                raise ValueError(
                    "has_ray=True 但 ray_origin_mask 缺失；Dataset/collate 未正确产出"
                )

        for sup_i, label in enumerate(self.supervision_labels):
            key = f"loss_{label}"
            valid_rows = sup_valid_mask[:, sup_i] > 0.5
            rows = torch.nonzero(valid_rows, as_tuple=False).reshape(-1).tolist()
            logits_list = []
            labels_list = []
            masks_list = []
            kept_rows: list[int] = []
            for b in rows:
                step_value = int(sup_step_indices[b, sup_i].item())
                local_step = step_map.get(step_value, None)
                if local_step is None:
                    continue
                logits_list.append(step_logits[b, local_step])
                labels_list.append(sup_labels[b, sup_i])
                masks_list.append(sup_masks[b, sup_i])
                kept_rows.append(b)

            if not logits_list:
                per_step_loss[key] = 0.0
                per_step_count[key] = 0
                continue

            logits_i = torch.stack(logits_list, dim=0)
            labels_i = torch.stack(labels_list, dim=0)
            masks_i = torch.stack(masks_list, dim=0)

            loss_kwargs: dict[str, torch.Tensor] = {}
            if has_ray:
                # (B_eff,K,3) / (B_eff,K,R) / (B_eff,K) —— row-mask gather，
                # 一次拿全。ray_sup_valid 为 0 的行对应的 origin_mask 本身就是 0，
                # RayLoss 会自己过滤。
                row_idx = torch.tensor(kept_rows, device=ray_origin.device, dtype=torch.long)
                sup_sel = torch.full_like(row_idx, sup_i)
                loss_kwargs["ray_origins"] = ray_origin[row_idx, sup_sel]
                loss_kwargs["gt_dist"] = ray_gt_dist[row_idx, sup_sel]
                # origin_mask 对 ray_sup_valid=0 的行整段清零，让 RayLoss 早退
                sup_valid_vec = ray_sup_valid[row_idx, sup_sel].to(ray_origin_mask.dtype)
                loss_kwargs["origin_mask"] = (
                    ray_origin_mask[row_idx, sup_sel] * sup_valid_vec.unsqueeze(-1)
                )

            loss_i = self.loss_fn(logits_i, labels_i, masks_i, **loss_kwargs)
            weight = float(self.supervision_weights[sup_i])
            weighted_total = loss_i["total"] * weight
            weighted_focal = loss_i["focal"] * weight
            weighted_aux = loss_i["aux"] * weight
            total = total + weighted_total
            total_focal = total_focal + weighted_focal
            total_aux = total_aux + weighted_aux
            if "ray_total" in loss_i:
                # ray_total 已经含 lambda_ray，这里再乘以 sup 权重保持整体加权一致。
                total_ray = total_ray + loss_i["ray_total"] * weight
                total_ray_hit = total_ray_hit + loss_i["ray_hit"] * weight
                total_ray_empty = total_ray_empty + loss_i["ray_empty"] * weight
                total_ray_pre_free = total_ray_pre_free + loss_i.get("ray_pre_free", torch.zeros((), device=step_logits.device)) * weight
                total_ray_depth = total_ray_depth + loss_i["ray_depth"] * weight
                if int(loss_i.get("ray_valid_rays", torch.tensor(0)).item()) > 0:
                    ray_sup_count += 1
            per_step_loss[key] = float(weighted_total.detach().item())
            per_step_count[key] = int(len(logits_list))
            active_any = True

        if not active_any:
            # 无有效监督时返回可反传的零损失，避免 backward 报错。
            zero = step_logits.sum() * 0.0
            total = zero
            total_focal = zero
            total_aux = zero
            total_ray = zero
            total_ray_hit = zero
            total_ray_empty = zero
            total_ray_pre_free = zero
            total_ray_depth = zero

        return {
            "total": total,
            "focal": total_focal,
            "aux": total_aux,
            "ray_total": total_ray.detach(),
            "ray_hit": total_ray_hit.detach(),
            "ray_empty": total_ray_empty.detach(),
            "ray_pre_free": total_ray_pre_free.detach(),
            "ray_depth": total_ray_depth.detach(),
            "ray_sup_count": torch.tensor(ray_sup_count, device=step_logits.device),
        }, per_step_loss, per_step_count

    def _forward_stepwise(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        # OnlineNcdeAligner200 统一用 mode 参数，原版用独立方法
        model = self.model.module if hasattr(self.model, "module") else self.model
        kwargs = dict(
            fast_logits=sample["fast_logits"],
            slow_logits=sample["slow_logits"],
            frame_ego2global=sample["frame_ego2global"],
            frame_timestamps=sample.get("frame_timestamps", None),
            frame_dt=sample.get("frame_dt", None),
        )
        if hasattr(model, "forward_stepwise_train"):
            return model.forward_stepwise_train(**kwargs, max_step_index=self.stepwise_max_step_index)
        return self.model(**kwargs, mode="stepwise_train", max_step_index=self.stepwise_max_step_index)

    def _select_primary_supervision_batch(
        self,
        step_logits: torch.Tensor,
        step_indices: torch.Tensor,
        sample: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.primary_supervision_label not in self.supervision_labels:
            return None
        sup_i = self.supervision_labels.index(self.primary_supervision_label)
        sup_labels = cast(torch.Tensor, sample["sup_labels"])
        sup_masks = cast(torch.Tensor, sample["sup_masks"])
        sup_step_indices = cast(torch.Tensor, sample["sup_step_indices"])
        sup_valid_mask = cast(torch.Tensor, sample["sup_valid_mask"])

        step_map = {int(v): i for i, v in enumerate(step_indices.detach().cpu().tolist())}
        valid_rows = sup_valid_mask[:, sup_i] > 0.5
        rows = torch.nonzero(valid_rows, as_tuple=False).reshape(-1).tolist()
        logits_list = []
        labels_list = []
        masks_list = []
        for b in rows:
            step_value = int(sup_step_indices[b, sup_i].item())
            local_step = step_map.get(step_value, None)
            if local_step is None:
                continue
            logits_list.append(step_logits[b, local_step])
            labels_list.append(sup_labels[b, sup_i])
            masks_list.append(sup_masks[b, sup_i])
        if not logits_list:
            return None
        return (
            torch.stack(logits_list, dim=0),
            torch.stack(labels_list, dim=0),
            torch.stack(masks_list, dim=0),
        )

    def _resolve_eval_targets_from_stepwise(
        self,
        step_logits: torch.Tensor,
        step_indices: torch.Tensor,
        sample: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = (
            step_logits[:, -1]
            if step_logits.shape[1] > 0
            else cast(torch.Tensor, sample["fast_logits"])[:, -1]
        )
        eval_labels = cast(torch.Tensor, sample["gt_labels"])
        eval_masks = cast(torch.Tensor, sample["gt_mask"])

        if self.rollout_mode == "one_second_tminus1":
            selected = self._select_primary_supervision_batch(
                step_logits=step_logits,
                step_indices=step_indices,
                sample=sample,
            )
            if selected is not None:
                return selected
        return logits, eval_labels, eval_masks

    def _run_stepwise_and_compute_loss(
        self,
        sample: Dict[str, Any],
        for_eval: bool,
    ) -> tuple[
        Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]],
        dict[str, torch.Tensor],
        dict[str, float],
        dict[str, int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # eval 带 log_multistep_losses=False 时可以不依赖 sup_*，其它情况都必须有。
        if not for_eval or self.log_multistep_losses:
            self._require_multistep_supervision(
                sample, context="eval" if for_eval else "train"
            )

        outputs_step = self._forward_stepwise(sample)
        step_logits = cast(torch.Tensor, outputs_step["step_logits"])
        step_indices = cast(torch.Tensor, outputs_step["step_indices"])

        ray_gt_dist = sample.get("ray_gt_dist", None)
        ray_origin = sample.get("ray_origin", None)
        ray_origin_mask = sample.get("ray_origin_mask", None)
        ray_sup_valid = sample.get("ray_sup_valid", None)

        if for_eval:
            logits, eval_labels, eval_masks = self._resolve_eval_targets_from_stepwise(
                step_logits=step_logits,
                step_indices=step_indices,
                sample=sample,
            )
            # eval 不关心 ray 指标，单点 loss 和 multistep sup loss 都不传 ray，
            # 避免每个 val batch 白跑 4 次 ray forward。
            loss_dict = self.loss_fn(logits, eval_labels, eval_masks)
            sup_loss_batch: dict[str, float] = {}
            sup_count_batch: dict[str, int] = {}
            if self.log_multistep_losses:
                _, sup_loss_batch, sup_count_batch = self._compute_multistep_loss(
                    step_logits=step_logits,
                    step_indices=step_indices,
                    sup_labels=cast(torch.Tensor, sample["sup_labels"]),
                    sup_masks=cast(torch.Tensor, sample["sup_masks"]),
                    sup_step_indices=cast(torch.Tensor, sample["sup_step_indices"]),
                    sup_valid_mask=cast(torch.Tensor, sample["sup_valid_mask"]),
                )
            return (
                outputs_step,
                loss_dict,
                sup_loss_batch,
                sup_count_batch,
                logits,
                eval_labels,
                eval_masks,
            )

        loss_dict, sup_loss_batch, sup_count_batch = self._compute_multistep_loss(
            step_logits=step_logits,
            step_indices=step_indices,
            sup_labels=cast(torch.Tensor, sample["sup_labels"]),
            sup_masks=cast(torch.Tensor, sample["sup_masks"]),
            sup_step_indices=cast(torch.Tensor, sample["sup_step_indices"]),
            sup_valid_mask=cast(torch.Tensor, sample["sup_valid_mask"]),
            ray_gt_dist=ray_gt_dist,
            ray_origin=ray_origin,
            ray_origin_mask=ray_origin_mask,
            ray_sup_valid=ray_sup_valid,
        )
        logits = (
            step_logits[:, -1]
            if step_logits.shape[1] > 0
            else cast(torch.Tensor, sample["fast_logits"])[:, -1]
        )
        eval_labels = cast(torch.Tensor, sample["gt_labels"])
        eval_masks = cast(torch.Tensor, sample["gt_mask"])
        return (
            outputs_step,
            loss_dict,
            sup_loss_batch,
            sup_count_batch,
            logits,
            eval_labels,
            eval_masks,
        )

    def train_one_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        total_focal = 0.0
        total_aux = 0.0
        total_delta = 0.0
        total_fast_kl = 0.0
        total_ray = 0.0
        total_ray_hit = 0.0
        total_ray_empty = 0.0
        total_ray_pre_free = 0.0
        total_ray_depth = 0.0
        total_ray_sup_count = 0
        total_sup_loss: Dict[str, float] = {}
        total_sup_count: Dict[str, int] = {}

        total_steps = len(loader)
        _pb = make_pbar(total_steps, prefix=f"[train][epoch={epoch}] ") if (make_pbar is not None and self.is_main) else None
        pbar = _pb.start() if _pb is not None else None
        for step, sample in enumerate(loader, start=1):
            sample = move_to_device(sample, self.device)
            outputs, loss_dict, sup_loss_batch, sup_count_batch, _, _, _ = (
                self._run_stepwise_and_compute_loss(sample=sample, for_eval=False)
            )
            loss = cast(torch.Tensor, loss_dict["total"])
            fast_kl_tensor = outputs.get("fast_kl", None)
            if isinstance(fast_kl_tensor, torch.Tensor):
                total_fast_kl += float(fast_kl_tensor.detach().item())
                loss = loss + self.lambda_fast_kl * fast_kl_tensor

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += float(loss.item())
            total_focal += float(loss_dict["focal"].item())
            total_aux += float(loss_dict["aux"].item())
            if "ray_total" in loss_dict:
                ray_sup_cnt = int(
                    cast(torch.Tensor, loss_dict.get("ray_sup_count", torch.tensor(0))).item()
                )
                if ray_sup_cnt > 0:
                    total_ray += float(loss_dict["ray_total"].item())
                    total_ray_hit += float(loss_dict["ray_hit"].item())
                    total_ray_empty += float(loss_dict["ray_empty"].item())
                    total_ray_pre_free += float(loss_dict.get("ray_pre_free", torch.tensor(0.0)).item())
                    total_ray_depth += float(loss_dict["ray_depth"].item())
                    total_ray_sup_count += ray_sup_cnt
            for key, value in sup_loss_batch.items():
                cnt = sup_count_batch.get(key, 0)
                if cnt <= 0:
                    continue
                total_sup_loss[key] = total_sup_loss.get(key, 0.0) + float(value) * float(cnt)
                total_sup_count[key] = total_sup_count.get(key, 0) + int(cnt)

            diag = self._pack_diag(cast(list[dict[str, torch.Tensor]], outputs.get("diagnostics", [])))
            total_delta += diag.get("delta_scene_abs_mean", 0.0)

            if pbar is not None:
                pbar.update(step)
            elif self.is_main and (step % self.log_interval == 0 or step == total_steps):
                print(f"[train] epoch={epoch} step={step}/{total_steps} loss={total_loss / step:.4f}")

        if pbar is not None:
            pbar.finish()

        main_stats = self._all_reduce_sums([
            total_loss,
            total_focal,
            total_aux,
            total_delta,
            total_fast_kl,
            total_ray,
            total_ray_hit,
            total_ray_empty,
            total_ray_pre_free,
            total_ray_depth,
            float(total_ray_sup_count),
            float(total_steps),
        ])
        (
            total_loss,
            total_focal,
            total_aux,
            total_delta,
            total_fast_kl,
            total_ray,
            total_ray_hit,
            total_ray_empty,
            total_ray_pre_free,
            total_ray_depth,
            total_ray_sup_count_f,
            total_steps_f,
        ) = main_stats
        total_ray_sup_count = int(round(total_ray_sup_count_f))
        denom = max(int(round(total_steps_f)), 1)

        sup_keys = [f"loss_{label}" for label in self.supervision_labels]
        sup_stats = self._all_reduce_sums(
            [total_sup_loss.get(key, 0.0) for key in sup_keys]
            + [float(total_sup_count.get(key, 0)) for key in sup_keys]
        )
        sup_split = len(sup_keys)
        total_sup_loss = {
            key: float(value) for key, value in zip(sup_keys, sup_stats[:sup_split])
        }
        total_sup_count = {
            key: int(round(value)) for key, value in zip(sup_keys, sup_stats[sup_split:])
        }

        metrics = {
            "loss": total_loss / denom,
            "focal": total_focal / denom,
            "aux": total_aux / denom,
            "delta_scene_abs_mean": total_delta / denom,
        }
        if self.lambda_fast_kl > 0.0:
            metrics["fast_kl"] = total_fast_kl / denom
        if total_ray_sup_count > 0:
            metrics["ray"] = total_ray / denom
            metrics["ray_hit"] = total_ray_hit / denom
            metrics["ray_empty"] = total_ray_empty / denom
            metrics["ray_pre_free"] = total_ray_pre_free / denom
            metrics["ray_depth"] = total_ray_depth / denom
        for key, value in total_sup_loss.items():
            count = max(total_sup_count.get(key, 0), 1)
            metrics[key] = value / count
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        collect_predictions: bool = False,
        use_ema: bool = True,
    ) -> Dict[str, float | list[str] | list[float]]:
        """评估并返回 loss + mIoU。

        Args:
            collect_predictions: 为 True 时额外收集每个样本的 dense pred/gt/token，
                用于后续 RayIoU 等需要完整预测结果的指标计算。
                结果存于返回字典的 ``"predictions"`` 键。
            use_ema: 若为 True 且 self.ema 存在，则用 EMA 权重评估。
        """
        original_model = self.model
        if use_ema and self.ema is not None:
            self.model = self.ema.module
        try:
            return self._evaluate_impl(loader, collect_predictions)
        finally:
            self.model = original_model

    @torch.no_grad()
    def _evaluate_impl(
        self,
        loader: DataLoader,
        collect_predictions: bool,
    ) -> Dict[str, float | list[str] | list[float]]:
        self.model.eval()
        total_loss = 0.0
        total_focal = 0.0
        total_aux = 0.0
        total_sup_loss: Dict[str, float] = {}
        total_sup_count: Dict[str, int] = {}
        collected: list[dict] = []
        metric = MetricMiouOcc3D(
            num_classes=self.num_classes,
            use_image_mask=True,
            use_lidar_mask=False,
        )

        total_steps = len(loader)
        _pb = make_pbar(total_steps, prefix="[eval] ") if (make_pbar is not None and self.is_main) else None
        pbar = _pb.start() if _pb is not None else None
        for step, sample in enumerate(loader, start=1):
            sample = move_to_device(sample, self.device)
            _, loss_dict, sup_loss_batch, sup_count_batch, logits, eval_labels, eval_masks = (
                self._run_stepwise_and_compute_loss(sample=sample, for_eval=True)
            )
            total_loss += float(loss_dict["total"].item())
            total_focal += float(loss_dict["focal"].item())
            total_aux += float(loss_dict["aux"].item())
            for key, value in sup_loss_batch.items():
                cnt = sup_count_batch.get(key, 0)
                if cnt <= 0:
                    continue
                sup_key = f"sup_{key}"
                total_sup_loss[sup_key] = total_sup_loss.get(sup_key, 0.0) + float(value) * float(cnt)
                total_sup_count[sup_key] = total_sup_count.get(sup_key, 0) + int(cnt)

            gt_labels_rs, gt_mask_rs = resize_labels_and_mask_to_logits(
                logits, eval_labels, eval_masks
            )
            if self.free_conf_thresh is not None:
                preds = apply_free_threshold(logits, self.free_index, self.free_conf_thresh)
            else:
                preds = logits.argmax(dim=1)
            preds_np = preds.detach().cpu().numpy()
            gt_np = gt_labels_rs.detach().cpu().numpy()
            mask_np = gt_mask_rs.detach().cpu().numpy() if gt_mask_rs is not None else None
            for b in range(preds_np.shape[0]):
                metric.add_batch(
                    semantics_pred=preds_np[b],
                    semantics_gt=gt_np[b],
                    mask_lidar=None,
                    mask_camera=mask_np[b] if mask_np is not None else None,
                )

            # 收集 dense 预测用于 RayIoU 等后续指标
            if collect_predictions:
                meta_list = sample.get("meta", [])
                if isinstance(meta_list, dict):
                    meta_list = [meta_list]
                for b in range(preds_np.shape[0]):
                    token = meta_list[b].get("token", "") if b < len(meta_list) else ""
                    collected.append({
                        "pred": preds_np[b].astype(np.uint8),
                        "gt": gt_np[b].astype(np.uint8),
                        "token": token,
                    })

            if pbar is not None:
                pbar.update(step)
            elif self.is_main and (step % self.log_interval == 0 or step == total_steps):
                print(f"[eval] step={step}/{total_steps}")

        if pbar is not None:
            pbar.finish()

        denom = max(total_steps, 1)
        miou = metric.count_miou(verbose=False)
        miou_d = metric.count_miou_d(verbose=False)
        per_class = metric.get_per_class_iou()
        per_class = np.nan_to_num(per_class, nan=0.0).tolist()
        metrics: Dict[str, float | list[str] | list[float]] = {
            "loss": total_loss / denom,
            "focal": total_focal / denom,
            "aux": total_aux / denom,
            "miou": miou,
            "miou_d": miou_d,
            "per_class_iou": per_class,
            "class_names": metric.class_names,
        }
        for key, value in total_sup_loss.items():
            count = max(total_sup_count.get(key, 0), 1)
            metrics[key] = value / count
        if collect_predictions:
            metrics["predictions"] = collected
        return metrics

    def save_checkpoint(self, path: str, epoch: int | None = None, extra: dict | None = None) -> None:
        """保存模型参数及 optimizer 状态（自动解包 DDP）。"""
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        merged_extra = dict(extra) if extra else {}
        if self.ema is not None:
            merged_extra["ema"] = self.ema.state_dict()
        _save_checkpoint(path, model=raw_model, optimizer=self.optimizer, epoch=epoch, extra=merged_extra or None)
