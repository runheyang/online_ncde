"""online_ncde 训练与评估流程。"""

from __future__ import annotations

from typing import Any, Dict, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from online_ncde.losses import resize_labels_and_mask_to_logits
from online_ncde.metrics import MetricMiouOcc3D, apply_free_threshold
from online_ncde.utils.checkpoints import save_checkpoint as _save_checkpoint

try:
    import progressbar
    from online_ncde.utils.progress import make_pbar
except Exception:  # pragma: no cover
    progressbar = None
    make_pbar = None


def online_ncde_collate(batch):
    """组 batch，并处理可选时序字段。"""
    fast_logits = torch.stack([item["fast_logits"] for item in batch], dim=0)
    slow_logits = torch.stack([item["slow_logits"] for item in batch], dim=0)
    frame_ego2global = torch.stack([item["frame_ego2global"] for item in batch], dim=0)

    frame_timestamps = [item.get("frame_timestamps", None) for item in batch]
    frame_dt = [item.get("frame_dt", None) for item in batch]
    if all(ts is not None for ts in frame_timestamps):
        frame_timestamps = torch.stack(cast(list[torch.Tensor], frame_timestamps), dim=0)
    else:
        frame_timestamps = None
    if all(dt is not None for dt in frame_dt):
        frame_dt = torch.stack(cast(list[torch.Tensor], frame_dt), dim=0)
    else:
        frame_dt = None

    gt_labels = torch.stack([item["gt_labels"] for item in batch], dim=0)
    gt_mask = torch.stack([item["gt_mask"] for item in batch], dim=0)
    sup_labels = [item.get("sup_labels", None) for item in batch]
    sup_masks = [item.get("sup_masks", None) for item in batch]
    sup_step_indices = [item.get("sup_step_indices", None) for item in batch]
    sup_valid_mask = [item.get("sup_valid_mask", None) for item in batch]
    if all(x is not None for x in sup_labels):
        sup_labels = torch.stack(cast(list[torch.Tensor], sup_labels), dim=0)
    else:
        sup_labels = None
    if all(x is not None for x in sup_masks):
        sup_masks = torch.stack(cast(list[torch.Tensor], sup_masks), dim=0)
    else:
        sup_masks = None
    if all(x is not None for x in sup_step_indices):
        sup_step_indices = torch.stack(cast(list[torch.Tensor], sup_step_indices), dim=0)
    else:
        sup_step_indices = None
    if all(x is not None for x in sup_valid_mask):
        sup_valid_mask = torch.stack(cast(list[torch.Tensor], sup_valid_mask), dim=0)
    else:
        sup_valid_mask = None
    meta = [item.get("meta", {}) for item in batch]

    return {
        "fast_logits": fast_logits,
        "slow_logits": slow_logits,
        "frame_ego2global": frame_ego2global,
        "frame_timestamps": frame_timestamps,
        "frame_dt": frame_dt,
        "gt_labels": gt_labels,
        "gt_mask": gt_mask,
        "sup_labels": sup_labels,
        "sup_masks": sup_masks,
        "sup_step_indices": sup_step_indices,
        "sup_valid_mask": sup_valid_mask,
        "meta": meta,
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
        use_multistep_supervision: bool = False,
        supervision_labels: list[str] | None = None,
        supervision_weights: list[float] | None = None,
        supervision_weight_normalize: bool = True,
        log_multistep_losses: bool = True,
        rollout_mode: str = "full",
        primary_supervision_label: str = "t-1.0",
        stepwise_max_step_index: int | None = None,
        is_main: bool = True,
    ) -> None:
        self.model = model
        self.is_main = is_main
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.num_classes = num_classes
        self.free_index = free_index
        self.free_conf_thresh = free_conf_thresh
        self.log_interval = log_interval
        self.clip_norm = clip_norm
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.use_multistep_supervision = bool(use_multistep_supervision)
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

    def _has_multistep_fields(self, sample: Dict[str, Any]) -> bool:
        return (
            sample.get("sup_labels", None) is not None
            and sample.get("sup_masks", None) is not None
            and sample.get("sup_step_indices", None) is not None
            and sample.get("sup_valid_mask", None) is not None
        )

    def _compute_multistep_loss(
        self,
        step_logits: torch.Tensor,
        step_indices: torch.Tensor,
        sup_labels: torch.Tensor,
        sup_masks: torch.Tensor,
        sup_step_indices: torch.Tensor,
        sup_valid_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, int]]:
        """按 sidecar 指定 step 做 4 时刻联合监督（不做 detach）。"""
        step_map = {int(v): i for i, v in enumerate(step_indices.detach().cpu().tolist())}
        num_sup = len(self.supervision_labels)
        if sup_labels.shape[1] != num_sup:
            raise ValueError(
                f"监督时刻数不一致: labels.shape[1]={sup_labels.shape[1]} vs expected={num_sup}"
            )

        total = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        total_focal = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        total_aux = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
        per_step_loss: dict[str, float] = {}
        per_step_count: dict[str, int] = {}
        active_any = False

        for sup_i, label in enumerate(self.supervision_labels):
            key = f"loss_{label}"
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
                per_step_loss[key] = 0.0
                per_step_count[key] = 0
                continue

            logits_i = torch.stack(logits_list, dim=0)
            labels_i = torch.stack(labels_list, dim=0)
            masks_i = torch.stack(masks_list, dim=0)
            loss_i = self.loss_fn(logits_i, labels_i, masks_i)
            weight = float(self.supervision_weights[sup_i])
            weighted_total = loss_i["total"] * weight
            weighted_focal = loss_i["focal"] * weight
            weighted_aux = loss_i["aux"] * weight
            total = total + weighted_total
            total_focal = total_focal + weighted_focal
            total_aux = total_aux + weighted_aux
            per_step_loss[key] = float(weighted_total.detach().item())
            per_step_count[key] = int(len(logits_list))
            active_any = True

        if not active_any:
            # 无有效监督时返回可反传的零损失，避免 backward 报错。
            zero = step_logits.sum() * 0.0
            total = zero
            total_focal = zero
            total_aux = zero

        return {
            "total": total,
            "focal": total_focal,
            "aux": total_aux,
        }, per_step_loss, per_step_count

    def _forward_stepwise(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        return self.model(
            fast_logits=sample["fast_logits"],
            slow_logits=sample["slow_logits"],
            frame_ego2global=sample["frame_ego2global"],
            frame_timestamps=sample.get("frame_timestamps", None),
            frame_dt=sample.get("frame_dt", None),
            mode="stepwise_train",
            max_step_index=self.stepwise_max_step_index,
        )

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
        outputs_step = self._forward_stepwise(sample)
        step_logits = cast(torch.Tensor, outputs_step["step_logits"])
        step_indices = cast(torch.Tensor, outputs_step["step_indices"])

        if for_eval:
            logits, eval_labels, eval_masks = self._resolve_eval_targets_from_stepwise(
                step_logits=step_logits,
                step_indices=step_indices,
                sample=sample,
            )
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
        total_sup_loss: Dict[str, float] = {}
        total_sup_count: Dict[str, int] = {}

        total_steps = len(loader)
        pbar = make_pbar(total_steps, prefix=f"[train][epoch={epoch}] ").start() if (make_pbar is not None and self.is_main) else None
        for step, sample in enumerate(loader, start=1):
            sample = move_to_device(sample, self.device)
            sup_loss_batch: Dict[str, float] = {}
            sup_count_batch: Dict[str, int] = {}
            if self.use_multistep_supervision and self._has_multistep_fields(sample):
                outputs, loss_dict, sup_loss_batch, sup_count_batch, _, _, _ = (
                    self._run_stepwise_and_compute_loss(sample=sample, for_eval=False)
                )
            else:
                outputs = self.model(
                    fast_logits=sample["fast_logits"],
                    slow_logits=sample["slow_logits"],
                    frame_ego2global=sample["frame_ego2global"],
                    frame_timestamps=sample.get("frame_timestamps", None),
                    frame_dt=sample.get("frame_dt", None),
                )
                logits = cast(torch.Tensor, outputs["aligned"])
                loss_dict = self.loss_fn(
                    logits,
                    sample["gt_labels"],
                    sample["gt_mask"],
                )
            loss = cast(torch.Tensor, loss_dict["total"])

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            total_loss += float(loss.item())
            total_focal += float(loss_dict["focal"].item())
            total_aux += float(loss_dict["aux"].item())
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

        denom = max(total_steps, 1)
        metrics = {
            "loss": total_loss / denom,
            "focal": total_focal / denom,
            "aux": total_aux / denom,
            "delta_scene_abs_mean": total_delta / denom,
        }
        for key, value in total_sup_loss.items():
            count = max(total_sup_count.get(key, 0), 1)
            metrics[key] = value / count
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        collect_predictions: bool = False,
    ) -> Dict[str, float | list[str] | list[float]]:
        """评估并返回 loss + mIoU。

        Args:
            collect_predictions: 为 True 时额外收集每个样本的 dense pred/gt/token，
                用于后续 RayIoU 等需要完整预测结果的指标计算。
                结果存于返回字典的 ``"predictions"`` 键。
        """
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
        pbar = make_pbar(total_steps, prefix="[eval] ").start() if (make_pbar is not None and self.is_main) else None
        for step, sample in enumerate(loader, start=1):
            sample = move_to_device(sample, self.device)
            sup_loss_batch: Dict[str, float] = {}
            sup_count_batch: Dict[str, int] = {}
            if self.use_multistep_supervision and self._has_multistep_fields(sample):
                _, loss_dict, sup_loss_batch, sup_count_batch, logits, eval_labels, eval_masks = (
                    self._run_stepwise_and_compute_loss(sample=sample, for_eval=True)
                )
            else:
                eval_labels = cast(torch.Tensor, sample["gt_labels"])
                eval_masks = cast(torch.Tensor, sample["gt_mask"])
                outputs = self.model(
                    fast_logits=sample["fast_logits"],
                    slow_logits=sample["slow_logits"],
                    frame_ego2global=sample["frame_ego2global"],
                    frame_timestamps=sample.get("frame_timestamps", None),
                    frame_dt=sample.get("frame_dt", None),
                )
                logits = cast(torch.Tensor, outputs["aligned"])
                loss_dict = self.loss_fn(
                    logits,
                    eval_labels,
                    eval_masks,
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

    def save_checkpoint(self, path: str, epoch: int | None = None) -> None:
        """保存模型参数及 optimizer 状态（自动解包 DDP）。"""
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        _save_checkpoint(path, model=raw_model, optimizer=self.optimizer, epoch=epoch)
