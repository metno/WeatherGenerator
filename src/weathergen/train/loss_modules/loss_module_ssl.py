# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

logger = logging.getLogger(__name__)


class LossLatentSSLStudentTeacher(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model pretraining using
    DINO/iBOT/JEPA/BYOL style losses.

    This class handles the initialization and application of various loss functions,
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
    """

    valid_loss_names = set(["DINO", "iBOT", "JEPA"])

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **losses):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.num_class_tokens = cf.num_class_tokens
        self.name = "LossLatentSSLStudentTeacher"

        # Dynamically load loss functions based on configuration and stage
        self.losses = {
            name: (local_conf["weight"], get_loss_function_ssl(name), local_conf["loss_extra_args"])
            for name, local_conf in losses.items()
        }

        # Deep self-supervision (V-JEPA 2.1): one weight per representation, in the order
        # [local2global] + [global:k ascending] + [final]
        deep_ssl_cfg = mode_cfg.get("deep_ssl", None)
        if deep_ssl_cfg and deep_ssl_cfg.get("enabled", False) and deep_ssl_cfg.get("tap_after"):
            num_levels = len(deep_ssl_cfg.tap_after) + 1
            self.level_weights = list(deep_ssl_cfg.get("level_weights", None) or [])
            if not self.level_weights:
                self.level_weights = [1.0 / num_levels] * num_levels
            assert len(self.level_weights) == num_levels, (
                f"deep_ssl.level_weights length ({len(self.level_weights)}) must match "
                f"len(tap_after) + 1 ({num_levels})"
            )
        else:
            self.level_weights = None

        # Context loss (V-JEPA 2.1): L1 on context (student-visible) tokens, linear warmup
        context_cfg = mode_cfg.get("context_loss", None)
        if context_cfg and context_cfg.get("enabled", True):
            self.context_loss_weight = float(context_cfg.get("weight", 1.0))
            self.context_loss_warmup_steps = int(context_cfg.get("warmup_steps", 0))
            dw = context_cfg.get("distance_weighting", None) or {}
            self.context_distance_weighting = bool(dw.get("enabled", False))
            self.context_distance_lambda = float(dw.get("lambda", 1.0))
        else:
            self.context_loss_weight = 0.0
            self.context_loss_warmup_steps = 0
            self.context_distance_weighting = False
            self.context_distance_lambda = 1.0

        # multi-level JEPA (head multilevel_transformer)
        jepa_conf = losses.get("JEPA")
        self.jepa_ml = (
            jepa_conf is not None
            and str(jepa_conf.get("head", "")).lower() == "multilevel_transformer"
        )
        assert self.level_weights is None or "JEPA" in losses, (
            "deep_ssl is implemented for the JEPA loss (single-level heads and "
            "multilevel_transformer); DINO / iBOT use the final representation only"
        )
        self.jepa_level_weights = {}
        if self.jepa_ml:
            self.jepa_level_weights = {
                int(k): float((v or {}).get("weight", 1.0)) for k, v in jepa_conf.levels.items()
            }
        # unit-sphere cell centres and mean cell size per latent level (distance weighting)
        self._cells_xyz: dict[int, torch.Tensor] = {}
        self._cell_size: dict[int, float] = {}
        if self.jepa_ml and self.context_distance_weighting:
            from weathergen.datasets.domain_pyramid import build_domain_pyramid
            from weathergen.model.latent_predictor import cell_xyz

            pyramid = build_domain_pyramid(cf)
            for lvl in pyramid.levels:
                xyz = cell_xyz(*pyramid.domain(lvl).centres_lonlat())
                self._cells_xyz[lvl] = torch.as_tensor(xyz, dtype=torch.float32)
                self._cell_size[lvl] = float(np.sqrt(4.0 * np.pi / (12 * 4**lvl)))

        # mask plan strategies, for per-strategy logging (fixed set of keys every step)
        plan_cfg = mode_cfg.get("mask_plan", None) or {}
        self.plan_strategies = [
            str(n)
            for n, v in (plan_cfg.get("strategies", None) or {}).items()
            if (v or {}).get("enabled", True)
        ]

    def _context_weight(self, istep: int) -> tuple[float, float]:
        """(effective context loss weight, warmup factor)."""
        if self.context_loss_weight <= 0.0:
            return 0.0, 0.0
        warmup = (
            min(1.0, istep / self.context_loss_warmup_steps)
            if self.context_loss_warmup_steps > 0
            else 1.0
        )
        return self.context_loss_weight * warmup, warmup

    def compute_loss(self, preds, targets, metadata, istep: int = 0, **kwargs) -> LossValues:
        if self.jepa_ml:
            return self.compute_loss_multilevel(preds, targets, metadata, istep)

        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all: dict[str, float] = {loss: 0.0 for loss in self.losses}

        source2target_matching_idxs, output_info, target2source_matching_idxs, _ = metadata

        preds = preds.latent[0]  # [0] because we always want the first fstep
        target_info = targets.aux_outputs
        targets = targets.latent

        ctx_weight, warmup = self._context_weight(istep)
        has_context_loss = ctx_weight > 0.0 and "JEPA" in self.losses
        if self.context_loss_weight > 0.0:
            losses_all["context_warmup"] = warmup

        # deep SSL (V-JEPA 2.1, ecmwf develop-ssl-diffusion-v2): JEPA on every representation
        has_deep_ssl = (
            self.level_weights is not None and "JEPA_deep" in preds and "JEPA_deep" in targets
        )

        for name, (weight, loss_fn, extra_args) in self.losses.items():
            if name == "JEPA" and has_deep_ssl:
                levels = list(
                    zip(self.level_weights, preds["JEPA_deep"], targets["JEPA_deep"], strict=True)
                )
            else:
                levels = [(1.0, preds[name], targets[name])]
            for level_idx, (level_w, s_data, t_data) in enumerate(levels):
                suffix = f"_L{level_idx}" if len(levels) > 1 else ""
                preds_for_loss = self.gather_preds_for_loss(
                    name, s_data, output_info, target2source_matching_idxs
                )
                targets_for_loss = self.gather_targets_for_loss(
                    name, t_data, target_info, target2source_matching_idxs
                )

                loss_value = loss_fn(**preds_for_loss, **targets_for_loss, **extra_args).mean()
                loss = loss + level_w * weight * loss_value
                losses_all[f"{name}{suffix}"] = loss_value.item()

                # Context loss (V-JEPA 2.1) reuses the gathered tensors
                if name == "JEPA" and has_context_loss:
                    ctx_loss_value = context_loss(
                        student_patches=preds_for_loss["student_patches_masked"],
                        student_masks=preds_for_loss["student_masks"],
                        teacher_patches=targets_for_loss["teacher_patches_masked"],
                        teacher_masks=targets_for_loss["teacher_masks"],
                    )
                    loss = loss + level_w * ctx_weight * ctx_loss_value
                    losses_all[f"context{suffix}"] = ctx_loss_value.item()
                elif name == "JEPA" and self.context_loss_weight > 0.0:
                    losses_all[f"context{suffix}"] = 0.0
            if len(levels) > 1:
                # weighted sum over the representations, as the term that enters the loss
                losses_all[name] = sum(
                    w * losses_all[f"{name}_L{i}"] for i, (w, _, _) in enumerate(levels)
                )

        return LossValues(loss=loss, losses_all=losses_all, stddev_all={})

    # ------------------------------------------------------------ multi-level ----

    def compute_loss_multilevel(self, preds, targets, metadata, istep: int) -> LossValues:
        """
        Multi-level JEPA (+ context loss, + deep SSL) for head multilevel_transformer.

        For every student sample s, its teacher sample t and every configured latent level l:
            hidden_l  : cells without student input (from the model, `JEPA_hidden`)
            valid_l   : cells with teacher input (`JEPA_valid`)
            JEPA loss : L1 at hidden_l & valid_l          (prediction of what was not seen)
            context   : L1 at ~hidden_l & valid_l         (V-JEPA 2.1 context loss)
        Reduction (as ecmwf develop-ssl-diffusion-v2): mean over features, mean over the
        selected tokens of a sample, mean over samples with a non-empty selection.
        Total = sum_l w_l sum_k a_k (w_JEPA * JEPA_lk + w_ctx(t) * context_lk), k over the
        deep SSL representations (a_k = deep_ssl.level_weights; one representation, a=1,
        without deep SSL).
        """
        source2target, output_info, _, _ = metadata
        weight_jepa = self.losses["JEPA"][0]
        ctx_weight, warmup = self._context_weight(istep)
        has_context = self.context_loss_weight > 0.0

        p_latent = preds.latent[0]
        t_latent = targets.latent
        # samples that take part in the JEPA loss
        sidx = [i for i, info in enumerate(output_info) if "JEPA" in info.global_params["loss"]]
        tidx = [int(np.asarray(source2target[i]).reshape(-1)[0]) for i in sidx]

        if self.level_weights is not None:
            assert "JEPA_deep" in p_latent and "JEPA_deep" in t_latent, (
                "deep_ssl enabled but no deep predictions / targets (encoder taps missing?)"
            )
            s_reps, t_reps, rep_w = p_latent["JEPA_deep"], t_latent["JEPA_deep"], self.level_weights
        else:
            s_reps = {lvl: [x] for lvl, x in p_latent["JEPA"].items()}
            t_reps = {lvl: [x] for lvl, x in t_latent["JEPA"].items()}
            rep_w = [1.0]

        hidden_all = p_latent["JEPA_hidden"]
        valid_all = t_latent.get("JEPA_valid", {})

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        losses_all: dict[str, float] = {"JEPA": 0.0}
        if has_context:
            losses_all["context"] = 0.0
            losses_all["context_warmup"] = warmup
        per_sample_total = None

        for lvl, w_lvl in sorted(self.jepa_level_weights.items()):
            key = f"hl{lvl}"
            if lvl not in s_reps or lvl not in t_reps:
                if w_lvl > 0.0:
                    _warn_once(
                        f"multi-level JEPA: no {'prediction' if lvl not in s_reps else 'target'}"
                        f" for latent level {lvl}; level skipped (teacher without this level?)"
                    )
                losses_all[f"JEPA_{key}"] = 0.0
                if has_context:
                    losses_all[f"context_{key}"] = 0.0
                continue

            device = s_reps[lvl][0].device
            hidden = hidden_all[lvl][sidx].to(device)
            if lvl in valid_all:
                valid = valid_all[lvl]
            elif -1 in valid_all and len(valid_all) == 1:
                valid = valid_all[-1]
            else:
                valid = torch.ones_like(t_reps[lvl][0][..., 0], dtype=torch.bool)
            valid = valid[tidx].to(device)
            m_jepa = hidden & valid
            m_ctx = (~hidden) & valid
            w_ctx_tok = None
            if has_context and self.context_distance_weighting:
                w_ctx_tok = self._distance_weights(lvl, hidden, m_ctx)

            jepa_lvl, ctx_lvl = 0.0, 0.0
            for k, (a_k, s_x, t_x) in enumerate(zip(rep_w, s_reps[lvl], t_reps[lvl], strict=True)):
                s_x = s_x[sidx]
                t_x = t_x[tidx].to(s_x.dtype)
                l_j, per_s = _masked_l1_per_sample(s_x, t_x, m_jepa)
                term = weight_jepa * l_j
                jepa_lvl += a_k * l_j.item()
                if k == len(rep_w) - 1:
                    per_sample_total = (
                        per_s if per_sample_total is None else per_sample_total + per_s
                    )
                if has_context:
                    l_c, _ = _masked_l1_per_sample(s_x, t_x, m_ctx, w_ctx_tok)
                    term = term + ctx_weight * l_c
                    ctx_lvl += a_k * l_c.item()
                    if self.level_weights is not None:
                        losses_all[f"context_{key}_L{k}"] = l_c.item()
                if self.level_weights is not None:
                    losses_all[f"JEPA_{key}_L{k}"] = l_j.item()
                loss = loss + w_lvl * a_k * term

            losses_all[f"JEPA_{key}"] = jepa_lvl
            losses_all["JEPA"] += w_lvl * jepa_lvl
            if has_context:
                losses_all[f"context_{key}"] = ctx_lvl
                losses_all["context"] += w_lvl * ctx_lvl
            # fraction of hidden / teacher-valid cells (diagnostics of the masking)
            losses_all[f"frac_hidden_{key}"] = hidden.float().mean().item()
            losses_all[f"frac_valid_{key}"] = valid.float().mean().item()

        # per mask-plan strategy: JEPA loss (final representation, summed over levels) and
        # number of samples drawn with the strategy (keys fixed every step for DDP logging)
        if self.plan_strategies:
            strat = [
                info.params.get("mask_plan", {}).get("strategy", None)
                if isinstance(info.params, dict) or hasattr(info.params, "get")
                else None
                for info in (output_info[i] for i in sidx)
            ]
            for name in self.plan_strategies:
                sel = [j for j, st in enumerate(strat) if st == name]
                losses_all[f"n_{name}"] = float(len(sel))
                if per_sample_total is not None and len(sel) > 0:
                    losses_all[f"JEPA_{name}"] = per_sample_total[sel].mean().item()
                else:
                    losses_all[f"JEPA_{name}"] = 0.0

        return LossValues(loss=loss, losses_all=losses_all, stddev_all={})

    def _distance_weights(self, lvl, hidden, m_ctx) -> torch.Tensor:
        """
        V-JEPA 2.1 context-loss weights lambda / sqrt(d_min): d_min = distance of a context
        cell to the nearest hidden cell, in units of the level's mean cell size (>= 1).
        Samples without hidden cells get weight lambda.
        """
        xyz = self._cells_xyz[lvl].to(hidden.device)
        w = torch.full(hidden.shape, self.context_distance_lambda, device=hidden.device)
        for b in range(hidden.shape[0]):
            h = hidden[b]
            if not h.any() or not m_ctx[b].any():
                continue
            # chord distance ~ angle for small angles
            d = torch.cdist(xyz[m_ctx[b]], xyz[h]).min(dim=1).values / self._cell_size[lvl]
            w[b, m_ctx[b]] = self.context_distance_lambda / torch.sqrt(d.clamp(min=1.0))
        return w

    def gather_preds_for_loss(self, name, preds, metadata, target2source_matching_idxs):
        if name == "JEPA":
            """
            Important this assumes that there is 1 masked version for each global view
            ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]
            """
            return {
                "student_patches_masked": torch.stack(
                    [
                        p
                        for p, info in zip(preds, metadata, strict=False)
                        if "JEPA" in info.global_params["loss"]
                    ],
                    dim=0,
                ),
                "student_masks": torch.stack(
                    [info.mask for info in metadata if "JEPA" in info.global_params["loss"]],
                    dim=0,
                ).unsqueeze(1),
            }
        elif name == "iBOT":
            """
            Important this assumes that there is 1 masked version for each global view
            ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]

            Note the class token of iBOT is still missing
            """
            return {
                "student_patches_masked": torch.stack(
                    [
                        p[self.num_class_tokens :]
                        for p, info in zip(preds, metadata, strict=False)
                        if "iBOT" in info.global_params["loss"]
                    ],
                    dim=0,
                ),
                "student_masks": torch.stack(
                    [info.mask for info in metadata if "iBOT" in info.global_params["loss"]],
                    dim=0,
                ).unsqueeze(1),
                "student_class_masked": torch.stack(
                    [
                        p[: self.num_class_tokens]
                        for p, info in zip(preds, metadata, strict=False)
                        if "iBOT" in info.global_params["loss"]
                    ],
                    dim=0,
                ),
            }
        elif name == "DINO":
            local2global_dino_student = []
            for student_indices in target2source_matching_idxs:
                local_preds = [
                    preds[sidx]
                    for sidx in student_indices
                    if "DINO" in metadata[sidx].global_params["loss"]
                    and metadata[sidx].global_params["relationship"] != "identity"
                ]
                local2global_dino_student.append(local_preds)
            local2global_dino_student = [
                torch.stack(latents, dim=0)
                for latents in zip(*local2global_dino_student, strict=False)
            ]
            return {
                "local2global_dino_student": local2global_dino_student,
                "global2global_dino_student": torch.stack(
                    [
                        p
                        for p, info in zip(preds, metadata, strict=False)
                        if "DINO" in info.global_params["loss"]
                        and info.global_params["relationship"] == "identity"
                    ],
                    dim=0,
                ),
            }
        else:
            raise NotImplementedError(
                f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
            )

    def gather_targets_for_loss(self, name, targets, metadata, target2source_matching_idxs):
        if name == "JEPA":
            """
            Important this assumes that there is 1 masked version for each global view
            ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]
            """
            return {
                "teacher_patches_masked": torch.stack(
                    [
                        p
                        for p, info in zip(targets, metadata, strict=True)
                        if "JEPA" in info.global_params["loss"]
                    ],
                    dim=0,
                ),
                "teacher_masks": torch.stack(
                    [info.mask for info in metadata if "JEPA" in info.global_params["loss"]],
                    dim=0,
                ).unsqueeze(1),
            }
        elif name == "iBOT":
            """
            Important this assumes that there is 1 masked version for each global view
            ie. student_patches_masked.shape[0] == teacher_patches_masked.shape[0]

            Note the class token of iBOT is still missing
            """
            return {
                "teacher_patches_masked": torch.stack(
                    [
                        p[self.num_class_tokens :]
                        for p, info in zip(targets, metadata, strict=False)
                    ],
                    dim=0,
                ),
                "teacher_masks": torch.stack(
                    [info.mask for info in metadata if "iBOT" in info.global_params["loss"]],
                    dim=0,
                ).unsqueeze(1),
                "teacher_class_masked": torch.stack(
                    [
                        p[: self.num_class_tokens]
                        for p, info in zip(targets, metadata, strict=False)
                    ],
                    dim=0,
                ),
            }
        elif name == "DINO":
            return {
                "local2global_dino_teacher": torch.stack(
                    [p for p, info in zip(targets, metadata, strict=False)],
                    dim=0,
                ),
                "global2global_dino_teacher": torch.stack(
                    list(reversed([p for p, info in zip(targets, metadata, strict=False)])),
                    dim=0,
                ),
            }
        else:
            raise NotImplementedError(
                f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
            )


def _masked_l1_per_sample(student, teacher, mask, token_weights=None):
    """
    L1 on a masked token set: mean over features, (weighted) mean over the selected tokens
    of a sample, mean over the samples with a non-empty selection.

    token_weights (optional, same shape as mask): per-token weights; the sample mean is then
    sum(w * l) / num_selected (so that the weights scale the loss).
    Returns (loss, per-sample loss (0 for empty samples)).
    """
    per_token = F.l1_loss(student, teacher, reduction="none").mean(dim=-1)
    if token_weights is not None:
        per_token = per_token * token_weights.to(per_token.dtype)
    mask_f = mask.to(per_token.dtype)
    counts = mask_f.sum(dim=-1)
    valid = counts > 0
    # torch.where (not multiply): NaN at excluded positions must not poison the sum
    per_sample = torch.where(mask, per_token, 0.0).sum(dim=-1) / counts.clamp(min=1.0)
    if not valid.any():
        return student.sum() * 0.0, per_sample.detach()
    return per_sample[valid].mean(), per_sample.detach()


def _masked_l1_loss_per_sample(
    student_patches, teacher_patches, mask, empty_warning: str | None = None
):
    """Average L1 on a masked token set, normalized on the set's own support.

    (ecmwf develop-ssl-diffusion-v2) Reduction order:
    1. Mean over feature dimension for each selected token.
    2. Mean over selected tokens per sample.
    3. Mean over samples with at least one selected token.
    """

    assert mask.shape[0] == student_patches.shape[0], (
        "mask.shape[0], batch dimension, has to match batch dimension for student_patches."
    )
    assert teacher_patches.shape[0] in (1, mask.shape[0]), (
        "teacher_patches batch dimension must be 1 or match mask batch dimension."
    )

    if mask.sum() == 0:
        if empty_warning is not None:
            logger.warning(empty_warning)
        return student_patches.sum() * 0.0

    teacher_patches = teacher_patches.expand((mask.shape[0], -1, -1))
    loss, _ = _masked_l1_per_sample(student_patches, teacher_patches, mask.bool())
    return loss


def jepa_loss(
    student_patches_masked, student_masks, teacher_patches_masked, teacher_masks, temporal=False
):
    # TODO remove as we deal with batch dimension
    assert teacher_masks.shape[0] == 1 or teacher_masks.shape[0] == student_masks.shape[0]
    student_masks = student_masks.squeeze(dim=1).to(student_patches_masked.device)
    teacher_masks = teacher_masks.squeeze(dim=1).to(student_patches_masked.device)

    if temporal:
        # Temporal JEPA: predict teacher's representation at ALL teacher-visible cells
        mask = teacher_masks
    else:
        # Standard JEPA: predict only at cells the teacher sees but the student doesn't
        mask = torch.logical_and(teacher_masks, torch.logical_not(student_masks))

    return _masked_l1_loss_per_sample(
        student_patches_masked,
        teacher_patches_masked,
        mask,
        empty_warning="jepa_loss mask is all zeros, likely incorrect masking config.",
    )


def context_loss(student_patches, student_masks, teacher_patches, teacher_masks):
    """V-JEPA 2.1 context loss: L1 on context tokens, normalized on context support."""
    student_masks = student_masks.squeeze(dim=1).to(student_patches.device)
    teacher_masks = teacher_masks.squeeze(dim=1).to(student_patches.device)

    # Context = positions visible to the student (AND visible to teacher)
    mask = torch.logical_and(student_masks, teacher_masks)
    return _masked_l1_loss_per_sample(student_patches, teacher_patches, mask)


_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        logger.warning(msg)


def ibot_loss(
    student_patches_masked,
    student_masks,
    teacher_patches_masked,
    teacher_masks,
    student_class_masked,
    teacher_class_masked,
    student_temp,
):
    student_masks = student_masks.squeeze(dim=1)
    teacher_masks = teacher_masks.squeeze(dim=1)
    loss = loss_fns.masked_student_teacher_patch_softmax(
        student_patches_masked, teacher_patches_masked, student_masks, teacher_masks, student_temp
    )
    loss = loss + loss_fns.student_teacher_softmax(
        student_class_masked, teacher_class_masked, student_temp
    )
    return loss / 2


def dino_loss(
    local2global_dino_student,
    local2global_dino_teacher,
    global2global_dino_student,
    global2global_dino_teacher,
    student_temp,
):
    loss = loss_fns.student_teacher_global_softmax(
        local2global_dino_student, local2global_dino_teacher, student_temp
    ) + loss_fns.student_teacher_softmax(
        global2global_dino_student, global2global_dino_teacher, student_temp
    )
    return loss / 2


def get_loss_function_ssl(name):
    if name == "iBOT":
        return ibot_loss
    elif name == "DINO":
        return dino_loss
    elif name == "JEPA":
        return jepa_loss
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )
