# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import Any

import torch
from omegaconf import OmegaConf

from weathergen.common.config import Config, load_run_config, merge_configs
from weathergen.datasets.domain_pyramid import build_domain_pyramid
from weathergen.model.model import build_model_params
from weathergen.model.model_interface import get_model
from weathergen.model.ssl_target_processing import (
    DINOTargetProcessing,
    JEPATargetProcessing,
    iBOTPatchTargetProcessing,
)
from weathergen.train.frozen_run_utils import DatasetStreamSubset, frozen_run_streams
from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, TargetAuxOutput
from weathergen.train.teacher_utils import (
    load_encoder_from_checkpoint,
    prepare_encoder_teacher,
)
from weathergen.train.utils import select_sample_metadata

logger = logging.getLogger(__name__)


def teacher_valid_cells(batch) -> dict[int, torch.Tensor]:
    """{level: (num_samples, num_cells) bool}: cells with >= 1 input token of the teacher."""
    tokens_lens = batch.tokens_lens
    if not isinstance(tokens_lens, dict):
        # single latent level: the level is not known here; key -1 = "the only level"
        tokens_lens = {-1: tokens_lens}
    return {lvl: tl.sum(dim=(0, 2)) > 0 for lvl, tl in tokens_lens.items()}


class _LevelSubsetBatch:
    """
    View of a (multi-level) BatchSamples for a frozen teacher whose latent has a subset of the
    student's latent levels: `tokens_lens` is restricted to the teacher's levels (a plain
    tensor for a single-level teacher); everything else is the student batch.
    """

    def __init__(self, batch, levels: list[int]):
        self._batch = batch
        tl = batch.tokens_lens
        self.tokens_lens = tl[levels[0]] if len(levels) == 1 else {lvl: tl[lvl] for lvl in levels}

    def __len__(self):
        return len(self._batch)

    def __getattr__(self, name):
        return getattr(self._batch, name)


class EncoderTeacher(TargetAndAuxModuleBase):
    """Base class for SSL teacher models.

    Handles shared logic: SSL loss extraction, target postprocessing, compute loop.
    Subclasses must implement forward_teacher().
    """

    def __init__(self, teacher_model, training_cfg, **kwargs):
        self.teacher_model = teacher_model

        # Extract SSL loss configs
        losses_cfg = [
            v.loss_fcts
            for k, v in training_cfg.losses.items()
            if v.type == "LossLatentSSLStudentTeacher"
        ]
        # TODO: support multiple LossLatentSSLStudentTeacher loss terms
        self.postprocess_targets = get_target_postprocessing(losses_cfg[0], training_cfg, **kwargs)

    def forward_teacher(self, model_params, batch) -> Any:
        raise NotImplementedError("Subclasses must implement forward_teacher()")

    def compute(self, bidx, batch, model_params, model) -> TargetAuxOutput:
        with torch.no_grad():
            outputs = self.forward_teacher(model_params, batch).get_latent_prediction(0)
            targets = {}
            for loss_name, target_module in self.postprocess_targets.items():
                if isinstance(outputs[loss_name], dict):
                    # multi-level JEPA: {level: targets} (identity target processing)
                    assert isinstance(target_module, JEPATargetProcessing)
                    targets[loss_name] = outputs[loss_name]
                else:
                    targets[loss_name] = target_module(outputs[loss_name])
            if "JEPA_deep" in outputs:
                targets["JEPA_deep"] = outputs["JEPA_deep"]
            if isinstance(targets.get("JEPA"), dict):
                # cells of each latent level with at least one teacher input token: targets
                # elsewhere are inferred by the teacher itself and are not used by the loss
                targets["JEPA_valid"] = teacher_valid_cells(batch)

            # collect target meta-information for selected samples
            aux_outputs = [select_sample_metadata(sample) for sample in batch.get_samples()]

            targets_out = TargetAuxOutput(batch.get_output_len(), batch.get_output_idxs())
            targets_out.latent = targets
            targets_out.aux_outputs = aux_outputs

            return targets_out

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        return

    def to_device(self, device) -> EncoderTeacher:
        for _, module in self.postprocess_targets.items():
            module.to(device)
        return self

    def get_current_beta(self, cur_step: int) -> float:
        beta = self.ema_model.get_current_beta(cur_step)
        return beta


class EMATeacher(EncoderTeacher):
    """SSL teacher using exponential moving average of student weights."""

    def __init__(self, model, ema_model, batch_size, training_cfg, **kwargs):
        super().__init__(model, training_cfg, **kwargs)
        self.ema_model = ema_model
        self.batch_size = batch_size
        self.reset()

    def forward_teacher(self, model_params, batch):
        return self.ema_model.forward_eval(model_params, batch)

    def reset(self, batch_size=None):
        self.ema_model.reset()
        if batch_size is not None:
            self.batch_size = batch_size

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        if self.ema_model.is_model_sharded:
            self.ema_model.ema_model.reshard()
        self.ema_model.update(istep, self.batch_size)

    def get_current_beta(self, cur_step: int) -> float:
        """Return the current EMA interpolation beta for monitoring."""
        return self.ema_model.get_current_beta(cur_step)


class FrozenTeacher(EncoderTeacher):
    """SSL teacher using a frozen pre-trained encoder.

    The encoder is loaded from a checkpoint and never updated. Non-encoder
    parts are discarded; latent heads are created fresh based on the student's
    SSL loss config.
    """

    def __init__(self, teacher_model, training_cfg, teacher_model_params=None, teacher_levels=None):
        super().__init__(teacher_model, training_cfg)
        self.teacher_model_params = teacher_model_params
        # latent levels of the teacher when they are a strict subset of the student's
        # (e.g. CARRA2 autoencoder at hl8 only, student at hl5 + hl8); None otherwise
        self.teacher_levels = teacher_levels

        # Freeze all parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

    @classmethod
    def from_pretrained(cls, cf: Config, dataset, device, params: dict) -> FrozenTeacher:
        """Create a FrozenTeacher from a pre-trained checkpoint.

        Args:
            cf: Full training config
            dataset: Dataset for model creation
            device: Target device
            params: Dict with 'teacher_run_id' and optional 'teacher_mini_epoch'
        """

        teacher_run_id = params["teacher_run_id"]
        teacher_mini_epoch = params.get("teacher_mini_epoch", -1)

        # Load teacher's config, create model with teacher's architecture
        teacher_config = load_run_config(teacher_run_id, teacher_mini_epoch, model_path=None)
        teacher_config = merge_configs(teacher_config, {"with_ddp": False, "with_fsdp": False})

        # deep SSL: the teacher has to tap the same encoder layers as the student
        deep_ssl = cf.training_config.get("deep_ssl", None)
        if deep_ssl is not None and deep_ssl.get("enabled", False):
            teacher_config = merge_configs(
                teacher_config, {"training_config": {"deep_ssl": OmegaConf.to_container(deep_ssl)}}
            )

        # The teacher runs on the student's batch. Build it on the student's streams at the
        # teacher's latent levels, in the student's order, so that its embedders and the
        # stream axis of the batch's tokens_lens line up (see frozen_run_streams).
        teacher_config, stream_idxs, teacher_levels = frozen_run_streams(cf, teacher_config)
        teacher_dataset = DatasetStreamSubset(dataset, stream_idxs)

        teacher_model = get_model(teacher_config, "student", teacher_dataset, {})

        # Load only encoder weights
        load_encoder_from_checkpoint(teacher_model, cf, teacher_run_id, teacher_mini_epoch, device)

        # Strip to encoder + create fresh heads (modules created here still have to be moved)
        prepare_encoder_teacher(teacher_model, cf.training_config, teacher_config)
        teacher_model.to(device)

        # Create model params matching teacher's architecture
        teacher_model_params = build_model_params(teacher_config).create(teacher_config).to(device)

        student_levels = build_domain_pyramid(cf).levels
        subset = teacher_levels if sorted(teacher_levels) != sorted(student_levels) else None
        return cls(teacher_model, cf.training_config, teacher_model_params, subset)

    def forward_teacher(self, model_params, batch):
        params = (
            self.teacher_model_params if self.teacher_model_params is not None else model_params
        )
        if self.teacher_levels is not None:
            batch = _LevelSubsetBatch(batch, self.teacher_levels)
        return self.teacher_model(params, batch)

    def reset(self, batch_size=None):
        pass

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        pass


def get_target_postprocessing(
    target_losses: dict[str, Any], training_cfg, **kwargs
) -> dict[str, torch.nn.Module]:
    """Create target postprocessing modules for each SSL loss type.

    Args:
        target_losses: Dict mapping loss name → loss config
        training_cfg: Training configuration

    Returns:
        Dict mapping loss name → target processing module
    """
    return_dict = {}
    for loss_name, conf in target_losses.items():
        if loss_name == "iBOT":
            for key in ("out_dim", "center_momentum", "teacher_temp", "teacher_style"):
                if key not in conf:
                    raise KeyError(f"iBOT config missing required key {key!r}")
            return_dict[loss_name] = iBOTPatchTargetProcessing(
                patch_out_dim=conf["out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["loss_extra_args"]["student_temp"],
                teacher_temp=conf["teacher_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "DINO":
            for key in ("out_dim", "center_momentum", "teacher_style"):
                if key not in conf:
                    raise KeyError(f"DINO config missing required key {key!r}")
            return_dict[loss_name] = DINOTargetProcessing(
                out_dim=conf["out_dim"],
                center_momentum=conf["center_momentum"],
                student_temp=conf["loss_extra_args"]["student_temp"],
                teacher_style=conf["teacher_style"],
            )
        elif loss_name == "JEPA":
            return_dict[loss_name] = JEPATargetProcessing()
        else:
            # We skip losses that are not handled by the teacher
            logger.debug(f"Skipping unknown loss type {loss_name!r} in target postprocessing")
            continue
    return return_dict
