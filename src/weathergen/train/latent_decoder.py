# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Latent downscaler output: decode the JEPA predictor output with the decoder of a frozen
autoencoder run (e.g. the CARRA2 autoencoder whose encoder is the FrozenTeacher).

    student encoder (ERA5 + satellites) -> latent cascade -> JEPA predictor
        -> predicted CARRA2 latent at `level` -> CARRA2 autoencoder decoder -> CARRA2 fields

Config (top level):

    latent_decoder:
      enabled: true
      run_id: <autoencoder run>      # e.g. the FrozenTeacher's teacher_run_id
      mini_epoch: -1                 # -1 = latest checkpoint
      level: 8                       # latent level of the predicted latent to decode
      streams: [CARRA2]              # streams to decode (default: all streams of that run)
      during_training: false         # also decode in training steps
      backprop: false                # training: let a physical loss (e.g. MSE + Haar on
                                     # CARRA2) train the student + predictor through the
                                     # frozen decoder (needs during_training: true)

The decoded fields are written as the physical predictions of those streams (they replace
the student's own decoder output for them), so validation / inference output and physical
losses of the current run see them. The decoder is frozen; it runs under no_grad unless
`backprop` is set (then gradients of a physical loss reach the predictor and the student).

The decoder was trained on the autoencoder's raw latent: use `target_norm: none` in the
JEPA config so that the predictor regresses that latent (with `layernorm` the predictor
output is a normalised latent and the decoder has to be fine-tuned on it first, stage 3;
point `run_id` to that fine-tuned run then).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from weathergen.common.config import Config, get_path_model, load_run_config, merge_configs
from weathergen.model.model import build_model_params
from weathergen.model.model_interface import get_model
from weathergen.train.frozen_run_utils import DatasetStreamSubset, frozen_run_streams
from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)

_DECODER_PREFIXES = ("embed_target_coords.", "target_token_engines.", "pred_heads.")


def _checkpoint_path(cf: Config, run_id: str, mini_epoch: int | None) -> Path:
    model_path = cf.get("model_path", None)
    path_run = Path(model_path if model_path else get_path_model(run_id=run_id)) / run_id
    tag = f"chkpt{mini_epoch:05d}" if mini_epoch is not None and mini_epoch != -1 else "latest"
    return path_run / f"{run_id}_{tag}.chkpt"


class PredictedLatentDecoder:
    """
    Frozen decoder of another run applied to a predicted latent. Not an nn.Module on purpose:
    it is attached to the student Model as a plain attribute, so it is not part of the
    student's state_dict, optimizer or FSDP sharding.
    """

    def __init__(
        self, ae_model, ae_params, level: int, streams: list[str], during_training, backprop
    ):
        self.ae_model = ae_model
        self.ae_params = ae_params
        self.level = level
        self.streams = streams
        self.during_training = during_training
        self.backprop = backprop

    @classmethod
    def from_config(cls, cf: Config, dataset, device) -> PredictedLatentDecoder | None:
        dcfg = cf.get("latent_decoder", None)
        if not dcfg or not dcfg.get("enabled", False):
            return None
        run_id = dcfg["run_id"]
        mini_epoch = dcfg.get("mini_epoch", -1)

        ae_config = load_run_config(run_id, mini_epoch, model_path=None)
        ae_config = merge_configs(ae_config, {"with_ddp": False, "with_fsdp": False})
        ae_run_streams = list((ae_config.get("streams", None) or {}).keys())
        ae_config, stream_idxs, ae_levels = frozen_run_streams(cf, ae_config)
        level = int(dcfg.get("level", max(ae_levels)))
        assert len(ae_levels) == 1 and ae_levels[0] == level, (
            f"latent_decoder: run {run_id} has latent levels {ae_levels}; a single-level "
            f"autoencoder at level {level} is supported"
        )
        streams = list(dcfg.get("streams", None) or ae_run_streams)
        for s in streams:
            assert s in ae_run_streams, f"latent_decoder: stream {s} is not a stream of {run_id}"

        # increment learning (datasets/increment_baseline.py): a decoder trained on increments
        # needs the baseline at the target coords, which the data loader only builds when
        # the stream of THIS run has the same `increment_baseline` block
        for s in streams:
            ae_inc = (ae_config.streams[s] or {}).get("increment_baseline", None) is not None
            own_inc = (cf.streams.get(s, None) or {}).get("increment_baseline", None) is not None
            if ae_inc != own_inc:
                raise ValueError(
                    f"latent_decoder: stream {s} of run {run_id} was trained "
                    f"{'with' if ae_inc else 'without'} increment_baseline, the stream {s} of "
                    f"this run is configured {'with' if own_inc else 'without'} it; use the "
                    "same increment_baseline block in both"
                )

        ae_model = get_model(ae_config, "student", DatasetStreamSubset(dataset, stream_idxs), {})

        # decoders of the decoded streams only
        params = torch.load(
            _checkpoint_path(cf, run_id, mini_epoch),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        dec = {
            k: v
            for k, v in params.items()
            if k.startswith(_DECODER_PREFIXES) and k.split(".")[1] in streams
        }
        assert len(dec) > 0, f"latent_decoder: no decoder weights for {streams} in {run_id}"
        missing, _ = ae_model.load_state_dict(dec, strict=False)
        missing_dec = [
            k for k in missing if k.startswith(_DECODER_PREFIXES) and k.split(".")[1] in streams
        ]
        assert not missing_dec, (
            f"latent_decoder: decoder weights missing in {run_id}: {missing_dec}"
        )
        for name in list(ae_model.pred_heads.keys()):
            if name not in streams:
                del ae_model.pred_heads[name]
        # the encoder and the forecast part are not needed
        ae_model.encoder = None
        ae_model.forecast_engine = None
        ae_model.requires_grad_(False)
        ae_model.eval()
        ae_model.to(device)
        ae_params = build_model_params(ae_config).create(ae_config).to(device)

        if is_root():
            logger.info(
                f"latent_decoder: decoding the predicted hl{level} latent with the {streams} "
                f"decoder(s) of run {run_id}"
            )
        during_training = bool(dcfg.get("during_training", False))
        backprop = bool(dcfg.get("backprop", False))
        assert during_training or not backprop, "latent_decoder.backprop needs during_training"
        return cls(ae_model, ae_params, level, streams, during_training, backprop)

    def decode(self, step: int, pred, batch, output, num_aux_tokens: int, training: bool):
        """
        pred : JEPA prediction, {level: (B, N, D)} (multi-level head) or (B, N, D).
        Writes the decoded fields of `streams` into output.physical[step].
        """
        if training and not self.during_training:
            return output
        if isinstance(pred, dict):
            if self.level not in pred:
                return output
            pred = pred[self.level]
        ae = self.ae_model
        grad = training and self.backprop
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            aux = torch.zeros(
                (pred.shape[0], ae.num_aux_tokens, pred.shape[-1]),
                dtype=pred.dtype,
                device=pred.device,
            )
            tokens = torch.cat([aux, pred if grad else pred.detach()], dim=1)
            ae.predict_decoders(self.ae_params, step, tokens, batch, output)
        return output


def attach_latent_decoder(model, cf: Config, dataset, device) -> None:
    """Attach the configured PredictedLatentDecoder (if any) to a (wrapped) Model."""
    decoder = PredictedLatentDecoder.from_config(cf, dataset, device)
    if decoder is None:
        return
    jepa = None
    for v in cf.training_config.get("losses", {}).values():
        if v.get("type", None) == "LossLatentSSLStudentTeacher":
            jepa = v.loss_fcts.get("JEPA", None)
    if jepa is not None and str(jepa.get("target_norm", "layernorm")).lower() != "none":
        logger.warning(
            "latent_decoder: JEPA target_norm is not 'none': the predictor regresses a "
            "normalised latent, which the autoencoder decoder was not trained on (fine-tune the "
            "decoder first and point latent_decoder.run_id to that run)."
        )
    base = getattr(model, "module", model)
    object.__setattr__(base, "latent_decoder", decoder)
