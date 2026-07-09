# pylint: disable=bad-builtin
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
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.train.utils import TRAIN, VAL, Stage

_logger = logging.getLogger(__name__)


def get_num_samples(config) -> np.typing.NDArray:
    """
    Get number of samples in source/target config
    """
    return np.array([s_cfg.get("num_samples", 1) for _, s_cfg in config.items()])


class DynamicLossEMA:
    """
    Tracks and applies dynamic channel weights using an Exponential Moving Average (EMA)
    of inverse MSE, as described in Samudra 2.
    """

    def __init__(self, cfg: dict | None, streams_cfg: dict, device: str):
        self.enabled = cfg is not None
        if self.enabled:
            self.window = cfg.get("window", 100)
            self.L = cfg.get("L", 20.0)
            self.channel_weights_ema = {}
            for stream_name, stream_info in streams_cfg.items():
                num_channels = len(stream_info.train_target_channels)
                self.channel_weights_ema[stream_name] = torch.ones(num_channels, device=device)

    def get_weights(
        self, stream_name: str, weights_channels_static: torch.Tensor | None
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None

        ema = self.channel_weights_ema[stream_name]
        if ema.numel() > 0:
            l_min = ema.min().clamp(min=1e-6)
            # Clamp max weight to L * min weight as per Samudra 2 paper
            clamped_ema = ema.clamp(max=self.L * l_min)
            # Normalize so mean is 1.0 to preserve overall learning rate scale
            weights_channels = clamped_ema / clamped_ema.mean()
        else:
            weights_channels = ema.clone()

        if weights_channels_static is not None and weights_channels_static.numel() > 0:
            weights_channels = weights_channels * weights_channels_static

        return weights_channels

    def update(self, stream_name: str, loss_lfct_chs: torch.Tensor):
        if not self.enabled:
            return

        with torch.no_grad():
            mse_per_chan = loss_lfct_chs.detach().clamp(min=1e-6)
            inv_mse = 1.0 / mse_per_chan
            self.channel_weights_ema[stream_name] = (
                1.0 - 1.0 / self.window
            ) * self.channel_weights_ema[stream_name] + (1.0 / self.window) * inv_mse


class LossPhysical(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model during
    training and validation stages.

    This class handles the initialization and application of various loss functions,
    applies channel-specific weights, constructs masks for missing data, and
    aggregates losses across different data streams, channels, and forecast steps.
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.mode_cfg = mode_cfg
        self.stage = stage
        self.device = device
        self.name = "LossPhysical"

        # Dynamic Loss state (extract it before parsing the actual loss functions)
        self.dynamic_loss_cfg = loss_fcts.get("dynamic_loss")
        self.forecast_offset = self.mode_cfg.forecast.offset

        # dynamically load loss functions based on configuration and stage.
        # custom losses (wavelet, healpix cell, fft) are handled by dedicated
        # branches in compute_loss and have no direct function object in
        # loss_functions.py that matches the standard signature
        _custom_losses = (
            "haar_wavelet_cell",
            "global_haar_wavelet",
            "global_haar_wavelet_reshape",
            "global_haar_wavelet_reshape_geoweighted",
            "global_haar_wavelet_reshape_varweighted",
            "global_haar_wavelet_reshape_varweighted_crps",
            "global_haar_ll_reshape_varweighted",
            "healpix_cell_mse",
            "global_fft_mse",
        )
        self.loss_fcts = []
        for name, params in loss_fcts.items():
            if name == "dynamic_loss":
                continue
            fn = None if name in _custom_losses else getattr(loss_fns, name)
            self.loss_fcts.append([
                fn,
                params.get("weight", 1.0),
                name,
                {k: v for k, v in params.items() if k != "weight"},
            ])

        # neighbourhood structure for haar_wavelet_cell — set externally via
        # loss_calculator._hp_nbours = model_params.hp_nbours.cpu() in trainer.py
        self._hp_nbours = None

        self.dynamic_loss_ema = DynamicLossEMA(
            self.dynamic_loss_cfg if self.stage == TRAIN else None,
            self.cf.streams,
            self.device,
        )

    def _get_weights(self, stream_name, stream_info):
        """
        Get weights for current stream
        """

        device = self.device

        # Determine stream and channel loss weights based on the current stage
        if self.stage == TRAIN:
            # set loss_weights to 1. when not specified
            stream_info_loss_weight = stream_info.get("loss_weight", 1.0)
            weights_channels_static = (
                torch.tensor(stream_info["target_channel_weights"]).to(
                    device=device, non_blocking=True
                )
                if stream_info.get("target_channel_weights")
                else None
            )
        elif self.stage == VAL:
            # in validation mode, always unweighted loss
            stream_info_loss_weight = 1.0
            weights_channels_static = None

        if self.dynamic_loss_ema.enabled:
            weights_channels = self.dynamic_loss_ema.get_weights(
                stream_name, weights_channels_static
            )
        else:
            weights_channels = (
                weights_channels_static
                if weights_channels_static is None or weights_channels_static.numel() > 0
                else None
            )

        return stream_info_loss_weight, weights_channels

    def _get_output_step_weights(self, len_forecast_steps):
        timestep_weight_config = self.mode_cfg.get("forecast", {}).get("timestep_weight", {})
        if len(timestep_weight_config) == 0:
            return [1.0 for _ in range(len_forecast_steps)]
        weights_timestep_fct = getattr(loss_fns, list(timestep_weight_config.keys())[0])
        decay_factor = list(timestep_weight_config.values())[0]["decay_factor"]
        return weights_timestep_fct(len_forecast_steps, decay_factor)

    def _get_location_weights(self, stream_info, target_coords, substep_masks):
        location_weight_type = stream_info.get("location_weight", None)
        if location_weight_type is None:
            return [None for _ in substep_masks]

        target_coords = target_coords.to(self.device, non_blocking=True)
        weights_locations_fct = getattr(loss_fns, location_weight_type)
        weights_locations = [weights_locations_fct(target_coords[mask]) for mask in substep_masks]

        return weights_locations

    def _get_substep_masks(self, stream_info, output_step, target_times):
        """
        Find substeps and create corresponding masks (reused across loss functions)
        """

        tok_spacetime = stream_info.get("tokenize_spacetime", None)
        target_times_unique = np.unique(target_times) if tok_spacetime else [target_times]
        substep_masks = []
        for t in target_times_unique:
            # find substep
            mask_t = torch.tensor(t == target_times).to(self.device, non_blocking=True)
            substep_masks.append(mask_t)

        return substep_masks

    @staticmethod
    def _loss_per_loss_function(
        loss_fct,
        target: torch.Tensor,
        pred: torch.Tensor,
        substep_masks: list[torch.Tensor],
        weights_channels: torch.Tensor,
        weights_locations: list[torch.Tensor],
        loss_fct_params: dict | None = None,
    ):
        """
        Compute loss for given loss function
        """

        loss_lfct = torch.tensor(0.0, device=target.device, requires_grad=True)
        losses_chs = torch.zeros(target.shape[-1], device=target.device, dtype=torch.float32)

        ctr_substeps = 0
        for i_t, mask_t in enumerate(substep_masks):
            assert (
                mask_t.sum() == len(weights_locations[i_t])
                if weights_locations[i_t] is not None
                else True
            )

            loss, loss_chs = loss_fct(
                target[mask_t], pred[:, mask_t], 
                weights_channels, weights_locations[i_t],
                **(loss_fct_params or {}),
            )

            # accumulate loss
            loss_lfct = loss_lfct + loss
            losses_chs = losses_chs + loss_chs.detach() if len(loss_chs) > 0 else losses_chs
            ctr_substeps += 1 if loss > 0.0 else 0

        # normalize over forecast steps in window
        losses_chs /= ctr_substeps if ctr_substeps > 0 else 1.0

        # TODO: substep weight
        loss_lfct = loss_lfct / (ctr_substeps if ctr_substeps > 0 else 1.0)

        return loss_lfct, losses_chs

    @staticmethod
    def _loss_wavelet_per_cell(
        target: torch.Tensor,
        pred: torch.Tensor,
        target_coords: torch.Tensor,
        target_coords_lens: torch.Tensor,
        hp_nbours: torch.Tensor,
        weights_channels: torch.Tensor | None,
        geoinfo_offset: int,
        grid_size: int = 32,
        detail_weight: float = 2.0,
        num_levels: int = 2,
        min_points: int = 50,
        stream_name: str = "",
    ):
        num_cells    = target_coords_lens.shape[0]
        num_channels = target.shape[-1]
        dev          = target.device

        cumlen = torch.cat([
            torch.zeros(1, dtype=torch.long, device=dev),
            target_coords_lens.long().to(dev).cumsum(0),
        ])

        loss_total = torch.tensor(0.0, device=dev, requires_grad=True)
        loss_chs   = torch.zeros(num_channels, device=dev)
        n_active   = 0

        for cell_id in range(num_cells):
            i0 = int(cumlen[cell_id].item())
            i1 = int(cumlen[cell_id + 1].item())
            if i1 - i0 == 0:
                continue

            nbour_ids = hp_nbours[cell_id, 1:9].cpu().tolist()
            nbour_ids = [int(n) for n in nbour_ids if int(n) != cell_id]

            patch_target_parts = [target[i0:i1]]
            patch_pred_parts   = [pred[:, i0:i1]]
            patch_xy_parts     = [
                target_coords[i0:i1, geoinfo_offset + 1 : geoinfo_offset + 3]
            ]

            for nbr in nbour_ids:
                if nbr >= num_cells:
                    continue
                j0 = int(cumlen[nbr].item())
                j1 = int(cumlen[nbr + 1].item())
                if j1 <= j0:
                    continue
                patch_target_parts.append(target[j0:j1])
                patch_pred_parts.append(pred[:, j0:j1])
                patch_xy_parts.append(
                    target_coords[j0:j1, geoinfo_offset + 1 : geoinfo_offset + 3]
                )

            patch_target = torch.cat(patch_target_parts, dim=0)
            patch_pred   = torch.cat(patch_pred_parts,   dim=1)
            patch_xy     = torch.cat(patch_xy_parts,     dim=0)

            if patch_target.shape[0] < min_points:
                continue

            cell_loss, cell_loss_chs = loss_fns.haar_wavelet_mse_local_patch(
                patch_target,
                patch_pred,
                patch_xy,
                grid_size=grid_size,
                detail_weight=detail_weight,
                num_levels=num_levels,
                min_points=min_points,
                stream_name=stream_name,
            )

            if cell_loss > 0.0:
                loss_total = loss_total + cell_loss
                loss_chs   = loss_chs + cell_loss_chs
                n_active  += 1

        if n_active > 0:
            loss_total = loss_total / n_active
            loss_chs   = loss_chs   / n_active

        if weights_channels is not None:
            loss_total = torch.mean(loss_chs * weights_channels.to(dev))

        return loss_total, loss_chs

    @staticmethod
    def _loss_global_haar(
        target, pred, target_coords_raw, weights_channels,
        stream_name="", grid_resolution_deg=0.03,
        detail_weight=2.0, num_levels=3,
    ):
        if target.shape[0] == 0:
            return (
                torch.tensor(0.0, device=target.device, requires_grad=True),
                torch.zeros(target.shape[-1], device=target.device),
            )
        target_coords_raw = target_coords_raw.to(target.device)
        return loss_fns.global_haar_wavelet_mse(
            target, pred, target_coords_raw,
            weights_channels=weights_channels, weights_points=None,
            grid_resolution_deg=grid_resolution_deg,
            detail_weight=detail_weight, num_levels=num_levels,
            stream_name=stream_name,
        )

    @staticmethod
    def _loss_global_haar_reshape(
        target, pred, target_coords_raw, weights_channels,
        stream_name="", template_path="",
        detail_weight=2.0, num_levels=3,
    ):
        if target.shape[0] == 0:
            return (
                torch.tensor(0.0, device=target.device, requires_grad=True),
                torch.zeros(target.shape[-1], device=target.device),
            )
        target_coords_raw = target_coords_raw.to(target.device)
        return loss_fns.global_haar_wavelet_reshape(
            target, pred, target_coords_raw,
            weights_channels=weights_channels, weights_points=None,
            template_path=template_path,
            detail_weight=detail_weight, num_levels=num_levels,
            stream_name=stream_name,
        )

    @staticmethod
    def _loss_global_haar_geoweighted(
        target, pred, target_coords_raw, target_coords_local, weights_channels,
        stream_name="", template_path="",
        detail_weight=2.0, num_levels=3,
        geoinfo_channel_idx=0, geoinfo_invert=False,
    ):
        if target.shape[0] == 0:
            return (
                torch.tensor(0.0, device=target.device, requires_grad=True),
                torch.zeros(target.shape[-1], device=target.device),
            )
        target_coords_raw   = target_coords_raw.to(target.device)
        target_coords_local = target_coords_local.to(target.device)
        return loss_fns.global_haar_wavelet_reshape_geoweighted(
            target, pred, target_coords_raw, target_coords_local,
            weights_channels=weights_channels, weights_points=None,
            template_path=template_path,
            detail_weight=detail_weight, num_levels=num_levels,
            geoinfo_channel_idx=geoinfo_channel_idx,
            geoinfo_invert=geoinfo_invert,
            stream_name=stream_name,
        )

    @staticmethod
    def _loss_global_haar_varweighted(
        target, pred, target_coords_raw, weights_channels,
        stream_name="", template_path="",
        detail_weight=2.0, num_levels=3, var_weight_epsilon=1e-3,
    ):
        if target.shape[0] == 0:
            return (
                torch.tensor(0.0, device=target.device, requires_grad=True),
                torch.zeros(target.shape[-1], device=target.device),
            )
        target_coords_raw = target_coords_raw.to(target.device)
        return loss_fns.global_haar_wavelet_reshape_varweighted(
            target, pred, target_coords_raw,
            weights_channels=weights_channels, weights_points=None,
            template_path=template_path,
            detail_weight=detail_weight, num_levels=num_levels,
            var_weight_epsilon=var_weight_epsilon,
            stream_name=stream_name,
        )

    @staticmethod
    def _loss_global_haar_varweighted_crps(
        target, pred, target_coords_raw, weights_channels,
        stream_name="", template_path="",
        num_levels=3, var_weight_epsilon=1e-3,
        fair=True, normalization="std",
    ):
        if target.shape[0] == 0:
            return (
                torch.tensor(0.0, device=target.device, requires_grad=True),
                torch.zeros(target.shape[-1], device=target.device),
            )
        target_coords_raw = target_coords_raw.to(target.device)
        return loss_fns.global_haar_wavelet_reshape_varweighted_crps(
            target, pred, target_coords_raw,
            weights_channels=weights_channels, weights_points=None,
            template_path=template_path,
            num_levels=num_levels,
            var_weight_epsilon=var_weight_epsilon,
            fair=fair, normalization=normalization,
            ll_weight=ll_weight,
            stream_name=stream_name,
        )

    @staticmethod
    def _loss_global_haar_ll_varweighted(
        target, pred, target_coords_raw, weights_channels,
        stream_name="", template_path="",
        num_levels=3, var_weight_epsilon=1e-3,
    ):
        if target.shape[0] == 0:
            return (
                torch.tensor(0.0, device=target.device, requires_grad=True),
                torch.zeros(target.shape[-1], device=target.device),
            )
        target_coords_raw = target_coords_raw.to(target.device)
        return loss_fns.global_haar_ll_reshape_varweighted(
            target, pred, target_coords_raw,
            weights_channels=weights_channels, weights_points=None,
            template_path=template_path,
            num_levels=num_levels,
            var_weight_epsilon=var_weight_epsilon,
            stream_name=stream_name,
        )

    @staticmethod
    def _loss_healpix_cell_mse(
        target, pred, target_coords_lens, weights_channels, stream_name="",
    ):
        if target.shape[0] == 0 or target_coords_lens.sum() == 0:
            return (
                torch.tensor(0.0, device=target.device, requires_grad=True),
                torch.zeros(target.shape[-1], device=target.device),
            )
        return loss_fns.healpix_cell_mse(
            target, pred, target_coords_lens.to(target.device),
            weights_channels, weights_points=None, stream_name=stream_name,
        )

    @staticmethod
    def _loss_global_fft(
        target, pred, target_coords_raw, weights_channels,
        stream_name="", template_path="", freq_weight_power=0.0,
    ):
        if target.shape[0] == 0:
            return (
                torch.tensor(0.0, device=target.device, requires_grad=True),
                torch.zeros(target.shape[-1], device=target.device),
            )
        target_coords_raw = target_coords_raw.to(target.device)
        return loss_fns.global_fft_mse(
            target, pred, target_coords_raw,
            weights_channels=weights_channels, weights_points=None,
            template_path=template_path,
            freq_weight_power=freq_weight_power,
            stream_name=stream_name,
        )

    def compute_loss(self, preds: dict, targets: dict, metadata) -> LossValues:
        """
        Computes the total loss for a given batch of predictions and corresponding
        stream data.

        The computed loss is:

        Mean_{stream}( Mean_{output_steps}( Mean_{loss_fcts}( loss_fct( target, pred, weigths) )))

        This method orchestrates the calculation of the overall loss by iterating through
        different data streams, forecast steps, channels, and configured loss functions.
        It applies weighting, handles NaN values through masking, and accumulates
        detailed loss metrics for logging.

        Args:
            preds: A nested list of prediction tensors. The outer list represents forecast steps,
                   the inner list represents streams. Each tensor contains predictions for that
                   step and stream.
            streams_data: A nested list representing the input batch data. The outer list is for
                          batch items, the inner list for streams. Each element provides an object
                          (e.g., dataclass instance) containing target data and metadata.

        Returns:
            A ModelLoss dataclass instance containing:
            - loss: The loss for back-propagation.
            - losses_all: A dictionary mapping stream names to a tensor of per-channel and
                          per-loss-function losses, normalized by non-empty targets/forecast steps.
            - stddev_all: A dictionary mapping stream names to a tensor of mean standard deviations
                          of predictions for channels with statistical loss functions, normalized.
        """

        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        # counter for non-empty targets
        ctr_streams = 0

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all = defaultdict(dict)
        stddev_all = defaultdict(dict)
        _spread_acc: dict[str, list] = defaultdict(list)

        source2target_idxs, output_info, target2source_idxs, target_info = metadata

        # TODO: iterate over batch dimension
        for stream_name, stream_info in self.cf.streams.items():
            # TODO: avoid this
            target_channels = (
                stream_info.val_target_channels
                if self.stage == "val"
                else stream_info.train_target_channels
            )

            losses_all[stream_name] = defaultdict(dict)

            stream_loss_weight, weights_channels = self._get_weights(stream_name, stream_info)
            if self.dynamic_loss_ema.enabled and weights_channels is not None:
                losses_all[stream_name][str(self.forecast_offset)]["mse_ema_weight"] = {}
                for ch_n, w in zip(target_channels, weights_channels, strict=True):
                    losses_all[stream_name][str(self.forecast_offset)]["mse_ema_weight"][ch_n] = w.item()

            # TODO: make nicer
            output_step_loss_weights = self._get_output_step_weights(len(targets.output_idxs))
            if len(targets.physical) - len(targets.output_idxs) > 0:
                output_step_loss_weights.insert(0, None)

            # loss_stream: loss for given stream
            loss_stream = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_timesteps = 0
            for timestep_idx, (preds_cur, target_cur) in enumerate(
                zip(preds.physical, targets.physical, strict=True)
            ):
                preds_batch = preds_cur.get(stream_name, [])
                if not preds_batch:
                    # skip to next timestep if preds of current timestep are empty
                    continue

                targets_batch = target_cur[stream_name]["target"]
                targets_coords_batch = target_cur[stream_name]["target_coords"]
                targets_coords_local_batch = target_cur[stream_name].get(
                    "target_coords_local", None
                )
                targets_coords_lens_batch = target_cur[stream_name].get(
                    "target_coords_lens", None
                )
                targets_times_batch = target_cur[stream_name]["target_times"]
                targets_params = target_cur[stream_name]["target_metda_data"]
                targets_is_spoof = target_cur[stream_name]["is_spoof"]

                output_step_weight = output_step_loss_weights[timestep_idx]

                # loss_timestep: loss for given timestep
                loss_timestep = torch.tensor(0.0, device=self.device, requires_grad=True)
                ctr_batch = 0
                for pred, pred_params in zip(preds_batch, output_info, strict=True):
                    # source has a unique target but index is not invariant with multiple
                    # target_aux calculators
                    target_idx_native = pred_params.global_params.get("correspondence", -1)
                    target_idx = [
                        i
                        for i, t in enumerate(targets_params)
                        if t[stream_name].global_params["idx"] == target_idx_native
                    ]
                    # source/model_input has no target for physical loss
                    if len(target_idx) == 0:
                        continue
                    # source -> target correspondence has to be unique
                    assert len(target_idx) == 1
                    target_idx = target_idx[0]

                    if pred.shape[0] > 1 and not targets_is_spoof[target_idx] and pred.shape[1] > 0:
                        _spread_acc[stream_name].append(
                            pred.detach().to(torch.float32).std(dim=0).mean()
                        )

                    # current target data
                    target = targets_batch[target_idx]
                    target_times = targets_times_batch[target_idx]

                    # get masks for sub-time steps
                    substep_masks = self._get_substep_masks(stream_info, timestep_idx, target_times)

                    # get weights for locations
                    weights_locations = self._get_location_weights(
                        stream_info, targets_coords_batch[target_idx], substep_masks
                    )

                    # loss_st_corr: loss for give source-target correspondence
                    loss_st_corr = torch.tensor(0.0, device=self.device, requires_grad=True)
                    ctr_loss_fcts = 0
                    for loss_fct, loss_fct_weight, loss_fct_name, loss_fct_params in self.loss_fcts:
                        # skip is loss is not computed for this sample
                        if loss_fct_name not in pred_params.global_params["loss"]:
                            continue

                        # spoofed inputs are masked in the output calculations
                        is_spoof = targets_is_spoof[target_idx]
                        sw = 0.0 if is_spoof else 1.0
                        spoof_weight = torch.tensor(sw, device=self.device, requires_grad=False)

                        # skip if either target or prediction has no data points
                        if not (target.shape[0] > 0 and pred.shape[0] > 0):
                            continue

                        pred = pred.reshape([pred.shape[0], *target.shape])
                        assert pred.shape[1] > 0

                        losses_all[stream_name][str(timestep_idx)][loss_fct_name] = defaultdict(
                            dict
                        )

                        # ---- dispatch: custom losses vs standard path ----
                        if loss_fct_name == "haar_wavelet_cell":
                            tc_lens = (
                                targets_coords_lens_batch[target_idx]
                                if targets_coords_lens_batch is not None else None
                            )
                            tc_local = (
                                targets_coords_local_batch[target_idx]
                                if targets_coords_local_batch is not None else None
                            )
                            n_geoinfo = len(stream_info.get("geoinfo_channels", []))
                            geoinfo_offset = 1 + 6 + n_geoinfo

                            if tc_lens is not None and tc_local is not None \
                                    and self._hp_nbours is not None:
                                loss_lfct, loss_lfct_chs = self._loss_wavelet_per_cell(
                                    target, pred, tc_local,
                                    tc_lens.to(self.device),
                                    self._hp_nbours.to(self.device),
                                    weights_channels,
                                    geoinfo_offset=geoinfo_offset,
                                    stream_name=stream_name,
                                    **loss_fct_params,
                                )
                            else:
                                _logger.warning(
                                    "[haar_wavelet_cell] missing target_coords_lens/"
                                    "target_coords_local/_hp_nbours; loss set to zero."
                                )
                                loss_lfct = torch.tensor(
                                    0.0, device=self.device, requires_grad=True
                                )
                                loss_lfct_chs = torch.zeros(
                                    target.shape[-1], device=self.device
                                )

                        elif loss_fct_name == "global_haar_wavelet":
                            stream_grid_res = stream_info.get(
                                "grid_resolution_deg",
                                loss_fct_params.get("grid_resolution_deg", 0.03),
                            )
                            loss_lfct, loss_lfct_chs = self._loss_global_haar(
                                target, pred, targets_coords_batch[target_idx],
                                weights_channels, stream_name=stream_name,
                                grid_resolution_deg=stream_grid_res,
                                **{k: v for k, v in loss_fct_params.items()
                                   if k != "grid_resolution_deg"},
                            )

                        elif loss_fct_name == "global_haar_wavelet_reshape":
                            loss_lfct, loss_lfct_chs = self._loss_global_haar_reshape(
                                target, pred, targets_coords_batch[target_idx],
                                weights_channels, stream_name=stream_name,
                                template_path=stream_info.get("template_path", ""),
                                **{k: v for k, v in loss_fct_params.items()
                                   if k != "template_path"},
                            )

                        elif loss_fct_name == "global_haar_wavelet_reshape_geoweighted":
                            tc_local = (
                                targets_coords_local_batch[target_idx]
                                if targets_coords_local_batch is not None else None
                            )
                            if tc_local is not None:
                                loss_lfct, loss_lfct_chs = self._loss_global_haar_geoweighted(
                                    target, pred, targets_coords_batch[target_idx],
                                    tc_local, weights_channels,
                                    stream_name=stream_name,
                                    template_path=stream_info.get("template_path", ""),
                                    **{k: v for k, v in loss_fct_params.items()
                                       if k != "template_path"},
                                )
                            else:
                                loss_lfct = torch.tensor(
                                    0.0, device=self.device, requires_grad=True
                                )
                                loss_lfct_chs = torch.zeros(
                                    target.shape[-1], device=self.device
                                )

                        elif loss_fct_name == "global_haar_wavelet_reshape_varweighted":
                            loss_lfct, loss_lfct_chs = self._loss_global_haar_varweighted(
                                target, pred, targets_coords_batch[target_idx],
                                weights_channels, stream_name=stream_name,
                                template_path=stream_info.get("template_path", ""),
                                **{k: v for k, v in loss_fct_params.items()
                                   if k != "template_path"},
                            )

                        elif loss_fct_name == "global_haar_wavelet_reshape_varweighted_crps":
                            loss_lfct, loss_lfct_chs = self._loss_global_haar_varweighted_crps(
                                target, pred, targets_coords_batch[target_idx],
                                weights_channels, stream_name=stream_name,
                                template_path=stream_info.get("template_path", ""),
                                **{k: v for k, v in loss_fct_params.items()
                                    if k != "template_path"},
                            )

                        elif loss_fct_name == "global_haar_ll_reshape_varweighted":
                            loss_lfct, loss_lfct_chs = self._loss_global_haar_ll_varweighted(
                                target, pred, targets_coords_batch[target_idx],
                                weights_channels, stream_name=stream_name,
                                template_path=stream_info.get("template_path", ""),
                                **{k: v for k, v in loss_fct_params.items()
                                   if k != "template_path"},
                            )

                        elif loss_fct_name == "healpix_cell_mse":
                            tc_lens = (
                                targets_coords_lens_batch[target_idx]
                                if targets_coords_lens_batch is not None else None
                            )
                            if tc_lens is not None:
                                loss_lfct, loss_lfct_chs = self._loss_healpix_cell_mse(
                                    target, pred, tc_lens, weights_channels,
                                    stream_name=stream_name,
                                )
                            else:
                                loss_lfct = torch.tensor(
                                    0.0, device=self.device, requires_grad=True
                                )
                                loss_lfct_chs = torch.zeros(
                                    target.shape[-1], device=self.device
                                )

                        elif loss_fct_name == "global_fft_mse":
                            loss_lfct, loss_lfct_chs = self._loss_global_fft(
                                target, pred, targets_coords_batch[target_idx],
                                weights_channels, stream_name=stream_name,
                                template_path=stream_info.get("template_path", ""),
                                **{k: v for k, v in loss_fct_params.items()
                                   if k != "template_path"},
                            )

                        else:
                            # standard pointwise loss functions
                            loss_lfct, loss_lfct_chs = self._loss_per_loss_function(
                                loss_fct,
                                target,
                                pred,
                                substep_masks,
                                weights_channels,
                                weights_locations,
                                loss_fct_params,
                            )
                        # ---- end dispatch ----

                        for ch_n, v in zip(target_channels, loss_lfct_chs, strict=True):
                            losses_all[stream_name][str(timestep_idx)][loss_fct_name][ch_n] = (
                                spoof_weight * v if v != 0.0 and not is_spoof else torch.nan
                            )

                        # Update EMA for dynamic loss if enabled
                        if (
                            self.dynamic_loss_ema.enabled
                            and timestep_idx == self.forecast_offset
                            and loss_fct_name == "mse"
                            and not is_spoof
                        ):
                            self.dynamic_loss_ema.update(stream_name, loss_lfct_chs)

                        # Add the weighted and normalized loss from this loss function to the total
                        # batch loss
                        loss_cur_w = spoof_weight * loss_fct_weight * loss_lfct * output_step_weight
                        loss_st_corr = loss_st_corr + loss_cur_w
                        ctr_loss_fcts += 1 if (loss_cur_w > 0.0 and not is_spoof) else 0

                    loss_timestep = loss_timestep + loss_st_corr
                    ctr_batch += 1 if ctr_loss_fcts > 0.0 else 0

                loss_stream = loss_stream + loss_timestep
                ctr_timesteps += 1 if ctr_batch > 0 else 0

            denom = ctr_timesteps if ctr_timesteps > 0 else 1.0
            loss = loss + (stream_loss_weight * loss_stream) / denom

            ctr_streams += 1 if ctr_timesteps > 0 else 0

        # normalize by all targets and forecast steps that were non-empty
        # (with each having an expected loss of 1 for an uninitalized neural net)
        if loss == 0.0:
            _logger.warning(
                "Loss is 0.0, likely incorrect configuration. Check stream"
                " support time and training configuration."
            )
        loss = loss / ctr_streams if ctr_streams > 0 else loss

        def _nested_dict():
            return defaultdict(dict)

        # Reorder losses_all to [stream_name][loss_fct_name][ch_n][output_step]
        reordered_losses = defaultdict(dict)
        for stream_name, output_step_dict in losses_all.items():
            reordered_losses[stream_name] = defaultdict(_nested_dict)
            for output_step, lfct_dict in output_step_dict.items():
                for loss_fct_name, ch_dict in lfct_dict.items():
                    for ch_n, v in ch_dict.items():
                        reordered_losses[stream_name][loss_fct_name][ch_n][output_step] = v

        # Calculate per stream, per lfct average across channels and output_steps
        for stream_name, lfct_dict in reordered_losses.items():
            for loss_fct_name, ch_dict in lfct_dict.items():
                reordered_losses[stream_name][loss_fct_name]["avg"] = 0
                count = 0
                for ch_n, output_step_dict in ch_dict.items():
                    if ch_n != "avg":
                        for _, v in output_step_dict.items():
                            v = 0.0 if type(v) is float and np.isnan(v) else v
                            reordered_losses[stream_name][loss_fct_name]["avg"] += v
                            count += 1
                reordered_losses[stream_name][loss_fct_name]["avg"] /= count

        for sname, vals in _spread_acc.items():
            stddev_all[sname]["stddev_avg"] = torch.stack(vals).mean()

        # Return all computed loss components encapsulated in a ModelLoss dataclass
        return LossValues(loss=loss, losses_all=reordered_losses, stddev_all=None)
