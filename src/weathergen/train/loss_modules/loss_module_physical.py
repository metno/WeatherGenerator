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

        # dynamically load loss functions based on configuration and stage
#        self.loss_fcts = [
#            [
#                getattr(loss_fns, name),
#                params.get("weight", 1.0),
#                name,
#            ]
#            for name, params in loss_fcts.items()
#        ]
        # CrL
        self.loss_fcts = []
        for name, params in loss_fcts.items():
            if name == "haar_wavelet_cell":
                fn = None   # handled separately in compute_loss via _loss_wavelet_per_cell
            else:
                fn = getattr(loss_fns, name)
            self.loss_fcts.append([
                fn,
                params.get("weight", 1.0),
                name,
                {k: v for k, v in params.items() if k not in ("weight",)},
            ])
        # store hp_nbours for wavelet loss — set externally after model init
        self._hp_nbours = None   # set via loss_calculator._hp_nbours = model_params.hp_nbours

    def _get_weights(self, stream_info):
        """
        Get weights for current stream
        """

        device = self.device

        # Determine stream and channel loss weights based on the current stage
        if self.stage == TRAIN:
            # set loss_weights to 1. when not specified
            stream_info_loss_weight = stream_info.get("loss_weight", 1.0)
            weights_channels = (
                torch.tensor(stream_info["target_channel_weights"]).to(
                    device=device, non_blocking=True
                )
                if "target_channel_weights" in stream_info
                else None
            )
        elif self.stage == VAL:
            # in validation mode, always unweighted loss
            stream_info_loss_weight = 1.0
            weights_channels = None

        return stream_info_loss_weight, weights_channels

    def _get_output_step_weights(self, len_forecast_steps):
        timestep_weight_config = self.mode_cfg.get("forecast", {}).get("timestep_weight", {})
        if len(timestep_weight_config) == 0:
            return [1.0 for _ in range(len_forecast_steps)]
        weights_timestep_fct = getattr(loss_fns, list(timestep_weight_config.keys())[0])
        decay_factor = list(timestep_weight_config.values())[0]["decay_factor"]
        return weights_timestep_fct(len_forecast_steps, decay_factor)

    def _get_location_weights(self, stream_info, target_coords):
        location_weight_type = stream_info.get("location_weight", None)
        if location_weight_type is None:
            return None
        weights_locations_fct = getattr(loss_fns, location_weight_type)
        weights_locations = weights_locations_fct(target_coords)
        weights_locations = weights_locations.to(device=self.device, non_blocking=True)

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
        weights_locations: torch.Tensor,
    ):
        """
        Compute loss for given loss function
        """

        assert loss_fct is not None, (
            "_loss_per_loss_function called with None loss function — "
            "haar_wavelet_cell must be handled before this point"
        )

        loss_lfct = torch.tensor(0.0, device=target.device, requires_grad=True)
        losses_chs = torch.zeros(target.shape[-1], device=target.device, dtype=torch.float32)

        ctr_substeps = 0
        for mask_t in substep_masks:
            assert mask_t.sum() == len(weights_locations) if weights_locations is not None else True

            loss, loss_chs = loss_fct(
                target[mask_t], pred[:, mask_t], weights_channels, weights_locations
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

#########################################################################
# CrL
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
    ):
        """
        Iterate over HEALPix cells. For cells with >= min_points target points
        (counting the central cell + 1-ring together), compute the 2D Haar
        wavelet MSE on the local patch and accumulate.
    
        Args:
            target             : (total_points, num_channels)
            pred               : (ens_size, total_points, num_channels)
            target_coords      : (total_points, coord_dim) — full descriptor
            target_coords_lens : (num_cells,) int32 — points per cell
            hp_nbours          : (num_cells, 9) int32 — col 0 = self, 1-8 = 1-ring
            weights_channels   : (num_channels,) or None
            geoinfo_offset     : index of the first local coord column in target_coords
                                 = 1 + n_time + n_geoinfo
            grid_size, detail_weight, num_levels, min_points: passed to haar_wavelet_mse_local_patch
        """
        import weathergen.train.loss_modules.loss_functions as _lf
    
        num_cells    = target_coords_lens.shape[0]
        num_channels = target.shape[-1]
        dev          = target.device
    
        # build cumulative offsets into the flat target/pred tensors
        cumlen = torch.cat([
            torch.zeros(1, dtype=torch.long, device=dev),
            target_coords_lens.long().to(dev).cumsum(0)
        ])
    
        loss_total  = torch.tensor(0.0, device=dev, requires_grad=True)
        loss_chs    = torch.zeros(num_channels, device=dev)
        n_active    = 0
    
        for cell_id in range(num_cells):
            # --- collect central cell points ---
            i0 = int(cumlen[cell_id].item())
            i1 = int(cumlen[cell_id + 1].item())
            n_central = i1 - i0
            if n_central == 0:
                continue
    
            # --- collect 1-ring neighbour points ---
            # hp_nbours[cell_id, 0] = self, [1:9] = neighbours
            nbour_ids = hp_nbours[cell_id, 1:9].cpu().tolist()
            nbour_ids = [int(n) for n in nbour_ids if int(n) != cell_id]
    
            patch_target_parts = [target[i0:i1]]
            patch_pred_parts   = [pred[:, i0:i1]]
            patch_xy_parts     = [target_coords[i0:i1, geoinfo_offset + 1 : geoinfo_offset + 3]]
    
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
    
            n_patch = patch_target.shape[0]
            if n_patch < min_points:
                continue
    
            # --- compute wavelet loss for this patch ---
            cell_loss, cell_loss_chs = _lf.haar_wavelet_mse_local_patch(
                patch_target,
                patch_pred,
                patch_xy,
                grid_size=grid_size,
                detail_weight=detail_weight,
                num_levels=num_levels,
                min_points=min_points,
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

        source2target_idxs, output_info, target2source_idxs, target_info = metadata

        # TODO: iterate over batch dimension
        for stream_info in self.cf.streams:
            stream_name = stream_info["name"]
            # TODO: avoid this
            target_channels = (
                stream_info.val_target_channels
                if self.stage == "val"
                else stream_info.train_target_channels
            )

            losses_all[stream_name] = defaultdict(dict)

            stream_loss_weight, weights_channels = self._get_weights(stream_info)

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

                    # get weights for locations
                    weights_locations = self._get_location_weights(
                        stream_info, targets_coords_batch[target_idx]
                    )

                    # loss_st_corr: loss for give source-target correspondence
                    loss_st_corr = torch.tensor(0.0, device=self.device, requires_grad=True)
                    ctr_loss_fcts = 0

                    for loss_fct, loss_fct_weight, loss_fct_name, loss_fct_params in self.loss_fcts:

                        print(f"[DEBUG] loss_fct_name='{loss_fct_name}', fn={loss_fct}")
                        
                        # skip if loss is not computed for this sample
                        if loss_fct_name not in pred_params.global_params["loss"]:
                            continue
                    
                        target      = targets_batch[target_idx]
                        target_times = targets_times_batch[target_idx]
                    
                        # spoofed inputs are masked in the output calculations
                        is_spoof    = targets_is_spoof[target_idx]
                        sw          = 0.0 if is_spoof else 1.0
                        spoof_weight = torch.tensor(sw, device=self.device, requires_grad=False)
                    
                        # skip if either target or prediction has no data points
                        if not (target.shape[0] > 0 and pred.shape[0] > 0):
                            continue
                    
                        # reshape prediction tensor to match target's dimensions: extract
                        # data/coords and remove token dimension if it exists.
                        # expected shape of pred is [ensemble_size, num_samples, num_channels].
                        pred = pred.reshape([pred.shape[0], *target.shape])
                        assert pred.shape[1] > 0
                    
                        # get masks for sub-time steps
                        substep_masks = self._get_substep_masks(
                            stream_info, timestep_idx, target_times
                        )
                    
                        losses_all[stream_name][str(timestep_idx)][loss_fct_name] = defaultdict(dict)
                    
                        if loss_fct_name == "haar_wavelet_cell":
                            # determine geoinfo_offset for this stream:
                            # 1 (stream_id) + n_time (5) + n_geoinfo
                            n_geoinfo      = len(stream_info.get("geoinfo_channels", []))
                            geoinfo_offset = 1 + 5 + n_geoinfo
                    
                            # target_coords_lens: points per cell for this sample/step
                            # retrieve from the batch target sample directly
                            # (not available from the flattened target tensor alone)
#                            tc_lens = targets_params[target_idx][stream_name].global_params.get(
#                                "target_coords_lens", None
#                            )
                            tc_lens_list = target_cur[stream_name]["target_coords_lens"]
                            tc_lens      = tc_lens_list[target_idx]
                    
                            if tc_lens is not None and hasattr(self, '_hp_nbours') \
                                    and self._hp_nbours is not None:
                                loss_lfct, loss_lfct_chs = self._loss_wavelet_per_cell(
                                    target,
                                    pred,
                                    targets_coords_batch[target_idx],
                                    tc_lens.to(self.device),
                                    self._hp_nbours.to(self.device),
                                    weights_channels,
                                    geoinfo_offset=geoinfo_offset,
                                    **{k: v for k, v in loss_fct_params.items()},
                                )
                            else:
                                # fallback to zero loss if neighbourhood info not available
                                loss_lfct    = torch.tensor(0.0, device=self.device, requires_grad=True)
                                loss_lfct_chs = torch.zeros(target.shape[-1], device=self.device)
                            # log and accumulate — then skip to next loss function
                            for ch_n, v in zip(target_channels, loss_lfct_chs, strict=True):
                                losses_all[stream_name][str(timestep_idx)][loss_fct_name][ch_n] = (
                                    spoof_weight * v if v != 0.0 and not is_spoof else torch.nan
                                )
                            loss_cur_w   = spoof_weight * loss_fct_weight * loss_lfct * output_step_weight
                            loss_st_corr = loss_st_corr + loss_cur_w
                            ctr_loss_fcts += 1 if (loss_cur_w > 0.0 and not is_spoof) else 0
                            continue   # <-- this skips _loss_per_loss_function entirely

                        # loss_lfct: loss for given loss function aggregated over all channels
                        # loss_lfct_chs: loss for given loss function per channel
                        loss_lfct, loss_lfct_chs = self._loss_per_loss_function(
                            loss_fct, target, pred, substep_masks,
                            weights_channels, weights_locations,
                        )
                    
                        for ch_n, v in zip(target_channels, loss_lfct_chs, strict=True):
                            losses_all[stream_name][str(timestep_idx)][loss_fct_name][ch_n] = (
                                spoof_weight * v if v != 0.0 and not is_spoof else torch.nan
                            )
                        # Add the weighted and normalized loss from this loss function to the total
                        # batch loss
                    
                        loss_cur_w   = spoof_weight * loss_fct_weight * loss_lfct * output_step_weight
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
                            reordered_losses[stream_name][loss_fct_name]["avg"] += v
                            count += 1
                reordered_losses[stream_name][loss_fct_name]["avg"] /= count

        # Return all computed loss components encapsulated in a ModelLoss dataclass
        return LossValues(loss=loss, losses_all=reordered_losses, stddev_all=None)
