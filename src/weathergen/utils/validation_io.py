# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import astropy_healpix as hp
import numpy as np
import torch
from numpy.typing import NDArray

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler

_logger = logging.getLogger(__name__)


def write_output(
    cf, val_cfg, batch_size, mini_epoch, batch_idx, dn_data, batch, model_output, target_aux_out
):
    """
    Interface for writing model output
    """

    # TODO: how to handle multiple physical loss terms
    outputs_physical = [
        loss_name
        for i, (loss_name, loss_term) in enumerate(val_cfg.losses.items())
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    target_aux_out = target_aux_out[outputs_physical[0]]

    # collect all target / prediction-related information
    fp32 = torch.float32
    preds_all, targets_all, targets_coords_all, targets_times_all = [], [], [], []

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    forecast_offset = timestep_idxs[0]
    targets_lens = []

    # TODO Maybe stopping at forecast_steps explained #1657
    for t_idx in timestep_idxs:
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        targets_lens += [[]]
        for sname in cf.streams.keys():
            # handle spoof data: do not write since it might corrupt validation (spoofing invisible
            # there)
            if target_aux_out.physical[t_idx][sname]["is_spoof"][0]:
                targets = target_aux_out.physical[t_idx][sname]["target"]
                # for-loop to make sure we have a consistent number of samples
                preds_s = [np.zeros((1, 0, t.shape[1])) for t in targets]
                targets_s = [np.zeros((0, t.shape[1])) for t in targets]
                t_coords_s = [np.zeros((0, 2)) for t in targets]
                t_times_s = [np.array([]).astype("datetime64[ns]") for t in targets]

            else:
                preds = model_output.get_physical_prediction(t_idx, sname)
                targets = target_aux_out.physical[t_idx][sname]["target"]

                preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []

                # handle forcing streams or if sample is empty
                if preds is None:
                    # preds are empty so create copy of target and add ensemble dimension
                    assert targets[0].shape[0] == 0, "Empty preds but non-empty targets."
                    preds = [target.clone().unsqueeze(0) for target in targets]

                for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                    target_data = target_aux_out.physical[t_idx][sname]
                    t_coords = target_data["target_coords"][i_batch]
                    t_times = target_data["target_times"][i_batch]

                    idxs_inv = target_aux_out.physical[t_idx][sname]["idxs_inv"][i_batch]
                    if idxs_inv is not None:
                        pred = pred[:, idxs_inv]
                        target = target[idxs_inv]
                        t_coords = t_coords[idxs_inv]
                        t_times = t_times[idxs_inv]

                    # denormalize data if requested and map to storage format
                    preds_s += [dn_data(sname, pred.to(fp32)).detach().cpu().numpy()]
                    targets_s += [dn_data(sname, target.to(fp32)).detach().cpu().numpy()]

                    # extract original target coords and times from target data
                    t_coords_s += [t_coords.cpu().numpy()]
                    t_times_s += [t_times.astype("datetime64[ns]")]

            targets_lens[-1] += [[]]
            targets_lens[-1][-1] += [t.shape[0] for t in targets_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]

    # output stream names to be written, use specified ones or all if nothing specified
    stream_names = list(cf.streams.keys())
    stream_infos = list(cf.streams.values())
    if val_cfg.get("output").get("streams") is not None:
        output_stream_names = val_cfg.output.streams
    else:
        output_stream_names = stream_names
    latent_requested = io.LATENT_STREAM in output_stream_names

    if (
        len(preds_all) == 0 or np.array([p.shape[1] for pp in preds_all for p in pp]).sum() == 0
    ) and not latent_requested:
        _logger.warning("Writing no data since predictions are empty.")
        return

    # collect source information
    sources = []
    for sample in batch.get_source_samples().get_samples():
        sources += [[]]
        for _, stream_data in sample.streams_data.items():
            # TODO: support multiple input steps
            sources[-1] += [stream_data.source_raw[0]]

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

    # more prep work

    output_streams = {
        name: stream_names.index(name) for name in output_stream_names if name != io.LATENT_STREAM
    }
    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in stream_infos]
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in stream_infos]

    geoinfo_channels = [[] for _ in stream_infos]  # TODO obtain channels

    # calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * batch_size

    # write output

    start_date = val_cfg.start_date
    end_date = val_cfg.end_date

    twh = TimeWindowHandler(
        start_date,
        end_date,
        val_cfg.time_window_len,
        val_cfg.time_window_step,
    )
    source_windows = (twh.window(idx) for idx in sample_idxs)
    source_intervals = [TimeRange(window.start, window.end) for window in source_windows]
    latent_outputs = get_latent_output(batch, model_output) if latent_requested else []

    data = io.OutputBatchData(
        sources,
        source_intervals,
        targets_all,
        preds_all,
        targets_coords_all,
        targets_times_all,
        targets_lens,
        output_streams,
        target_channels,
        source_channels,
        geoinfo_channels,
        sample_start,
        forecast_offset,
    )
    with zarrio_writer(config.get_path_results(cf, mini_epoch)) as zio:
        for subset in data.items():
            zio.write_zarr(subset)
        if latent_outputs:
            _write_latent_outputs(
                zio,
                cf,
                batch,
                latent_outputs,
                sample_start,
                timestep_idxs,
            )


def get_latent_output(batch, model_output):
    """Collect latent outputs per forecast step and sample as CPU numpy arrays."""

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    n_samples = len(batch.get_source_samples().get_samples())
    latents_all: list[list[dict[str, NDArray]]] = []

    for t_idx in timestep_idxs:
        latent_pred = model_output.get_latent_prediction(t_idx)
        latents_step: list[dict[str, NDArray]] = []
        for sample_idx in range(n_samples):
            latents_sample: dict[str, NDArray] = {}
            for latent_name, latent_value in latent_pred.items():
                for output_name, tensor in _iter_latent_tensors(latent_name, latent_value):
                    if tensor is None:
                        continue
                    latents_sample[output_name] = _as_sample_array(tensor, sample_idx)
            latents_step.append(latents_sample)
        latents_all.append(latents_step)

    return latents_all


def _iter_latent_tensors(latent_name: str, latent_value):
    if latent_value is None:
        return

    if hasattr(latent_value, "z_pre_norm"):
        yield latent_name, latent_value.z_pre_norm
        yield f"{latent_name}_register_tokens", getattr(latent_value, "register_tokens", None)
        yield f"{latent_name}_class_token", getattr(latent_value, "class_token", None)
        return

    if isinstance(latent_value, torch.Tensor):
        yield latent_name, latent_value


def _as_sample_array(tensor: torch.Tensor, sample_idx: int) -> NDArray:
    sample_tensor = tensor[sample_idx] if tensor.ndim > 0 else tensor
    return sample_tensor.detach().to(torch.float32).cpu().numpy()


def _write_latent_outputs(
    zio,
    cf,
    batch,
    latent_outputs: list[list[dict[str, NDArray]]],
    sample_start: int,
    timestep_idxs: list[int],
) -> None:
    for rel_step, latents_for_step in enumerate(latent_outputs):
        forecast_step = timestep_idxs[rel_step]
        for sample_idx, latents_for_sample in enumerate(latents_for_step):
            if not latents_for_sample:
                continue

            group_path = f"{sample_start + sample_idx}/{io.LATENT_STREAM}/{forecast_step}"
            npoints = _infer_latent_points(latents_for_sample)
            metadata = _build_latent_metadata(cf, batch, sample_idx, npoints)
            attrs = {
                "num_register_tokens": metadata["num_register_tokens"],
                "num_class_tokens": metadata["num_class_tokens"],
                "num_extra_tokens": metadata["num_extra_tokens"],
                "spatial_points": metadata["coords_len"],
                "coords_order": "lat_lon",
            }

            group = zio.data_root.get(group_path)
            if group is None:
                group = zio.data_root.create_group(group_path, attributes=attrs)

            for latent_name, latent_array in latents_for_sample.items():
                array = _strip_extra_tokens(latent_array, metadata["coords_len"], attrs)
                _write_array(group, latent_name, array)

            _write_array(group, "coords", metadata["coords"])
            _write_array(group, "geoinfo", metadata["geoinfo"])
            _write_array(group, "times", metadata["times"])


def _infer_latent_points(latents_for_sample: dict[str, NDArray]) -> int | None:
    for key in ("latent_state", "z_pre_norm", "patch_tokens"):
        if key in latents_for_sample and latents_for_sample[key].ndim >= 1:
            return latents_for_sample[key].shape[0]
    for latent_array in latents_for_sample.values():
        if latent_array.ndim >= 1:
            return latent_array.shape[0]
    return None


def _build_latent_metadata(cf, batch, sample_idx: int, npoints: int | None):
    num_register_tokens = int(cf.get("num_register_tokens", 0))
    num_class_tokens = int(cf.get("num_class_tokens", 0))
    num_extra_tokens = num_register_tokens + num_class_tokens

    coords = _healpix_coords(int(cf.healpix_level))
    coords = _apply_sample_mask(coords, batch, sample_idx)
    coords_len = coords.shape[0]

    if npoints is not None and npoints == coords_len + num_extra_tokens:
        npoints = coords_len
    if npoints is not None and npoints != coords_len:
        coords = np.zeros((npoints, 2), dtype=np.float32)
        coords_len = npoints

    return {
        "coords": coords.astype(np.float32),
        "geoinfo": np.zeros((coords_len, 0), dtype=np.float32),
        "times": np.full((coords_len,), np.datetime64("NaT"), dtype="datetime64[ns]"),
        "coords_len": coords_len,
        "num_register_tokens": num_register_tokens,
        "num_class_tokens": num_class_tokens,
        "num_extra_tokens": num_extra_tokens,
    }


def _healpix_coords(healpix_level: int) -> NDArray:
    nside = 2**healpix_level
    ipix = np.arange(12 * 4**healpix_level)
    lon, lat = hp.healpix_to_lonlat(ipix, nside, order="nested")
    return np.stack([lat.to_value("deg"), lon.to_value("deg")], axis=1)


def _apply_sample_mask(coords: NDArray, batch, sample_idx: int) -> NDArray:
    samples = batch.get_source_samples().get_samples()
    if sample_idx >= len(samples):
        return coords
    for meta in getattr(samples[sample_idx], "meta_info", {}).values():
        mask = getattr(meta, "mask", None)
        if mask is None:
            continue
        mask = mask.detach().cpu().numpy().astype(bool) if isinstance(mask, torch.Tensor) else mask
        if mask.shape[0] == coords.shape[0]:
            return coords[mask]
    return coords


def _strip_extra_tokens(
    latent_array: NDArray,
    coords_len: int,
    attrs: dict[str, int | str],
) -> NDArray:
    num_extra_tokens = int(attrs["num_extra_tokens"])
    if (
        num_extra_tokens > 0
        and latent_array.ndim >= 1
        and latent_array.shape[0] == coords_len + num_extra_tokens
    ):
        return latent_array[num_extra_tokens:]
    return latent_array


def _write_array(group, name: str, data: NDArray) -> None:
    if name not in group:
        group.create_array(name, data=data)
