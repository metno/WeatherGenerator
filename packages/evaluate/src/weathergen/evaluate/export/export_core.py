import logging
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

from weathergen.common.config import (
    get_model_results,
)
from weathergen.common.io import zarrio_reader
from weathergen.evaluate.export.parser_factory import CfParserFactory
from weathergen.evaluate.export.reshape import detect_grid_type

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# Per-worker cache: zarr_path → open zarrio_reader context.
_WORKER_ZIO_CACHE: dict[str, object] = {}


def _init_worker() -> None:
    """Pool initializer: reset the per-worker zarr store cache."""
    global _WORKER_ZIO_CACHE
    _WORKER_ZIO_CACHE = {}


def _get_or_open_zio(zarr_path: str):
    """Return a cached zarrio_reader for *zarr_path*, opening it on first access."""
    global _WORKER_ZIO_CACHE
    key = str(zarr_path)
    if key not in _WORKER_ZIO_CACHE:
        zio = zarrio_reader(Path(zarr_path))
        zio.__enter__()
        _WORKER_ZIO_CACHE[key] = zio
    return _WORKER_ZIO_CACHE[key]


def get_data_worker(args: tuple) -> tuple[int, int, xr.DataArray]:
    """
    Worker function to retrieve data for a single (sample, fstep) pair.

    Reads the raw zarr arrays as numpy (bypassing dask) and builds a
    lightweight xarray DataArray that can be pickled back to the main
    process with all data already in memory.

    Returns
    -------
        Tuple of (sample, fstep, xarray.DataArray) with data fully in memory.
    """
    global_sample, local_sample, zarr_path, fstep, stream, dtype = args

    zio = _get_or_open_zio(zarr_path)

    # Navigate directly to the zarr group for this (sample, stream, fstep, dtype).
    group_path = f"{local_sample}/{stream}/{fstep}/{dtype}"
    ds_group = zio.data_root.get(group_path)

    if ds_group is None:
        raise FileNotFoundError(f"Zarr group '{group_path}' not found in {zarr_path}")

    # Read raw arrays as numpy — no dask, no chunking overhead.
    data_arr = np.asarray(ds_group["data"])  # (npoints, nchannels) or (npoints, nchannels, nens)
    coords_arr = np.asarray(ds_group["coords"])  # (npoints, 2)
    times_arr = np.asarray(ds_group["times"]).astype("datetime64[ns]")  # (npoints,)
    channels = list(ds_group.attrs["channels"])

    # Build a lightweight xarray DataArray with the same structure
    # that process_sample / assign_coords expects:
    #   dims = [ipoint, channel]
    #   coords: forecast_step, channel, valid_time, lat, lon
    npoints = data_arr.shape[0]

    # Handle optional ensemble dimension: squeeze it out if present.
    data_dims = ["ipoint", "channel"]
    if data_arr.ndim == 3:
        if data_arr.shape[2] == 1:
            data_arr = data_arr[:, :, 0]
        else:
            data_dims.append("mem")

    data_coords = {
        "ipoint": np.arange(npoints),
        "channel": channels,
        "forecast_step": fstep,
        "valid_time": ("ipoint", times_arr),
        "lat": ("ipoint", coords_arr[:, 0]),
        "lon": ("ipoint", coords_arr[:, 1]),
    }

    da_result = xr.DataArray(data_arr, dims=data_dims, coords=data_coords)

    return (global_sample, fstep, da_result)


def get_fsteps(fsteps, fname_zarr: str):
    """
    Retrieve available forecast steps from the Zarr store and filter
    based on requested forecast steps.

    Parameters
    ----------
        fsteps : list
            List of requested forecast steps.
            If None, retrieves all available forecast steps.
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        list[int]
            List of forecast steps to be used for data retrieval.
    """
    with zarrio_reader(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])

    if fsteps is None:
        return zio_forecast_steps

    requested = sorted([int(fstep) for fstep in fsteps])
    available_set = set(zio_forecast_steps)
    valid = [f for f in requested if f in available_set]
    missing = [f for f in requested if f not in available_set]

    if missing:
        _logger.warning(
            f"Requested forecast steps {missing} are not available in the zarr store "
            f"(available: {zio_forecast_steps}). They will be skipped."
        )

    if not valid:
        raise ValueError(
            f"None of the requested forecast steps {requested} exist in the zarr store. "
            f"Available forecast steps: {zio_forecast_steps}"
        )

    return valid


def get_samples(samples, fname_zarr: str):
    """
    Retrieve available samples from the Zarr store
    and filter based on requested samples.
    Parameters
    ----------
        samples : list
            List of requested samples. If None, retrieves all available samples.
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        list[int]
            List of samples to be used for data retrieval.
    """
    with zarrio_reader(fname_zarr) as zio:
        zio_samples = sorted([int(sample) for sample in zio.samples])

    if samples is None:
        return zio_samples

    requested = sorted([int(sample) for sample in samples])
    available_set = set(zio_samples)
    valid = [s for s in requested if s in available_set]
    missing = [s for s in requested if s not in available_set]

    if missing:
        _logger.warning(
            f"Requested samples {missing} are not available in the zarr store "
            f"(available range: {zio_samples[0]}–{zio_samples[-1]}). They will be skipped."
        )

    if not valid:
        raise ValueError(
            f"None of the requested samples {requested} exist in the zarr store. "
            f"Available samples: {zio_samples}"
        )

    return valid


def get_channels(channels, stream: str, fname_zarr: str) -> list[str]:
    """
    Retrieve available channels from the Zarr store and filter based on requested channels.
    Parameters
    ----------
        channels : list
            List of requested channels. If None, retrieves all available channels.
        stream : str
            Stream name to retrieve data for (e.g., 'ERA5').
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        list[str]
            List of channels to be used for data retrieval.
    """
    with zarrio_reader(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        all_channels = dummy_out.target.channels
        if channels is not None:
            existing_channels = set(all_channels) & set(channels)
            if existing_channels != set(channels):
                missing_channels = set(channels) - set(existing_channels)
                _logger.warning(
                    "The following requested channels are"
                    f"not available in the data and will be skipped: {missing_channels}"
                )
        return all_channels if channels is None else list(existing_channels)


def get_grid_type(data_type, stream: str, fname_zarr: str) -> str:
    """
    Determine the grid type of the data (regular or gaussian).
    Parameters
    ----------
        data_type : str
            Type of data to retrieve ('target' or 'prediction').
        stream : str
            Stream name to retrieve data for (e.g., 'ERA5').
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        str
            Grid type ('regular' or 'gaussian').
    """
    with zarrio_reader(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        data = dummy_out.target if data_type == "target" else dummy_out.prediction
        return detect_grid_type(data.as_xarray().squeeze())


def _int_keys(group) -> list[int]:
    """Return integer-named subgroups sorted numerically."""
    keys = []
    for key in group.group_keys():
        try:
            keys.append(int(key))
        except ValueError:
            continue
    return sorted(keys)


def _streams_with_source(root, sample: int) -> list[str]:
    """Return streams containing a source group for the given sample."""
    streams = []
    sample_group = root.get(str(sample))
    if sample_group is None:
        return streams

    for candidate_stream in sorted(sample_group.group_keys()):
        stream_group = root.get(f"{sample}/{candidate_stream}")
        if stream_group is None:
            continue
        if any(
            root.get(f"{sample}/{candidate_stream}/{fstep}/source") is not None
            for fstep in _int_keys(stream_group)
        ):
            streams.append(candidate_stream)
    return streams


def _resolve_source_stream(root, samples: list[int], stream: str) -> str | None:
    """Choose a stream containing source data, if one exists."""
    candidates = _streams_with_source(root, samples[0])
    if not candidates:
        return None
    if stream in candidates:
        return stream

    chosen = candidates[0]
    _logger.warning(
        f"Stream '{stream}' has no source group; using source times from '{chosen}'. "
        f"Available source streams: {candidates}."
    )
    return chosen


def _first_valid_time(root, sample: int, stream: str, fstep: int) -> np.datetime64:
    """Return the earliest valid time from a prediction or target group."""
    for data_type in ("prediction", "target"):
        group = root.get(f"{sample}/{stream}/{fstep}/{data_type}")
        if group is not None:
            return np.asarray(group["times"]).astype("datetime64[ns]").min()
    raise FileNotFoundError(
        f"No prediction or target group found at '{sample}/{stream}/{fstep}'."
    )


def _derive_source_interval(
    root, sample: int, stream: str, fstep_hours: int
) -> tuple[np.datetime64, np.datetime64]:
    """Derive a one-step source interval when the store has no source groups."""
    stream_group = root.get(f"{sample}/{stream}")
    fsteps = _int_keys(stream_group) if stream_group is not None else []
    fsteps = [
        fstep
        for fstep in fsteps
        if any(
            root.get(f"{sample}/{stream}/{fstep}/{candidate_type}") is not None
            for candidate_type in ("prediction", "target")
        )
    ]
    if not fsteps:
        raise FileNotFoundError(
            f"Stream '{stream}' has no prediction or target data for sample {sample}."
        )

    first_fstep = fsteps[0]
    first_valid_time = _first_valid_time(root, sample, stream, first_fstep)
    if len(fsteps) >= 2:
        second_fstep = fsteps[1]
        step_duration = (
            _first_valid_time(root, sample, stream, second_fstep) - first_valid_time
        ) / (second_fstep - first_fstep)
    else:
        step_duration = np.timedelta64(fstep_hours, "h").astype("timedelta64[ns]")

    reference_time = first_valid_time - first_fstep * step_duration
    return reference_time - step_duration, reference_time


def get_source_info(
    fname_zarr,
    stream,
    samples,
    fstep_hours: int = 6,
) -> tuple[list[np.datetime64], list[np.datetime64]]:
    """
    Retrieve source intervals without assuming that forecast step 0 exists.

    Parameters
    ----------
    fname_zarr : str
        Path to the Zarr store.
    stream : str
        Stream name to retrieve data for (e.g., 'ERA5').
    samples : list
        List of samples to process.
    fstep_hours : int
        Forecast-step duration used when source times must be derived from a
        store containing only one prediction or target step.

    Returns
    -------
    tuple[list, list]
        ``(source_starts, source_ends)`` — one entry per sample,
        all as ``datetime64[ns]``.
    """
    _logger.info(f"Retrieving source info for {len(samples)} samples...")

    source_starts = []
    source_ends = []
    with zarrio_reader(fname_zarr) as zio:
        root = zio.data_root
        resolved_source_stream = _resolve_source_stream(root, samples, stream)
        if resolved_source_stream is None:
            _logger.warning(
                "No source group exists in the store; deriving source intervals "
                "from prediction or target valid times."
            )

        for sample in tqdm(samples, desc="Getting source info"):
            if resolved_source_stream is None:
                source_start, source_end = _derive_source_interval(
                    root, sample, stream, fstep_hours
                )
            else:
                stream_group = root.get(f"{sample}/{resolved_source_stream}")
                source_path = None
                if stream_group is not None:
                    for fstep in _int_keys(stream_group):
                        candidate_path = (
                            f"{sample}/{resolved_source_stream}/{fstep}/source"
                        )
                        if root.get(candidate_path) is not None:
                            source_path = candidate_path
                            break
                if source_path is None:
                    raise FileNotFoundError(
                        f"No source group found for sample {sample} under stream "
                        f"'{resolved_source_stream}'."
                    )

                source_group = root.get(source_path)
                times_arr = np.asarray(source_group["times"]).astype("datetime64[ns]")
                source_start = np.min(times_arr)
                source_end = np.max(times_arr)

            _logger.debug(f"Sample {sample}: source_interval=[{source_start} .. {source_end}]")
            source_starts.append(source_start)
            source_ends.append(source_end)

    return source_starts, source_ends


def get_streams(stream, fname_zarr):
    with zarrio_reader(fname_zarr) as zio:
        zio_streams = zio.streams
    streams = zio_streams if stream is None else [stream]
    return streams


def export_model_outputs(data_type: str, config: OmegaConf, **kwargs) -> None:
    """
    Retrieve data from Zarr store and export to the requested format.

    Iterates over all rank files.  Each rank gets its own parser instance
    (and therefore its own output GRIB file pair named with the rank label).

    Parameters
    ----------
    data_type: str
        Type of data to retrieve ('target' or 'prediction').
    config : OmegaConf
            Loaded config for cf_parser function.
    kwargs:
        Additional keyword arguments for the parser.
    """
    kwargs = OmegaConf.create(kwargs)

    run_id = kwargs.run_id
    samples_cfg = kwargs.samples
    fsteps_cfg = kwargs.fsteps
    stream = kwargs.stream
    channels_cfg = kwargs.channels
    n_processes = kwargs.n_processes
    epoch = kwargs.epoch
    rank = kwargs.rank
    init_time_reference = kwargs.get("init_time_reference", "source_start")
    if init_time_reference not in ("source_start", "source_end"):
        raise ValueError(
            f"Invalid init_time_reference: {init_time_reference}. "
            "Must be 'source_start' or 'source_end'."
        )

    if data_type not in ["target", "prediction"]:
        raise ValueError(f"Invalid type: {data_type}. Must be 'target' or 'prediction'.")

    # --- Discover rank files ---
    # get_model_results accepts lists of epochs and ranks ("all" or list of ints).
    rank_arg = ["all"] if rank == "all" else (rank if isinstance(rank, list) else [rank])
    rank_files = get_model_results(run_id, [epoch], rank_arg)
    if not rank_files:
        raise FileNotFoundError(
            f"No rank files found for run_id={run_id}, epoch={epoch}, rank={rank}"
        )
    _logger.info(f"Discovered {len(rank_files)} rank file(s).")

    first_zarr = rank_files[0]
    fsteps = get_fsteps(fsteps_cfg, first_zarr)
    streams = get_streams(stream, first_zarr)

    processed_samples = []  # for verif

    for stream in streams:
        grid_type = get_grid_type(data_type, stream, first_zarr)
        stream_channels = get_channels(channels_cfg, stream, first_zarr)
        kwargs["stream"] = stream
        kwargs["grid_type"] = grid_type
        kwargs["channels"] = stream_channels
        kwargs["data_type"] = data_type
        for rank_file in rank_files:
            rank_label = rank_file.stem.split("rank")[-1]  # e.g. "0000"
            _logger.info(f"RUN {run_id}: Processing rank {rank_label} ({rank_file.name})")

            samples = get_samples(samples_cfg, rank_file)
            source_starts, source_ends = get_source_info(
                rank_file,
                stream,
                samples,
                fstep_hours=kwargs.get("fstep_hours", 6),
            )

            kwargs["rank_label"] = rank_label
            parser = CfParserFactory.get_parser(config=config, **kwargs)

            n_fsteps = len(fsteps)
            total_tasks = len(samples) * n_fsteps
            batch_size = max(1, n_processes * 2)
            n_batches = (len(samples) + batch_size - 1) // batch_size

            _logger.info(
                f"Exporting {len(samples)} samples × {n_fsteps} fsteps "
                f"({total_tasks} total tasks) in {n_batches} batch(es) of up to "
                f"{batch_size} samples, using {n_processes} workers. "
                f"Reading and writing are interleaved within each batch."
            )

            with Pool(processes=n_processes, initializer=_init_worker) as pool:
                samples_written = 0

                for batch_idx in range(n_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, len(samples))
                    batch_samples = samples[batch_start:batch_end]
                    batch_source_starts = source_starts[batch_start:batch_end]
                    batch_source_ends = source_ends[batch_start:batch_end]

                    # Map sample -> index within this batch for ref_times lookup.
                    sample_to_batch_idx = {s: i for i, s in enumerate(batch_samples)}

                    batch_tasks = [
                        (s, s, str(rank_file), fstep, stream, data_type)
                        for s in batch_samples
                        for fstep in fsteps
                    ]

                    _logger.info(
                        f"Batch {batch_idx + 1}/{n_batches}: "
                        f"samples {batch_start}–{batch_end - 1} "
                        f"({len(batch_samples)} samples, {len(batch_tasks)} tasks)"
                    )

                    # Interleaved read/write: as soon as all fsteps for a sample
                    # arrive, write it immediately while workers continue reading.
                    sample_results: dict[int, list] = defaultdict(list)
                    batch_written = 0

                    pbar = tqdm(
                        total=len(batch_tasks),
                        desc=f"  Rank {rank_label} batch {batch_idx + 1}/{n_batches}",
                    )

                    for global_s, _fstep, data in pool.imap_unordered(
                        get_data_worker, batch_tasks, chunksize=1
                    ):
                        sample_results[global_s].append(data)
                        pbar.update(1)

                        # Check if this sample is complete (all fsteps received).
                        if len(sample_results[global_s]) == n_fsteps:
                            b_idx = sample_to_batch_idx[global_s]
                            source_start = batch_source_starts[b_idx]
                            source_end = batch_source_ends[b_idx]
                            # The forecast init time is either the start or the end
                            # of the source (conditioning) window, selected via
                            # `init_time_reference` (e.g. a 00-05 UTC window has
                            # init = 00:00 for "source_start" or 05:00 for "source_end").
                            init_time = (
                                source_start
                                if init_time_reference == "source_start"
                                else source_end
                            )
                            processed_sample = parser.process_sample(
                                iter(sample_results[global_s]),
                                ref_time=init_time,
                                source_interval_start=source_start,
                                source_interval_end=init_time,
                            )
                            processed_samples.append(processed_sample)
                            # Free memory immediately.
                            del sample_results[global_s]
                            batch_written += 1

                    pbar.close()

                    samples_written += batch_written
                    if batch_written != len(batch_samples):
                        _logger.error(
                            f"Batch {batch_idx + 1}: expected {len(batch_samples)} "
                            f"samples but only wrote {batch_written}. "
                            f"Incomplete: {list(sample_results.keys())}"
                        )

                    # Free any remaining refs before next batch.
                    del sample_results

            # Flush and close the parser's file handles for this rank.
            if hasattr(parser, "close"):
                parser.close()

            _logger.info(f"Rank {rank_label}: wrote {samples_written}/{len(samples)} samples.")

    # Only save here if need to merge samples (i.e. verif), otherwise saved in process_sample
    if processed_samples[0] is not None:
        parser.save(processed_samples)

    _logger.info(f"Export complete across {len(rank_files)} rank(s).")
