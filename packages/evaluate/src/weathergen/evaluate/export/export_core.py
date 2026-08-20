import logging
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

from weathergen.common.config import get_model_results
from weathergen.common.io import zarrio_reader
from weathergen.evaluate.export.parser_factory import CfParserFactory
from weathergen.evaluate.export.reshape import detect_grid_type

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# Module-level cache for the zarr path and open store — resolved once per worker.
_CACHED_FNAME_ZARR: str | None = None
_CACHED_ZIO = None


def _init_worker(fname_zarr: str) -> None:
    """Pool initializer: open the zarr store once and keep it for the worker's lifetime."""
    global _CACHED_FNAME_ZARR, _CACHED_ZIO
    _CACHED_FNAME_ZARR = fname_zarr
    _CACHED_ZIO = zarrio_reader(fname_zarr)
    _CACHED_ZIO.__enter__()


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
    sample, fstep, stream, dtype = args

    # Navigate directly to the zarr group for this (sample, stream, fstep, dtype).
    group_path = f"{sample}/{stream}/{fstep}/{dtype}"
    ds_group = _CACHED_ZIO.data_root.get(group_path)

    if ds_group is None:
        raise FileNotFoundError(f"Zarr group '{group_path}' not found in {_CACHED_FNAME_ZARR}")

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
    common_coords = {
        "ipoint": np.arange(npoints),
        "channel": channels,
        "forecast_step": fstep,
        "valid_time": ("ipoint", times_arr),
        "lat": ("ipoint", coords_arr[:, 0]),
        "lon": ("ipoint", coords_arr[:, 1]),
    }
    if data_arr.ndim == 3:   # (npoints, nchannels, nens)
        da_result = xr.DataArray(
            data_arr,
            dims=["ipoint", "channel", "ensemble_member"],
            coords={**common_coords, "ensemble_member": np.arange(data_arr.shape[2])},
        )
    else:                    # (npoints, nchannels)
        da_result = xr.DataArray(data_arr, dims=["ipoint", "channel"], coords=common_coords)

    # Handle optional ensemble dimension: squeeze it out if present.
#    if data_arr.ndim == 3 and data_arr.shape[2] == 1:
#        data_arr = data_arr[:, :, 0]
#
#    da_result = xr.DataArray(
#        data_arr,
#        dims=["ipoint", "channel"],
#        coords={
#            "ipoint": np.arange(npoints),
#            "channel": channels,
#            "forecast_step": fstep,
#            "valid_time": ("ipoint", times_arr),
#            "lat": ("ipoint", coords_arr[:, 0]),
#            "lon": ("ipoint", coords_arr[:, 1]),
#        },
#    )

    return (sample, fstep, da_result)

def _int_keys(group) -> list[int]:
    """Integer-named subgroups of `group`, sorted numerically."""
    keys = []
    for k in group.group_keys():
        try:
            keys.append(int(k))
        except ValueError:
            continue
    return sorted(keys)


def _find_stream_example(root, stream: str) -> tuple[int, list[int]]:
    """
    First sample that actually holds `stream` on disk, plus its forecast steps.

    Replaces the store-level `example_key` probe, which picks sample/stream
    itself and fails when that combination was never written.
    """
    for sample in _int_keys(root):
        sgrp = root.get(f"{sample}/{stream}")
        if sgrp is None:
            continue
        fsteps = _int_keys(sgrp)
        if fsteps:
            return sample, fsteps
    raise FileNotFoundError(f"Stream '{stream}' has no groups in the zarr store.")


def _streams_with_source(root, sample: int) -> list[str]:
    """Streams that have a `source` group at some forecast step for `sample`."""
    out = []
    for st in sorted(root[str(sample)].group_keys()):
        sgrp = root.get(f"{sample}/{st}")
        if sgrp is None:
            continue
        for f in _int_keys(sgrp):
            if root.get(f"{sample}/{st}/{f}/source") is not None:
                out.append(st)
                break
    return out


def _resolve_source_stream(root, samples, stream: str, preferred: str | None = None) -> str:
    """
    Pick the stream to read the conditioning window from.

    Diagnostic streams hold only `target`/`prediction`, so the reference time
    must come from a stream that was actually conditioned on.
    """
    candidates = _streams_with_source(root, samples[0])
    if not candidates:
        raise FileNotFoundError(
            f"No stream has a 'source' group for sample {samples[0]}; "
            "cannot determine the forecast reference time."
        )
    if preferred is not None:
        if preferred not in candidates:
            raise ValueError(
                f"--source-stream '{preferred}' has no source group. Available: {candidates}"
            )
        return preferred
    if stream in candidates:
        return stream

    chosen = candidates[0]
    _logger.warning(
        f"Stream '{stream}' is diagnostic (no 'source' group). Taking the reference "
        f"time from '{chosen}' instead. Candidates: {candidates}. "
        "Override with --source-stream if this is not the conditioning stream."
    )
    return chosen


def get_fsteps(fsteps, fname_zarr: str, stream: str):
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
        _, zio_forecast_steps = _find_stream_example(zio.data_root, stream)

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


def get_samples(samples, fname_zarr: str, stream: str):
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
        root = zio.data_root
        zio_samples = [s for s in _int_keys(root) if root.get(f"{s}/{stream}") is not None]

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


def get_channels(channels, stream: str, fname_zarr: str, data_type: str = "target") -> list[str]:
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
        root = zio.data_root
        sample, fsteps = _find_stream_example(root, stream)
        grp = root.get(f"{sample}/{stream}/{fsteps[0]}/{data_type}")
        if grp is None:
            raise FileNotFoundError(
                f"Zarr group '{sample}/{stream}/{fsteps[0]}/{data_type}' not found."
            )
        all_channels = list(grp.attrs["channels"])

    if channels is None:
        return all_channels

    missing = [c for c in channels if c not in all_channels]
    if missing:
        _logger.warning(f"Requested channels not available, will be skipped: {missing}")
    return [c for c in channels if c in all_channels]

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
        root = zio.data_root
        sample, fsteps = _find_stream_example(root, stream)
        grp = root.get(f"{sample}/{stream}/{fsteps[0]}/{data_type}")
        coords_arr = np.asarray(grp["coords"])

    probe = xr.DataArray(
        np.zeros(coords_arr.shape[0], dtype="float32"),
        dims=["ipoint"],
        coords={
            "lat": ("ipoint", coords_arr[:, 0]),
            "lon": ("ipoint", coords_arr[:, 1]),
        },
    )
    return detect_grid_type(probe)


# TODO: this will change after restructuring the lead time.
def get_source_info(
    fname_zarr, stream, samples, source_stream: str | None = None
) -> tuple[list[np.datetime64], list[np.datetime64]]:
    """
    Retrieve source interval boundaries from the source group at forecast step 0.

    Values are derived from the actual ``times`` array of the **source**
    group at forecast step 0:
    - ``source_start = min(source_times)``
    - ``source_end   = max(source_times)``

    The ``source_end`` also serves as the reference (initialisation) time.

    Parameters
    ----------
    fname_zarr : str
        Path to the Zarr store.
    stream : str
        Stream name to retrieve data for (e.g., 'ERA5').
    samples : list
        List of samples to process.

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
        src_stream = _resolve_source_stream(root, samples, stream, source_stream)

        for sample in tqdm(samples, desc="Getting source info"):
            sgrp = root.get(f"{sample}/{src_stream}")
            group_path = None
            if sgrp is not None:
                for f in _int_keys(sgrp):
                    if root.get(f"{sample}/{src_stream}/{f}/source") is not None:
                        group_path = f"{sample}/{src_stream}/{f}/source"
                        break
            if group_path is None:
                raise FileNotFoundError(
                    f"No 'source' group for sample {sample} under stream '{src_stream}'."
                )

            source_group = root.get(group_path)
            times_arr = np.asarray(source_group["times"]).astype("datetime64[ns]")
            source_start = np.min(times_arr)
            source_end = np.max(times_arr)

            _logger.debug(f"Sample {sample}: source_interval=[{source_start} .. {source_end}]")
            source_starts.append(source_start)
            source_ends.append(source_end)

    return source_starts, source_ends


def get_streams(stream, fname_zarr):
    with zarrio_reader(fname_zarr) as zio:
        root = zio.data_root
        on_disk: list[str] = []
        for s in _int_keys(root):
            for st in root[str(s)].group_keys():
                if st not in on_disk:
                    on_disk.append(st)

    if stream is None:
        return on_disk
    if stream not in on_disk:
        raise ValueError(f"Stream '{stream}' not in store. Available: {on_disk}")
    return [stream]

def export_model_outputs(data_type: str, config: OmegaConf, **kwargs) -> None:
    """
    Retrieve data from Zarr store and export to the requested format.

    All (sample, fstep) pairs are submitted to the pool at once so that
    every worker stays busy.  Results are grouped by sample and handed to
    the parser in sample order.

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
    req_samples = kwargs.samples
    req_fsteps = kwargs.fsteps
    stream = kwargs.stream
    req_channels = kwargs.channels
    n_processes = kwargs.n_processes
    epoch = kwargs.epoch
    rank = kwargs.rank

    if data_type not in ["target", "prediction"]:
        raise ValueError(f"Invalid type: {data_type}. Must be 'target' or 'prediction'.")

    fname_zarr = get_model_results(run_id, epoch, rank)
    streams = get_streams(stream, fname_zarr)
    for stream in streams:
        fsteps = get_fsteps(req_fsteps, fname_zarr, stream)
        samples = get_samples(req_samples, fname_zarr, stream)
        channels = get_channels(req_channels, stream, fname_zarr, data_type)
        grid_type = get_grid_type(data_type, stream, fname_zarr)
        source_starts, source_ends = get_source_info(
            fname_zarr, stream, samples, kwargs.get("source_stream")
        )

        kwargs["grid_type"] = grid_type
        kwargs["channels"] = channels
        kwargs["data_type"] = data_type

        parser = CfParserFactory.get_parser(config=config, **kwargs)

        n_fsteps = len(fsteps)
        total_tasks = len(samples) * n_fsteps

        # Batch size in *samples*. Limits how many samples can be in-flight at once,
        # bounding peak memory while still allowing read/write overlap within each batch.
        batch_size = max(1, n_processes * 2)
        n_batches = (len(samples) + batch_size - 1) // batch_size

        _logger.info(
            f"Exporting {len(samples)} samples × {n_fsteps} fsteps "
            f"({total_tasks} total tasks) in {n_batches} batch(es) of up to "
            f"{batch_size} samples, using {n_processes} workers. "
            f"Reading and writing are interleaved within each batch."
        )

        # Initialise each worker with the zarr path so it is resolved only once.
        with Pool(
            processes=n_processes,
            initializer=_init_worker,
            initargs=(fname_zarr,),
        ) as pool:
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
                    (sample, fstep, stream, data_type)
                    for sample in batch_samples
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
                    desc=f"  Batch {batch_idx + 1}/{n_batches}",
                )

                processed_samples = []

                for sample, _fstep, data in pool.imap_unordered(
                    get_data_worker, batch_tasks, chunksize=1
                ):
                    sample_results[sample].append(data)
                    pbar.update(1)

                    # Check if this sample is complete (all fsteps received).
                    if len(sample_results[sample]) == n_fsteps:
                        b_idx = sample_to_batch_idx[sample]
                        source_start = batch_source_starts[b_idx]
                        source_end = batch_source_ends[b_idx]
                        results_iter = iter(sample_results[sample])
                        processed = parser.process_sample(
                            results_iter,
                            ref_time=source_end,
                            source_interval_start=source_start,
                            source_interval_end=source_end,
                        )
                        processed_samples.append(processed)

                        # Free memory immediately.
                        del sample_results[sample]
                        batch_written += 1

                # Only save here if need to merge samples, otherwise saved in process_sample
                if processed_samples and processed_samples[0] is not None:
                    parser.save(processed_samples)
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

        _logger.info(f"Export complete. Wrote {samples_written}/{len(samples)} samples.")
