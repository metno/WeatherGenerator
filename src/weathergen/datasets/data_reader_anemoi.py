# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override

import anemoi.datasets as anemoi_datasets
import numpy as np
from anemoi.datasets.data import MissingDateError
from anemoi.datasets.data.dataset import Dataset
from numpy.typing import NDArray
from omegaconf import OmegaConf

from weathergen.common.config import timedelta_to_str
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage
from weathergen.utils.distributed import is_root

_logger = logging.getLogger(__name__)


class DataReaderAnemoi(DataReaderTimestep):
    "Wrapper for Anemoi datasets"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
        domain=None,
    ) -> None:
        """
        Construct data reader for anemoi dataset

        Parameters
        ----------
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        # use anemoi_config if it's defined; ignore filename in this case
        data_paths = stream_info.get("data_paths", [])
        anemoi_config = stream_info.get("anemoi_config")
        if anemoi_config:
            # convert OmegaConf DictConfig to a plain dict for anemoi.open_dataset.
            filename = OmegaConf.to_container(anemoi_config, resolve=True)
            # add additional data paths
            for path in data_paths:
                anemoi_datasets.add_dataset_path(path)
            # provide some visibility since we ignore filename
            if is_root():
                _logger.info("Ignoring filename and using anemoi_config option.")

        # open  dataset to peak that it is compatible with requested parameters
        ds0: Dataset = anemoi_datasets.open_dataset(filename)
        # If there is no overlap with the time range, the dataset will be empty
        if tw_handler.t_start >= ds0.dates[-1] or tw_handler.t_end <= ds0.dates[0]:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info, domain=domain)
            self.init_empty()
            return

        kwargs = {}
        if "frequency" in stream_info:
            frequency = timedelta_to_str(stream_info["frequency"])
            kwargs["frequency"] = frequency
        if "subsampling_rate" in stream_info:
            name = stream_info["name"]
            _logger.warning(
                f"subsampling_rate specified for anemoi dataset for stream {name}. "
                + "Use frequency instead."
            )
        ds: Dataset = anemoi_datasets.open_dataset(
            ds0, **kwargs, start=tw_handler.t_start, end=tw_handler.t_end
        )

        period = np.timedelta64(ds.frequency)
        data_start_time = ds.dates[0]
        data_end_time = ds.dates[-1]
        assert data_start_time is not None and data_end_time is not None, (
            data_start_time,
            data_end_time,
        )
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
            domain=domain,
        )
        # If there is no overlap with the time range, no need to keep the dataset.
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            self.init_empty()
            return
        else:
            self.ds = ds
            self.len = len(ds)
            # anemoi exposes the missing indices of the (already time-sliced) dataset
            self.missing_idxs = set(getattr(ds, "missing", set()))

        # caches lats and lons
        self.latitudes = _clip_lat(ds.latitudes)
        self.longitudes = _clip_lon(ds.longitudes)

        # ADD THIS BLOCK
        # Static per-gridpoint domain mask; the anemoi grid is fixed so compute once.
        # This is a PRE-FILTER only: the authoritative filter is the cell remap in
        # hpy_cell_splits(). bbox_point_mask() is deliberately permissive (it widens
        # the box to cover pad_rings), so it never drops a point the tokenizer keeps.
        if domain is not None and not domain.is_global:
            self.domain_mask = domain.bbox_point_mask(self.latitudes, self.longitudes)
            n_keep = int(self.domain_mask.sum())
            _logger.info(
                "%s: domain pre-filter keeps %d of %d grid points (%.2f%%)",
                stream_info["name"], n_keep, len(self.domain_mask),
                100.0 * n_keep / max(len(self.domain_mask), 1),
            )
            if n_keep == 0:
                _logger.warning(
                    "%s: no grid points inside the domain; stream will be empty.",
                    stream_info["name"],
                )
            # Pre-subset the cached coords: they are read ONLY to build `latlon` in
            # _get() (verified by repo-wide grep -- 4 hits, all in this file), so
            # subsetting here means _get() needs no coords change at all.
            self.latitudes = self.latitudes[self.domain_mask]
            self.longitudes = self.longitudes[self.domain_mask]

        # select/filter requested source channels
        if stream_info.get(str(stage) + "_source_channels") is None:
            self.source_idx = self.select_channels(ds, "source")
            self.source_channels = [ds.variables[i] for i in self.source_idx]
        else:
            self.source_channels = stream_info.get(str(stage) + "_source_channels")
            self.source_idx = [ds.variables.index(ch) for ch in self.source_channels]

        # select/filter requested target channels
        if stream_info.get(str(stage) + "_target_channels") is None:
            self.target_idx = self.select_channels(ds, "target")
            self.target_channels = [ds.variables[i] for i in self.target_idx]
        else:
            self.target_channels = stream_info.get(str(stage) + "_target_channels")
            self.target_idx = [ds.variables.index(ch) for ch in self.target_channels]

        # get target channel weights from stream config
        if stream_info.get("target_channel_weights") is None:
            self.target_channel_weights = self.parse_target_channel_weights()
        else:
            self.target_channel_weights = stream_info.get("target_channel_weights")

        # select/filter requested geoinfo channels (can be any variable, not just constant-in-time)
        if stream_info.get("geoinfo_channels") is None:
            self.geoinfo_idx = self.select_geoinfo_channels(ds)
            self.geoinfo_channels = [ds.variables[i] for i in self.geoinfo_idx]
        else:
            self.geoinfo_channels = stream_info.get("geoinfo_channels")
            self.geoinfo_idx = [ds.variables.index(ch) for ch in self.geoinfo_channels]

        # set geoinfo normalization statistics
        if len(self.geoinfo_idx) > 0:
            self.mean_geoinfo = ds.statistics["mean"][self.geoinfo_idx]
            self.stdev_geoinfo = ds.statistics["stdev"][self.geoinfo_idx]
        else:
            self.mean_geoinfo = np.zeros(0)
            self.stdev_geoinfo = np.ones(0)

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: geoinfo channels: {self.geoinfo_channels}")

        self.properties = {
            "stream_id": 0,
        }
        self.mean = ds.statistics["mean"]
        self.stdev = ds.statistics["stdev"]

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0
        self.missing_idxs = set()

    @override
    def length(self) -> int:
        return self.len

    def window_has_missing(self, idx) -> bool:
        """True if the time window at `idx` overlaps a missing date."""
        if getattr(self, "ds", None) is None:
            return False
        try:
            (t_idxs, _) = self._get_dataset_idxs(idx)
        except Exception:
            return True
        if len(t_idxs) == 0:
            return True
        return any(int(t) in self.missing_idxs for t in t_idxs)

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        try:
            data = self.ds[didx_start:didx_end][:, :, 0].astype(np.float32)
        except MissingDateError as e:
            _logger.debug(f"Date not present in anemoi dataset: {str(e)}. Skipping.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # coords-first representation and collapse multiple steps
        data = data.transpose([0, 2, 1]).reshape((data.shape[0] * data.shape[2], -1))

        if self.domain_mask is not None:
            data = data[np.tile(self.domain_mask, len(t_idxs))]

        # extract geoinfo channels (can be time-varying, so read from dataset)
        geoinfos = data[:, list(self.geoinfo_idx)]
        # extract channels
        data = data[:, list(channels_idx)]

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            axis=0,
        ).transpose()
        # repeat latlon len(t_idxs) times
        coords = np.vstack((latlon,) * len(t_idxs))

        # date time matching #data points of data
        # Assuming a fixed frequency for the dataset
        datetimes = np.repeat(self.ds.dates[didx_start:didx_end], len(data) // len(t_idxs))

        # TEMP §11 layout check -- remove after verifying
#        _logger.warning(
#            "§11 %s: T=%d G=%d nvars=%d | geoinfo_idx=%s -> geoinfos%s | "
#            "data%s coords%s datetimes%s | G==len(lat)? %s",
#            self.stream_info["name"], len(t_idxs), len(self.latitudes),
#            len(self.ds.variables), list(self.geoinfo_idx), geoinfos.shape,
#            data.shape, coords.shape, datetimes.shape,
#            coords.shape[0] == len(t_idxs) * len(self.latitudes),
#        )

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd

    def select_channels(self, ds0: anemoi_datasets, ch_type: str) -> NDArray[np.int64]:
        """
        Select source or target channels

        Parameters
        ----------
        ds0 :
            raw anemoi dataset with available channels
        ch_type :
            "source" or "target", i.e channel type to select

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes

        """

        channels = self.stream_info.get(ch_type)
        channels_exclude = self.stream_info.get(ch_type + "_exclude", [])
        # sanity check
        is_empty = len(channels) == 0 if channels is not None else False
        if is_empty:
            stream_name = self.stream_info["name"]
            _logger.warning(f"No channel for {stream_name} for {ch_type}.")

        chs_idx = np.sort(
            [
                ds0.name_to_index[k]
                for (k, v) in ds0.typed_variables.items()
                if (
                    not v.is_computed_forcing
                    and not v.is_constant_in_time
                    and (
                        np.array([f == k for f in channels]).any() if channels is not None else True
                    )
                    and not np.array([f == k for f in channels_exclude]).any()
                )
            ]
        )

        return np.array(chs_idx, dtype=np.int64)

    def select_geoinfo_channels(self, ds0: anemoi_datasets) -> NDArray[np.int64]:
        """
        Select geoinfo channels (can be any variable, not just constant-in-time)

        Parameters
        ----------
        ds0 :
            raw anemoi dataset with available channels

        Returns
        -------
        NDArray of channel indices for geoinfo variables

        """

        geoinfo_channels = self.stream_info.get("geoinfo_channels", [])

        if len(geoinfo_channels) == 0:
            return np.array([], dtype=np.int64)

        # Select channels that match the geoinfo list (exact match required)
        chs_idx = np.sort(
            [ds0.name_to_index[k] for k in ds0.typed_variables.keys() if k in geoinfo_channels]
        )

        if len(chs_idx) == 0 and len(geoinfo_channels) > 0:
            stream_name = self.stream_info["name"]
            _logger.warning(
                f"No matching geoinfo channels found for {stream_name}. "
                f"Requested: {geoinfo_channels}"
            )

        return np.array(chs_idx, dtype=np.int64)


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """
    Clip latitudes to the range [-90, 90] and ensure periodicity.
    """
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """
    Clip longitudes to the range [-180, 180] and ensure periodicity.
    """
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)
