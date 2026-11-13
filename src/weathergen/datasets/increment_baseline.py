# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Increment (residual) learning w.r.t. an interpolated baseline field.

With the stream option `increment_baseline`, a target stream (e.g. CARRA2) gets a baseline:
a coarse dataset (e.g. ERA5) interpolated onto the target points, at the target times. The
decoder output is then an increment: the prediction that enters the loss (and that is written
to the output) is

    pred = baseline + decoder_output          (in the normalised space of the target stream)

so that the decoder only learns the correction "target - interpolated baseline". The losses
are unchanged.

The baseline is computed in the data loader, from the target reader's own (domain-cropped)
grid, so the interpolation weights are computed once. It is appended to the target data as
extra channels before sub-sampling / tokenisation, which keeps it aligned point by point with
the targets and the target coordinates, and it is split off again after tokenisation.

Stream config (target stream, anemoi type only):

  increment_baseline:
    filenames: ['aifs-ea-an-oper-0001-mars-n320-1979-2024-6h-v1-for-single-v2.zarr']
    # target channel -> baseline variable; default: every target channel that also exists
    # in the baseline dataset, under the same name. Target channels without a baseline
    # variable get baseline 0 (= climatological mean in normalised space), i.e. they are
    # predicted as full fields, as without this option.
    channels: { 2t: 2t, 10u: 10u }
    # optional unit conversion, applied as baseline * scale + offset (target units)
    scale: { tp: 1000.0 }
    offset: {}
    method: linear      # linear (barycentric on a Delaunay triangulation) or nearest
    margin_deg: 2.0     # baseline points kept around the target grid's lat/lon box
"""

import logging
import pathlib

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, cKDTree

_logger = logging.getLogger(__name__)


def has_increment_baseline(stream_info) -> bool:
    """True if the stream predicts increments w.r.t. an interpolated baseline"""
    return stream_info.get("increment_baseline") is not None


def _stereographic(lats, lons, lat0, lon0):
    """Stereographic projection (degrees in, unit-sphere plane coordinates out) about
    (lat0, lon0). Conformal, so triangles of a quasi-regular grid stay well shaped."""
    phi, lam = np.deg2rad(lats), np.deg2rad(lons)
    phi0, lam0 = np.deg2rad(lat0), np.deg2rad(lon0)
    dlam = lam - lam0
    cos_c = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(dlam)
    k = 2.0 / (1.0 + cos_c)
    x = k * np.cos(phi) * np.sin(dlam)
    y = k * (np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(dlam))
    return np.stack([x, y], axis=-1)


def interpolation_weights(
    src_lats: NDArray,
    src_lons: NDArray,
    dst_lats: NDArray,
    dst_lons: NDArray,
    method: str = "linear",
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """
    Interpolation weights from scattered source points to destination points.

    Returns (idxs, weights), both (num_dst, 3) for "linear" and (num_dst, 1) for "nearest";
    the interpolated value is sum_k weights[:, k] * src[idxs[:, k]]. Destination points
    outside the convex hull of the source points fall back to the nearest source point.
    """
    lat0 = float(np.mean(dst_lats))
    # circular mean of the longitudes (robust across the date line)
    lon0 = float(
        np.rad2deg(
            np.arctan2(np.sin(np.deg2rad(dst_lons)).mean(), np.cos(np.deg2rad(dst_lons)).mean())
        )
    )
    src_xy = _stereographic(src_lats, src_lons, lat0, lon0)
    dst_xy = _stereographic(dst_lats, dst_lons, lat0, lon0)

    _, nearest = cKDTree(src_xy).query(dst_xy)
    if method == "nearest":
        return nearest[:, None].astype(np.int64), np.ones((len(dst_xy), 1), dtype=np.float32)
    if method != "linear":
        raise ValueError(f"increment_baseline: unknown method '{method}'")

    tri = Delaunay(src_xy)
    simplex = tri.find_simplex(dst_xy)
    inside = simplex >= 0
    s = np.where(inside, simplex, 0)
    trans = tri.transform[s]  # (num_dst, 3, 2)
    b = np.einsum("nij,nj->ni", trans[:, :2], dst_xy - trans[:, 2])
    weights = np.concatenate([b, 1.0 - b.sum(axis=1, keepdims=True)], axis=1)
    idxs = tri.simplices[s]

    # outside the hull: nearest neighbour
    idxs[~inside] = nearest[~inside, None]
    weights[~inside] = np.array([1.0, 0.0, 0.0])
    if (~inside).any():
        _logger.warning(
            "increment_baseline: %d of %d target points outside the baseline grid hull; "
            "using nearest neighbour there.",
            int((~inside).sum()),
            len(dst_xy),
        )
    return idxs.astype(np.int64), weights.astype(np.float32)


class IncrementBaseline:
    """
    Baseline field for a target reader: a (coarse) anemoi dataset interpolated onto the
    target reader's grid points, for the target channels.

    Call with the ReaderData returned by the target reader (before any cropping/shuffling);
    returns an array (num_points, num_target_channels) in physical (target) units, NaN where
    no baseline exists (unmapped channel or missing date).
    """

    def __init__(self, cfg: dict, target_reader, data_paths: list | None = None):
        import anemoi.datasets as anemoi_datasets

        self.target_channels = list(target_reader.target_channels)
        self.num_points = len(target_reader.latitudes)

        filename = self._resolve(cfg, data_paths or [])
        ds = anemoi_datasets.open_dataset(filename)

        # target channel -> baseline variable
        ch_map = cfg.get("channels")
        if ch_map is None:
            ch_map = {ch: ch for ch in self.target_channels if ch in ds.name_to_index}
        ch_map = {str(k): str(v) for k, v in dict(ch_map).items()}
        for t_ch, b_ch in ch_map.items():
            if t_ch not in self.target_channels:
                raise ValueError(f"increment_baseline: '{t_ch}' is not a target channel")
            if b_ch not in ds.name_to_index:
                raise ValueError(f"increment_baseline: '{b_ch}' not in {filename}")
        # (position in target channels, index in baseline dataset)
        self.t_pos = np.array([self.target_channels.index(t) for t in ch_map], dtype=np.int64)
        self.b_idx = np.array([ds.name_to_index[b] for b in ch_map.values()], dtype=np.int64)
        scale, offset = dict(cfg.get("scale", {}) or {}), dict(cfg.get("offset", {}) or {})
        self.scale = np.array([float(scale.get(t, 1.0)) for t in ch_map], dtype=np.float32)
        self.offset = np.array([float(offset.get(t, 0.0)) for t in ch_map], dtype=np.float32)

        missing = [ch for ch in self.target_channels if ch not in ch_map]
        _logger.info(f"increment_baseline: {filename}: {ch_map}")
        if missing:
            _logger.info(f"increment_baseline: no baseline (full-field prediction) for {missing}")

        # baseline points in the target grid's lat/lon box (+ margin); then weights, once
        t_lats = np.asarray(target_reader.latitudes, dtype=np.float64)
        t_lons = np.asarray(target_reader.longitudes, dtype=np.float64)
        b_lats = np.asarray(ds.latitudes, dtype=np.float64)
        b_lons = ((np.asarray(ds.longitudes, dtype=np.float64) + 180.0) % 360.0) - 180.0
        m = float(cfg.get("margin_deg", 2.0))
        sel = (b_lats >= t_lats.min() - m) & (b_lats <= t_lats.max() + m)
        # longitude box only when the target grid does not wrap around
        if t_lons.max() - t_lons.min() < 180.0:
            m_lon = m / max(np.cos(np.deg2rad(min(abs(t_lats).max() + m, 89.0))), 1e-3)
            sel &= (b_lons >= t_lons.min() - m_lon) & (b_lons <= t_lons.max() + m_lon)
        self.b_pts = np.flatnonzero(sel)
        assert len(self.b_pts) >= 3, "increment_baseline: no baseline points around the target"
        self.idxs, self.weights = interpolation_weights(
            b_lats[self.b_pts], b_lons[self.b_pts], t_lats, t_lons, cfg.get("method", "linear")
        )
        _logger.info(
            f"increment_baseline: {len(self.b_pts)} baseline points -> {self.num_points} "
            f"target points ({cfg.get('method', 'linear')})"
        )

        self.ds = ds
        self.dates = np.asarray(ds.dates).astype("datetime64[s]")
        self.ens_member = int(cfg.get("ensemble_member", 0))

    @staticmethod
    def _resolve(cfg, data_paths):
        names = cfg.get("filenames") or [cfg.get("filename")]
        assert len(names) == 1 and names[0], "increment_baseline: give exactly one filename"
        fname = pathlib.Path(names[0])
        if fname.exists():
            return str(fname)
        for p in data_paths:
            if (pathlib.Path(p) / fname).exists():
                return str(pathlib.Path(p) / fname)
        raise FileNotFoundError(f"increment_baseline: did not find {fname} in {data_paths}")

    def interpolate(self, field: NDArray) -> NDArray[np.float32]:
        """field: (channels, baseline points) -> (target points, channels)"""
        vals = field[:, self.idxs]  # (channels, num_target, k)
        return np.einsum("cnk,nk->nc", vals, self.weights).astype(np.float32)

    def __call__(self, rdata) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """
        Returns (baseline, valid): baseline (num rows, num target channels) in target units,
        NaN for channels without a baseline variable; valid (num rows) is False where the
        baseline dataset has no data at that time. Those rows must be dropped: their mapped
        channels would otherwise be scored as full fields instead of increments.
        """
        num_ch = len(self.target_channels)
        out = np.full((len(rdata.data), num_ch), np.nan, dtype=np.float32)
        valid = np.ones(len(rdata.data), dtype=bool)
        if len(rdata.data) == 0 or len(self.t_pos) == 0:
            return out, valid
        n_t, rem = divmod(len(rdata.data), self.num_points)
        assert rem == 0, (
            "increment_baseline: target data is not on the reader grid "
            f"({len(rdata.data)} rows, {self.num_points} grid points)"
        )
        times = np.asarray(rdata.datetimes).astype("datetime64[s]")
        for it in range(n_t):
            rows = slice(it * self.num_points, (it + 1) * self.num_points)
            field = self._read(times[rows.start])
            if field is None:
                valid[rows] = False
                continue
            out[rows, self.t_pos] = self.interpolate(field) * self.scale + self.offset
        return out, valid

    def _read(self, t) -> NDArray[np.float32] | None:
        """baseline channels at the baseline points at time t: (channels, points), or None"""
        from anemoi.datasets.data import MissingDateError

        i = int(np.searchsorted(self.dates, t))
        if i >= len(self.dates) or self.dates[i] != t:
            _logger.warning(f"increment_baseline: no baseline at {t}; target dropped there")
            return None
        try:
            # same access pattern as DataReaderAnemoi: (time, channels, ensemble, points)
            x = self.ds[i : i + 1][0, :, self.ens_member]
        except MissingDateError:
            _logger.warning(f"increment_baseline: baseline missing at {t}; target dropped there")
            return None
        return x[self.b_idx][:, self.b_pts].astype(np.float32)
