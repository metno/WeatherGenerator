"""Station reader (type: station) options on synthetic files with the layout of the metno
station netCDF files (CPU):
  - stats_file: mean / stdev taken from another file (same normalisation for every file)
  - drop_empty_rows: station rows with every requested channel missing are dropped
  - stations with a NaN geoinfo are reported
  - defaults unchanged (no stats_file, no drop_empty_rows)"""
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.data_reader_synop import DataReaderSynop

logging.basicConfig(level=logging.WARNING)

CH = ["2t", "2r", "sp", "10si", "wind_speed_of_gust", "tp"]
GEO = ["altitude", "SFX.ZS", "SFX.FRAC_SEA", "FOREST"]
FILL = np.float32(9.96921e36)


def make_file(path, start, nt, nloc, t_offset, seed, nan_geo_station=None):
    """6-hourly station file; station 0 always missing, station 1 missing every other step."""
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, periods=nt, freq="6h")
    ds = xr.Dataset(coords={"time": time, "location": np.arange(nloc)})
    ds["latitude"] = ("location", rng.uniform(58, 70, nloc))
    ds["longitude"] = ("location", rng.uniform(5, 30, nloc))
    for ch in CH:
        v = rng.normal(0, 1, (nt, nloc)).astype(np.float32)
        if ch == "2t":
            v = v * 5 + 275 + t_offset
        v[:, 0] = np.nan
        v[::2, 1] = np.nan
        ds[ch] = (("time", "location"), v)
        ds[ch].encoding["_FillValue"] = FILL
    for g in GEO:
        v = rng.uniform(0, 1000, nloc).astype(np.float32)
        if nan_geo_station is not None and g == "FOREST":
            v[nan_geo_station] = np.nan
        ds[g] = ("location", v)
    ds.to_netcdf(path)


def reader(fname, t0, t1, **extra):
    tw = TimeWindowHandler(
        np.datetime64(t0), np.datetime64(t1), np.timedelta64(6, "h"), np.timedelta64(6, "h")
    )
    si = {"name": "SYNOP", "source": CH, "target": CH, "geoinfos": GEO, **extra}
    return DataReaderSynop(tw, fname, si)


with tempfile.TemporaryDirectory() as d:
    f_train, f_val = Path(d) / "train.nc", Path(d) / "val.nc"
    make_file(f_train, "2015-01-01", 4 * 365, 20, t_offset=0.0, seed=1)
    make_file(f_val, "2023-01-01", 4 * 365, 6, t_offset=4.0, seed=2, nan_geo_station=3)

    # 1. defaults: own stats, all rows kept (station 0 enters with all NaN)
    r_tr = reader(f_train, "2015-02-01", "2015-11-30T18:00")
    r_va0 = reader(f_val, "2023-02-01", "2023-11-30T18:00")
    i2t = r_tr.channels_file.index("2t")
    print(f"default  : mean 2t train {r_tr.mean[i2t]:.2f}, val (own) {r_va0.mean[i2t]:.2f}")
    assert abs(r_va0.mean[i2t] - r_tr.mean[i2t]) > 3.0, "synthetic files should differ"
    rd = r_tr.get_source(0)
    print(f"default  : rows {len(rd.data)} (20 stations), all-NaN rows "
          f"{int((~np.isfinite(rd.data).any(axis=1)).sum())}")
    assert len(rd.data) == 20

    # 2. stats_file: val file normalised with the training stats (incl. geoinfos)
    r_va = reader(f_val, "2023-02-01", "2023-11-30T18:00", stats_file=str(f_train))
    print(f"stats_file: mean 2t val {r_va.mean[i2t]:.2f} (train {r_tr.mean[i2t]:.2f})")
    assert np.allclose(r_va.mean, r_tr.mean, equal_nan=True)
    assert np.allclose(r_va.stdev, r_tr.stdev, equal_nan=True)
    assert np.allclose(r_va.mean_geoinfo, r_tr.mean_geoinfo)

    # 3. drop_empty_rows: station 0 always dropped, station 1 dropped on even steps
    r_d = reader(f_train, "2015-02-01", "2015-11-30T18:00", drop_empty_rows=True)
    for idx in range(4):
        s, t = r_d.get_source(idx), r_d.get_target(idx)
        ref = r_tr.get_source(idx)
        n_empty = int((~np.isfinite(ref.data).any(axis=1)).sum())
        print(f"drop_rows: window {idx}: rows {len(s.data)} (was {len(ref.data)}, "
              f"{n_empty} empty), target rows {len(t.data)}")
        assert len(s.data) == len(ref.data) - n_empty == len(t.data)
        assert np.isfinite(s.data).any(axis=1).all()
        assert len(s.coords) == len(s.geoinfos) == len(s.datetimes) == len(s.data)
        # kept rows unchanged, in the same order
        keep = np.isfinite(ref.data).any(axis=1)
        assert np.array_equal(s.coords, ref.coords[keep])
        assert np.array_equal(s.data, ref.data[keep], equal_nan=True)

    # 4. NaN geoinfo station is reported (warning above) and later removed by the base class
    s = r_va.get_source(0)
    s = s.remove_nan_coords_and_geoinfos()
    print(f"nan geo  : rows after remove_nan_coords_and_geoinfos {len(s.data)} (6 stations)")
    assert len(s.data) == 5

print("OK")
