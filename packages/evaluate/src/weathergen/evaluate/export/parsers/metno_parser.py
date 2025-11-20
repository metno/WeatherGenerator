import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
import pyproj

from weathergen.evaluate.export.cf_utils import CfParser
from weathergen.evaluate.export.reshape import find_pl

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Usage:

uv run export --run-id ciga1p9c --stream ERA5 
--output-dir ./test_output1 
--format metno --samples 1 2  --fsteps 1 2 3
"""


class MetnoParser(CfParser):
    """
    Child class for handling MET Norway's NetCDF output format.
    """

    def __init__(self, config: OmegaConf, **kwargs):
        """
        CF-compliant parser that handles both regular grids.

        Parameters
        ----------
        config : OmegaConf
            Configuration defining variable mappings and dimension metadata.
        ds : xr.Dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            CF-compliant dataset with consistent naming and attributes.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, "proj4_str"):
            raise ValueError("proj4_str must be provided for MET Norway parser.")
        if not hasattr(self, "field_shape"):
            raise ValueError("field_shape must be provided for MET Norway parser.")

        super().__init__(config=config, grid_type=self.grid_type)

        # self.field_shape = [13*73, 1069]
        # self.proj4_str = "+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs"

        self.mapping = config.get("variables", {})
        # self.dim_mapping = config.get("dimensions"), {})

    def process_sample(
        self,
        fstep_iterator_results: iter,
        ref_time: np.datetime64,
    ):
        """
        Process results from get_data_worker: reshape, concatenate, add metadata, and save.
        Parameters
        ----------
            fstep_iterator_results : Iterator over results from get_data_worker.
            ref_time : Forecast reference time for the sample.
        Returns
        -------
            None
        """
        da = self.concatenate(fstep_iterator_results)
        num_points = len(da.lat.values)
        if num_points != self.field_shape[0] * self.field_shape[1]:
            fs = self.field_shape
            raise ValueError(f"Wrong field_shape: {fs}. Num gridpoints: {num_points}).")

        coords = dict()

        time_step_seconds = self.fstep_hours.astype("int64") * 3600
        forecast_reference_time = ref_time.astype('datetime64[s]').astype(np.int32)
        times = da.forecast_step.data.astype("int64") * time_step_seconds + forecast_reference_time
        times = times.astype(np.double)
        coords["time"] = times

        lats = self.reshape(da.lat.values)
        lons = self.reshape(da.lon.values)
        x, y = get_xy(lats, lons, self.proj4_str)
        coords["y"] = y.astype(np.double)
        coords["x"] = x.astype(np.double)

        ds = xr.Dataset(coords)

        ds["latitude"] = (("y", "x"), lats)
        ds["longitude"] = (("y", "x"), lons)
        ds["forecast_reference_time"] = ([], forecast_reference_time)

        for name in ["forecast_reference_time", "time"]:
            ds[name].attrs["units"] = "seconds since 1970-01-01T00:00:00Z"
            ds[name].attrs["standard_name"] = name

        ds["latitude"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
        ds["longitude"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
        ds["x"].attrs = {"units": "m", "standard_name": "projection_x_coordinate"}
        ds["y"].attrs = {"units": "m", "standard_name": "projection_y_coordinate"}

        ds["projection"] = ([], np.array(1, np.int32))
        ds["projection"].attrs = get_proj_attributes(self.proj4_str)

        ds = self.add_attributes(ds)

        for channel in self.channels:
            index = np.where(da.channel == channel)[0][0]
            mapping = self.mapping.get(channel)
            new_name = mapping["var"]
            val = self.reshape(da[:, :, index].values)

            """
            import matplotlib.pylab as mpl
            mpl.scatter(da.lon.values, da.lat.values, c=da[0, :, index].values,
                    vmin=270, vmax=290,cmap="RdBu_r")
            mpl.show()
            """

            ds[new_name] = (("time", "y", "x"), val)

            attrs_to_copy = {"std": "standard_name", "units": "units", "long": "long_name"}
            for iattr,oattr in attrs_to_copy.items():
                if iattr in mapping:
                    ds[new_name].attrs[oattr] = mapping[iattr]
            ds[new_name].attrs["coordinates"] = "latitude longitude"
            ds[new_name].attrs["grid_mapping"] = "projection"

        # _logger.info(f"Retrieved {len(ds)} forecast steps for type {self.data_type}.")

        ds = self.assign_coords(ds, ref_time)
        ds = self.add_metadata(ds)
        self.save(ds, ref_time)
        _logger.info(f"Saved sample data to {self.output_format} in {self.output_dir}.")

    def add_attributes(self, ds: xr.Dataset) -> xr.Dataset:
        for name,attrs in self.config.dimensions.items():
            if name in ds:
                for k,v in attrs.items():
                    ds[name].attrd[k] = v

        return ds


    def get_output_filename(self, forecast_ref_time: np.datetime64) -> Path:
        """
        Generate output filename based on prefix (should refer to type e.g. pred/targ),
        run_id, sample index, output directory, format and forecast_ref_time.

        Parameters
        ----------
            forecast_ref_time : Forecast reference time to include in the filename.

        Returns
        -------
            Full path to the output file.
        """

        frt = np.datetime_as_string(forecast_ref_time, unit="h")
        out_fname = (
            Path(self.output_dir) / f"{self.data_type}_{frt}_{self.run_id}.{self.file_extension}"
        )
        return out_fname

    def reshape(self, data: xr.DataArray) -> xr.Dataset:
        """
        Reshape dataset while preserving grid structure (regular or Gaussian).

        Parameters
        ----------
        data : np.array
            Input data with dimensions (ipoint, channel)

        Returns
        -------
        np.array
            Reshaped dataset appropriate for the grid type
        """
        if len(data.shape) == 1:
            reshaped = np.reshape(data, self.field_shape)
        else:
            shape = [data.shape[0], self.field_shape[0], self.field_shape[1]]
            reshaped = np.reshape(data, shape)
        return reshaped

    def concatenate(self, iterator):
        """
        Uses list of pred/target xarray DataArrays to save one sample to a NetCDF file.

        Parameters
        iterator : Iterator of xr.DataArray
            Iterator of DataArrays to concatenate.

        Returns
        -------
        xr.Dataset
            Concatenated xarray Dataset.
        """
        da_fs = []
        for result in iterator:
            if result is None:
                continue

            result = result.as_xarray().squeeze()
            result = result.sel(channel=self.channels).drop(["valid_time"])
            # result = self.reshape2(result)
            da_fs.append(result)

        da = xr.concat(
            da_fs,
            coords="minimal",
            compat="override",
            dim="forecast_step",
        ).sortby("forecast_step")

        return da

    def assign_coords(self, ds: xr.Dataset, reference_time: np.datetime64) -> xr.Dataset:
        """
        Assign forecast reference time coordinate to the dataset.

        Parameters
        ----------
            ds : xarray Dataset to assign coordinates to.
            reference_time : Forecast reference time to assign.

        Returns
        -------
            xarray Dataset with assigned forecast reference time coordinate.
        """
        # ds = ds.assign_coords(forecast_reference_time=reference_time.astype(np.double))

        if "sample" in ds.coords:
            ds = ds.drop_vars("sample")

        n_hours = self.fstep_hours.astype("int64")
        # ds["forecast_period"] = ds["forecast_step"] * n_hours

        return ds

    def add_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add CF-compliant attributes to the dataset variables.

        Parameters
        ----------
            ds : xarray Dataset to add attributes to.
        Returns
        -------
            xarray Dataset with CF-compliant variable attributes.
        """

        variables = {}
        dims = self.config.get("dimensions", {})
        ds_attrs = self._assign_dim_attrs(ds, dims)
        mapping = self.mapping

        for var_name, da in ds.data_vars.items():
            var_cfg = mapping.get(var_name)
            if var_cfg is None:
                continue

            dims = ["pressure", "valid_time", "latitude", "longitude"]
            if var_cfg.get("level_type") == "sfc":
                dims.remove("pressure")

            coords = self._build_coordinate_mapping(ds, var_cfg, ds_attrs)
            print(coords)

            attrs = {
                "standard_name": var_cfg.get("std", var_name),
                "units": var_cfg.get("std_unit", "unknown"),
            }

            mapped_name = var_cfg.get("var", var_name)
            ds[var_name].attrs = attrs
            """
            variables[mapped_name] = xr.DataArray(
                data=da.values,
                dims=dims,
                coords={**coords, "valid_time": ds["valid_time"].values},
                attrs=attrs,
                name=mapped_name,
            )
            """

        return ds

    def _assign_latlon_attrs(self, ds: xr.Dataset) -> None:
        """Add CF-compliant attributes to lat/lon coordinates if they exist.
        Parameters
        ----------
            ds : xr.Dataset
                Input dataset.
        Returns
        -------
            None
        """
        if "latitude" in ds.coords:
            ds.coords["latitude"].attrs.update(
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                }
            )
        if "longitude" in ds.coords:
            ds.coords["longitude"].attrs.update(
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                }
            )

    def _assign_dim_attrs(
        self, ds: xr.Dataset, dim_cfg: dict[str, Any]
    ) -> dict[str, dict[str, str]]:
        """
        Assign CF attributes from given config file.
        Parameters
        ----------
            ds : xr.Dataset
                Input dataset.
            dim_cfg : Dict[str, Any]
                Dimension configuration from mapping.
        Returns
        -------
            Dict[str, Dict[str, str]]:
                Attributes for each dimension.
        """
        ds_attrs = {}

        for dim_name, meta in dim_cfg.items():
            wg_name = meta.get("wg", dim_name)
            if dim_name in ds.dims and dim_name != wg_name:
                ds = ds.rename_dims({dim_name: wg_name})

            dim_attrs = {"standard_name": meta.get("std", wg_name)}
            if meta.get("std_unit"):
                dim_attrs["units"] = meta["std_unit"]
            ds_attrs[wg_name] = dim_attrs

        return ds_attrs

    def _build_coordinate_mapping(
        self, ds: xr.Dataset, var_cfg: dict[str, Any], attrs: dict[str, dict[str, str]]
    ) -> dict[str, Any]:
        """Create coordinate mapping for a given variable.
        Parameters
        ----------
            ds : xr.Dataset
                Input dataset.
            var_cfg : Dict[str, Any]
                Variable configuration from mapping.
            attrs : Dict[str, Dict[str, str]]
                Attributes for dimensions.
        Returns
        -------
            Dict[str, Any]:
                Coordinate mapping for the variable.
        """
        coords = {}
        coord_map = self.config.get("coordinates", {}).get(var_cfg.get("level_type"), {})

        for coord, new_name in coord_map.items():
            coords[new_name] = (
                ds.coords[coord].dims,
                ds.coords[coord].values,
                attrs[new_name],
            )

        return coords

    def _add_grid_attrs(self, ds: xr.Dataset, grid_info: dict | None = None) -> xr.Dataset:
        """
        Add Gaussian grid metadata following CF conventions.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to add metadata to
        grid_info : dict, optional
            Dictionary with grid information:
            - 'N': Gaussian grid number (e.g., N320)
            - 'reduced': Whether it's a reduced Gaussian grid

        Returns
        -------
        xr.Dataset
            Dataset with added grid metadata
        """

        if self.grid_type != "gaussian":
            return ds

        # ds = ds.copy()
        # Add grid mapping information
        ds.attrs["grid_type"] = "gaussian"

        # If grid info provided, add it
        if grid_info:
            ds.attrs["gaussian_grid_number"] = grid_info.get("N", "unknown")
            ds.attrs["gaussian_grid_type"] = (
                "reduced" if grid_info.get("reduced", False) else "regular"
            )

        return ds

    def add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add CF conventions to the dataset attributes.

        Parameters
        ----------
            ds : Input xarray Dataset to add conventions to.
        Returns
        -------
            xarray Dataset with CF conventions added to attributes.
        """
        # ds = ds.copy()
        ds.attrs["title"] = f"WeatherGenerator Output for {self.run_id} using stream {self.stream}"
        ds.attrs["institution"] = "WeatherGenerator Project"
        ds.attrs["source"] = "WeatherGenerator v0.0"
        ds.attrs["history"] = (
            "Created using the export_inference.py script on "
            + np.datetime_as_string(np.datetime64("now"), unit="s")
        )
        ds.attrs["Conventions"] = "CF-1.12"
        return ds

    def save(self, ds: xr.Dataset, forecast_ref_time: np.datetime64) -> None:
        """
        Save the dataset to a NetCDF file.

        Parameters
        ----------
            ds : xarray Dataset to save.
            forecast_ref_time : Forecast reference time to include in the filename.

        Returns
        -------
            None
        """
        out_fname = self.get_output_filename(forecast_ref_time)
        _logger.info(f"Saving to {out_fname}.")

        # Compress all variables
        encoding = {}
        for name in ds.keys():
            encoding[name] = {"zlib": True}

        ds.to_netcdf(out_fname,
                engine="netcdf4",
                unlimited_dims=["time"],
                encoding=encoding)
        _logger.info(f"Saved NetCDF file to {out_fname}.")

def get_xy(lats, lons, proj4_str):
    """Reverse engineer x and y vectors from lats and lons"""

    proj_from = pyproj.Proj("proj+=longlat")
    proj_to = pyproj.Proj(proj4_str)

    transformer = pyproj.transformer.Transformer.from_proj(proj_from, proj_to)

    xx, yy = transformer.transform(lons, lats)
    x = xx[0, :]
    y = yy[:, 0]

    return x, y


def get_proj_attributes(proj4_str):
    crs = pyproj.CRS.from_proj4(proj4_str)
    attrs = crs.to_cf()

    del attrs["crs_wkt"]
    attrs = {k: v for k, v in attrs.items() if v != "unknown"}

    return attrs


