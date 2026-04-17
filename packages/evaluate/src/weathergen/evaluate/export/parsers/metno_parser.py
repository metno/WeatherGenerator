import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import scipy.interpolate
from omegaconf import OmegaConf

from weathergen.evaluate.export.cf_utils import CfParser

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Usage:

uv run export --run-id ciga1p9c --stream ERA5 
--output-dir ./test_output1 
--format metno --samples 1 2  --fsteps 1 2 3
"""

def get_proj_name(ds: xr.Dataset) -> str:
    """
    Determines the name of the variable that contains projection information
    """
    if "grid_mapping" in ds.latitude.attrs:
        return ds.latitude.grid_mapping

    for name in ds.variables:
        if "projection" in name:
            return name

    return None


class MetnoParser(CfParser):
    """
    Child class for handling MET Norway's NetCDF output format.
    """

    def __init__(self, config: OmegaConf, **kwargs):
        """
        CF-compliant parser that outputs fields on 2D projected grids.

        Parameters
        ----------
        config : OmegaConf
            Configuration defining variable mappings and dimension metadata.
        ds : xr.Dataset
            Input dataset.
        metno_template: a path to a NetCDF template file with grid information.

        Returns
        -------
        xr.Dataset
            CF-compliant dataset with consistent naming and attributes.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, "metno_template"):
            raise ValueError("Template file must be provided for Metno format.")

        # TODO: Why does the base class need this?
        grid_type = 1
        super().__init__(config=config, grid_type=grid_type)

        self.mapping = config.get("variables", {})

        self.template = xr.open_dataset(self.metno_template)

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
        da_fs = []

        # Loop over each time step
        for result in fstep_iterator_results:
            if result is None:
                continue
            # result is already a materialized xarray DataArray (built in the worker).
            if not isinstance(result, xr.DataArray):
                # Use squeeze to remove singleton dimensions (e.g., sample or stream) that may be present
                # If the input has shape (1, 1, forecast_step, ipoint, channel, ensemble_member),
                # squeeze() will remove the first two axes (sample, stream) if their size is 1.
                # This means only forecast_step, ipoint, channel, ensemble_member remain for further processing.
                result = result.as_xarray().squeeze()
            result = result.sel(channel=self.channels)
            da_fs.append(result)

        _logger.info(f"Retrieved {len(da_fs)} forecast steps for type {self.data_type}.")
        _logger.info(f"Saved sample data to {self.output_format} in {self.output_dir}.")

        if da_fs:
            da_fs = self.concatenate(da_fs, dim="forecast_step", sortby_dim="forecast_step")
            da_fs = self.regrid(da_fs)
            da_fs = self.add_global_attributes(da_fs)
            self.save(da_fs, ref_time)

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

    def concatenate(
        self,
        array_list,
        dim="valid_time",
        data_vars="minimal",
        coords="different",
        compat="equals",
        combine_attrs="drop",
        sortby_dim="valid_time",
    ) -> xr.Dataset:
        """
        Uses list of pred/target xarray DataArrays to save one sample to a NetCDF file.

        Parameters
        ----------
        type_str : str
            Type of data ('pred' or 'targ') to include in the filename.
        array_list : list of xr.DataArray
            List of DataArrays to concatenate.
        dim : str, optional
            Dimension along which to concatenate. Default is 'valid_time'.
        data_vars : str, optional
            How to handle data variables during concatenation. Default is 'minimal'.
        coords : str, optional
            How to handle coordinates during concatenation. Default is 'different'.
        compat : str, optional
            Compatibility check for variables. Default is 'equals'.
        combine_attrs : str, optional
            How to combine attributes. Default is 'drop'.
        sortby_dim : str, optional
            Dimension to sort the final dataset by. Default is 'valid_time'.

        Returns
        -------
        xr.Dataset
            Concatenated xarray Dataset.
        """

        data = xr.concat(
            array_list,
            dim=dim,
            #data_vars=data_vars,
            coords=coords,
            compat=compat,
            combine_attrs=combine_attrs,
        ).sortby(sortby_dim)

        return data

    def regrid(self, ds: xr.DataArray) -> xr.Dataset:
        # The export worker returns one sample/stream at a time and concatenates over
        # forecast steps, so expected dims are:
        # - without ensemble:   (forecast_step, ipoint, channel)
        # - with ensemble:      (forecast_step, ipoint, channel, ensemble_member)
        if "forecast_step" not in ds.dims:
            ds = ds.expand_dims("forecast_step")

        if "ensemble_member" not in ds.dims:
            ds = ds.expand_dims("ensemble_member")

        ds = ds.transpose("forecast_step", "ipoint", "channel", "ensemble_member")

        # After this, ds.values will have shape (Nsteps, Npoints, Nchannels, Nmembers).
        # The original data may have had 6 dimensions: (sample, stream, forecast_step, ipoint, channel, ensemble_member).
        # Here, the sample and stream dimensions are omitted because the export worker processes one sample/stream at a time
        # and concatenates over forecast steps, so only the relevant dimensions for output are kept.
        Nsteps = ds.sizes["forecast_step"]
        Npoints = ds.sizes["ipoint"]
        Nchannels = ds.sizes["channel"]
        Nmembers = ds.sizes["ensemble_member"]

        x = self.template.x.values[:]
        y = self.template.y.values[:]

        has_ens = Nmembers > 1

        valid_times = ds.valid_time
        if "forecast_step" in valid_times.dims:
            times = valid_times.isel(ipoint=0).astype("datetime64[s]").astype("float64").values
        else:
            # Fallback for single-step inputs where valid_time may only depend on ipoint.
            times = np.array(
                [valid_times.isel(ipoint=0).astype("datetime64[s]").astype("float64").item()]
            )

        coords = {"time": times, "x": x, "y": y}
        if has_ens:
            coords["ensemble_member"] = range(Nmembers)

        # Attributes on coordinate variables
        new_ds = xr.Dataset(coords)
        for name in ["x", "y"]:
            new_ds[name].attrs["standard_name"] = f"projection_{name}_coordinate"
            new_ds[name].attrs["units"] = "m"
        new_ds["time"].attrs["units"] = "seconds since 1970-01-01T00:00:00 +00:00"
        new_ds["forecast_reference_time"] = ([], times[0], {}, {"dtype": "double"})
        new_ds["forecast_reference_time"].attrs["units"] = "seconds since 1970-01-01T00:00:00 +00:00"
        if has_ens:
            new_ds["ensemble_member"].attrs["standard_name"] = "realization"

        olat = self.template.latitude.values[:]
        olon = self.template.longitude.values[:]
        assert len(olat.shape) == 2, olat.shape
        assert len(olon.shape) == 2, olon.shape

        lat_attrs = {"units": "degree_north", "standard_name": "latitude", "grid_mapping": "projection"}
        lon_attrs = {"units": "degree_east", "standard_name": "longitude", "grid_mapping": "projection"}
        new_ds["latitude"] = (("y", "x"), olat, lat_attrs)
        new_ds["longitude"] = (("y", "x"), olon, lon_attrs)

        # Set up projection variable
        proj_var_name = get_proj_name(self.template)
        if proj_var_name is not None:
            proj_attrs = self.template[proj_var_name].attrs
            new_ds["projection"] = ([], 0, proj_attrs, {"dtype": "int32"})

        ilat = ds.lat.values[:]
        ilon = ds.lon.values[:]
        Isort = self.get_sorting(ilat, ilon, olat.flatten(), olon.flatten())

        if Npoints != len(Isort):
            raise ValueError(
                f"Point count mismatch between input ({Npoints}) and template ({len(Isort)})."
            )

        # forecast_step, ipoint, channel, ensemble_member
        all_values = ds.values

        # Add variables
        for i, channel in enumerate(ds.channel.to_numpy()):
            _logger.info(f"Processing {channel}")
            values = all_values[:, Isort, i, :]
            if has_ens:
                new_shape = [len(times), Nmembers, len(y), len(x)]
                dims =  ["time", "ensemble_member", "y", "x"]
            else:
                new_shape = [len(times), len(y), len(x)]
                dims =  ["time", "y", "x"]
                values = values[..., 0]

            values = np.reshape(values, new_shape)
            attrs = {"coordinates": "longitude latitude"}
            if proj_var_name is not None:
                attrs["grid_mapping"] = proj_var_name
            var_name = self.channel_to_variable_name(channel)

            new_ds[var_name] = (dims, values, attrs)

        # Copy static fields from templae
        for name in self.template.variables:
            var = self.template[name]
            if var.dims == ("y", "x") and name not in new_ds:
                new_ds[name] = (("y", "x"), var.values[:], var.attrs)

        return new_ds

    def channel_to_variable_name(self, channel):
        if channel == "2t":
            return "air_temperature_2m"
        elif channel == "10si":
            return "wind_speed_10m"
        elif channel == "10u":
            return "u_wind_at_10m"
        elif channel == "10v":
            return "v_wind_at_10m"
        else:
            return channel

    def get_sorting(self,
            ilat: np.ndarray,
            ilon: np.ndarray,
            olat: np.ndarray,
            olon: np.ndarray) -> np.ndarray:
        """
        Computes for each output point the index into the input points corresponding to the nearest
        neighbour

        Parameters
        ----------
            ilat: input latitudes in degrees
            ilon: input longitudes in degrees
            olat: output latitudes in degrees
            olon: output longitudes in degrees
        Returns
        -------
            Numpy array of integers
        """
        ipoints = np.concatenate([ilat[:, None], ilon[:, None]], axis=1)
        opoints = np.concatenate([olat[:, None], olon[:, None]], axis=1)

        interpolator = scipy.interpolate.NearestNDInterpolator(ipoints, np.arange(len(ilat)))
        Isort = interpolator(opoints).astype(int)
        return Isort

    def add_global_attributes(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add CF conventions to the dataset attributes.

        Parameters
        ----------
            ds : Input xarray Dataset to add conventions to.
        Returns
        -------
            xarray Dataset with CF conventions added to attributes.
        """
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
            data_type : Type of data ('pred' or 'targ') to include in the filename.
            forecast_ref_time : Forecast reference time to include in the filename.

        Returns
        -------
            None
        """
        out_fname = self.get_output_filename(forecast_ref_time)
        _logger.info(f"Saving to {out_fname}.")
        ds.to_netcdf(out_fname, unlimited_dims=["time"])
        _logger.info(f"Saved NetCDF file to {out_fname}.")
