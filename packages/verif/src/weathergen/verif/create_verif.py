import argparse
import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from time import time

from pathlib import Path

from weathergen.common.io import ZarrIO

from weathergen.evaluate.score import Scores

from weathergen.verif.diana_io import diana_io

from weathergen.verif.verif_config import Variables

from weathergen.verif.verif_processers import Processer_factory

from weathergen.verif.verif_interpolator import Interpolator_factory


def readarg():
    parser = argparse.ArgumentParser(
        description="Create verif files from a zarr file and observation file"
    )

    parser.add_argument(
        "-z",
        "--zarr",
        dest="zarrfile",
        required=True,
        help="Zarr file (.zarr)",
    )

    parser.add_argument(
        "-b",
        "--obs",
        dest="obsfile",
        required=False,
        # default="data/metno_observations_v3.nc",
        default="/lustre/storeB/project/nwp/weathergen/datasets/metno_observations_v3.nc",
        help="Observation file (.nc)",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="outfiles",
        default="output/verif/%S/%V/verif_%S_%V_%M.nc",
        required=False,
        help="Template for the output nc filenames, default will be to create output/verif/%S/%V repertories where \
              %S, %V, %d are replaced by the streams, variable and date",
    )

    parser.add_argument(
        "-d",
        "--date",
        type=str,
        dest="datefromto",
        required=False,
        default=None,
        help="From to date in format %Y%m%d%H:%Y%m%d%H or %Y%m%d:%Y%m%d, \
              excluding the second date for instance 2024010100:2024020200",
    )

    # could have a default date but not sure what makes sense
    # d = datetime.now(timezone.utc).strftime("%Y%m%d%H") + ":" + (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y%m%d%H")

    parser.add_argument(
        "-v",
        "--variables",
        default=["2t"],
        dest="variables",
        nargs="*",
        help="Do verif for these variables. Default: 2t",
    )

    parser.add_argument(
        "-s",
        "--streams",
        default=None,
        dest="streams",
        nargs="*",
        help="Do verif for this streams. Default: Infer from .zarr file", # ex. ERA5, CERRA
    )

    parser.add_argument(
        "-m",
        "--method",
        default="2d",
        dest="method",
        choices=["2d", "lat_lon", "nearest", "pyresample_nearest", "pyresample_bilinear", "pyresample_gauss"],
        help="Interpolation method. Default: 2d_interpolation",
    )

    args = parser.parse_args()

    return args

def create_output_paths(stream, variable, outfiles, method):
    """
    Create output directories for the verif files
    and return path to output file
    Args:
        stream (string)
        variables (list[string])
        outfiles (string): template for the output files
    Outputs:
        None
    """
    outfile = Path(outfiles.replace("%S", stream).replace("%V", variable).replace("%M", method))
    pathdir = outfile.parent
    print(f"Output directory: {pathdir}")
    pathdir.mkdir(exist_ok=True, parents=True)
    return outfile

def generate_time_coordinates(zarrio, stream):
    """
    Read samples and steps from ZarrIO object
    and convert to xarray data objects
    to be used as coordinates in verrif dataset
    """

    # Initial times are stored as numpy.datetime64 objects in verif
    # Get the valid time of the first step for each sample
    verif_times = [np.datetime64("nat","h")]*len(zarrio.samples)
    for sample in zarrio.samples:
        item = zarrio.get_data(sample=sample, stream=stream, forecast_step=1)
        if "source_interval_start" in item.prediction.as_xarray().coords and "source_interval_end" in item.prediction.as_xarray().coords:
            source_interval_start = item.prediction.as_xarray().source_interval_start.values[0]
            source_interval_end = item.prediction.as_xarray().source_interval_end.values[0]
            verif_times[int(sample)] = source_interval_start
            dt = source_interval_end - source_interval_start
        else:
            item = zarrio.get_data(sample=sample, stream=stream, forecast_step=1)
            onetime = item.prediction.as_xarray().valid_time.values[0]
            item = zarrio.get_data(sample=sample, stream=stream, forecast_step=2)
            twotime = item.prediction.as_xarray().valid_time.values[0]
            item = zarrio.get_data(sample=sample, stream=stream, forecast_step="1")
            verif_times[int(sample)] = (item.prediction.as_xarray().valid_time.values[0] - dt)
            dt = (twotime - onetime)

    xrtime = xr.DataArray(
        verif_times,
        name = "time",
        dims = ["time"],
        coords = {"time":verif_times},
        attrs = {"standard_name":"forecast_reference_time"})

    
    dt = dt.astype("timedelta64[h]")

    # Lead times are stored as float32 in verif
    # Assume all time steps are the same,
    # so loop over steps and multiply the time step size by index
    leadtimes = np.ndarray(len(zarrio.forecast_steps), dtype=np.float32)
    for i in range(len(zarrio.forecast_steps)):
        leadtimes[i] = (i+1)*dt

    xrleadtime = xr.DataArray(
        leadtimes,
        name = "leadtime",
        dims = ["leadtime"],
        coords = {"leadtime":leadtimes},
        attrs = {"units":"hour"})

    return xrtime, xrleadtime

def get_streams(zarrio, arg_streams):
    """
    Determine the stream,
    either by getting streams from argument and check if they are in the zarr file
    or just use all the streams in zarrio
    Args:
        zarrio: ZarrIO object
        arg_streams: (list[string])
    Outputs:
        streams: (list[string])
    """
    if arg_streams:
        for stream in arg_streams:
            if stream not in zarrio.streams:
                raise Exception(
                    f"Stream {stream} is not present in .zarr file. zarrio.streams: {zarrio.streams}"
                )
        return arg_streams
    else:
        return zarrio.streams

def get_variables(xdata: xr.DataArray, config_file: Path, arg_variables: list, stream: str) -> list:
    """
    Go through argument variables,
    check if they are in the config_file and return
    a list ov variables.
    If no arguments are given,
    return list of variables found in file.
    """

    config_variables = Variables(config_file)

    config_names = (cv.name for cv in config_variables)

    variables = []
    if arg_variables:

        # Check if there's a config for requested variables
        for av in arg_variables:
            if not av in config_names:
                raise Exception(
                    f'Variable {av} does not have an entry in the config file'
                )

        # Add requested variables to list of variables
        for cv in config_variables:
            if cv.name in arg_variables:
                variables += [cv]

    else:
        variables = [v for v in config_variables]

    #Check what variables exist in zarr file
    vvars = []
    for v in variables:
        if isinstance(v.zarr_name, str):
            if v.zarr_name in xdata.channel.values:
                vvars += [v]
        else:
            if (len(set(v.zarr_name).intersection(xdata.channel.values)) == len(v.zarr_name)):
                vvars += [v]
    variables = vvars

    if not variables:
        raise Exception("No variables with configuration found in zarr file.")

    return variables

def get_obs_coordinates(obs):
    """
    Extract latitude, longitude and altitude
    from observation dataset
    Args:
        obs: Dataset
    Outputs:
        lat: DataArray
        lon: DataArray
        alt: DataArray
    """

    lat = obs.latitude.astype("float32")
    lat.name = "lat"

    lon = obs.longitude.astype("float32")
    lon.name = "lon"

    alt = obs.altitude.astype("float32")

    return lat, lon, alt


def get_point_map(xdata, old_coords):
    """
    Map point to point using lat-lon
    """

    new_coords = np.column_stack((xdata.ipoint.lat.values, xdata.ipoint.lon.values))

    pointmap = np.ndarray(old_coords.shape[0], dtype=np.int32)

    oldlist = list(map(tuple, old_coords.tolist()))
    newlist = list(map(tuple, new_coords.tolist()))

    n = old_coords.shape[0]

    for i in range(len(pointmap)):

        ni = max(0, i - 9)
        nf = min(n, i + 10)

        try:
            pointmap[i] = ni + newlist[ni:nf].index(oldlist[i])
        except:
            print("             i: ", i)
            print("            ni: ", ni)
            print("            nf: ", nf)
            print("    oldlist[i]: ", oldlist[i])
            print("newlist[ni:nf]: ", newlist[ni:nf])
            print()
            exit()

    return pointmap


def plot_norway_points(grid_coords, obs_coords, nearest_grid_coords=None, interpolated_values=None):
    """
    Plot grid and observation points on a map centered on Norway.
    If nearest_grid_coords is provided, plot those in green and draw lines from each obs to its nearest grid point.
    If interpolated_values is provided, color nearest grid points orange if their value is NaN.
    """
    fig = plt.figure(figsize=(10, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([4, 32, 57, 72], crs=ccrs.PlateCarree())  # Norway bounding box

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')

    ax.scatter(grid_coords[:, 1], grid_coords[:, 0], c='blue', s=10, label='Grid Points', alpha=0.5)
    ax.scatter(obs_coords[:, 1], obs_coords[:, 0], c='red', s=20, label='Observation Points', alpha=0.7)
    if nearest_grid_coords is not None:
        # Default: all nearest grid points green
        colors = ['green'] * nearest_grid_coords.shape[0]
        if interpolated_values is not None:
            # Color orange if interpolated value is NaN
            colors = ['orange' if np.isnan(val) else 'green' for val in interpolated_values]
        ax.scatter(nearest_grid_coords[:, 1], nearest_grid_coords[:, 0], c=colors, s=30, label='Nearest Grid Points', alpha=0.7)
        # Draw lines from each observation to its nearest grid point
        for obs, nearest in zip(obs_coords, nearest_grid_coords):
            ax.plot([obs[1], nearest[1]], [obs[0], nearest[0]], c='gray', linewidth=0.8, alpha=0.6)

    plt.legend()
    plt.title('Grid, Observation, and Nearest Points - Norway')
    plt.show()


def main():

    print("Start creating verif files")

    args = readarg()

    print("zarrfile:", args.zarrfile)
    print("obsfile:", args.obsfile)
    print("outputfile template:", args.outfiles)


    obs = xr.open_dataset(args.obsfile)
    lat, lon, alt = get_obs_coordinates(obs)
    obs_coords = np.column_stack((lat.values, lon.values))

    print()
    print(obs)
    print()

    config_file = Path(__file__).parent/"verif_config.yaml"

    method_factory = Interpolator_factory(args.method)

    with ZarrIO(args.zarrfile) as zarrio:

        streams = get_streams(zarrio, args.streams)

        t_start = time()

        for stream in streams:
            # Load a sample to get grid coordinates
            sample = zarrio.samples[0]
            xdata = zarrio.get_data(sample=sample, stream=stream, forecast_step=1).prediction.as_xarray()
            zarr_coords = np.column_stack((xdata.ipoint.lat.values, xdata.ipoint.lon.values))

            interpolator = method_factory.get_interpolator(zarr_coords, obs_coords)
            interpolator.prepare()

            # --- Write coordinates to Diana files ---
            diana_io(Path(f"output/verif/{stream}_grid_coords.diana")).write(zarr_coords)
            diana_io(Path(f"output/verif/{stream}_obs_coords.diana")).write(obs_coords)
            nearest_grid_coords = None
            if args.method == "nearest" or args.method == "pyresample_nearest":
                nearest_indices = interpolator.indices.flatten()
                nearest_grid_coords = zarr_coords[nearest_indices]
                diana_io(Path(f"output/verif/{stream}_nearest_grid_coords.diana")).write(nearest_grid_coords)
            # ------------------------------------------

            # --- Plot points on map for visual check (only for nearest methods) ---
            if nearest_grid_coords is not None:
                plot_norway_points(zarr_coords, obs_coords, nearest_grid_coords)
            # ----------------------------------------------------------------------

            data_shape = (len(zarrio.samples), len(zarrio.forecast_steps), obs.location.shape[0])

            processers = Processer_factory(zarrio, obs, stream, interpolator)

            for v in variables:

                vt_start = time()

                print("variable: ", v.name)

                fcstdata = np.ndarray(data_shape, dtype=np.float32)
                obsdata  = np.ndarray(data_shape, dtype=np.float32)

                p = processers.get_processer(v.name)

                p.get_data(v, fcstdata, obsdata)


                xrobsdata = xr.DataArray(obsdata,
                                         dims=["time", "leadtime", "location"],
                                         coords={"time": xrtime, "leadtime": xrleadtime,"location": obs.location},
                                         name="obs",
                                         attrs=v.attributes)

                xrfcstdata = xr.DataArray(fcstdata,
                                          dims=["time", "leadtime", "location"],
                                          coords={"time": xrtime, "leadtime": xrleadtime,"location": obs.location},
                                          name="fcst",
                                          attrs=v.attributes)


                merged = xr.merge([xrfcstdata,
                                   xrobsdata,
                                   lat,
                                   lon,
                                   alt])

                outfile = create_output_paths(stream, v.name, args.outfiles, args.method)

                merged.to_netcdf(outfile, encoding={"time": {"units": "seconds since 1970-01-01 00:00:00"}})

                vt_end = time()

                print(v.name, "time: ", vt_end - vt_start)
                print()

        t_end = time()

        print()
        print("all the time: ", t_end - t_start)
        print()

        print()
        print("merged")
        print(merged)
        print()


