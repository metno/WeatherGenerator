import argparse
import numpy as np
import xarray as xr

from time import time

from pathlib import Path

from weathergen.common.io import ZarrIO

from weathergen.verif.diana_io import diana_io

from weathergen.verif.verif_interpolator import verif_2D_interpolator
from weathergen.verif.verif_interpolator import verif_lat_lon_interpolator
from weathergen.verif.verif_interpolator import verif_nearest_interpolator

# from datetime import datetime, timedelta, timezone


def readarg():
    parser = argparse.ArgumentParser(
        description="Create verif files from a zarr file and observation file"
    )

    parser.add_argument(
        "-z",
        "--zarr",
        dest="zarrfile",
        required=False,
        default="data/validation_epoch00000_rank0000.zarr",
        help="Zarr file (.zarr)",
    )

    parser.add_argument(
        "-b",
        "--obs",
        dest="obsfile",
        required=False,
        default="data/metno_observations_v3.nc",
        help="Observation file (.nc)",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="outfiles",
        default="output/verif/%S/%V/verif_file_%d.nc",
        required=False,
        help="Template for the output nc filenames, default will be to create output/verif/%S/%V repertories where \
              %S, %V, %d are replaced by the stream, variable and date",
    )

    parser.add_argument(
        "-d",
        "--date",
        type=str,
        dest="datefromto",
        required=True,
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
        "--stream",
        default="ERA5",
        dest="stream",
        nargs="*",
        help="Do verif for this stream. Default: ERA5",
    )

    parser.add_argument(
        "-m",
        "--method",
        default="2d_interpolation",
        dest="method",
        choices=["2d", "lat_lon", "nearest"],
        help="Interpolation method. Default: 2d_interpolation",
    )

    args = parser.parse_args()

    # create output directories
    date_start, date_end = args.datefromto.split(":")
    print("start date", date_start)
    print("end date", date_end)
    if len(date_start) == 8:
        date_start = date_start + "00"
    if len(date_start) != 10:
        raise ValueError(
            f"date not in the right format expect date1:date2 as\
                          %Y%m%d%H:%Y%m%d%H or %Y%m%d:%Y%m%d, got date1 as {date_start}"
        )
    if len(date_end) == 8:
        date_end = date_end + "00"
    if len(date_end) != 10:
        raise ValueError(
            f"date not in the right format expect date1:date2 as\
                          %Y%m%d%H:%Y%m%d%H% or %Y%m%d:%Y%m%d, got date2 as {date_end}"
        )

    args.outfiles = args.outfiles.replace("%d", date_start + "_" + date_end)

    return args


def create_all_output_dir(stream, variables, outfiles):
    """Create output directories for the verif files
    Args:
        stream (list[string])
        variables (list[string])
        outfiles (string): template for the output files
    Outputs:
        None
    """
    for variable in variables:
        pathdir = Path(outfiles.replace("%S", stream).replace("%V", variable)).parent
        print(f"If not existing create directory {pathdir}")
        pathdir.mkdir(exist_ok=True, parents=True)


def generate_time_coordinates(zarrio):
    """
    Read samples and steps from ZarrIO object
    and convert to xarray data objects
    to be used as coordinates in verrif dataset
    """

    
    item = zarrio.get_data(sample=0, stream="ERA5", forecast_step=1)
    onetime = item.prediction.as_xarray().valid_time.values[0]
    item = zarrio.get_data(sample=0, stream="ERA5", forecast_step=2)
    twotime = item.prediction.as_xarray().valid_time.values[0]

    dt = (twotime - onetime)

    # Initial times are stored as numpy.datetime64 objects in verif
    # Get the valid time of the first step for each sample
    verif_times = [np.datetime64('nat','h')]*len(zarrio.samples)
    for sample in zarrio.samples:
        item = zarrio.get_data(sample=sample, stream="ERA5", forecast_step='1')
        verif_times[int(sample)] = (item.prediction.as_xarray().valid_time.values[0] - dt)

    xrtime = xr.DataArray(
        verif_times,
        name = "time",
        dims = ["time"],
        coords = {"time":verif_times},
        attrs = {"standard_name":"forecast_reference_time"})

    dt = dt.astype('timedelta64[h]')

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


def main():
    print("Start creating verif files")
    args = readarg()
    print("zarrfile:", args.zarrfile, args.zarrfile)
    print("obsfile:", args.obsfile)
    print("outputfile template:", args.outfiles)
    print("stream:", args.stream)

    # create verif directories
    create_all_output_dir(args.stream, args.variables, args.outfiles)

    with ZarrIO(args.zarrfile) as zarrio:

        print()
        print('zarrio.samples:        ', zarrio.samples, type(zarrio.samples))
        print('zarrio.streams:        ', zarrio.streams, type(zarrio.streams))
        print('zarrio.forecast_steps: ', zarrio.forecast_steps, type(zarrio.forecast_steps))
        print()

        gt_start = time()
        xrtime, xrleadtime = generate_time_coordinates(zarrio)
        gt_end = time()

        zc_start = time()
        item = zarrio.get_data(sample=0, stream=args.stream, forecast_step=1)
        xdata = item.prediction.as_xarray()
        zarr_coords = np.column_stack((xdata.ipoint.lat.values, xdata.ipoint.lon.values))
        zc_end = time()

        oc_start = time()
        obs = xr.open_dataset(args.obsfile)
        obs_lat_lon = obs.sel(time=xdata.valid_time.values[0])[["latitude", "longitude"]]
        obs_coords = np.column_stack((obs_lat_lon.latitude.values, obs_lat_lon.longitude.values))
        oc_end = time()

        obs_size = obs_coords.shape[0]

        if args.method == "2d":
            print()
            print("2D interpolation")

            setup_start = time()
            interpolator = verif_2D_interpolator(zarr_coords, obs_coords)
            setup_end = time()

        elif args.method == "lat_lon":
            print()
            print("lat-lon interpolation")

            setup_start = time()
            interpolator = verif_lat_lon_interpolator(zarr_coords, obs_coords)
            setup_end = time()

        elif args.method == "nearest":
            print()
            print("nearest neighbour interpolation")

            setup_start = time()
            interpolator = verif_nearest_interpolator(zarr_coords, obs_coords)
            setup_end = time()

        prep_start = time()
        interpolator.prepare()
        prep_end = time()

        lat_array = obs.latitude.astype('float32')
        lat_array.name = 'lat'
        lon_array = obs.latitude.astype('float32')
        lon_array.name = 'lon'
        alt_array = obs.altitude.astype('float32')

        vmap = {"2t":"air_temperature"}

        print()
        print(xrtime)
        print()

        inter_start = time()
        for v in args.variables:

            outfile = args.outfiles.replace("%S", args.stream).replace("%V", v)

            fcstdata = np.ndarray((len(zarrio.samples), len(zarrio.forecast_steps), obs_size), dtype=np.float32)
            obsdata = np.ndarray(fcstdata.shape, dtype=np.float32)

            for sample in range(len(zarrio.samples)):
                for step in range(len(zarrio.forecast_steps)):
                    item = zarrio.get_data(sample=sample, stream=args.stream, forecast_step=step+1)
                    xdata = item.prediction.as_xarray()
                    fcstdata[sample,step,:] = interpolator.interpolate(xdata.sel(sample=sample,
                                                                                 stream=args.stream,
                                                                                 forecast_step=step+1,
                                                                                 channel=v, 
                                                                                 ens=0).values)

                    obsdata[sample, step, :] = obs.data_vars["air_temperature"].sel(time=xdata.valid_time.values[0])


            temp_attrs = {
                "long_name":"2 meter temperature",
                "units":"K",
                "Conventions":"verif_1.0.0"
            }

            xrobsdata = xr.DataArray(obsdata, 
                                     dims=["time", "leadtime", "location"], 
                                     coords={"time": xrtime, "leadtime": xrleadtime,"location": obs.location}, 
                                     name="obs",
                                     attrs=temp_attrs)

            xrfcstdata = xr.DataArray(fcstdata, 
                                      dims=["time", "leadtime", "location"], 
                                      coords={"time": xrtime, "leadtime": xrleadtime,"location": obs.location}, 
                                      name="fcst",
                                      attrs=temp_attrs)


            merged = xr.merge([xrfcstdata,
                               xrobsdata, 
                               lat_array, 
                               lon_array, 
                               alt_array])

            print()
            print("outfile: ", outfile)
            print()
            merged.to_netcdf(outfile, encoding={'time': {'units': 'seconds since 1970-01-01 00:00:00'}})

        inter_end = time()

        print()
        print(fcstdata[0,0,0])
        print(obsdata[0,0,0])
        print(fcstdata[0,0,1])
        print(obsdata[0,0,1])
        print(fcstdata[0,0,2])
        print(obsdata[0,0,2])
        print(fcstdata[0,0,205])
        print(obsdata[0,0,205])
        print()

        print()
        print("   gt time: ", gt_end - gt_start)
        print("   zc time: ", zc_end - zc_start)
        print("   oc time: ", oc_end - oc_start)
        print("setup time: ", setup_end - setup_start)
        print(" prep time: ", prep_end - prep_start)
        print("inter time: ", inter_end - inter_start)
        print()

        verif = xr.open_dataset("data/MEPS_2.5km.nc")

        renamedict={'latitude':'lat', 'longitude':'lon'}

        print()
        print('merged')
        print(merged)
        print()

        print()
        print('verif')
        print(verif)
        print()



    Diana = diana_io(Path("wololo.txt"))
    Diana.write(obs_coords)

