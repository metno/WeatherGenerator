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

    parser = argparse.ArgumentParser(description='Create verif files from a zarr file and observation file')

    parser.add_argument(
        '-z',
        '--zarr',
        dest="zarrfile",
        required=False,
        default="/lustre/storeB/project/nwp/weathergen/experiments/era5_o96/validation_epoch00000_rank0000.zarr",
        help='Zarr file (.zarr)'
    )

    parser.add_argument(
        '-b',
        '--obs',
        dest="obsfile",
        required=False,
        default= '/lustre/storeB/project/nwp/weathergen/datasets/metno_observations_v3.nc',
        help='Observation file (.nc)',
    )

    parser.add_argument(
        '-o',
        '--output',
        dest="outfiles",
        default="output/verif/%S/%V/verif_file_%d.nc",
        required=False,
        help='Template for the output nc filenames, default will be to create output/verif/%S/%V repertories where \
              %S, %V, %d are replaced by the stream, variable and date'
    )

    parser.add_argument(
        '-d',
        '--date',
        type=str,
        dest='datefromto',
        required=True,
        help='From to date in format %Y%m%d%H:%Y%m%d%H or %Y%m%d:%Y%m%d, \
              excluding the second date for instance 2024010100:2024020200'
    )

    # could have a default date but not sure what makes sense
    # d = datetime.now(timezone.utc).strftime("%Y%m%d%H") + ":" + (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y%m%d%H")

    parser.add_argument(
        '-v',
        '--variables',
        default=['rr1', 'ta'],
        dest="variables",
        nargs='*',
        help='Do verif for these variables. Default: rr1, ta'
    )

    parser.add_argument(
        '-s',
        '--streams',
        default=['ERA5'],
        dest="streams",
        nargs='*',
        help='Do verif for these streams. Default: ERA5'
    )

    parser.add_argument(
        '-m',
        '--method',
        default='2d_interpolation',
        dest="method",
        choices=['2d', 'lat_lon', 'nearest'],
        help='Interpolation method. Default: 2d_interpolation'
    )

    args = parser.parse_args()

    # create output directories
    streams = args.streams
    variables = args.variables
    date_start, date_end = args.datefromto.split(":")
    print("start date", date_start)
    print("end date", date_end)
    if len(date_start) == 8:
        date_start = date_start + "00"
    if len(date_start) != 10:
        raise ValueError(f"date not in the right format expect date1:date2 as\
                          %Y%m%d%H:%Y%m%d%H or %Y%m%d:%Y%m%d, got date1 as {date_start}")
    if len(date_end) == 8:
        date_end = date_end + "00"
    if len(date_end) != 10:
        raise ValueError(f"date not in the right format expect date1:date2 as\
                          %Y%m%d%H:%Y%m%d%H% or %Y%m%d:%Y%m%d, got date2 as {date_end}")

    args.outfiles = args.outfiles.replace("%d", date_start + '_' + date_end)

    return(args)

def create_all_output_dir(streams, variables, outfiles):
    ''' Create output directories for the verif files
        Args:
            streams (list[string])
            variables (list[string])
            outfiles (string): template for the output files
        Outputs:
            None
    '''
    for stream in streams:
        for variable in variables:
            pathdir = Path(outfiles.replace('%S', stream).replace('%V', variable)).parent
            print(f"If not existing create directory {pathdir}")
            pathdir.mkdir(exist_ok=True, parents=True)

def main():
    print("Start creating verif files")
    args = readarg()
    print("zarrfile:", args.zarrfile)
    print("obsfile:", args.obsfile)


    # create verif directories
    create_all_output_dir(args.streams, args.variables, args.outfiles)

    with ZarrIO(args.zarrfile) as zarrio:

        item  = zarrio.get_data(sample="0", stream="ERA5", forecast_step="1")
        xdata = item.prediction.as_xarray()

        obs = xr.open_dataset(args.obsfile)

        zarr_coords = np.column_stack((xdata.ipoint.lat.values, xdata.ipoint.lon.values))

        obs_lat_lon = obs.sel(time=xdata.valid_time.values[0])[['latitude','longitude']]
        obs_coords = np.column_stack((obs_lat_lon.latitude.values, obs_lat_lon.longitude.values))


        zarr_temps = xdata.sel(channel='2t')[0,0,0,:,0].values
        obs_temps = obs.sel(time=xdata.valid_time.values[0]).air_temperature.values

        if (args.method == '2d'):

            print()
            print('2D interpolation')

            setup_start = time()
            interpolator = verif_2D_interpolator(zarr_coords, obs_coords)
            setup_end = time()

        elif (args.method == 'lat_lon'):

            print()
            print('lat-lon interpolation')

            setup_start = time()
            interpolator = verif_lat_lon_interpolator(zarr_coords, obs_coords)
            setup_end = time()

        elif (args.method == 'nearest'):

            print()
            print('nearest neighbour interpolation')

            setup_start = time()
            interpolator = verif_nearest_interpolator(zarr_coords, obs_coords)
            setup_end = time()

        prep_start = time()
        interpolator.prepare()
        prep_end = time()

        inter_start = time()
        t = interpolator.interpolate(zarr_temps)
        inter_end = time()

        print()
        print(t[0])
        print(t[1])
        print(t[2])
        print(t[205])

        print()
        print('setup time: ', setup_end - setup_start)
        print(' prep time: ', prep_end - prep_start)
        print('inter time: ', inter_end - inter_start)

    Diana = diana_io(Path('wololo.txt'))
    Diana.write(obs_coords)

    print("outputfile template:", args.outfiles)

