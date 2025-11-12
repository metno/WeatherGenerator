import os
import argparse
# from datetime import datetime, timedelta, timezone


def readarg():
    parser = argparse.ArgumentParser(description='Create verif files from a zarr file and observation file')
    parser.add_argument('-z', '--zarr', help='Zarr file (.zarr)', dest="zarrfile", required=False,\
                         default="/lustre/storeB/project/nwp/weathergen/experiments/era5_o96/validation_epoch00000_rank0000.zarr")
    parser.add_argument('-b', '--obs', help='Observation file (.nc)', dest="obsfile", required=False,\
                         default= '/lustre/storeB/project/nwp/weathergen/datasets/metno_observations_v3.nc')
    parser.add_argument('-o', '--output', help='Template for the output nc filenames,\
                         default will be to create output/verif/%S/%V repertories where\
                         %S, %V, %d are replaced by the stream, variable and date',\
                         dest="outfiles", default="output/verif/%S/%V/verif_file_%d.nc", required=False)
    parser.add_argument('-d', '--date', help='from to date in format %Y%m%d%H:%Y%m%d%H or %Y%m%d:%Y%m%d,'\
                        ' excluding the second date for instance 2024010100:2024020200',
                        type=str, dest='datefromto', required=True)
    # could have a default date but not sure what makes sense
    # d = datetime.now(timezone.utc).strftime("%Y%m%d%H") + ":" + (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y%m%d%H")
    parser.add_argument('-v', '--variables', default=['rr1', 'ta'], help='Do verif for these variables. Default: rr1, ta', dest="variables", nargs='*')
    parser.add_argument('-s', '--streams', default=['ERA5'], help='Do verif for these streams. Default: ERA5', dest="streams", nargs='*')
    parser.add_argument('-m', '--method', default='rolf_method', help='Interpolation method. Default: rolf_method', dest="method")

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

    return(args.zarrfile, args.obsfile, args.outfiles, streams, variables, date_start, date_end, args.method)

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
            pathdir = os.path.split(outfiles.replace('%S', stream).replace('%V', variable))[0]
            print(f"If not existing create repertory {pathdir}")
            os.makedirs(pathdir, exist_ok=True)

def main():
    print("Start creating verif files")
    zarrfile, obsfile, outfiles, streams, variables, date_start, date_end, method = readarg()
    print("zarrfile:", zarrfile)
    print("obsfile:", obsfile)
    print("start date", date_start)
    print("end date", date_end)


    # create verif directories
    create_all_output_dir(streams, variables, outfiles)

    print("outputfile template:", outfiles)

