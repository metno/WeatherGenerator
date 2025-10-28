from weathergen.common.io import ZarrIO

"""This script extracts predictions from a WG intermediate zarr file and writes to NetCDF"""

path = "/home/thomasn/validation_epoch00000_rank0000.zarr"
ofilename = "test.nc"

with ZarrIO(path) as io:
    print(f"Samples: {io.samples}")
    print(f"Streams: {io.streams}")
    print(f"Forecast steps: {io.forecast_steps}")

    # Get targets and predictions
    item = io.get_data(sample="0", stream="ERA5", forecast_step="1")
    print(item.key)

    # Get prediction dataset
    dataset = item.prediction
    print(f"Channels: {dataset.channels}")
    dataset.as_xarray().to_netcdf(ofilename)

    # interpolate prediction to observations

    # Write obs and forecast to verif file
