from weathergen.common.io import ZarrIO

import xarray as xr
import numpy as np
import pandas as pd

from pathlib import Path

from scipy.spatial import KDTree

def convert_coordinates(coords):
    #Convert lat-lon coordinates to cartesian coordinates in a unit box

    xyz_coords = np.ndarray((coords.shape[0],3))

    xyz_coords[:,0] = np.cos(np.pi*coords[:,0]/180.0)*np.cos(np.pi*coords[:,1]/180.0)
    xyz_coords[:,1] = np.cos(np.pi*coords[:,0]/180.0)*np.sin(np.pi*coords[:,1]/180.0)
    xyz_coords[:,2] = np.sin(np.pi*coords[:,0]/180.0)

    return xyz_coords

"""This script extracts predictions from a WG intermediate zarr file and writes to NetCDF"""

zarrpath = "/home/rolfhm/lustre/storeB/project/nwp/weathergen/experiments/era5_o96/validation_epoch00000_rank0000.zarr"
obspath  = "/home/rolfhm/lustre/storeB/project/nwp/weathergen/datasets/metno_observations_v3.nc"
ofilename = "test.nc"

trololo  = "trololo.nc"

diana_zarr = Path("diana_zarr.txt")
diana_obs = Path("diana_obs.txt")
diana_lalala = Path("lalala.txt")

with ZarrIO(zarrpath) as io:
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

    obs = xr.open_dataset(obspath, mask_and_scale=True)
#    obs = xr.open_dataset(obspath, mask_and_scale=False)


    xdata = dataset.as_xarray()

    print()
    print('zarr stuff')
    print()
    print('dims: ', xdata.dims)
    print()
    print('sizes: ', xdata.sizes)
    print()
    print('dtypes: ', xdata.dtype)
    print()
    print('coords: ', xdata.coords)
    print()
    print('attrs: ', xdata.attrs)
    print()
    print('encoding: ', xdata.encoding)
    print()
    print('indexes: ', xdata.indexes)
    print()
    print('xindexes: ', xdata.xindexes)
    print()
    print('chunks: ', xdata.chunks)
    print()
    print('chunksizes: ', xdata.chunksizes)
    print()
    print('nbytes: ', xdata.nbytes)
    print()
    print('valid_time 1: ', xdata.valid_time)
    print('valid_time 2: ', xdata.valid_time.values)
    print('valid_time 3: ', xdata.valid_time.values[0])
    print('valid_time 4: ', type(xdata.valid_time.values[0]))
    print('valid_time 5: ', xdata.sel(ipoint=4).valid_time)
    print('valid_time 6: ', xdata.sel(ipoint=4).valid_time.values)
    print('valid_time 7: ', type(xdata.sel(ipoint=4).valid_time.values))
    print()
    print()
    print()
    print()
    print('observation stuff')
    print()
    print('dims: ', obs.dims)
    print()
    print('sizes: ', obs.sizes)
    print()
    print('dtypes: ', obs.dtypes)
    print()
    print('data_vars: ', obs.data_vars)
    print()
    print('coords: ', obs.coords)
    print()
    print('attrs: ', obs.attrs)
    print()
    print('encoding: ', obs.encoding)
    print()
    print('indexes: ', obs.indexes)
    print()
    print('xindexes: ', obs.xindexes)
    print()
    print('chunks: ', obs.chunks)
    print()
    print('chunksizes: ', obs.chunksizes)
    print()
    print('nbytes: ', obs.nbytes)
    print()
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00'))
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').latitude)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').latitude.values)
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').longitude)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').longitude.values)
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').is_wmo_station)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').is_wmo_station.values)
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').air_temperature)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').air_temperature.values)
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').precipitation_amount_1h)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').precipitation_amount_1h.values)
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').relative_humidity)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').relative_humidity.values)
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').surface_air_pressure)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').surface_air_pressure.values)
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').wind_from_direction)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').wind_from_direction.values)
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').wind_speed)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').wind_speed.values)
    print()
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').wind_speed_of_gust)
    print(obs.sel(location=18700, time='2005-01-01 00:00:00').wind_speed_of_gust.values)
    print()
    print()


    print()
    print('types')
    print(type(obs.sel(time=xdata.valid_time.values[0])))
    print(type(xdata))
    print()
    print()
    print(xdata.indexes)
    print(xdata.coords)
    print()
    print()

    zarr_lat_lon = xdata.ipoint
    obs_lat_lon = obs.sel(time=xdata.valid_time.values[0])[['latitude','longitude']]
    print('values:      ', obs_lat_lon.values)
    print('18700 lat:   ', obs_lat_lon.sel(location=18700).latitude.values)
    print('18700 long:  ', obs_lat_lon.sel(location=18700).longitude.values)
    print()

    # interpolate prediction to observations

    zarr_coords = [zarr_lat_lon.ipoint.lat.values, zarr_lat_lon.ipoint.lon.values]
    zarr_coords = np.asarray(zarr_coords).T
    print('zarr shape: ', zarr_coords.shape)
    zarr_xyz = convert_coordinates(zarr_coords)

    tree = KDTree(zarr_xyz)

    obs_coords = [obs_lat_lon.latitude.values, obs_lat_lon.longitude.values]
    obs_coords = np.asarray(obs_coords).T
    print('obs shape: ', obs_coords.shape)
    obs_xyz = convert_coordinates(obs_coords)

    d, indices = tree.query(obs_xyz, k = 3)

    print()
    print('indices')
    print(indices)
    print()
    print('the d')
    print(d)
    print()
    print('obs  coords: ', obs_coords[0,0], obs_coords[0,1])
    print('zarr coords: ', zarr_coords[indices[0,0],0], zarr_coords[indices[0,0],1])
    print('zarr coords: ', zarr_coords[indices[0,1],0], zarr_coords[indices[0,1],1])
    print('zarr coords: ', zarr_coords[indices[0,2],0], zarr_coords[indices[0,2],1])
    print()
    print('obs  coords: ', obs_coords[9,0], obs_coords[9,1])
    print('zarr coords: ', zarr_coords[indices[9,0],0], zarr_coords[indices[9,0],1])
    print('zarr coords: ', zarr_coords[indices[9,1],0], zarr_coords[indices[9,1],1])
    print('zarr coords: ', zarr_coords[indices[9,2],0], zarr_coords[indices[9,2],1])
    print()
    print('obs  coords: ', obs_coords[284,0], obs_coords[284,1])
    print('zarr coords: ', zarr_coords[indices[284,0],0], zarr_coords[indices[284,0],1])
    print('zarr coords: ', zarr_coords[indices[284,1],0], zarr_coords[indices[284,1],1])
    print('zarr coords: ', zarr_coords[indices[284,2],0], zarr_coords[indices[284,2],1])
    print()


    with diana_lalala.open('w') as dianio:
        dianio.write('[NAME LALALA]\n')
        dianio.write('\n')
        dianio.write('[COLUMNS Lat:r Lon:r]\n')
        dianio.write('\n')
        dianio.write('[DATA]\n')
        dianio.write("   %3.5f  %3.5f\n"%(obs_coords[0,0], obs_coords[0,1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[0,0],0], zarr_coords[indices[0,0],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[0,1],0], zarr_coords[indices[0,1],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[0,2],0], zarr_coords[indices[0,2],1]))
        dianio.write("   %3.5f  %3.5f\n"%(obs_coords[9,0], obs_coords[9,1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[9,0],0], zarr_coords[indices[9,0],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[9,1],0], zarr_coords[indices[9,1],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[9,2],0], zarr_coords[indices[9,2],1]))
        dianio.write("   %3.5f  %3.5f\n"%(obs_coords[284,0], obs_coords[284,1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[284,0],0], zarr_coords[indices[284,0],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[284,1],0], zarr_coords[indices[284,1],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[284,2],0], zarr_coords[indices[284,2],1]))


    with diana_zarr.open('w') as dianio:
        dianio.write('[NAME ZARRCOORDS]\n')
        dianio.write('\n')
        dianio.write('[COLUMNS Lat:r Lon:r]\n')
        dianio.write('\n')
        dianio.write('[DATA]\n')

        for lat_lon in zarr_coords:
            dianio.write("   %3.5f  %3.5f\n"%(lat_lon[0], lat_lon[1]))

    with diana_obs.open('w') as dianio:
        dianio.write('[NAME OBSCOORDS]\n')
        dianio.write('\n')
        dianio.write('[COLUMNS Lat:r Lon:r]\n')
        dianio.write('\n')
        dianio.write('[DATA]\n')

        for lat_lon in obs_coords:
            dianio.write("   %3.5f  %3.5f\n"%(lat_lon[0], lat_lon[1]))


    # Write obs and forecast to verif file

