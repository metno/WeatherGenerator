from weathergen.common.io import ZarrIO

import xarray as xr
import numpy as np
import pandas as pd
import time

from pathlib import Path

from scipy.spatial import KDTree, Delaunay
from scipy.interpolate import LinearNDInterpolator

def convert_coordinates(coords):
    #Convert lat-lon coordinates to cartesian coordinates in a unit box

    xyz_coords = np.ndarray((coords.shape[0],3))

    xyz_coords[:,0] = np.cos(np.pi*coords[:,0]/180.0)*np.cos(np.pi*coords[:,1]/180.0)
    xyz_coords[:,1] = np.cos(np.pi*coords[:,0]/180.0)*np.sin(np.pi*coords[:,1]/180.0)
    xyz_coords[:,2] = np.sin(np.pi*coords[:,0]/180.0)

    return xyz_coords


def compute_weights(obs_points, model_points, indices):

    weights = np.ndarray((obs_points.shape[0],3))

    for i, (obs, indix) in enumerate(zip(obs_points, indices)):
        
        vertex1 = model_points[indix[1]] - model_points[indix[0]]
        vertex2 = model_points[indix[2]] - model_points[indix[0]]
        vertexp = obs - model_points[indix[0]]

        cross1 = np.cross(vertex1, vertex2)
        area1  = np.linalg.norm(cross1)

        cross2 = np.cross(vertex1, vertexp)
        area2  = np.linalg.norm(cross2)

        cross3 = np.cross(vertex2, vertexp)
        area3  = np.linalg.norm(cross3)

        vertex1 = model_points[indix[2]] - model_points[indix[1]]
        vertexp = obs - model_points[indix[1]]

        cross4 = np.cross(vertex1, vertexp)
        area4  = np.linalg.norm(cross4)

        weights[i,0] = area4/(area2+area3+area4)
        weights[i,1] = area3/(area2+area3+area4)
        weights[i,2] = area2/(area2+area3+area4)
    
    return weights



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

    obs = xr.open_dataset(obspath)


    xdata = dataset.as_xarray()

    print()
    print('zarr stuff')
    print()
    print()
    print('indexes')
    print(xdata.indexes)
    print()
    print('coords')
    print(xdata.coords)
    print()
    print('channel')
    print(xdata.channel)
    print()
    print('2t')
    print(xdata.sel(channel='2t'))
    print(xdata.sel(channel='2t')[0,0,0,:,0].values)
    print()
    print()

    zarr_temps = xdata.sel(channel='2t')[0,0,0,:,0].values
    obs_temps = obs.sel(time=xdata.valid_time.values[0]).air_temperature.values

    zarr_lat_lon = xdata.ipoint
    obs_lat_lon = obs.sel(time=xdata.valid_time.values[0])[['latitude','longitude']]
    print('values:      ', obs_lat_lon.values)
    print('18700 lat:   ', obs_lat_lon.sel(location=18700).latitude.values)
    print('18700 long:  ', obs_lat_lon.sel(location=18700).longitude.values)
    print()

    # interpolate prediction to observations

    print()
    print()
    print()

    zarr_coords = np.column_stack((zarr_lat_lon.ipoint.lat.values, zarr_lat_lon.ipoint.lon.values))

    convert_start = time.time()
    zarr_xyz = convert_coordinates(zarr_coords)
    convert_end = time.time()

    obs_coords = np.column_stack((obs_lat_lon.latitude.values, obs_lat_lon.longitude.values))
    obs_xyz = convert_coordinates(obs_coords)

    
    print('zarr shape: ', zarr_xyz.shape)
    print('obs shape:  ', obs_xyz.shape)
    print()

    tree_start = time.time()
    tree = KDTree(zarr_xyz)
    tree_end = time.time()

    query_start = time.time()
    d, indices = tree.query(obs_xyz, k = 3)
    query_end = time.time()

    weights_start = time.time()
    weights = compute_weights(obs_xyz, zarr_xyz, indices)
    weights_end = time.time()

    weigh_start = time.time()

    weigh_temps = np.ndarray((obs_temps.shape))
    for i, indix in enumerate(indices):

        if np.isnan(obs_temps[i]):
            weigh_temps[i] = np.nan
        else:        
            weigh_temps[i] = weights[i,0]*zarr_temps[indix[0]] \
                           + weights[i,1]*zarr_temps[indix[1]] \
                           + weights[i,2]*zarr_temps[indix[2]]

    weigh_end = time.time()

    tri_start = time.time()
    triangulation = Delaunay(zarr_xyz)
    tri_end = time.time()

    setup_start = time.time()
    interpolator = LinearNDInterpolator(triangulation, zarr_temps)
    setup_end = time.time()

    inter_start = time.time()
    inter_temps = interpolator(obs_xyz)
    inter_end = time.time()

    print()
    print('obs_temps:         ', obs_temps[0])
    print('obs_temps:         ', obs_temps[1])
    print('obs_temps:         ', obs_temps[2])
    print('weigh_temps:       ', weigh_temps[0])
    print('weigh_temps:       ', weigh_temps[1])
    print('weigh_temps:       ', weigh_temps[2])
    print()
    print('zarr_temps 0,0:    ', zarr_temps[indices[0,0]])
    print('zarr_temps 0,1:    ', zarr_temps[indices[0,1]])
    print('zarr_temps 0,2:    ', zarr_temps[indices[0,2]])
    print()
    print('zarr_temps 1,0:    ', zarr_temps[indices[1,0]])
    print('zarr_temps 1,1:    ', zarr_temps[indices[1,1]])
    print('zarr_temps 1,2:    ', zarr_temps[indices[1,2]])
    print()
    print('zarr_temps 2,0:    ', zarr_temps[indices[2,0]])
    print('zarr_temps 2,1:    ', zarr_temps[indices[2,1]])
    print('zarr_temps 2,2:    ', zarr_temps[indices[2,2]])
    print()

    print()
    print('obs:               ', obs_coords[0,0],obs_coords[0,1])
    print('zarr 0:            ', zarr_coords[indices[0,0],0], zarr_coords[indices[0,0],1])
    print('zarr 1:            ', zarr_coords[indices[0,1],0], zarr_coords[indices[0,1],1])
    print('zarr 2:            ', zarr_coords[indices[0,2],0], zarr_coords[indices[0,2],1])
    print('distances:         ', d[0,:])
    print('weights:           ', weights[0,:])
    print()

    print('convert time:       ', convert_end - convert_start)
    print('tree time:          ', tree_end - tree_start)
    print('query time:         ', query_end - query_start)
    print('weights time:       ', weights_end - weights_start)
    print('weigh time:         ', weigh_end - weigh_start)
    print('triangulation time: ', tri_end - tri_start)
    print('setup time:         ', setup_end - setup_start)
    print('interpolation time: ', inter_end - inter_start)


#    print('inter_temps 1:     ', inter_temps[0])
#    print('inter_temps 2:     ', inter_temps[1])
#    print('inter_temps 3:     ', inter_temps[2])


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

