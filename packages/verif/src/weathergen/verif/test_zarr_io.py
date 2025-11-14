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


def crosses(P, A, B, C, D, E):

    eps = 0.01

    AB = B - A
    AC = C - A
    BC = C - B
    AP = P - A
    BP = P - B

    area_tot = np.linalg.norm(np.cross(AB, AC))
    area_a = np.linalg.norm(np.cross(BC, BP))
    area_b = np.linalg.norm(np.cross(AC, AP))
    area_c = np.linalg.norm(np.cross(AB, AP))

    if (1 - area_tot/(area_a + area_b + area_c) < eps):
        return (area_a, area_b, area_c)

    AC = D - A
    BC = D - B

    area_tot = np.linalg.norm(np.cross(AB, AC))
    area_a = np.linalg.norm(np.cross(BC, BP))
    area_b = np.linalg.norm(np.cross(AC, AP))

    if (1 - area_tot/(area_a + area_b + area_c) < eps):
        return (area_a, area_b, area_c)

    AC = E - A
    BC = E - B

    area_tot = np.linalg.norm(np.cross(AB, AC))
    area_a = np.linalg.norm(np.cross(BC, BP))
    area_b = np.linalg.norm(np.cross(AC, AP))

    return (area_a, area_b, area_c)


def normalise(x):
    return x[:]/np.sum(x[:])


def compute_weights(obs_points, model_points, indices):

    weights = np.ndarray((obs_points.shape[0],3))

    eps = 0.01

    for i, (obs, indix) in enumerate(zip(obs_points, indices)):

        AB = model_points[indix[1]] - model_points[indix[0]]
        AC = model_points[indix[2]] - model_points[indix[0]]
        BC = model_points[indix[2]] - model_points[indix[1]]
        AP = obs                    - model_points[indix[0]]
        BP = obs                    - model_points[indix[1]]

        area_tot     = np.linalg.norm(np.cross(AB, AC))
        weights[i,0] = np.linalg.norm(np.cross(BC, BP))
        weights[i,1] = np.linalg.norm(np.cross(AC, AP))
        weights[i,2] = np.linalg.norm(np.cross(AB, AP))

        if (1 - area_tot/np.sum(weights[i,:]) < eps):
            continue

        indix[2] = indix[3]

        AC = model_points[indix[2]] - model_points[indix[0]]
        BC = model_points[indix[2]] - model_points[indix[1]]

        area_tot     = np.linalg.norm(np.cross(AB, AC))
        weights[i,0] = np.linalg.norm(np.cross(BC, BP))
        weights[i,1] = np.linalg.norm(np.cross(AC, AP))

        if (1 - area_tot/np.sum(weights[i,:]) < eps):
            continue

        indix[2] = indix[4]

        AC = model_points[indix[2]] - model_points[indix[0]]
        BC = model_points[indix[2]] - model_points[indix[1]]

        area_tot     = np.linalg.norm(np.cross(AB, AC))
        weights[i,0] = np.linalg.norm(np.cross(BC, BP))
        weights[i,1] = np.linalg.norm(np.cross(AC, AP))

    weights = normalise(weights)

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
    d, indices = tree.query(obs_xyz, k = 5)
    query_end = time.time()

    weights_start = time.time()
    weights = compute_weights(obs_xyz, zarr_xyz, indices)
    weights_end = time.time()

    weigh_start = time.time()
    weigh_temps = np.ndarray((obs_temps.shape))
    weigh_temps[:] = weights[:,0]*zarr_temps[indices[:,0]] + weights[:,1]*zarr_temps[indices[:,1]] + weights[:,2]*zarr_temps[indices[:,2]]
    weigh_end = time.time()

    tri_start = time.time()
    triangulation = Delaunay(zarr_coords)
    tri_end = time.time()

    setup_start = time.time()
    interpolator = LinearNDInterpolator(triangulation, zarr_temps)
    setup_end = time.time()

    inter_start = time.time()
    inter_temps = interpolator(obs_coords)
    inter_end = time.time()

    print()
    print('obs_temps:         ', obs_temps[0])
    print('obs_temps:         ', obs_temps[1])
    print('obs_temps:         ', obs_temps[2])
    print('obs_temps:         ', obs_temps[205])
    print()
    print('weigh_temps:       ', weigh_temps[0])
    print('weigh_temps:       ', weigh_temps[1])
    print('weigh_temps:       ', weigh_temps[2])
    print('weigh_temps:       ', weigh_temps[205])
    print()
    print('inter_temps:       ', inter_temps[0])
    print('inter_temps:       ', inter_temps[1])
    print('inter_temps:       ', inter_temps[2])
    print('inter_temps:       ', inter_temps[205])
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
    print('zarr_temps 205,0:  ', zarr_temps[indices[205,0]])
    print('zarr_temps 205,1:  ', zarr_temps[indices[205,1]])
    print('zarr_temps 205,2:  ', zarr_temps[indices[205,2]])
    print()

    print()
    print('obs  0:            ', obs_coords[0,0],obs_coords[0,1])
    print('obs  1:            ', obs_coords[1,0],obs_coords[1,1])
    print('obs  2:            ', obs_coords[2,0],obs_coords[2,1])
    print('obs  205:          ', obs_coords[205,0],obs_coords[205,1])
    print('weights:           ', weights[0,:])
    print('weights:           ', weights[1,:])
    print('weights:           ', weights[2,:])
    print('weights:           ', weights[205,:])
    print()

    print('convert time:       ', convert_end - convert_start)
    print('tree time:          ', tree_end - tree_start)
    print('query time:         ', query_end - query_start)
    print('weights time:       ', weights_end - weights_start)
    print('weigh time:         ', weigh_end - weigh_start)
    print('triangulation time: ', tri_end - tri_start)
    print('setup time:         ', setup_end - setup_start)
    print('interpolation time: ', inter_end - inter_start)



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
        dianio.write("   %3.5f  %3.5f\n"%(obs_coords[1,0], obs_coords[1,1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[1,0],0], zarr_coords[indices[1,0],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[1,1],0], zarr_coords[indices[1,1],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[1,2],0], zarr_coords[indices[1,2],1]))
        dianio.write("   %3.5f  %3.5f\n"%(obs_coords[2,0], obs_coords[2,1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[2,0],0], zarr_coords[indices[2,0],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[2,1],0], zarr_coords[indices[2,1],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[2,2],0], zarr_coords[indices[2,2],1]))
        dianio.write("   %3.5f  %3.5f\n"%(obs_coords[205,0], obs_coords[205,1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[205,0],0], zarr_coords[indices[205,0],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[205,1],0], zarr_coords[indices[205,1],1]))
        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[205,2],0], zarr_coords[indices[205,2],1]))


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

