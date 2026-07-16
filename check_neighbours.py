# check_neighbours.py
from astropy_healpix.healpy import pix2ang
import astropy_healpix as hp
import numpy as np

nside, c = 32, 1500
nb = hp.neighbours(np.array([c]), nside, order="nested").flatten()
th, ph = pix2ang(nside, np.array([c]), nest=True)
th_n, ph_n = pix2ang(nside, nb[nb >= 0], nest=True)
d = np.degrees(np.arccos(np.sin(np.pi/2-th)*np.sin(np.pi/2-th_n) +
    np.cos(np.pi/2-th)*np.cos(np.pi/2-th_n)*np.cos(ph-ph_n)))
print("neighbour distances (deg):", np.round(d, 2), " cell size ~", round(58.6/nside, 2))
