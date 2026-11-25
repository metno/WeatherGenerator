"""Geographic centres of WeatherGenerator's nested HEALPix token cells."""

from functools import cache

import astropy.units as u
import astropy_healpix as hp
import numpy as np

NSIDE = 32
N_CELLS = 12 * NSIDE**2
LON_SHIFT = 180.0


@cache
def cell_centres():
    """Return token centre latitude and longitude in degrees.

    The tokenizer uses HEALPix longitudes shifted by 180 degrees relative to
    geographic longitudes; stored prediction coordinates can be mislabeled.
    """
    lon, lat = hp.healpix_to_lonlat(np.arange(N_CELLS), NSIDE, order="nested")
    return lat.to_value(u.deg), (lon.to_value(u.deg) - LON_SHIFT + 180.0) % 360.0 - 180.0
