
import xarray as xr
import numpy as np

from weathergen.evaluate.score import Scores

from weathergen.common.io import ZarrIO
from weathergen.verif.verif_config import Variable
from weathergen.verif.verif_interpolator import Verif_interpolator

class Processer:

    def __init__(self, zarrio:ZarrIO, obs:xr.DataArray, stream: str, interpolator: Verif_interpolator):

        self.zarrio = zarrio
        self.obs = obs
        self.stream = stream
        self.interpolator = interpolator

        self.xdata = zarrio.get_data(sample=0, stream=stream, forecast_step=1).prediction.as_xarray()

    def get_data(self, v:Variable, fcstdata, obsdata):

        for sample in range(len(self.zarrio.samples)):
            for step in range(len(self.zarrio.forecast_steps)):

                item = self.zarrio.get_data(sample=sample, stream=self.stream, forecast_step=step+1)
                ydata = Scores.sort_by_coords(item.prediction.as_xarray(), self.xdata)

                obsdata[sample, step, :]  = self.get_obsdata(self.obs, v.obs_name, ydata.valid_time.values[0])

                fcstdata[sample, step, :] = self.get_fcstdata(ydata, v.zarr_name, sample, step+1)


    def get_obsdata(self, obs:xr.DataArray, name: str, time: np.datetime64):

        return obs.data_vars[name].sel(time=time)

    def get_fcstdata(self, ydata:xr.DataArray, name: str, sample: int,  step: int):

        return self.interpolator.interpolate(ydata.sel(sample=sample,
                                                       stream=self.stream,
                                                       forecast_step=step,
                                                       channel=name,
                                                       ens=0).values)

class MSLP_processer(Processer):

    def get_obsdata(self, obs:xr.DataArray, name: str, time: np.datetime64):
        return self.compute_mslp(obs, time)

    def compute_mslp(self, obs:xr.DataArray, time: np.datetime64):

        g   = 9.80665 # Gravitational acceleration (m/s**2)
        R   = 8.31447 # Universal gas constant (J/mol*K)

        a   = 0.0065  # Temperature lapse rate (K/m)
        Ch  = 0.0012  # (K/Pa)

        A   = 17.625
        B   = 243.03
        C   = 6.1094

        P   = obs.data_vars["surface_air_pressure"].sel(time=time)
        T   = obs.data_vars["air_temperature"].sel(time=time)
        rh  = obs.data_vars["relative_humidity"].sel(time=time)

        altitude = obs.altitude

        e = rh * 6.11 * np.power(10.0, ((7.5 * (T - 273.15))/(T - 38.85)))

        dewpoint = np.where(~np.isnan(e),
                            B * np.log(e/C)/(A - np.log(e/C)),
                            T - 276.15)

        e = np.where(np.isnan(e), 0, e)

        Tv = T / (1. - 0.379 * (6.11 * np.power(10.,((7.5 * dewpoint)/(237.7 + dewpoint))) / P))

#        mslp = np.where(altitude >= 50.,
#                        P * np.exp((g * altitude / R) / (T + 0.5 * a * altitude + e * Ch)),
#                        P + P * altitude / (29.27 * Tv))

        mslp = P + P * altitude / (29.27 * Tv)

        return mslp


class Wind_processer(Processer):

    def get_fcstdata(self, ydata:xr.DataArray, name: str, sample: int,  step: int):

        u = self.interpolator.interpolate(ydata.sel(sample=sample,
                                          stream=self.stream,
                                          forecast_step=step,
                                          channel="10u",
                                          ens=0).values)

        v = self.interpolator.interpolate(ydata.sel(sample=sample,
                                          stream=self.stream,
                                          forecast_step=step,
                                          channel="10v",
                                          ens=0).values)

        return np.sqrt(np.square(u) + np.square(v))


class Processer_factory():

    def __init__(self, zarrio:ZarrIO, obs:xr.DataArray, stream: str, interpolator: Verif_interpolator):

        self.zarrio = zarrio
        self.obs = obs
        self.stream = stream
        self.interpolator = interpolator

    def get_processer(self, name: str) -> Processer:

        if (name == "mslp"):
            return MSLP_processer(self.zarrio, self.obs, self.stream, self.interpolator)
        elif (name == "wind"):
            return Wind_processer(self.zarrio, self.obs, self.stream, self.interpolator)
        else:
            return Processer(self.zarrio, self.obs, self.stream, self.interpolator)


