
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
        self.data_shape = (len(zarrio.samples), len(zarrio.forecast_steps), obs.location.shape[0])

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
