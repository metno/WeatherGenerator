from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from weathergen.common.config import parse_timedelta
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.data_reader_obs import DataReaderObs


def make_reader(start_hour, end_hour, window_hours=6, step_hours=24):
    reader = object.__new__(DataReaderObs)
    reader.base_datetime = np.datetime64("2024-01-01T00:00")
    reader.time_window_handler = TimeWindowHandler(
        reader.base_datetime + np.timedelta64(start_hour, "h"),
        reader.base_datetime + np.timedelta64(end_hour, "h"),
        np.timedelta64(window_hours, "h"),
        np.timedelta64(step_hours, "h"),
    )
    reader.hrly_index = np.arange(49)
    reader.filename = Path("test-observations.zarr")
    reader.stream_info = {"name": "TEST_OBS"}
    reader.data = np.zeros((48, 3), dtype=np.float32)
    reader.dt = (reader.base_datetime + np.arange(48).astype("timedelta64[h]"))[:, None]
    reader.coords_idx = [0, 1]
    reader.geoinfo_idx = []
    return reader


@pytest.mark.parametrize("start_hour", [0, 24, 42, 43, 48, 49, 60])
@pytest.mark.parametrize("step_hours", [6, 24])
def test_windows_at_end_of_hourly_index(start_hour, step_hours, caplog):
    reader = make_reader(start_hour, start_hour + 96, step_hours=step_hours)
    reader._setup_sample_index()

    assert "TEST_OBS" in caplog.text
    assert "test-observations.zarr" in caplog.text
    assert "2024-01-03T00:00" in caplog.text
    num_windows = 96 // step_hours
    assert len(reader.indices_start) == len(reader.indices_end) == num_windows
    for window_idx in range(num_windows):
        data = reader._get(window_idx, [2])
        window = reader.time_window_handler.window(window_idx)
        assert np.all(data.datetimes >= window.start)
        assert np.all(data.datetimes < window.end)
        if start_hour + step_hours * window_idx >= 48:
            assert data.data.shape == (0, 1)
        else:
            assert len(data.data) > 0


def test_in_range_hourly_index_is_unchanged():
    reader = make_reader(0, 24, step_hours=6)
    reader._setup_sample_index()

    np.testing.assert_array_equal(reader.indices_start, [0, 6, 12, 18])
    np.testing.assert_array_equal(reader.indices_end, [5, 11, 17, 23])


def test_crps_inference_windows_resolve_distinct_forecast_leads():
    config_dir = Path(__file__).parents[1] / "config"
    config = OmegaConf.merge(
        OmegaConf.load(config_dir / "default_config.yml"),
        OmegaConf.load(config_dir / "config_noise_injection_crps_inference.yml"),
    )
    time_step = parse_timedelta(config.training_config.forecast.time_step)
    window_step = parse_timedelta(config.test_config.time_window_step)
    num_steps = config.test_config.forecast.num_steps
    target_indices = [step * time_step // window_step for step in range(1, num_steps + 1)]

    np.testing.assert_array_equal(target_indices, np.arange(1, num_steps + 1))
    assert config.test_config.output.num_samples >= config.test_config.samples_per_mini_epoch