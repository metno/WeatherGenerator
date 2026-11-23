from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.multi_stream_data_sampler import (
    MultiStreamDataSampler,
    _initialization_stride,
)


def sampler_for_year(year, *, stride=5, steps=3, offset=1, ranks=1, samples=400):
    sampler = object.__new__(MultiStreamDataSampler)
    sampler.time_window_handler = TimeWindowHandler(
        np.datetime64(f"{year}-01-01T00:00"),
        np.datetime64(f"{year + 1}-01-01T00:00"),
        np.timedelta64(6, "h"),
        np.timedelta64(6, "h"),
    )
    sampler.index_range = sampler.time_window_handler.get_index_range()
    sampler.max_input_steps = 1
    sampler.time_step = np.timedelta64(6, "h")
    sampler.step_timedelta = np.timedelta64(6, "h")
    sampler.len_timedelta = np.timedelta64(6, "h")
    sampler.output_offset = offset
    sampler.initialization_stride = stride
    sampler.samples_per_mini_epoch = samples
    sampler.batch_size = 1
    sampler.world_size = ranks
    sampler.repeat_data = False
    sampler.check_samples(steps)
    return sampler


@pytest.mark.parametrize("year,steps,offset", [(2023, 0, 0), (2024, 3, 1)])
def test_strided_initializations_preserve_forecast_windows(year, steps, offset):
    sampler = sampler_for_year(year, steps=steps, offset=offset)
    indices = sampler._calc_baseperms(steps)
    times = np.array([sampler.time_window_handler.window(i).start for i in indices])
    assert len(indices) == sampler.len == 292
    assert np.all(np.diff(times) == np.timedelta64(30, "h"))
    assert set(times.astype("datetime64[h]").astype(int) % 24) == {0, 6, 12, 18}
    for index in indices:
        for lead in range(offset, max(1, offset + steps)):
            actual = sampler.time_window_handler.window(index + lead).start
            expected = sampler.time_window_handler.window(index).start + lead * np.timedelta64(
                6, "h"
            )
            assert actual == expected
            assert actual < np.datetime64(f"{year + 1}-01-01")


def test_rank_partition_covers_unique_indices_and_reports_truncation(caplog):
    sampler = sampler_for_year(2024, ranks=3)
    assert sampler.len == 97
    assert sampler.samples_per_mini_epoch == 291
    assert "Dropping 1 initializations" in caplog.text
    indices = sampler._calc_baseperms(3)
    chunks = [indices[rank * sampler.len : (rank + 1) * sampler.len] for rank in range(3)]
    assert len(set(np.concatenate(chunks))) == 291


def test_sample_limit_and_legacy_indexing():
    sampler = sampler_for_year(2024, samples=10)
    assert sampler.len == 10
    sampler.initialization_stride = 1
    assert np.array_equal(sampler._calc_baseperms(3)[:6], np.arange(1, 7))


@pytest.mark.parametrize("stride", [0, -1, True, 2.5, "5"])
def test_invalid_stride(stride):
    with pytest.raises(ValueError, match="positive integer"):
        _initialization_stride(OmegaConf.create({"initialization_stride": stride}), 1, False)


@pytest.mark.parametrize(
    "shuffle,batch_size,repeat", [(True, 1, False), (False, 2, False), (False, 1, True)]
)
def test_stride_rejects_incompatible_sampling(shuffle, batch_size, repeat):
    cfg = OmegaConf.create({"initialization_stride": 5, "shuffle": shuffle})
    with pytest.raises(ValueError, match="requires shuffle=False"):
        _initialization_stride(cfg, batch_size, repeat)


def test_strided_invalid_initialization_is_skipped_not_replaced(caplog):
    sampler = object.__new__(MultiStreamDataSampler)
    sampler.initialization_stride = 5
    sampler.batch_size = 1
    sampler.len = 3
    sampler.mode_cfg = {"training_mode": ["masking"]}
    sampler.worker_workset = lambda: (0, 3)
    sampler.reset = lambda: (np.array([1, 6, 11]), np.array([0, 0, 0]))

    def get_batch(idx, _steps):
        return SimpleNamespace(
            idx=idx,
            sources_empty=lambda: idx == 6,
            is_nan=lambda: False,
            targets_empty=lambda: False,
        )

    sampler._get_batch = get_batch
    sampler._fft_grid_issue = lambda _: None
    assert [b.idx for b in sampler] == [1, 11]
    assert "Skipping strided initialization 6" in caplog.text
