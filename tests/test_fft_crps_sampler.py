from types import SimpleNamespace

import numpy as np
import torch

from weathergen.datasets.data_reader_anemoi import _get_anemoi_missing_dates
from weathergen.datasets.multi_stream_data_sampler import (
    MultiStreamDataSampler,
    _load_fft_crps_point_counts,
)


def test_fft_crps_point_counts_respect_stream_scope(tmp_path):
    template_path = tmp_path / "grid.npz"
    lat, lon = np.meshgrid(
        np.linspace(40.0, 44.0, 4),
        np.linspace(5.0, 9.0, 4),
        indexing="ij",
    )
    np.savez(template_path, lat=lat, lon=lon, ny=4, nx=4)
    mode_config = {
        "losses": {
            "physical": {
                "loss_fcts": {
                    "global_fft_crps": {"stream_names": ["MEPS"]},
                }
            }
        }
    }
    streams = {
        "MEPS": {"template_path": str(template_path)},
        "ERA55": {"template_path": "unused.npz"},
    }

    assert _load_fft_crps_point_counts(mode_config, streams) == {"MEPS": 16}


def test_anemoi_missing_indices_are_mapped_to_timestamps():
    dates = np.array(
        ["2020-03-23T00:00", "2020-03-23T06:00", "2020-03-23T12:00"],
        dtype="datetime64[m]",
    )
    dataset = SimpleNamespace(dates=dates, missing={1})

    assert _get_anemoi_missing_dates(dataset) == frozenset(
        {np.datetime64("2020-03-23T06:00", "ns")}
    )


def _sampler_and_batch(
    coords, times, *, tokenize_spacetime=False, missing_dates=frozenset()
):
    lat, lon = np.meshgrid(
        np.linspace(40.0, 44.0, 4),
        np.linspace(5.0, 9.0, 4),
        indexing="ij",
    )

    sampler = object.__new__(MultiStreamDataSampler)
    sampler.fft_crps_point_counts = {"MEPS": lat.size}
    sampler.streams_datasets = {
        "MEPS": SimpleNamespace(
            info={"tokenize_spacetime": tokenize_spacetime},
            readers=[SimpleNamespace(missing_dates=missing_dates)],
        )
    }

    stream_data = SimpleNamespace(
        sample_idx=37,
        target_coords_raw=[torch.as_tensor(coords, dtype=torch.float32)],
        target_times_raw=[np.asarray(times)],
        target_tokens=[torch.ones((len(coords), 1), dtype=torch.float32)],
    )
    sample = SimpleNamespace(streams_data={"MEPS": stream_data})
    batch = SimpleNamespace(target_samples=SimpleNamespace(samples=[sample], output_idxs=[0]))
    return sampler, batch


def test_fft_grid_issue_reports_timestamp_with_incomplete_grid():
    lat, lon = np.meshgrid(
        np.linspace(40.0, 44.0, 4),
        np.linspace(5.0, 9.0, 4),
        indexing="ij",
    )
    coords = np.stack((lat.ravel(), lon.ravel()), axis=1)[:-1]
    times = np.full(len(coords), np.datetime64("2021-03-01T06:00"))
    sampler, batch = _sampler_and_batch(coords, times)

    issue = sampler._fft_grid_issue(batch)

    assert "sample 37" in issue
    assert "2021-03-01T06:00" in issue
    assert "got 15 points, expected 16" in issue


def test_fft_grid_issue_skips_anemoi_reported_missing_timestamp():
    missing_date = np.datetime64("2021-03-01T06:00", "ns")
    sampler, batch = _sampler_and_batch(
        np.empty((0, 2)),
        np.array([], dtype="datetime64[ns]"),
        missing_dates={missing_date},
    )
    sampler.time_step = np.timedelta64(6, "h")
    sampler.step_timedelta = np.timedelta64(6, "h")
    sampler.time_window_handler = SimpleNamespace(
        window=lambda _idx: SimpleNamespace(
            start=missing_date, end=missing_date + np.timedelta64(6, "h")
        )
    )

    issue = sampler._fft_grid_issue(batch)

    assert issue is not None
    assert "Anemoi dataset marks target timestamp(s)" in issue
    assert "2021-03-01T06:00:00" in issue


def test_fft_grid_issue_rejects_fully_filtered_target():
    sampler, batch = _sampler_and_batch(np.empty((0, 2)), np.array([], dtype="datetime64[ns]"))

    issue = sampler._fft_grid_issue(batch)

    assert "got 0 points, expected 16" in issue


def test_fft_grid_issue_rejects_missing_target_window():
    sampler, batch = _sampler_and_batch(np.empty((0, 2)), np.array([], dtype="datetime64[ns]"))
    batch.target_samples.samples[0].streams_data["MEPS"].target_coords_raw[0] = []

    issue = sampler._fft_grid_issue(batch)

    assert "got 0 points, expected 16" in issue


def test_fft_grid_issue_checks_each_spacetime_substep():
    lat, lon = np.meshgrid(
        np.linspace(40.0, 44.0, 4),
        np.linspace(5.0, 9.0, 4),
        indexing="ij",
    )
    full_grid = np.stack((lat.ravel(), lon.ravel()), axis=1)
    coords = np.concatenate((full_grid, full_grid[:-1]))
    times = np.concatenate(
        (
            np.full(16, np.datetime64("2021-03-01T00:00")),
            np.full(15, np.datetime64("2021-03-01T06:00")),
        )
    )
    sampler, batch = _sampler_and_batch(coords, times, tokenize_spacetime=True)

    issue = sampler._fft_grid_issue(batch)

    assert "2021-03-01T06:00" in issue
    assert "got 15 points, expected 16" in issue


def test_fft_grid_issue_accepts_complete_point_count():
    lat, lon = np.meshgrid(
        np.linspace(40.0, 44.0, 4),
        np.linspace(5.0, 9.0, 4),
        indexing="ij",
    )
    coords = np.stack((lat.ravel(), lon.ravel()), axis=1)
    times = np.full(len(coords), np.datetime64("2021-03-01T06:00"))
    sampler, batch = _sampler_and_batch(coords, times)

    assert sampler._fft_grid_issue(batch) is None


def test_fft_grid_issue_rejects_nonfinite_target_values():
    lat, lon = np.meshgrid(
        np.linspace(40.0, 44.0, 4),
        np.linspace(5.0, 9.0, 4),
        indexing="ij",
    )
    coords = np.stack((lat.ravel(), lon.ravel()), axis=1)
    times = np.full(len(coords), np.datetime64("2021-03-01T06:00"))
    sampler, batch = _sampler_and_batch(coords, times)
    batch.target_samples.samples[0].streams_data["MEPS"].target_tokens[0][4, 0] = torch.nan

    issue = sampler._fft_grid_issue(batch)

    assert "found 1 non-finite target values" in issue
    assert "2021-03-01T06:00" in issue


def test_sampler_replaces_incomplete_grid_batch_with_next_sample():
    lat, lon = np.meshgrid(
        np.linspace(40.0, 44.0, 4),
        np.linspace(5.0, 9.0, 4),
        indexing="ij",
    )
    full_grid = np.stack((lat.ravel(), lon.ravel()), axis=1)
    invalid_batch_times = np.full(15, np.datetime64("2021-03-01T06:00"))
    sampler, invalid_batch = _sampler_and_batch(full_grid[:-1], invalid_batch_times)
    _, valid_batch = _sampler_and_batch(full_grid, np.full(16, np.datetime64("2021-03-01T06:00")))
    batches = {0: invalid_batch, 1: valid_batch}
    attempted_indices = []

    for batch in batches.values():
        batch.sources_empty = lambda: False
        batch.is_nan = lambda: False
        batch.targets_empty = lambda: False

    sampler.batch_size = 1
    sampler.len = 1
    sampler.mode_cfg = {"training_mode": ["masking"]}
    sampler.worker_workset = lambda: (0, 1)
    sampler.reset = lambda: (np.array([0, 1]), np.array([0]))

    def get_batch(index, _forecast_steps):
        attempted_indices.append(index)
        return batches[index]

    sampler._get_batch = get_batch

    assert list(sampler) == [valid_batch]
    assert attempted_indices == [0, 1]
