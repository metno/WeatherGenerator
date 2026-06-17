from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

import weathergen.utils.validation_io as validation_io
from weathergen.common.io import IOReaderData


class FakeSourceSamples:
    def __init__(self, samples):
        self._samples = samples

    def get_samples(self):
        return self._samples


class FakeBatch:
    def __init__(self):
        source_raw = IOReaderData(
            coords=np.array([[60.0, 10.0]], dtype=np.float32),
            geoinfos=np.empty((1, 0), dtype=np.float32),
            data=np.array([[0.5]], dtype=np.float32),
            datetimes=np.array(["2021-10-10T00:00"], dtype="datetime64[ns]"),
        )
        stream_data = SimpleNamespace(sample_idx=0, source_raw=[source_raw])
        self.samples = [SimpleNamespace(streams_data={"ERA5": stream_data}, meta_info={})]

    def __len__(self):
        return len(self.samples)

    def get_output_idxs(self):
        return [0]

    def get_source_samples(self):
        return FakeSourceSamples(self.samples)


class FakeLatentState:
    def __init__(self):
        self.z_pre_norm = torch.arange(28, dtype=torch.float32).reshape(1, 14, 2)
        self.register_tokens = torch.ones((1, 1, 2), dtype=torch.float32)
        self.class_token = torch.full((1, 1, 2), 2.0, dtype=torch.float32)
        self.patch_tokens = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2)


class FakeModelOutput:
    def get_physical_prediction(self, fstep, stream_name):
        assert fstep == 0
        assert stream_name == "ERA5"
        return [torch.tensor([[[1.5], [2.5]]], dtype=torch.float32)]

    def get_latent_prediction(self, fstep):
        assert fstep == 0
        return {"latent_state": FakeLatentState()}


@pytest.fixture
def output_config(tmp_path, monkeypatch):
    cf = OmegaConf.create(
        {
            "healpix_level": 0,
            "num_register_tokens": 1,
            "num_class_tokens": 1,
            "streams": {
                "ERA5": {
                    "val_target_channels": ["2t"],
                    "val_source_channels": ["2t"],
                }
            },
        }
    )
    store_path = tmp_path / "results.zarr"
    monkeypatch.setattr(validation_io.config, "get_path_results", lambda cf, mini_epoch: store_path)
    monkeypatch.setattr(
        validation_io,
        "TimeWindowHandler",
        lambda *args: SimpleNamespace(
            window=lambda idx: SimpleNamespace(
                start=np.datetime64("2021-10-10T00:00", "ns"),
                end=np.datetime64("2021-10-10T01:00", "ns"),
            )
        ),
    )
    return cf, store_path


def make_validation_config(output_streams):
    return OmegaConf.create(
        {
            "losses": {"physical": {"type": "LossPhysical"}},
            "output": {"streams": output_streams},
            "start_date": "2021-10-10T00:00",
            "end_date": "2021-10-11T00:00",
            "time_window_len": 1,
            "time_window_step": 1,
        }
    )


def make_target_aux():
    physical = [
        {
            "ERA5": {
                "is_spoof": [False],
                "target": [torch.tensor([[1.0], [2.0]], dtype=torch.float32)],
                "target_coords": [torch.tensor([[60.0, 10.0], [61.0, 11.0]])],
                "target_times": [np.array(["2021-10-10T01:00", "2021-10-10T02:00"])],
                "idxs_inv": [None],
            }
        }
    ]
    return {"physical": SimpleNamespace(physical=physical)}


def test_write_output_writes_physical_output_without_latent(output_config):
    cf, store_path = output_config

    validation_io.write_output(
        cf=cf,
        val_cfg=make_validation_config(["ERA5"]),
        batch_size=1,
        mini_epoch=0,
        batch_idx=0,
        dn_data=lambda _stream, tensor: tensor,
        batch=FakeBatch(),
        model_output=FakeModelOutput(),
        target_aux_out=make_target_aux(),
    )

    root = zarr.open_group(store_path, mode="r")
    assert "ERA5" in root["0"]
    assert "latent" not in root["0"]
    assert root["0"]["ERA5"]["0"]["prediction"]["data"].shape == (2, 1, 1)


def test_write_output_writes_latent_output_as_pseudo_stream(output_config):
    cf, store_path = output_config

    validation_io.write_output(
        cf=cf,
        val_cfg=make_validation_config(["ERA5", "latent"]),
        batch_size=1,
        mini_epoch=0,
        batch_idx=0,
        dn_data=lambda _stream, tensor: tensor,
        batch=FakeBatch(),
        model_output=FakeModelOutput(),
        target_aux_out=make_target_aux(),
    )

    root = zarr.open_group(store_path, mode="r")
    latent_group = root["0"]["latent"]["0"]
    assert latent_group.attrs["num_register_tokens"] == 1
    assert latent_group.attrs["num_class_tokens"] == 1
    assert latent_group["latent_state"].shape == (12, 2)
    assert latent_group["latent_state_register_tokens"].shape == (1, 2)
    assert latent_group["latent_state_class_token"].shape == (1, 2)
    assert latent_group["coords"].shape == (12, 2)
