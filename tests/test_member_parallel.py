import ast
import dataclasses
import datetime
import importlib.util
import logging
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from weathergen.model.layers import NoiseInjector
from weathergen.utils.distributed import (
    gather_ensemble_predictions,
    get_ensemble_parallel_size,
    init_ensemble_parallel,
    is_ensemble_root,
    validate_ensemble_targets,
)


class MemberModel(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.noise = NoiseInjector(3, noise_dim=4, hidden_dim=8, seed=seed)
        self.head = torch.nn.Linear(3, 1)

    def forward(self, tokens):
        return self.head(self.noise(tokens))


def _crps():
    path = Path(__file__).parents[1] / "src/weathergen/train/loss_modules/loss_functions.py"
    spec = importlib.util.spec_from_file_location("member_parallel_losses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel_crps


def _model_classes():
    namespace = {
        "__name__": __name__,
        "dataclass": dataclasses.dataclass,
        "dataclasses": dataclasses,
        "torch": torch,
        "nn": torch.nn,
        "typing": typing,
        "logger": logging.getLogger(__name__),
        "checkpoint": checkpoint,
        "gather_ensemble_predictions": gather_ensemble_predictions,
        "validate_ensemble_targets": validate_ensemble_targets,
    }
    for filename, names in (
        ("engines.py", {"ForecastingEngine", "LatentState"}),
        ("model.py", {"Model", "ModelOutput"}),
    ):
        path = Path(__file__).parents[1] / "src/weathergen/model" / filename
        classes = [
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name in names
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *classes,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Model"], namespace["ForecastingEngine"]


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = torch.nn.Parameter(torch.randn(24, 8))
        self.calls = 0

    def forward(self, samples):
        self.calls += 1
        return self.tokens.reshape(2, 12, 8)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)

    def forward(self, latent, output, latent_lens, output_lens, coordinates):
        batch_size = latent.shape[0] // (12 * 9)
        states = latent.reshape(batch_size, 12, 9, 8)[:, 0, 0]
        lengths = output_lens[1:].reshape(batch_size, 12).sum(dim=1)
        return self.linear(states).repeat_interleave(lengths, dim=0)


class Head(torch.nn.Module):
    def forward(self, tokens):
        return tokens.unsqueeze(0)


class Sample:
    target_timestep_hours = 1
    target_mask = None

    def __init__(self, points):
        self.points = points

    def get_target_coords(self, step):
        coords = torch.arange(self.points * 2, dtype=torch.float32).reshape(self.points, 2)
        coords = coords + step
        lengths = torch.tensor([self.points, *([0] * 11)])
        return coords, lengths


class Batch:
    def __init__(self):
        self.samples = []
        for points in (1, 2):
            coords, lengths = zip(
                *(Sample(points).get_target_coords(step) for step in range(3)), strict=True
            )
            stream = SimpleNamespace(target_coords=list(coords), target_coords_lens=list(lengths))
            self.samples.append(SimpleNamespace(streams_data={"ERA5": stream}))

    def __len__(self):
        return len(self.samples)

    def get_num_steps(self):
        return 1

    def get_output_idxs(self):
        return [1, 2]

    def stack_sample_meta_info(self):
        return {}


def _rollout_check(rank, crps):
    model_type, engine_type = _model_classes()
    torch.manual_seed(42)
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.cf = OmegaConf.create(
        {
            "ae_local_num_queries": 1,
            "ae_global_dim_embed": 8,
            "decoder_type": "Transformer",
            "training_config": {"training_mode": ["masking"]},
        }
    )
    model.is_diffusion = False
    model.is_denoising = False
    model.with_class_token = False
    model.num_register_tokens = 0
    model.num_aux_tokens = 0
    model.register_token_idxs = []
    model.class_token_idxs = []
    model.use_class_token = False
    model.use_patch_token = True
    model.num_healpix_cells = 12
    model.hp_nbours = torch.arange(12)[:, None].expand(12, 9)
    model.member_parallel = True
    model.noise_ensemble_size = 2
    model.local_noise_ensemble_size = 1
    model.source_encoder = Encoder()
    model.encoder = lambda params, samples: (model.source_encoder(samples), None, [])
    engine = engine_type.__new__(engine_type)
    torch.nn.Module.__init__(engine)
    engine.cf = OmegaConf.create({})
    engine.noise_injector = NoiseInjector(8, noise_dim=4, hidden_dim=8, seed=100 + rank)
    engine.fe_blocks = torch.nn.ModuleList()
    engine.out_proj = None
    model.forecast_engine = engine
    model.streams = {"ERA5": {}}
    model.embed_target_coords = torch.nn.ModuleDict({"ERA5": torch.nn.Identity()})
    model.target_token_engines = torch.nn.ModuleDict({"ERA5": Decoder()})
    model.pred_heads = torch.nn.ModuleDict({"ERA5": Head()})
    model.latent_heads = torch.nn.ModuleDict()
    model.latent_norm = torch.nn.Identity()
    distributed_model = DistributedDataParallel(model, find_unused_parameters=True)
    samples = SimpleNamespace(batch_samples=Batch(), latent=[])
    params = SimpleNamespace(hp_nbours=model.hp_nbours, rope_coords=None)
    for _ in range(2):
        distributed_model.zero_grad()
        output = distributed_model(params, samples, [1, 2])
        assert len(output.latent) == 3
        loss = torch.zeros(())
        for step in (1, 2):
            predictions = output.get_physical_prediction(step)["ERA5"]
            for points, prediction in zip((1, 2), predictions, strict=True):
                assert prediction.shape == (2, points, 1)
                assert not torch.equal(prediction[0], prediction[1])
                term, _ = crps(torch.zeros(points, 1), prediction, None, None, alpha=0.95)
                loss = loss + term
        loss.backward()
        for parameter in distributed_model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            expected = parameter.grad.detach().clone()
            dist.broadcast(expected, src=0)
            torch.testing.assert_close(parameter.grad, expected)
    distributed_model.eval()
    with torch.no_grad():
        first = distributed_model(params, samples, [1])
        continuation = distributed_model(params, first, [2])
        assert model.source_encoder.calls == 3
        assert continuation.latent[-1]["latent_state"].z_pre_norm.shape == (2, 12, 8)
        assert continuation.get_physical_prediction(0)["ERA5"][1].shape == (2, 2, 1)


def _distributed_check(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        config = init_ensemble_parallel(
            OmegaConf.create(
                {"noise_injector": {"enabled": True, "member_parallel": True, "ensemble_size": 2}}
            )
        )
        assert config.data_parallel_rank == rank // 2
        assert config.data_parallel_size == world_size // 2
        assert get_ensemble_parallel_size() == 2
        assert is_ensemble_root() == (rank % 2 == 0)
        torch.manual_seed(42)
        model = MemberModel(100 + rank)
        wrapped = DistributedDataParallel(model)
        reference = MemberModel(None)
        reference.load_state_dict(model.state_dict())
        data_size = config.data_parallel_size
        inputs = torch.arange(data_size * 9.0).reshape(data_size, 3, 3) / 10
        targets = torch.arange(data_size * 3.0).reshape(data_size, 3, 1) / 10
        coordinates = torch.arange(6.0).reshape(3, 2)
        local = wrapped(inputs[config.data_parallel_rank]).unsqueeze(0)
        ensemble = gather_ensemble_predictions(local, coordinates, [1, 2])
        assert not torch.equal(ensemble[0], ensemble[1])
        crps = _crps()
        loss, _ = crps(targets[config.data_parallel_rank], ensemble, None, None, alpha=0.95)
        expected_loss = torch.zeros(())
        for data_rank in range(data_size):
            members = []
            for member in range(2):
                reference.noise.seed = 100 + 2 * data_rank + member
                reference.noise._generator = None
                members.append(reference(inputs[data_rank]))
            member_loss, _ = crps(targets[data_rank], torch.stack(members), None, None, alpha=0.95)
            if data_rank == config.data_parallel_rank:
                torch.testing.assert_close(loss, member_loss)
            expected_loss = expected_loss + member_loss / data_size
        loss.backward()
        expected_loss.backward()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            torch.testing.assert_close(gather_ensemble_predictions(local), ensemble)
        _rollout_check(rank, crps)
        with pytest.raises(ValueError, match="coordinates"):
            validate_ensemble_targets(coordinates + rank % 2, [1, 2])
        with pytest.raises(ValueError, match="coordinates"):
            validate_ensemble_targets(coordinates[: rank % 2], [rank % 2])
        with pytest.raises(ValueError, match="shapes"):
            gather_ensemble_predictions(torch.zeros(1, rank % 2 + 1, 1))
        with pytest.raises(ValueError, match="shapes"):
            gather_ensemble_predictions(torch.zeros((1, 1) if rank % 2 else (1, 1, 1)))
        init_ensemble_parallel(OmegaConf.create({}))
        assert get_ensemble_parallel_size() == 1
        assert gather_ensemble_predictions(local) is local
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_member_parallel_crps_and_gradients(tmp_path, world_size):
    mp.start_processes(
        _distributed_check,
        args=(world_size, str(tmp_path / "rendezvous")),
        nprocs=world_size,
        start_method="spawn",
        join=True,
    )


def test_member_parallel_requires_enough_ranks():
    with pytest.raises(ValueError, match="divisible"):
        init_ensemble_parallel(
            OmegaConf.create(
                {"noise_injector": {"enabled": True, "member_parallel": True, "ensemble_size": 2}}
            )
        )