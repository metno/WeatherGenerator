# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import math

import pytest
import torch
from omegaconf import OmegaConf
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from weathergen.train.lr_scheduler import LearningRateScheduler
from weathergen.train.optimizer import AdEMAMix, build_optimizer


@pytest.mark.parametrize("warmup", [None, 2])
def test_ademamix_scalar_updates(warmup):
    parameter = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.float64))
    optimizer = AdEMAMix(
        [parameter],
        lr=0.02,
        betas=(0.9, 0.95, 0.9999),
        alpha=8,
        beta3_warmup=warmup,
        alpha_warmup=warmup,
        weight_decay=0.1,
        eps=1e-7,
    )
    expected, fast, slow, variance = 1.5, 0.0, 0.0, 0.0
    for step, gradient in enumerate([0.3, -0.7, 0.1, 0.8], start=1):
        beta3, alpha = 0.9999, 8.0
        if warmup == 2 and step == 1:
            # Midpoint of the two half-lives, not the two beta coefficients.
            half_lives = [math.log(0.5) / math.log(b + 1e-8) for b in (0.9, 0.9999)]
            beta3 = 0.5 ** (2 / sum(half_lives))
            alpha = 4.0
        fast = 0.9 * fast + 0.1 * gradient
        slow = beta3 * slow + (1 - beta3) * gradient
        variance = 0.95 * variance + 0.05 * gradient**2
        direction = (fast / (1 - 0.9**step) + alpha * slow) / (
            math.sqrt(variance / (1 - 0.95**step)) + 1e-7
        )
        expected -= 0.02 * (direction + 0.1 * expected)
        parameter.grad = torch.tensor([gradient], dtype=torch.float64)
        optimizer.step()
        assert parameter.item() == pytest.approx(expected, rel=1e-12)
        assert optimizer.state[parameter]["step"] == step


def test_zero_alpha_matches_adamw_and_missing_gradient_does_not_advance():
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    second = torch.nn.Parameter(first.detach().clone())
    kwargs = dict(lr=0.01, weight_decay=0.1, eps=1e-7)
    mix = AdEMAMix([first], betas=(0.9, 0.95, 0.9999), alpha=0, **kwargs)
    adam = torch.optim.AdamW([second], betas=(0.9, 0.95), **kwargs)
    mix.step()
    assert mix.state[first]["step"] == 0
    for gradient in ([0.2, -0.3], [-0.1, 0.7], [0.0, 0.0]):
        first.grad = torch.tensor(gradient, dtype=torch.float64)
        second.grad = first.grad.clone()
        mix.step()
        adam.step()
        torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"betas": (0.9, 0.95)},
        {"betas": (0.9, 0.95, 1.0)},
        {"alpha": -1},
        {"eps": float("nan")},
        {"alpha_warmup": 0},
        {"beta3_warmup": 1.5},
    ],
)
def test_invalid_ademamix_settings(kwargs):
    with pytest.raises(ValueError):
        AdEMAMix([torch.nn.Parameter(torch.ones(1))], **kwargs)


def test_optimizer_selection_preserves_adamw_scaling_and_ademamix_settings():
    model = torch.nn.Linear(2, 1)
    cfg = OmegaConf.create(
        {
            "weight_decay": 0.1,
            "adamw": {"beta1": 0.975, "beta2": 0.9875, "eps": 2e-8},
            "ademamix": {
                "betas": [0.9, 0.95, 0.9999],
                "alpha": 8,
                "beta3_warmup": 100000,
                "alpha_warmup": 100000,
                "eps": 1e-7,
            },
        }
    )
    adam = build_optimizer(model, cfg, lr=1e-6, kappa=4)
    assert isinstance(adam, torch.optim.AdamW)
    assert adam.param_groups[0]["betas"] == pytest.approx((0.9, 0.95))
    assert adam.param_groups[0]["eps"] == 1e-8
    assert adam.defaults["fused"] is True
    cfg.name = "ademamix"
    mix = build_optimizer(model, cfg, lr=1e-6, kappa=64)
    assert isinstance(mix, AdEMAMix)
    assert mix.param_groups[0]["betas"] == (0.9, 0.95, 0.9999)
    assert mix.param_groups[0]["eps"] == 1e-7
    assert mix.param_groups[0]["alpha_warmup"] == 100000
    cfg.name = "unknown"
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(model, cfg, lr=1e-6, kappa=4)


def test_cosine_scheduler_preserves_ademamix_betas_and_effective_peak_lr():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = AdEMAMix([parameter], lr=1e-6)
    cfg = OmegaConf.create(
        {
            "lr_start": 1e-6,
            "lr_max": 6.4e-4,
            "lr_final_decay": 1e-6,
            "lr_final": 0,
            "num_steps_warmup": 4,
            "num_steps_cooldown": 2,
            "policy_warmup": "cosine",
            "policy_decay": "constant",
            "policy_cooldown": "linear",
            "parallel_scaling_policy": "const",
        }
    )
    scheduler = LearningRateScheduler(optimizer, 2, 64, 0, 20, cfg)
    rates = []
    for _ in range(19):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.95, 0.9999)
    assert max(rates) == pytest.approx(6.4e-4, rel=1e-12)
    assert rates[8] == pytest.approx(6.4e-4, rel=1e-12)


def _checkpoint_roundtrip(model, tmp_path):
    optimizer = AdEMAMix(model.parameters(), lr=0.01, beta3_warmup=10, alpha_warmup=10)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    for _ in range(3):
        optimizer.step()
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state = get_optimizer_state_dict(model, optimizer, options=options)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = tmp_path / "checkpoint.optim"
    if rank == 0:
        torch.save(state, path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored_model = copy.deepcopy(model)
    restored_model.zero_grad(set_to_none=True)
    restored = AdEMAMix(restored_model.parameters())
    set_optimizer_state_dict(
        restored_model, restored, state, options=StateDictOptions(full_state_dict=True)
    )
    for parameter in restored_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    for original, resumed in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert restored.state[resumed]["step"] == 3
        for key in ("exp_avg_fast", "exp_avg_slow", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[original][key], restored.state[resumed][key])
    optimizer.step()
    restored.step()
    for original, resumed in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(original, resumed)


def test_optimizer_checkpoint_preserves_next_update(tmp_path):
    _checkpoint_roundtrip(torch.nn.Linear(4, 2), tmp_path)


def test_checkpoint_before_first_update_does_not_advance_warmups():
    model = torch.nn.Linear(2, 1)
    optimizer = AdEMAMix(model.parameters(), beta3_warmup=100000, alpha_warmup=100000)
    before = copy.deepcopy(model.state_dict())
    state = get_optimizer_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    for parameter_state in state["state"].values():
        assert parameter_state["step"] == 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize("optimizer_name", ["ademamix", "adamw"])
def test_trainer_checkpoint_sidecar_and_missing_resume(tmp_path, optimizer_name):
    pytest.importorskip("flash_attn", reason="Trainer imports GPU model modules")
    from weathergen.train.trainer import Trainer

    trainer = Trainer(OmegaConf.create({}))
    trainer.cf = OmegaConf.create(
        {
            "general": {"run_id": "test"},
            "full_model_path": str(tmp_path),
            "with_ddp": False,
            "with_fsdp": False,
        }
    )
    trainer.training_cfg = OmegaConf.create({"num_mini_epochs": 4})
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = (
        AdEMAMix(trainer.model.parameters(), alpha_warmup=10, beta3_warmup=10)
        if optimizer_name == "ademamix"
        else torch.optim.AdamW(trainer.model.parameters())
    )
    for parameter in trainer.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer.optimizer.step()
    trainer.model.zero_grad(set_to_none=True)
    trainer.save_model(-1)
    assert (tmp_path / "test_latest.chkpt").exists()
    sidecar = tmp_path / "test_latest.optim"
    assert sidecar.exists() == (optimizer_name == "ademamix")
    if optimizer_name == "ademamix":
        full_state = trainer._get_full_optimizer_state_dict()
        assert all(state["step"] == 1 for state in full_state["state"].values())
        trainer.optimizer = AdEMAMix(trainer.model.parameters())
        trainer._load_optimizer_checkpoint("test", -1)
        assert all(state["step"] == 1 for state in trainer.optimizer.state.values())
        assert trainer.optimizer.param_groups[0]["alpha_warmup"] == 10
        with pytest.raises(FileNotFoundError, match="weights-only"):
            trainer._load_optimizer_checkpoint("test", 3)


def _sharded_checkpoint_worker(rank, rendezvous, tmp_path):
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{rendezvous}"
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        model = torch.nn.Linear(4, 2, bias=False)
        model.weight = torch.nn.Parameter(distribute_tensor(torch.ones(2, 4), mesh, [Shard(0)]))
        _checkpoint_roundtrip(model, tmp_path)
    finally:
        torch.distributed.destroy_process_group()


def test_sharded_optimizer_checkpoint_preserves_next_update(tmp_path):
    torch.multiprocessing.spawn(
        _sharded_checkpoint_worker,
        args=(str(tmp_path / "rendezvous"), tmp_path),
        nprocs=2,
    )
