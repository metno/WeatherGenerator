# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Warm start from several runs (init_from), freezing with exceptions, lr-scaled groups."""

import sys
import types

import pytest
import torch
from omegaconf import OmegaConf

try:  # pragma: no cover
    import flash_attn  # noqa: F401
except ImportError:  # pragma: no cover
    # flash-attn is only available on GPU nodes: CPU fallback (same as test_multilevel_jepa)
    _fa = types.ModuleType("flash_attn")

    def _flash_attn_func(q, k, v, softcap=0.0, dropout_p=0.0, **kwargs):
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
        )
        return o.transpose(1, 2).to(q.dtype)

    _fa.flash_attn_func = _flash_attn_func
    _fa.flash_attn_varlen_func = None
    sys.modules["flash_attn"] = _fa

from weathergen.model import model_interface as mi  # noqa: E402
from weathergen.train.lr_scheduler import (  # noqa: E402
    LearningRateScheduler,
    optimizer_param_groups,
)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.ModuleDict(
            {"embed": torch.nn.Linear(4, 4), "latent_cascade": torch.nn.Linear(4, 4)}
        )
        self.pred_heads = torch.nn.ModuleDict({"CARRA2": torch.nn.Linear(4, 2)})
        self.forecast_cascade = torch.nn.Linear(4, 4)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(mi, "get_path_model", lambda cf, run_id=None: tmp_path / run_id)
    a, b = Toy(), Toy()
    for run_id, m in (("jepa", a), ("ae", b)):
        (tmp_path / run_id).mkdir()
        torch.save(m.state_dict(), tmp_path / run_id / f"{run_id}_latest.chkpt")
    return a, b


def test_init_from_two_runs(runs):
    a, b = runs
    cf = OmegaConf.create(
        {
            "with_ddp": False,
            "with_fsdp": False,
            "init_from": [
                {"run_id": "jepa", "include": [r"encoder\..*"]},
                {"run_id": "ae", "include": r"pred_heads\.CARRA2\..*"},
            ],
        }
    )
    m = Toy()
    fresh = m.forecast_cascade.weight.clone()
    m = mi.load_init_from(cf, m, "cpu")
    assert torch.equal(m.encoder["embed"].weight, a.encoder["embed"].weight)
    assert torch.equal(m.pred_heads["CARRA2"].weight, b.pred_heads["CARRA2"].weight)
    assert torch.equal(m.forecast_cascade.weight, fresh)  # new part keeps its init


def test_init_from_shape_mismatch(runs):
    cf = OmegaConf.create({"with_ddp": False, "with_fsdp": False, "init_from": [{"run_id": "ae"}]})
    m = Toy()
    m.pred_heads["CARRA2"] = torch.nn.Linear(8, 2)
    with pytest.raises(AssertionError, match="shape mismatch"):
        mi.load_init_from(cf, m, "cpu")


def test_freeze_with_exceptions():
    m = Toy()
    cf = OmegaConf.create(
        {
            "freeze_modules": [r"encoder", r"pred_heads"],
            "trainable_modules": [r"encoder\.latent_cascade"],
        }
    )
    mi.apply_freezing(cf, m)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {
        "encoder.latent_cascade.weight",
        "encoder.latent_cascade.bias",
        "forecast_cascade.weight",
        "forecast_cascade.bias",
    }


def test_lr_scaled_groups():
    m = Toy()
    opt_cfg = OmegaConf.create({"param_groups": [{"modules": [r"encoder\..*"], "lr_scale": 0.1}]})
    groups = optimizer_param_groups(m, opt_cfg)
    assert len(groups) == 2 and groups[1]["lr_scale"] == 0.1
    opt = torch.optim.AdamW(groups, lr=1e-4)
    lr_cfg = OmegaConf.create(
        {
            "lr_start": 1e-6,
            "lr_max": 1e-4,
            "lr_final_decay": 1e-5,
            "lr_final": 0.0,
            "num_steps_warmup": 10,
            "num_steps_cooldown": 10,
            "policy_warmup": "cosine",
            "policy_decay": "cosine",
            "policy_cooldown": "linear",
            "parallel_scaling_policy": "sqrt",
        }
    )
    sched = LearningRateScheduler(opt, 1, 1, 0, 100, lr_cfg)
    for _ in range(15):
        sched.step()
        g0, g1 = opt.param_groups
        assert abs(g1["lr"] - 0.1 * g0["lr"]) < 1e-12
