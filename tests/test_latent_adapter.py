# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Residual multi-level predictor and the stage-3 latent adapter (kept predictor)."""

import copy

import pytest
import torch
from omegaconf import OmegaConf

import weathergen.common.config as wg_config
from tests.test_multilevel_jepa import BBOX, JEPA_CONF, LEVELS, model_cfg  # noqa: F401
from weathergen.datasets.domain_pyramid import DomainPyramid
from weathergen.model import latent_adapter as la
from weathergen.model.init_utils import reinit_parameters
from weathergen.model.latent_predictor import MultiLevelLatentPredictor, cell_xyz


def _pyramid():
    return DomainPyramid.from_config(OmegaConf.create({"latent_levels": LEVELS}))


def _predictor(conf, out_dims, dims=None):
    pyramid = _pyramid()
    dims = dims or {5: 8, 8: 16}
    xyz = {lvl: cell_xyz(*pyramid.domain(lvl).centres_lonlat()) for lvl in dims}
    n = {lvl: len(pyramid.domain(lvl)) for lvl in dims}
    return MultiLevelLatentPredictor(model_cfg(), "JEPA-head", conf, dims, out_dims, xyz, 2), n


def test_residual_predictor_is_identity_at_init_and_after_reinit():
    conf = {**JEPA_CONF, "residual": True}
    pred, n = _predictor(conf, {8: 16})
    tokens = {lvl: torch.randn(2, 2 + n[lvl], d) for lvl, d in {5: 8, 8: 16}.items()}
    hidden = {lvl: torch.rand(2, n[lvl]) < 0.5 for lvl in n}
    out, _ = pred(tokens, hidden)
    assert torch.allclose(out[8], tokens[8][:, 2:])
    # FSDP meta-device path: reinit_parameters keeps the zero output projection
    with torch.no_grad():
        pred.proj_out["8"].weight.normal_()
    reinit_parameters(pred)
    assert pred.proj_out["8"].weight.abs().sum() == 0
    with pytest.raises(AssertionError, match="residual"):
        _predictor(conf, {8: 12})


def _run_cfg(teacher: str):
    calc = (
        {"FrozenTeacher": {"teacher_run_id": "ae"}}
        if teacher == "frozen"
        else {"EMATeacher": {"ema_ramp_up_ratio": None}}
    )
    return model_cfg(
        losses={
            "student-teacher": {
                "type": "LossLatentSSLStudentTeacher",
                "loss_fcts": {"JEPA": dict(JEPA_CONF)},
                "target_and_aux_calc": calc,
            }
        }
    )


def test_resolve_latent_adapter(monkeypatch):
    run_cf = _run_cfg("frozen")
    monkeypatch.setattr(wg_config, "load_run_config", lambda *a, **k: copy.deepcopy(run_cf))
    cf = model_cfg()
    cf.latent_adapter = {"enabled": True, "run_id": "jepa"}
    assert la.latent_adapter_enabled(cf)
    res = la.resolve_latent_adapter(cf)
    assert res["teacher"] == "frozen" and res["replace_levels"] == [5, 8] and not res["deep"]
    assert res["out_levels"] == {"8": 12, "5": None}
    # stored in the config: resolved again without the run config
    monkeypatch.setattr(wg_config, "load_run_config", lambda *a, **k: 1 / 0)
    assert la.resolve_latent_adapter(cf)["run_id"] == "jepa"
    src = la.init_from_source(cf, 8)
    assert src["rename"][r"^latent_pre_norm\."] == "latent_adapter.post_norm.8."

    # an EMA-teacher run: the predictor is meant to be discarded
    ema_cf = _run_cfg("ema")
    monkeypatch.setattr(wg_config, "load_run_config", lambda *a, **k: copy.deepcopy(ema_cf))
    cf2 = model_cfg()
    cf2.latent_adapter = {"enabled": True, "run_id": "jepa_ema"}
    with pytest.raises(ValueError, match="EMA teacher"):
        la.resolve_latent_adapter(cf2)
    cf2.latent_adapter.allow_ema_teacher = True
    assert la.resolve_latent_adapter(cf2)["teacher"] == "ema"
    # a level the stage-2 predictor does not predict
    cf3 = model_cfg()
    cf3.latent_adapter = {"enabled": True, "run_id": "jepa", "levels": [7]}
    monkeypatch.setattr(wg_config, "load_run_config", lambda *a, **k: copy.deepcopy(run_cf))
    with pytest.raises(AssertionError, match="not predicted"):
        la.resolve_latent_adapter(cf3)


def test_adapter_replaces_only_its_levels():
    pred, n = _predictor(JEPA_CONF, {8: 12})
    post = torch.nn.ModuleDict({"5": torch.nn.LayerNorm(8), "8": torch.nn.LayerNorm(16)})
    adapter = la.LatentAdapter(pred, post, None, [8], num_aux_tokens=2)
    tokens = {lvl: torch.randn(2, 2 + n[lvl], d) for lvl, d in {5: 8, 8: 16}.items()}
    hidden = {lvl: torch.zeros(2, n[lvl], dtype=torch.bool) for lvl in n}
    out = adapter(tokens, hidden)
    assert torch.equal(out[5], tokens[5])
    assert out[8].shape == (2, 2 + n[8], 12)
    assert out[8][:, :2].abs().sum() == 0  # aux tokens: zeros when the size changes
    ref, _ = pred({5: post["5"](tokens[5]), 8: post["8"](tokens[8])}, hidden)
    assert torch.allclose(out[8][:, 2:], ref[8])
