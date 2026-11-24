# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Tests of the multi-level JEPA pieces: Masker + MaskPlanner integration, encoder taps,
multi-level predictor, multi-level JEPA / context / deep SSL loss, frozen-teacher stream set.
"""

import sys
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# flash-attn is only available on GPU nodes: CPU fallback for the attention of these tests
try:  # pragma: no cover
    import flash_attn  # noqa: F401
except ImportError:  # pragma: no cover
    _fa = types.ModuleType("flash_attn")

    def _flash_attn_func(q, k, v, softcap=0.0, dropout_p=0.0, **kwargs):
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
        )
        return o.transpose(1, 2).to(q.dtype)

    _fa.flash_attn_func = _flash_attn_func
    _fa.flash_attn_varlen_func = None
    sys.modules["flash_attn"] = _fa

from weathergen.datasets.batch import SampleMetaData  # noqa: E402
from weathergen.datasets.domain_pyramid import DomainPyramid  # noqa: E402
from weathergen.datasets.mask_plan import MaskPlanner  # noqa: E402
from weathergen.datasets.masking import Masker  # noqa: E402
from weathergen.model.engines import GlobalAssimilationEngine  # noqa: E402
from weathergen.model.latent_predictor import (  # noqa: E402
    MultiLevelLatentPredictor,
    cell_xyz,
    jepa_multilevel_cfg,
)
from weathergen.train.loss_modules.loss_module_ssl import (  # noqa: E402
    LossLatentSSLStudentTeacher,
)
from weathergen.train.utils import select_sample_metadata  # noqa: E402

BBOX = {"lon_min": 0.0, "lon_max": 30.0, "lat_min": 55.0, "lat_max": 72.0}
LEVELS = [{"level": 5, "domain": BBOX}, {"level": 8, "domain": BBOX}]

JEPA_CONF = {
    "weight": 4.0,
    "loss_extra_args": {},
    "head": "multilevel_transformer",
    "num_blocks": 1,
    "num_heads": 2,
    "intermediate_dim": 16,
    "dropout_rate": 0.0,
    "with_qk_lnorm": True,
    "levels": {8: {"weight": 1.0, "out_dim": 12}, 5: {"weight": 0.5}},
    "target_source_correspondence": {0: {0: "subset"}},
}


def model_cfg(**training):
    return OmegaConf.create(
        {
            "latent_levels": LEVELS,
            "ae_local_num_queries": 1,
            "with_flash_attention": True,
            "norm_type": "LayerNorm",
            "norm_eps": 1e-5,
            "mlp_norm_eps": 1e-5,
            "attention_dtype": "fp32",
            "num_class_tokens": 0,
            "num_register_tokens": 2,
            "ae_global_num_blocks": 3,
            "ae_global_num_heads": 2,
            "ae_global_dropout_rate": 0.0,
            "ae_global_with_qk_lnorm": True,
            "ae_global_att_dense_rate": 1.0,
            "ae_global_block_factor": 64,
            "ae_global_mlp_hidden_factor": 2,
            "ae_global_dim_embed": 16,
            "training_config": training,
        }
    )


@pytest.fixture(scope="module")
def pyramid():
    return DomainPyramid.from_config(OmegaConf.create({"latent_levels": LEVELS}))


# ------------------------------------------------------------------ masking --------


def test_masker_uses_shared_plan(pyramid):
    streams = {
        "era5": {"name": "era5", "train_source_channels": ["t"], "train_target_channels": []},
        "seviri": {"name": "seviri", "train_source_channels": ["b"], "train_target_channels": []},
    }
    mode_cfg = OmegaConf.create(
        {
            "training_mode": ["student_teacher"],
            "losses": {
                "student-teacher": {
                    "type": "LossLatentSSLStudentTeacher",
                    "loss_fcts": {"JEPA": JEPA_CONF},
                }
            },
            "model_input": {"student": {"masking_strategy": "mask_plan", "num_samples": 1}},
            "target_input": {"teacher": {"masking_strategy": "mask_plan", "num_samples": 1}},
            "mask_plan": {
                "strategies": {
                    "joint": {"levels": {5: {"pattern": "blocks", "hidden": 0.5}, 8: "follow:5"}}
                }
            },
        }
    )
    planner = MaskPlanner(pyramid, mode_cfg.mask_plan, {"era5": "both", "seviri": "both"})
    m5 = Masker(5, "train", OmegaConf.create(streams), mode_cfg, domain=pyramid.domain(5))
    m8 = Masker(8, "train", OmegaConf.create(streams), mode_cfg, domain=pyramid.domain(8))
    rng = np.random.default_rng(0)
    for m in (m5, m8):
        m.planner = planner
        m.reset_rng(rng)
        m.start_item()
    planner.start_item(rng)

    t5, s5, _ = m5.build_samples_for_stream(
        ["student_teacher"], len(pyramid.domain(5)), streams["era5"]
    )
    t8, s8, _ = m8.build_samples_for_stream(
        ["student_teacher"], len(pyramid.domain(8)), streams["seviri"]
    )
    # streams without target channels still get a (full) teacher view in student-teacher mode
    assert t5.masks[0].all() and t8.masks[0].all()
    # student masks are the plan's masks of the stream's level, hl8 follows hl5
    plan = s5.metadata[0].params["mask_plan"]
    assert torch.equal(s5.masks[0], plan["student"][5])
    assert torch.equal(s8.masks[0], plan["student"][8])
    assert s8.metadata[0].params["mask_plan"]["strategy"] == "joint"
    anc = pyramid.ancestor_of(8, 5)
    ok = anc >= 0
    assert torch.equal(s8.masks[0][ok], s5.masks[0][anc[ok]])


def test_select_sample_metadata_prefers_finest():
    class S:
        meta_info = {
            "era5": SampleMetaData(params={}, mask=torch.ones(66, dtype=torch.bool)),
            "seviri": SampleMetaData(params={}, mask=torch.ones(4305, dtype=torch.bool)),
        }

    assert select_sample_metadata(S()).mask.shape[0] == 4305


# --------------------------------------------------------------- encoder taps --------


def test_global_engine_taps():
    cf = model_cfg()
    eng = GlobalAssimilationEngine(cf, 10, 16, tap_global_layers={0, 2})
    x = torch.randn(2, 12, 16)
    out, inter = eng(x)
    assert out.shape == x.shape and len(inter) == 2
    eng_plain = GlobalAssimilationEngine(cf, 10, 16)
    assert isinstance(eng_plain(x), torch.Tensor)
    eng_l2g_only = GlobalAssimilationEngine(cf, 10, 16, tap_global_layers=set())
    out, inter = eng_l2g_only(x)
    assert inter == []


# --------------------------------------------------- predictor and loss --------


class _Info:
    def __init__(self, strategy):
        self.params = {"mask_plan": {"strategy": strategy}}
        self.global_params = {"loss": ["JEPA"]}


class _Out:
    def __init__(self, latent):
        self.latent = [latent]


class _Tgt:
    def __init__(self, latent):
        self.latent = latent
        self.aux_outputs = []


@pytest.mark.parametrize("deep", [False, True])
def test_predictor_and_multilevel_loss(pyramid, deep):
    training = {
        "losses": {
            "student-teacher": {
                "type": "LossLatentSSLStudentTeacher",
                "loss_fcts": {"JEPA": JEPA_CONF},
            }
        },
        "context_loss": {
            "enabled": True,
            "weight": 0.4,
            "warmup_steps": 10,
            "distance_weighting": {"enabled": True, "lambda": 1.0},
        },
        "mask_plan": {"strategies": {"joint": {}, "fine_only": {}}},
    }
    if deep:
        training["deep_ssl"] = {
            "enabled": True,
            "tap_after": ["local2global", "global:0"],
            "level_weights": [0.25, 0.25, 0.5],
        }
    cf = model_cfg(**training)
    ml = jepa_multilevel_cfg(cf.training_config)
    assert ml["levels"][8]["out_dim"] == 12 and ml["target_norm"] == "layernorm"

    dims = {5: 8, 8: 16}
    n = {lvl: len(pyramid.domain(lvl)) for lvl in dims}
    xyz = {lvl: cell_xyz(*pyramid.domain(lvl).centres_lonlat()) for lvl in dims}
    num_deep = 3 if deep else 0
    pred = MultiLevelLatentPredictor(
        cf, "JEPA-head", JEPA_CONF, dims, {8: 12, 5: 8}, xyz, 2, num_deep_targets=num_deep
    )
    bs = 3
    tokens = {lvl: torch.randn(bs, 2 + n[lvl], d, requires_grad=True) for lvl, d in dims.items()}
    hidden = {lvl: torch.rand(bs, n[lvl]) < 0.5 for lvl in dims}
    out, out_deep = pred(tokens, hidden)
    assert out[8].shape == (bs, n[8], 12) and out[5].shape == (bs, n[5], 8)

    if deep:
        assert len(out_deep[8]) == 3
        s_latent = {"JEPA": out, "JEPA_deep": out_deep, "JEPA_hidden": hidden}
        t_deep = {
            lvl: [torch.randn(1, n[lvl], o) for _ in range(3)] for lvl, o in [(8, 12), (5, 8)]
        }
        t_latent = {"JEPA": {lvl: x[-1] for lvl, x in t_deep.items()}, "JEPA_deep": t_deep}
    else:
        assert out_deep is None
        s_latent = {"JEPA": out, "JEPA_hidden": hidden}
        t_latent = {"JEPA": {8: torch.randn(1, n[8], 12), 5: torch.randn(1, n[5], 8)}}
    # teacher had no data on part of hl8 (e.g. a swath gap)
    valid8 = torch.ones(1, n[8], dtype=torch.bool)
    valid8[:, : n[8] // 4] = False
    t_latent["JEPA_valid"] = {8: valid8, 5: torch.ones(1, n[5], dtype=torch.bool)}

    loss_mod = LossLatentSSLStudentTeacher(
        cf,
        cf.training_config,
        "train",
        "cpu",
        **cf.training_config.losses["student-teacher"].loss_fcts,
    )
    infos = [_Info("joint"), _Info("fine_only"), _Info("joint")]
    metadata = (np.zeros(bs, dtype=np.int32), infos, [[0, 1, 2]], None)
    lv = loss_mod.compute_loss(_Out(s_latent), _Tgt(t_latent), metadata, istep=5)
    assert torch.isfinite(lv.loss) and lv.loss.item() > 0
    lv.loss.backward()
    assert tokens[8].grad is not None and tokens[5].grad is not None
    la = lv.losses_all
    assert la["context_warmup"] == 0.5
    for key in ("JEPA_hl8", "JEPA_hl5", "context_hl8", "n_joint", "JEPA_fine_only"):
        assert key in la, key
    assert la["n_joint"] == 2.0 and la["n_fine_only"] == 1.0
    assert abs(la["frac_valid_hl8"] - 0.75) < 0.01
    if deep:
        assert "JEPA_hl8_L2" in la


def test_loss_skips_level_without_target(pyramid):
    conf = dict(JEPA_CONF)
    cf = model_cfg(
        losses={"st": {"type": "LossLatentSSLStudentTeacher", "loss_fcts": {"JEPA": conf}}}
    )
    n8 = len(pyramid.domain(8))
    n5 = len(pyramid.domain(5))
    s_latent = {
        "JEPA": {8: torch.randn(1, n8, 12, requires_grad=True), 5: torch.randn(1, n5, 8)},
        "JEPA_hidden": {8: torch.rand(1, n8) < 0.5, 5: torch.zeros(1, n5, dtype=torch.bool)},
    }
    # frozen CARRA2 teacher at hl8 only; single-level teacher validity key -1
    t_latent = {"JEPA": {8: torch.randn(1, n8, 12)}, "JEPA_valid": {8: torch.ones(1, n8).bool()}}
    loss_mod = LossLatentSSLStudentTeacher(
        cf, cf.training_config, "train", "cpu", **cf.training_config.losses["st"].loss_fcts
    )
    lv = loss_mod.compute_loss(
        _Out(s_latent), _Tgt(t_latent), (np.zeros(1, dtype=np.int32), [_Info(None)], [[0]], None)
    )
    assert lv.losses_all["JEPA_hl5"] == 0.0 and lv.losses_all["JEPA_hl8"] > 0


# ------------------------------------------- deep SSL with single-level heads --------


def test_single_level_deep_ssl_loss():
    """JEPA_deep lists (single-level head) go through the ECMWF v2 per-representation loop."""
    conf = {
        "weight": 4.0,
        "loss_extra_args": {},
        "head": "transformer",
        "out_dim": 8,
        "target_source_correspondence": {0: {0: "subset"}},
    }
    cf = model_cfg(
        losses={"st": {"type": "LossLatentSSLStudentTeacher", "loss_fcts": {"JEPA": conf}}},
        context_loss={"enabled": True, "weight": 0.4, "warmup_steps": 0},
        deep_ssl={"enabled": True, "tap_after": ["global:0"], "level_weights": [0.3, 0.7]},
    )
    loss_mod = LossLatentSSLStudentTeacher(
        cf, cf.training_config, "train", "cpu", **cf.training_config.losses["st"].loss_fcts
    )
    n = 20
    s_deep = [torch.randn(2, n, 8, requires_grad=True) for _ in range(2)]
    t_deep = [torch.randn(1, n, 8) for _ in range(2)]

    class Info:
        def __init__(self, mask):
            self.mask = mask
            self.global_params = {"loss": ["JEPA"]}

    s_mask = torch.rand(n) < 0.5
    infos = [Info(s_mask), Info(~s_mask)]
    t_info = [Info(torch.ones(n, dtype=torch.bool))]

    class Tgt:
        latent = {"JEPA": t_deep[-1], "JEPA_deep": t_deep}
        aux_outputs = t_info

    lv = loss_mod.compute_loss(
        _Out({"JEPA": s_deep[-1], "JEPA_deep": s_deep}),
        Tgt(),
        (np.zeros(2, dtype=np.int32), infos, [[0, 1]], t_info),
    )
    la = lv.losses_all
    assert "JEPA_L0" in la and "JEPA_L1" in la and "context_L1" in la
    assert abs(la["JEPA"] - (0.3 * la["JEPA_L0"] + 0.7 * la["JEPA_L1"])) < 1e-5
    lv.loss.backward()
    assert s_deep[0].grad is not None
