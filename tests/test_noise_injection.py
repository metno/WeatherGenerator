"""Tests for AIFS-CRPS-style noise injection in the forecasting engine."""

import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    import flash_attn  # noqa: F401
except ImportError:
    flash_attn_stub = ModuleType("flash_attn")
    flash_attn_stub.flash_attn_func = None
    flash_attn_stub.flash_attn_varlen_func = None
    with patch.dict(sys.modules, {"flash_attn": flash_attn_stub}):
        from weathergen.model import attention, engines
else:
    from weathergen.model import attention, engines

from weathergen.model.noise import (
    NoiseInjector,
    combine_member_predictions,
    get_noise_injection_config,
    validate_noise_injection_setup,
)
from weathergen.model.norms import AdaLayerNorm

DIM = 16


@pytest.fixture(autouse=True)
def sdpa_attention(monkeypatch):
    def flash_attn_func(q, k, v, softcap=0.0, dropout_p=0.0):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p
        )
        return out.transpose(1, 2)

    monkeypatch.setattr(attention, "flash_attn_func", flash_attn_func)


def _engine_config(enabled=True, num_members=2):
    return OmegaConf.create(
        {
            "forecast_att_dense_rate": 1.0,
            "fe_num_blocks": 2,
            "ae_global_dim_embed": DIM,
            "fe_num_heads": 2,
            "fe_dropout_rate": 0.0,
            "fe_with_qk_lnorm": True,
            "with_flash_attention": True,
            "norm_type": "LayerNorm",
            "norm_eps": 1e-5,
            "mlp_norm_eps": 1e-5,
            "attention_dtype": "fp32",
            "fe_layer_norm_after_blocks": [],
            "fe_impute_latent_noise_std": 0.0,
            "ae_local_num_queries": 1,
            "ae_global_block_factor": 64,
            "fe_noise_injection": {
                "enabled": enabled,
                "num_members": num_members,
                "noise_channels": 4,
                "noise_mlp_hidden_dim": 8,
                "noise_std": 1.0,
            },
        }
    )


def _engine(enabled=True):
    torch.manual_seed(0)
    mode_cfg = OmegaConf.create({"forecast": {"policy": "fixed"}})
    return engines.ForecastingEngine(_engine_config(enabled), mode_cfg, num_healpix_cells=6)


def _member_tokens(num_members=2):
    torch.manual_seed(1)
    return torch.randn(1, 6, DIM).repeat(num_members, 1, 1)


def _randomise_conditioning_outputs(engine):
    with torch.no_grad():
        for module in engine.fe_blocks.modules():
            if isinstance(module, AdaLayerNorm):
                module.embed_aux[-1].weight.normal_(std=0.5)


def test_noise_config_disabled_by_default():
    assert get_noise_injection_config(OmegaConf.create({})) is None
    assert get_noise_injection_config(_engine_config(enabled=False)) is None
    assert get_noise_injection_config(_engine_config())["num_members"] == 2


def test_noise_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="num_members"):
        get_noise_injection_config(_engine_config(num_members=0))


def test_noise_setup_rejects_deterministic_ensemble_heads():
    cf = _engine_config()
    cf.training_config = {"forecast": {"policy": "fixed"}}

    validate_noise_injection_setup(cf, {"MEPS": {"pred_head": {"ens_size": 1}}})
    with pytest.raises(ValueError, match="ens_size: 1"):
        validate_noise_injection_setup(cf, {"MEPS": {"pred_head": {"ens_size": 2}}})


def test_noise_setup_requires_forecasting_engine():
    cf = _engine_config()
    cf.training_config = {"forecast": {"policy": "fixed"}}
    cf.fe_num_blocks = 0

    with pytest.raises(ValueError, match="fe_num_blocks"):
        validate_noise_injection_setup(cf, {})


def test_noise_injector_draws_fresh_layer_normalised_noise():
    injector = NoiseInjector(noise_channels=4, mlp_hidden_dim=8)
    tokens = torch.zeros(3, 5, DIM)

    first, second = injector(tokens), injector(tokens)

    assert first.shape == (3, 5, 4)
    assert not torch.allclose(first, second)
    assert torch.allclose(first.mean(-1), torch.zeros(3, 5), atol=1e-5)


def test_engine_without_noise_is_unchanged():
    engine = _engine(enabled=False)

    assert engine.noise_injector is None
    assert not any(isinstance(m, AdaLayerNorm) for m in engine.modules())


def test_zero_initialised_members_are_identical():
    engine = _engine()
    out = engine(_member_tokens(), fstep=0)

    assert torch.allclose(out[0], out[1])


def test_members_diverge_once_noise_conditioning_is_learned():
    engine = _engine()
    _randomise_conditioning_outputs(engine)
    tokens = _member_tokens()

    out = engine(tokens, fstep=0)
    out_next_step = engine(tokens, fstep=1)

    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out, out_next_step)


def test_zero_initialised_conditioning_receives_gradients():
    engine = _engine()
    engine(_member_tokens(), fstep=0).square().sum().backward()

    grads = [
        m.embed_aux[-1].weight.grad.abs().sum()
        for m in engine.fe_blocks.modules()
        if isinstance(m, AdaLayerNorm)
    ]
    assert grads and all(g > 0 for g in grads)


def test_reset_noise_conditioning_restores_zero_init_after_global_reset():
    engine = _engine()
    for module in engine.modules():
        if isinstance(module, torch.nn.Linear | torch.nn.LayerNorm):
            module.reset_parameters()
    tokens = _member_tokens()
    assert not torch.allclose(*engine(tokens, fstep=0))

    engine.reset_noise_conditioning()

    assert torch.allclose(*engine(tokens, fstep=0))


def test_combine_member_predictions_stacks_ensemble_dimension():
    lens = [2, 3]
    members = [{"MEPS": (torch.full((1, 5, 3), float(i)), lens)} for i in range(2)]

    combined = combine_member_predictions(members)

    pred, combined_lens = combined["MEPS"]
    assert pred.shape == (2, 5, 3)
    assert torch.equal(pred[:, 0, 0], torch.tensor([0.0, 1.0]))
    assert combined_lens == lens
    assert combine_member_predictions(members[:1]) is members[0]


def test_combine_member_predictions_passes_through_skipped_streams():
    members = [{"MEPS": (torch.tensor([]), [0])} for _ in range(2)]

    assert combine_member_predictions(members)["MEPS"][0].numel() == 0


def _lat_lon_along_meridian(distances_km):
    lat = torch.tensor(distances_km) / 6371.0 + 1.1
    return torch.stack([lat, torch.full_like(lat, 0.3)], dim=-1).unsqueeze(0)


def test_fourier_noise_is_smooth_in_space_and_independent_far_away():
    torch.manual_seed(0)
    injector = NoiseInjector(4, 8, noise_type="fourier", length_scales_km=[200.0])
    # neighbouring hl8-like cells (25 km apart) and a point far beyond the length scale
    coords = _lat_lon_along_meridian([0.0, 25.0, 3000.0])
    tokens = torch.zeros(4000, 3, DIM)

    noise = injector.smooth_noise(tokens, coords).flatten(0, 0).permute(1, 0, 2).reshape(3, -1)
    corr = torch.corrcoef(noise)

    assert noise.var(dim=1).sub(1.0).abs().max() < 0.1
    assert corr[0, 1] > 0.95
    assert corr[0, 2].abs() < 0.1


def test_fourier_noise_config_requires_rope_coordinates():
    cf = _engine_config()
    cf.training_config = {"forecast": {"policy": "fixed"}}
    cf.fe_noise_injection.noise_type = "fourier"

    with pytest.raises(ValueError, match="rope_2D"):
        validate_noise_injection_setup(cf, {})
    cf.rope_2D = True
    validate_noise_injection_setup(cf, {})

    cf.fe_noise_injection.noise_type = "smooth"
    with pytest.raises(ValueError, match="noise_type"):
        get_noise_injection_config(cf)


def test_engine_with_fourier_noise_diverges_members():
    cf = _engine_config()
    cf.fe_noise_injection.noise_type = "fourier"
    torch.manual_seed(0)
    mode_cfg = OmegaConf.create({"forecast": {"policy": "fixed"}})
    engine = engines.ForecastingEngine(cf, mode_cfg, num_healpix_cells=6)
    assert engine.noise_injector.noise_type == "fourier"
    _randomise_conditioning_outputs(engine)

    coords = _lat_lon_along_meridian([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
    out = engine(_member_tokens(), fstep=0, coords=coords)

    assert not torch.allclose(out[0], out[1])


def test_fourier_noise_survives_meta_device_initialisation():
    # FSDP path: built on the meta device, then to_empty() (uninitialised memory) or a
    # checkpoint load; the length scales must not depend on module state
    with torch.device("meta"):
        injector = NoiseInjector(4, 8, noise_type="fourier", length_scales_km=[200.0])
    injector.to_empty(device="cpu")
    injector.reset_parameters()
    torch.manual_seed(0)
    coords = _lat_lon_along_meridian([0.0, 25.0, 3000.0])

    noise = injector.smooth_noise(torch.zeros(4000, 3, DIM), coords)
    corr = torch.corrcoef(noise.permute(1, 0, 2).reshape(3, -1))

    assert corr[0, 1] > 0.95
    assert corr[0, 2].abs() < 0.1
    assert torch.isfinite(injector(torch.zeros(2, 3, DIM), coords)).all()


def test_fourier_noise_is_computed_in_fp32_under_bf16_autocast():
    injector = NoiseInjector(4, 8, noise_type="fourier", length_scales_km=[50.0])
    coords = _lat_lon_along_meridian([0.0, 25.0, 3000.0])
    tokens = torch.zeros(8, 3, DIM)

    torch.manual_seed(0)
    reference = injector.smooth_noise(tokens, coords)
    torch.manual_seed(0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        noise = injector.smooth_noise(tokens, coords)

    assert noise.dtype == torch.float32
    assert torch.allclose(noise, reference, atol=1e-4)
