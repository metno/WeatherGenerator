# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""AIFS-CRPS-style noise injection for generating exchangeable ensemble members.

For every ensemble member and forecast step, Gaussian noise is drawn for each latent token,
embedded with a small MLP followed by a layer norm, and used to condition all layer norms of
the forecasting engine (Lang et al. 2024, arXiv:2412.15832; defaults follow anemoi-models'
``NoiseConditioning`` / ``ConditionalLayerNorm`` with ``zero_init``). Members are stacked along the
batch dimension in member-major order: member ``m`` of sample ``b`` is at index
``m * batch_size + b``.

``noise_type``:
  * ``iid``     : independent noise per token (original behaviour). Neighbouring HEALPix cells get
                  unrelated perturbations, which show up as cell-shaped tiles in member
                  differences.
  * ``fourier`` : a spatially smooth Gaussian random field, evaluated at the token's cell centre via
                  random Fourier features of the 3D position (squared-exponential covariance, unit
                  variance). ``length_scales_km`` gives the correlation length(s); with several
                  values every feature picks one at random (mixture of scales).
"""

from __future__ import annotations

import math

import torch

from weathergen.model.norms import AdaLayerNorm

NOISE_CONFIG_KEY = "fe_noise_injection"
NOISE_TYPES = ("iid", "fourier")
EARTH_RADIUS_KM = 6371.0


def get_noise_injection_config(cf) -> dict | None:
    """Return the validated noise-injection settings, or None if noise injection is disabled."""
    cfg = cf.get(NOISE_CONFIG_KEY)
    if cfg is None or not cfg.get("enabled", False):
        return None

    settings = {
        "num_members": int(cfg.get("num_members", 2)),
        "noise_channels": int(cfg.get("noise_channels", 4)),
        "noise_mlp_hidden_dim": int(cfg.get("noise_mlp_hidden_dim", 32)),
        "noise_std": float(cfg.get("noise_std", 1.0)),
        "noise_type": str(cfg.get("noise_type", "iid")),
        "length_scales_km": [float(s) for s in cfg.get("length_scales_km", [150.0, 500.0])],
        "num_fourier_features": int(cfg.get("num_fourier_features", 256)),
    }
    for key in ("num_members", "noise_channels", "noise_mlp_hidden_dim", "num_fourier_features"):
        if settings[key] < 1:
            raise ValueError(f"{NOISE_CONFIG_KEY}.{key} must be >= 1, got {settings[key]}.")
    if settings["noise_std"] <= 0.0:
        raise ValueError(f"{NOISE_CONFIG_KEY}.noise_std must be > 0, got {settings['noise_std']}.")
    if settings["noise_type"] not in NOISE_TYPES:
        raise ValueError(
            f"{NOISE_CONFIG_KEY}.noise_type must be one of {NOISE_TYPES}, "
            f"got {settings['noise_type']!r}."
        )
    if not settings["length_scales_km"] or min(settings["length_scales_km"]) <= 0.0:
        raise ValueError(
            f"{NOISE_CONFIG_KEY}.length_scales_km must be a non-empty list of positive values, "
            f"got {settings['length_scales_km']}."
        )
    return settings


def validate_noise_injection_setup(cf, streams) -> None:
    """Reject configurations in which injected noise cannot produce exchangeable members."""
    noise_cfg = get_noise_injection_config(cf)
    if noise_cfg is None:
        return

    if noise_cfg["noise_type"] == "fourier" and not cf.get("rope_2D", False):
        raise ValueError(
            f"{NOISE_CONFIG_KEY}.noise_type: fourier needs the cell-centre coordinates of the "
            "latent tokens, which are only computed with rope_2D: True."
        )

    has_forecast_blocks = (
        cf.get("fe_num_blocks", 0) > 0
        and cf.training_config.get("forecast", {}).get("policy") is not None
    )
    if not has_forecast_blocks:
        raise ValueError(
            f"{NOISE_CONFIG_KEY} conditions the forecasting engine, which is disabled: set "
            "fe_num_blocks > 0 and training_config.forecast.policy."
        )

    for stream_name, stream_info in streams.items():
        pred_head = stream_info.get("pred_head")
        if pred_head is not None and pred_head.get("ens_size", 1) != 1:
            raise ValueError(
                f"Stream {stream_name!r} has pred_head.ens_size={pred_head.get('ens_size')}, but "
                f"{NOISE_CONFIG_KEY} requires ens_size: 1. Deterministic per-member heads would "
                "make members non-exchangeable; ensemble members come from the injected noise."
            )


class NoiseInjector(torch.nn.Module):
    """Draw per-token Gaussian noise and embed it for conditional layer norms."""

    def __init__(
        self,
        noise_channels: int,
        mlp_hidden_dim: int,
        noise_std: float = 1.0,
        noise_type: str = "iid",
        length_scales_km: list[float] | tuple[float, ...] = (150.0, 500.0),
        num_fourier_features: int = 256,
    ):
        super().__init__()
        self.noise_channels = noise_channels
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.num_fourier_features = num_fourier_features
        # plain floats, not a buffer: with FSDP the model is built on the meta device, and a
        # non-persistent buffer is neither in the checkpoint nor re-initialised after to_empty()
        self.length_scales_km = tuple(float(s) for s in length_scales_km)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(noise_channels, mlp_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_hidden_dim, noise_channels),
            torch.nn.LayerNorm(noise_channels),
        )

    @property
    def dim_embed(self) -> int:
        return self.noise_channels

    def reset_parameters(self) -> None:
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear | torch.nn.LayerNorm):
                module.reset_parameters()

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        """Return a noise embedding of shape ``(*tokens.shape[:-1], noise_channels)``.

        coords : (1 or batch, num_tokens, 2) token positions as (lat, lon) in radians; required
            for noise_type "fourier".
        """
        # Drawn outside any activation-checkpointed block, so recomputation reuses the same noise.
        dtype = self.mlp[0].weight.dtype
        if self.noise_type == "fourier":
            noise = self.smooth_noise(tokens, coords).to(dtype)
        else:
            noise = torch.randn(
                (*tokens.shape[:-1], self.noise_channels), device=tokens.device, dtype=dtype
            )
        return self.mlp(noise * self.noise_std)

    def smooth_noise(self, tokens: torch.Tensor, coords: torch.Tensor | None) -> torch.Tensor:
        """Unit-variance Gaussian random field per batch row and channel (random Fourier feat.)."""
        if coords is None:
            raise ValueError("noise_type 'fourier' requires the token coordinates (rope_2D: True).")
        batch, num_tokens = tokens.shape[0], tokens.shape[1]
        assert coords.shape[-2] == num_tokens, (coords.shape, tokens.shape)
        device = tokens.device

        lat, lon = coords[..., 0].float(), coords[..., 1].float()
        pos = EARTH_RADIUS_KM * torch.stack(
            [torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)],
            dim=-1,
        ).expand(batch, num_tokens, 3)

        nc, nf = self.noise_channels, self.num_fourier_features
        scales = torch.tensor(self.length_scales_km, device=device)
        ell = scales[torch.randint(len(scales), (batch, nc, nf), device=device)]
        omega = torch.randn(batch, nc, nf, 3, device=device) / ell.unsqueeze(-1)
        phase = 2.0 * math.pi * torch.rand(batch, nc, nf, device=device)
        amp = torch.randn(batch, nc, nf, device=device)

        # (batch, channels, tokens, features); fp32 even under bf16 autocast: positions are
        # ~6371 km, so bf16 rounding (~25 km) would be as large as an hl8 cell
        with torch.autocast(device_type=device.type, enabled=False):
            proj = torch.einsum("btd,bcfd->bctf", pos, omega) + phase.unsqueeze(2)
            field = torch.einsum("bctf,bcf->btc", torch.cos(proj), amp)
        return field * math.sqrt(2.0 / nf)


def zero_init_noise_conditioning(module: torch.nn.Module) -> None:
    """Make every AdaLayerNorm in ``module`` start as a plain layer norm (anemoi ``zero_init``).

    The output layer of each conditioning MLP is zeroed so that all members are identical at
    initialisation; the hidden layer keeps a standard initialisation so gradients can flow.
    """
    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, AdaLayerNorm):
                submodule.embed_aux[0].reset_parameters()
                torch.nn.init.zeros_(submodule.embed_aux[-1].weight)
                torch.nn.init.zeros_(submodule.embed_aux[-1].bias)


def combine_member_predictions(
    member_predictions: list[dict[str, tuple[torch.Tensor, list[int]]]],
) -> dict[str, tuple[torch.Tensor, list[int]]]:
    """Concatenate per-member decoder outputs along the ensemble dimension.

    Each entry maps a stream name to ``(pred, t_coords_lens)`` with ``pred`` shaped
    ``(ens, points, channels)``. Empty predictions (skipped streams) are passed through.
    """
    if len(member_predictions) == 1:
        return member_predictions[0]

    combined = {}
    for stream_name, (pred, t_coords_lens) in member_predictions[0].items():
        if pred.numel() > 0:
            pred = torch.cat([member[stream_name][0] for member in member_predictions], dim=0)
        combined[stream_name] = (pred, t_coords_lens)
    return combined
