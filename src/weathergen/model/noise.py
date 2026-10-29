# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""AIFS-CRPS-style noise injection for generating exchangeable ensemble members.

For every ensemble member and forecast step, independent Gaussian noise is drawn for each latent
token, embedded with a small MLP followed by a layer norm, and used to condition all layer norms of
the forecasting engine (Lang et al. 2024, arXiv:2412.15832; defaults follow anemoi-models'
``NoiseConditioning`` / ``ConditionalLayerNorm`` with ``zero_init``). Members are stacked along the
batch dimension in member-major order: member ``m`` of sample ``b`` is at index
``m * batch_size + b``.
"""

from __future__ import annotations

import torch

from weathergen.model.norms import AdaLayerNorm

NOISE_CONFIG_KEY = "fe_noise_injection"


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
    }
    for key in ("num_members", "noise_channels", "noise_mlp_hidden_dim"):
        if settings[key] < 1:
            raise ValueError(f"{NOISE_CONFIG_KEY}.{key} must be >= 1, got {settings[key]}.")
    if settings["noise_std"] <= 0.0:
        raise ValueError(f"{NOISE_CONFIG_KEY}.noise_std must be > 0, got {settings['noise_std']}.")
    return settings


def validate_noise_injection_setup(cf, streams) -> None:
    """Reject configurations in which injected noise cannot produce exchangeable members."""
    if get_noise_injection_config(cf) is None:
        return

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

    def __init__(self, noise_channels: int, mlp_hidden_dim: int, noise_std: float = 1.0):
        super().__init__()
        self.noise_channels = noise_channels
        self.noise_std = noise_std
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return a noise embedding of shape ``(*tokens.shape[:-1], noise_channels)``."""
        # Drawn outside any activation-checkpointed block, so recomputation reuses the same noise.
        noise = torch.randn(
            (*tokens.shape[:-1], self.noise_channels),
            device=tokens.device,
            dtype=self.mlp[0].weight.dtype,
        )
        return self.mlp(noise * self.noise_std)


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
