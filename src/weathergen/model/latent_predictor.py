# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
JEPA predictor for a multi-resolution latent (V-JEPA style predictor, one pass over all
latent levels).

Same structure as `LatentPredictionHeadTransformer` (Linear bottleneck -> num_blocks x
(self-attention + MLP) -> Linear), extended to several latent levels:

* input: the (post-norm) latent tokens of every latent level, aux (register / class)
  tokens included; each level has its own input projection D_l -> intermediate_dim;
* added to every cell token: a learned level embedding, a geographic code of the cell
  centre (Fourier features of the unit-sphere position, so that tokens of different levels
  are comparable in position), and a learned mask embedding at the cells the student did
  not see (no input tokens: hidden by the mask plan or no data). The encoder output at those
  cells already carries the positional encoding and context; the mask embedding tells the
  predictor where to predict;
* full self-attention over the concatenated tokens of all levels;
* output: one projection per latent level to that level's target size (out_dim, e.g. the
  latent size of a frozen teacher).

Only the levels listed in the JEPA `levels` config get an output; all levels are input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.model.attention import MultiSelfAttentionHead
from weathergen.model.layers import MLP
from weathergen.utils.utils import get_dtype


def cell_xyz(lons_deg: NDArray, lats_deg: NDArray) -> NDArray:
    """(N, 3) unit-sphere positions of cell centres."""
    lon, lat = np.radians(lons_deg), np.radians(lats_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


class GeoFourierCode(nn.Module):
    """Learned projection of multi-frequency Fourier features of unit-sphere positions."""

    def __init__(self, dim_out: int, num_freqs: int = 8, max_freq: float = 256.0):
        super().__init__()
        self._num_freqs, self._max_freq = num_freqs, max_freq
        self.register_buffer("freqs", self._freqs(), persistent=False)
        self.proj = nn.Linear(3 + 2 * 3 * num_freqs, dim_out, bias=False)

    def _freqs(self, device=None) -> torch.Tensor:
        return torch.exp(
            torch.linspace(0.0, float(np.log(self._max_freq)), self._num_freqs, device=device)
        )

    def rebuild_buffers(self, device) -> None:
        self.freqs = self._freqs(device)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        ang = xyz.unsqueeze(-1) * self.freqs  # (N, 3, F)
        feats = torch.cat([xyz, torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2)], dim=-1)
        return self.proj(feats.to(self.proj.weight.dtype))


class MultiLevelLatentPredictor(nn.Module):
    def __init__(
        self,
        cf: Config,
        name: str,
        loss_conf,
        dims_in: dict[int, int],
        out_dims: dict[int, int],
        cells_xyz: dict[int, NDArray],
        num_aux_tokens: int,
        num_deep_targets: int = 0,
    ):
        """
        dims_in    : latent size per latent level (all levels are predictor input)
        out_dims   : target size per predicted level (keys = predicted levels)
        cells_xyz  : (N_l, 3) unit-sphere cell centres per latent level (compact order)
        num_deep_targets : deep SSL: number of representations per level (taps + final);
            0 = no deep SSL. Each predicted level then gets one Linear(out, out) projection
            per representation (as DeepSSL in ecmwf develop-ssl-diffusion-v2).
        """
        super().__init__()
        self.name = name
        self.levels = sorted(dims_in.keys())
        self.out_levels = sorted(out_dims.keys())
        for lvl in self.out_levels:
            assert lvl in self.levels, f"{name}: predicted level {lvl} is not a latent level"
        self.num_aux = num_aux_tokens
        assert cf.ae_local_num_queries == 1, (
            "MultiLevelLatentPredictor assumes one latent token per cell (ae_local_num_queries=1)"
        )

        dim = loss_conf["intermediate_dim"]
        num_blocks = loss_conf["num_blocks"]
        num_heads = loss_conf["num_heads"]
        dropout_rate = loss_conf.get("dropout_rate", 0.0)
        with_qk_lnorm = loss_conf.get("with_qk_lnorm", True)

        self.proj_in = nn.ModuleDict(
            {str(lvl): nn.Linear(dims_in[lvl], dim, bias=False) for lvl in self.levels}
        )
        self.level_embed = nn.Parameter(torch.zeros(len(self.levels), dim))
        self.mask_embed = nn.Parameter(torch.zeros(dim))
        self.geo = GeoFourierCode(
            dim, loss_conf.get("geo_num_freqs", 8), loss_conf.get("geo_max_freq", 256.0)
        )
        # cell centres as non-persistent buffers (derived from the domains, not checkpointed);
        # rebuild_buffers() restores them after a meta-device (FSDP) construction
        self._cells_xyz_np = {
            lvl: np.asarray(cells_xyz[lvl], dtype=np.float32) for lvl in self.levels
        }
        for lvl in self.levels:
            self.register_buffer(
                f"xyz_{lvl}", torch.as_tensor(self._cells_xyz_np[lvl]), persistent=False
            )
        self.reset_extra_parameters()

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                MultiSelfAttentionHead(
                    dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    with_qk_lnorm=with_qk_lnorm,
                    with_flash=cf.with_flash_attention,
                    norm_type=cf.norm_type,
                    qk_norm_type=cf.get("qk_norm_type", None),
                    norm_eps=cf.norm_eps,
                    attention_dtype=get_dtype(cf.attention_dtype),
                )
            )
            self.blocks.append(
                MLP(
                    dim,
                    dim,
                    hidden_factor=4,
                    with_residual=True,
                    dropout_rate=dropout_rate,
                    norm_type=cf.norm_type,
                    norm_eps=cf.mlp_norm_eps,
                )
            )
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.ModuleDict(
            {str(lvl): nn.Linear(dim, out_dims[lvl], bias=False) for lvl in self.out_levels}
        )

        self.num_deep_targets = num_deep_targets
        if num_deep_targets > 0:
            self.deep_proj = nn.ModuleDict(
                {
                    str(lvl): nn.ModuleList(
                        nn.Linear(out_dims[lvl], out_dims[lvl], bias=False)
                        for _ in range(num_deep_targets)
                    )
                    for lvl in self.out_levels
                }
            )

    def reset_extra_parameters(self) -> None:
        """Init of the plain Parameters (Linear / LayerNorm are handled by reinit_parameters)."""
        with torch.no_grad():
            nn.init.normal_(self.level_embed, std=0.02)
            nn.init.normal_(self.mask_embed, std=0.02)

    def rebuild_buffers(self, device) -> None:
        """Recreate the non-persistent buffers on `device` (after to_empty, FSDP)."""
        for lvl in self.levels:
            setattr(self, f"xyz_{lvl}", torch.as_tensor(self._cells_xyz_np[lvl], device=device))
        self.geo.rebuild_buffers(device)

    def forward(
        self, tokens: dict[int, torch.Tensor], hidden: dict[int, torch.Tensor]
    ) -> tuple[dict[int, torch.Tensor], dict[int, list[torch.Tensor]] | None]:
        """
        tokens : {level: (B, num_aux + N_l, D_l)} post-norm latent tokens of the student
        hidden : {level: (B, N_l) bool} True where the student had no input
        Returns ({level: (B, N_l, out_l)}, deep {level: [(B, N_l, out_l)] per representation}
        or None).
        """
        xs, sizes = [], []
        for i, lvl in enumerate(self.levels):
            t = self.proj_in[str(lvl)](tokens[lvl])
            aux, cells = t[:, : self.num_aux], t[:, self.num_aux :]
            xyz = getattr(self, f"xyz_{lvl}")
            cells = cells + self.geo(xyz).to(cells.dtype).unsqueeze(0)
            cells = cells + hidden[lvl].unsqueeze(-1).to(cells.dtype) * self.mask_embed.to(
                cells.dtype
            )
            x_l = torch.cat([aux, cells], dim=1) + self.level_embed[i].to(cells.dtype)
            xs.append(x_l)
            sizes.append(x_l.shape[1])
        x = torch.cat(xs, dim=1)

        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        x = self.norm_out(x)

        out, deep = {}, ({} if self.num_deep_targets > 0 else None)
        for lvl, x_l in zip(self.levels, torch.split(x, sizes, dim=1), strict=True):
            if lvl not in self.out_levels:
                continue
            pred = self.proj_out[str(lvl)](x_l[:, self.num_aux :])
            out[lvl] = pred
            if deep is not None:
                deep[lvl] = [proj(pred) for proj in self.deep_proj[str(lvl)]]
        return out, deep


def jepa_multilevel_cfg(training_cfg) -> dict | None:
    """
    Multi-level JEPA config (JEPA loss with `head: multilevel_transformer`), None otherwise.

    Returns {"levels": {level: {"weight": w, "out_dim": d or None}}, "target_norm": str}.
    """
    if training_cfg is None or training_cfg.get("losses", None) is None:
        return None
    for v in training_cfg.losses.values():
        if v.get("type", None) != "LossLatentSSLStudentTeacher" or not v.get("enabled", True):
            continue
        conf = v.loss_fcts.get("JEPA", None)
        if conf is None or str(conf.get("head", "")).lower() != "multilevel_transformer":
            continue
        levels_cfg = conf.get("levels", None)
        assert levels_cfg, "JEPA head multilevel_transformer requires a `levels` block"
        levels = {}
        for k, lv in levels_cfg.items():
            lv = lv or {}
            levels[int(k)] = {
                "weight": float(lv.get("weight", 1.0)),
                "out_dim": lv.get("out_dim", None),
            }
        target_norm = str(conf.get("target_norm", "layernorm")).lower()
        assert target_norm in ("layernorm", "none"), f"JEPA target_norm {target_norm!r}"
        return {"levels": levels, "target_norm": target_norm}
    return None
