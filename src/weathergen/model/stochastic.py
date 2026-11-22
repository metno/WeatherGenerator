# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Stochastic latent: ensemble members from noise injected into one latent level.

The encoder runs once. For every member m the latent of `stochastic.level` (default the
finest level) goes through a few small blocks whose LayerNorms are conditioned on noise
(AdaLayerNorm, zero-initialised scale / shift / gate, as DiT). The rest of the model (cascade
already applied, forecast engine, decoders) then runs on that member's latent. The members are
stacked along the existing ensemble dimension of the predictions: [members, points, channels].

Noise: N(0, 1), `noise_dim` numbers per scale, one draw per parent cell at every level in
`noise_levels` (HEALPix nested ordering: parent id = id >> 2 (level - parent level)), shared by
all children of that parent. A level equal to `stochastic.level` gives every cell its own noise.
Optional smoothing over the 1-ring neighbourhood (`smooth_passes`) softens the parent edges.
The amplitude of the effect on the latent is learned (it starts at exactly zero).

Config (all optional except enabled):

    stochastic:
      enabled: True
      level: 8                 # latent level that receives the noise (default: finest)
      noise_levels: [6, 7]     # scales of the noise (parent levels); add 8 for per-cell noise
      noise_dim: 8             # numbers per scale
      smooth_passes: 1         # neighbour averaging passes of the noise (0 = none)
      num_blocks: 1            # noise-conditioned blocks
      att_dim: 512             # inner width of the neighbour attention
      att_heads: 8
      mlp_hidden_factor: 1
      members_train: 2
      members_val: 8
      seed: 1234
      weight_decay: 0.0        # weight decay for the noise blocks (trainer param group)

FSDP / meta device: the constructor creates parameters only (no tensor computation); the
parent index maps are built at the first forward call on the device of the latent and kept in
a plain Python dict (never in the state_dict). Every rank runs the same number of members.
With `enabled: False` (or no `stochastic` block) the module is not built and the model is
unchanged.
"""

import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from weathergen.model.norms import AdaLayerNorm


def _cfg(cf):
    sc = cf.get("stochastic", None) if hasattr(cf, "get") else None
    return sc if sc is not None else {}


def stochastic_enabled(cf) -> bool:
    return bool(_cfg(cf).get("enabled", False))


class NeighbourAttention(torch.nn.Module):
    """Each cell attends to its 1-ring (itself + 8 neighbours, hp_nbours order)."""

    def __init__(self, dim, inner, heads, norm_eps=1e-5):
        super().__init__()
        assert inner % heads == 0
        self.heads = heads
        self.dh = inner // heads
        self.norm_eps = norm_eps
        self.proj_q = torch.nn.Linear(dim, inner, bias=False)
        self.proj_k = torch.nn.Linear(dim, inner, bias=False)
        self.proj_v = torch.nn.Linear(dim, inner, bias=False)
        self.proj_out = torch.nn.Linear(inner, dim, bias=False)
        # one learned key offset per neighbour slot (direction); zero at init
        self.slot = torch.nn.Parameter(torch.empty(9, inner))
        torch.nn.init.zeros_(self.slot)

    def forward(self, x, nb):
        # x: (B, N, D), nb: (N, 9) long, compact indices
        B, N, _ = x.shape
        H, dh = self.heads, self.dh
        q = self.proj_q(x).view(B, N, H, dh)
        k = (self.proj_k(x)[:, nb] + self.slot).view(B, N, 9, H, dh)
        v = self.proj_v(x)[:, nb].view(B, N, 9, H, dh)
        q = F.layer_norm(q, (dh,), eps=self.norm_eps)
        k = F.layer_norm(k, (dh,), eps=self.norm_eps)
        att = torch.einsum("bnhd,bnshd->bnhs", q, k) / math.sqrt(dh)
        att = att.softmax(dim=-1)
        out = torch.einsum("bnhs,bnshd->bnhd", att, v).reshape(B, N, H * dh)
        return self.proj_out(out)


class NoiseBlock(torch.nn.Module):
    """x <- x + g1 * att(LN_z(x));  x <- x + g2 * mlp(LN_z(x)); g, scale, shift from z (zero init)."""

    def __init__(self, dim, z_dim, att_dim, att_heads, mlp_hidden, norm_eps=1e-5):
        super().__init__()
        self.norm_att = AdaLayerNorm(dim, z_dim, norm_eps=norm_eps, zero_init=True, with_gate=True)
        self.att = NeighbourAttention(dim, att_dim, att_heads, norm_eps=norm_eps)
        self.norm_mlp = AdaLayerNorm(dim, z_dim, norm_eps=norm_eps, zero_init=True, with_gate=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_hidden), torch.nn.GELU(), torch.nn.Linear(mlp_hidden, dim)
        )

    def forward(self, x, z, nb):
        h, g = self.norm_att(x, z)
        x = x + g * self.att(h, nb)
        h, g = self.norm_mlp(x, z)
        x = x + g * self.mlp(h)
        return x


class LatentNoise(torch.nn.Module):
    def __init__(self, cf, level: int):
        super().__init__()
        sc = _cfg(cf)
        self.level = int(sc.get("level", level))
        self.noise_levels = [int(v) for v in sc.get("noise_levels", [6, 7])]
        assert all(lv <= self.level for lv in self.noise_levels), (
            f"stochastic.noise_levels {self.noise_levels} must be <= level {self.level}"
        )
        self.noise_dim = int(sc.get("noise_dim", 8))
        self.smooth_passes = int(sc.get("smooth_passes", 1))
        self.members_train = int(sc.get("members_train", 2))
        self.members_val = int(sc.get("members_val", 8))
        self.seed = int(sc.get("seed", 1234))
        self.num_queries = int(cf.ae_local_num_queries)
        assert self.num_queries == 1, "stochastic latent: only ae_local_num_queries = 1"
        self.num_aux = int(cf.num_register_tokens + cf.num_class_tokens)
        dim = int(cf.ae_global_dim_embed)
        self.z_dim = self.noise_dim * len(self.noise_levels)
        hidden = int(sc.get("mlp_hidden_factor", 1) * dim)
        self.blocks = torch.nn.ModuleList(
            [
                NoiseBlock(dim, self.z_dim, int(sc.get("att_dim", 512)),
                           int(sc.get("att_heads", 8)), hidden, norm_eps=cf.get("norm_eps", 1e-5))
                for _ in range(int(sc.get("num_blocks", 1)))
            ]
        )
        # runtime state (not parameters, not in the state_dict)
        self._maps = {}
        self._train_calls = 0
        self._eval_calls = 0
        self._was_training = True

    def num_members(self) -> int:
        return self.members_train if self.training else self.members_val

    # ------------------------------------------------------------------ noise
    def _parent_maps(self, cells: np.ndarray, device):
        key = (str(device), cells.shape[0], int(cells[0]), int(cells[-1]))
        maps = self._maps.get(key)
        if maps is None:
            maps = []
            for lv in self.noise_levels:
                parent = cells.astype(np.int64) >> (2 * (self.level - lv))
                uniq, inv = np.unique(parent, return_inverse=True)
                maps.append((len(uniq), torch.as_tensor(inv, dtype=torch.long, device=device)))
            self._maps[key] = maps
        return maps

    def begin_forward(self):
        """Call once per model forward (before member 0): advances the seed counter."""
        if self.training:
            self._train_calls += 1
            self._was_training = True
        else:
            if self._was_training:
                self._eval_calls = 0  # every validation sees the same noise
            self._was_training = False
            self._eval_calls += 1

    def draw(self, batch_size, cells, nb, member, device):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        calls = self._train_calls if self.training else self._eval_calls
        mode = 0 if self.training else 1
        s = (self.seed * 1_000_003 + rank * 7_919 + mode * 104_729 + calls * 9_973 + member) % (2**62)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(s))
        zs = []
        for n_par, inv in self._parent_maps(cells, device):
            zp = torch.randn(batch_size, n_par, self.noise_dim, generator=gen, device=device)
            zs.append(zp[:, inv])
        z = torch.cat(zs, dim=-1)                                     # (B, N, z_dim)
        for _ in range(self.smooth_passes):
            z = z[:, nb].mean(dim=2)
        if self.smooth_passes > 0:
            z = z / z.std(dim=1, keepdim=True).clamp(min=1e-6)       # back to unit variance
        return z

    # ---------------------------------------------------------------- forward
    def forward(self, tokens, model_params, member: int):
        """tokens: (B, num_aux + N, D) latent of self.level -> same shape, member `member`."""
        mp = model_params.params_for(self.level) if hasattr(model_params, "params_for") else model_params
        nb = mp.hp_nbours.long()
        cells = np.asarray(mp.domain.active_cells)
        na = self.num_aux
        x = tokens[:, na:]
        z = self.draw(x.shape[0], cells, nb, member, x.device).to(x.dtype)
        for blk in self.blocks:
            x = blk(x, z, nb)
        return torch.cat([tokens[:, :na], x], dim=1) if na > 0 else x
