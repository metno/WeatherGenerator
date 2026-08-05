# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Latent cascade (Phase 4): information exchange BETWEEN latent levels.

Phase 3 built parallel per-level latents that never interact during encoding.
Phase 4 adds a cascade that lets adjacent levels inform each other:

    coarse -> fine (down):  upsample the coarse latent onto the fine grid (exact
                            HEALPix parent->children map) and inject it -> gives a
                            fine stream synoptic context.
    fine -> coarse (up):    pool the fine latent onto the coarse grid (average of
                            each coarse cell's active children) and write it back
                            -> gives the coarse latent the fine-obs constraint
                            where the fine level has support.

Only the GEOMETRY (which cells connect) is fixed -- it comes from the exact
parent/child compact maps in DomainPyramid. The TRANSFORM applied along those
connections is a pluggable operator:

    - "none"      : parameter-free copy (down) / average (up). Validates wiring.
    - "linear"    : a learned Linear per direction after the gather/scatter.
    - "attention" : (future) fine cells cross-attend to their parent coarse latent
                    and vice versa. Same call site; only the operator changes.

The cascade operates ONLY on the spatial cell tokens; prepended register/class
("aux") tokens are passed through untouched. With a single-level pyramid there
are no adjacent pairs, so the cascade is a no-op (baseline behaviour preserved).
"""

import logging

import numpy as np
import torch

_logger = logging.getLogger(__name__)


class _PairOperator(torch.nn.Module):
    """
    Transforms exchanged features for ONE adjacent (coarse, fine) level pair.

    Interface (fixed, so operators are swappable):
        down(coarse_cells)                 -> features to add into fine cells,
                                              shape (B, n_fine, q, D)
        up(fine_cells)                     -> features to add into coarse cells,
                                              shape (B, n_coarse, q, D)

    The geometry (parent_of_child / child_of_parent) is applied by the caller
    (LatentCascade); the operator only transforms feature vectors.
    """

    def __init__(self, dim: int, kind: str = "none"):
        super().__init__()
        self.kind = kind
        if kind == "none":
            self.down_proj = None
            self.up_proj = None
        elif kind == "linear":
            self.down_proj = torch.nn.Linear(dim, dim, bias=False)
            self.up_proj = torch.nn.Linear(dim, dim, bias=False)
            # init near-zero so the cascade starts as a gentle perturbation
            torch.nn.init.zeros_(self.down_proj.weight)
            torch.nn.init.zeros_(self.up_proj.weight)
        else:
            raise ValueError(f"unknown cascade operator kind={kind!r}")

    def down(self, x):
        return x if self.down_proj is None else self.down_proj(x)

    def up(self, x):
        return x if self.up_proj is None else self.up_proj(x)


class LatentCascade(torch.nn.Module):
    """
    Runs V-cycle(s) of coarse<->fine exchange across all adjacent level pairs.

    Args:
        domain_pyramid : provides the exact parent/child compact maps.
        dim            : latent feature dim (ae_global_dim_embed).
        num_queries    : ae_local_num_queries (per-cell query count).
        num_aux        : number of prepended register+class tokens per sample.
        num_cycles     : number of down+up passes. 0 => inert (Phase 4a default).
        operator       : "none" | "linear" | "attention".
    """

    def __init__(
        self,
        domain_pyramid,
        dim: int,
        num_queries: int,
        num_aux: int,
        num_cycles: int = 0,
        operator: str = "none",
    ):
        super().__init__()
        self.pyramid = domain_pyramid
        self.dim = dim
        self.num_queries = num_queries
        self.num_aux = num_aux
        self.num_cycles = num_cycles
        self.operator_kind = operator
        self.levels = domain_pyramid.levels

        # adjacent (coarse, fine) pairs, coarse->fine order
        self.pairs = list(zip(self.levels[:-1], self.levels[1:], strict=True))

        # one operator per pair; register the exact index maps as buffers so they
        # move with .to(device) and are picklable.
        self._ops = torch.nn.ModuleDict()
        for (lc, lf) in self.pairs:
            key = f"{lc}_{lf}"
            self._ops[key] = _PairOperator(dim, operator)
            poc = self.pyramid.parent_of_child(lc, lf)        # (n_fine,) coarse idx or -1
            cop = self.pyramid.child_of_parent(lc, lf)        # (n_coarse, K) fine idx or -1
            cv = self.pyramid.child_valid(lc, lf)             # (n_coarse, K) bool
            self.register_buffer(f"poc_{key}", torch.as_tensor(poc, dtype=torch.long), persistent=False)
            self.register_buffer(f"cop_{key}", torch.as_tensor(cop, dtype=torch.long), persistent=False)
            self.register_buffer(f"cv_{key}", torch.as_tensor(cv, dtype=torch.bool), persistent=False)

    # -- layout helpers ----------------------------------------------------
    def _split(self, latent):
        """(B, (num_aux+n_cells)*q, D) -> aux (B, num_aux*q, D), cells (B, n_cells, q, D)."""
        B, T, D = latent.shape
        aux_t = self.num_aux * self.num_queries
        aux = latent[:, :aux_t, :]
        cells = latent[:, aux_t:, :].reshape(B, -1, self.num_queries, D)
        return aux, cells

    def _merge(self, aux, cells):
        B = cells.shape[0]
        D = cells.shape[-1]
        return torch.cat([aux, cells.reshape(B, -1, D)], dim=1)

    # -- exchange ----------------------------------------------------------
    def is_inert(self) -> bool:
        return self.num_cycles == 0 or len(self.pairs) == 0

    def forward(self, tokens_by_level: dict) -> dict:
        """
        tokens_by_level: {level: latent (B, (num_aux+n_cells)*q, D)}.
        Returns an updated dict (same shapes). Inert -> returns input unchanged.
        """
        if self.is_inert():
            return tokens_by_level

        # split every level into (aux, cells)
        aux = {}
        cells = {}
        for lvl, lat in tokens_by_level.items():
            a, c = self._split(lat)
            aux[lvl] = a
            cells[lvl] = c

        for _ in range(self.num_cycles):
            # DOWN: coarse -> fine, from coarsest pair to finest
            for (lc, lf) in self.pairs:
                key = f"{lc}_{lf}"
                op = self._ops[key]
                poc = getattr(self, f"poc_{key}")             # (n_fine,)
                coarse = cells[lc]                            # (B, n_coarse, q, D)
                # gather each fine cell's parent coarse vector; -1 -> zero
                valid = (poc >= 0)
                idx = poc.clamp(min=0)
                parent = coarse[:, idx, :, :]                 # (B, n_fine, q, D)
                parent = parent * valid.view(1, -1, 1, 1).to(parent.dtype)
                cells[lf] = cells[lf] + op.down(parent)

            # UP: fine -> coarse, from finest pair back to coarsest
            for (lc, lf) in reversed(self.pairs):
                key = f"{lc}_{lf}"
                op = self._ops[key]
                cop = getattr(self, f"cop_{key}")             # (n_coarse, K)
                cv = getattr(self, f"cv_{key}")               # (n_coarse, K)
                fine = cells[lf]                              # (B, n_fine, q, D)
                B, _, q, D = fine.shape
                K = cop.shape[1]
                idx = cop.clamp(min=0)                        # (n_coarse, K)
                gathered = fine[:, idx.reshape(-1), :, :].reshape(
                    B, cop.shape[0], K, q, D
                )                                             # (B, n_coarse, K, q, D)
                mask = cv.view(1, cop.shape[0], K, 1, 1).to(gathered.dtype)
                summed = (gathered * mask).sum(dim=2)         # (B, n_coarse, q, D)
                count = cv.sum(dim=1).clamp(min=1).view(1, -1, 1, 1).to(gathered.dtype)
                pooled = summed / count                       # mean over active children
                cells[lc] = cells[lc] + op.up(pooled)

        # re-merge aux + updated cells
        out = {}
        for lvl in tokens_by_level:
            out[lvl] = self._merge(aux[lvl], cells[lvl])
        return out
