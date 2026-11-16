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

Options for the "attention" operator, coarse -> fine direction (both off by default):
    down_ring : each fine cell reads the 9 coarse cells around its parent (the parent and
                its 8 neighbours) instead of the parent alone. With a single source the
                attention weight is always 1 (the fine cell's own content plays no role
                and all children of a parent get the same update); with 9 sources the
                fine cell selects among them and can draw on the neighbouring parent.
    down_rope : 2D RoPE between the fine cell and each coarse source, so the attention
                sees the offset (fine cell centre - coarse cell centre) and the update can
                vary smoothly inside a parent. Offsets in units of coarse cells.
    down_sigma: > 0 adds a distance penalty -d^2 / (2 sigma^2) to the attention scores
                (d = distance fine cell centre - coarse cell centre, in coarse cells), i.e.
                a Gaussian preference for near sources. The set of 9 sources changes when a
                fine cell crosses a coarse border; far sources then have ~zero weight, so
                the update stays (nearly) continuous across the border. 0 = off.

The cascade operates ONLY on the spatial cell tokens; prepended register/class
("aux") tokens are passed through untouched. With a single-level pyramid there
are no adjacent pairs, so the cascade is a no-op (baseline behaviour preserved).
"""

import logging

import numpy as np
import torch

from weathergen.model.positional_encoding import (
    apply_rotary_2d_single,
    dec_rope_frame,
    dec_rope_project,
    rope_2d_wavelength_freqs,
    rotary_cos_sin_2d_wavelengths,
)

_logger = logging.getLogger(__name__)


class _PairOperator(torch.nn.Module):
    """
    Transforms exchanged features for ONE adjacent (coarse, fine) level pair.

    Uniform interface (so operators are swappable without touching the caller):

        down(query, source)  -> additive update for the fine cells
        up(query, source)    -> additive update for the coarse cells

    where, per target cell:
      query  : the target level's own latent    (B, N, q, D)   [attention only]
      source : the aligned source features       shape depends on `reduces_source`:
                 - reduces_source=True  (none/linear): pre-reduced (B, N, q, D)
                     down: parent copied to each fine cell; up: mean of children.
                 - reduces_source=False (attention):  per-cell neighbour SET
                     (B, N, S, q, D)  (down: S=1 parent; up: S=K children), with a
                     validity mask so padded/absent neighbours are ignored.

    `none`/`linear` ignore `query`. `attention` uses `query` as the attention
    query and `source` as keys/values. The GEOMETRY (which cells connect) is always
    applied by the caller; the operator only transforms features.
    """

    def __init__(self, dim: int, kind: str = "none", num_heads: int = 8,
                 down_rope_wavelengths=None):
        super().__init__()
        self.kind = kind
        self.dim = dim
        # attention consumes the per-cell neighbour SET (unreduced); none/linear
        # consume the pre-reduced source. This flag tells the caller which to pass.
        self.reduces_source = kind in ("none", "linear")

        if kind == "none":
            self.down_proj = None
            self.up_proj = None
        elif kind == "linear":
            self.down_proj = torch.nn.Linear(dim, dim, bias=False)
            self.up_proj = torch.nn.Linear(dim, dim, bias=False)
            torch.nn.init.zeros_(self.down_proj.weight)
            torch.nn.init.zeros_(self.up_proj.weight)
        elif kind == "attention":
            self.num_heads = num_heads
            assert dim % num_heads == 0, "dim must be divisible by num_heads"
            self.down_attn = _CrossAttn(dim, num_heads, rope_wavelengths=down_rope_wavelengths)
            self.up_attn = _CrossAttn(dim, num_heads)
        else:
            raise ValueError(f"unknown cascade operator kind={kind!r}")

    def down(self, query, source, mask=None, coords_q=None, coords_src=None, bias=None):
        if self.kind == "none":
            return source
        if self.kind == "linear":
            return self.down_proj(source)
        return self.down_attn(query, source, mask, coords_q, coords_src, bias)

    def up(self, query, source, mask=None):
        if self.kind == "none":
            return source
        if self.kind == "linear":
            return self.up_proj(source)
        return self.up_attn(query, source, mask)


class _CrossAttn(torch.nn.Module):
    """
    Minimal per-cell cross-attention: each target cell (query) attends over its
    set of source neighbours (keys/values). Output-projected to `dim` and
    ZERO-INITIALISED on the output projection, so the cascade starts as identity
    (adds nothing) and learns in -- same gentle-start property as linear.

    Shapes:
        query  : (B, N, q, D)
        source : (B, N, S, q, D)     S = neighbours per cell
        mask   : (B, N, S) bool or None   (True = valid neighbour)
    Returns:
        (B, N, q, D)   additive update
    """

    def __init__(self, dim: int, num_heads: int, rope_wavelengths=None):
        super().__init__()
        self.h = num_heads
        self.dh = dim // num_heads
        # optional 2D RoPE between a target cell and its sources: (shortest, longest)
        # wavelength in the units of the coordinates passed to forward(). No weights.
        self.rope_wavelengths = (
            None if rope_wavelengths is None
            else (float(rope_wavelengths[0]), float(rope_wavelengths[1]))
        )
        if self.rope_wavelengths is not None:
            assert self.dh % 4 == 0, f"cascade RoPE needs a head dimension divisible by 4; got {self.dh}"
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.o_proj = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.zeros_(self.o_proj.weight)  # identity start

    def forward(self, query, source, mask=None, coords_q=None, coords_src=None, bias=None):
        """
        coords_q (N, 2), coords_src (N, S, 2): (y, x) positions of the target cells and of
        their sources, used only if this attention was built with rope_wavelengths.
        bias (N, S) or None: added to the attention scores (same for all heads).
        """
        B, N, q, D = query.shape
        S = source.shape[2]
        # fold the per-cell query count q into the batch of independent attentions
        Q = self.q_proj(query).reshape(B, N, q, self.h, self.dh)
        K = self.k_proj(source).reshape(B, N, S, q, self.h, self.dh)
        V = self.v_proj(source).reshape(B, N, S, q, self.h, self.dh)
        if self.rope_wavelengths is not None:
            if coords_q is None or coords_src is None:
                raise ValueError("cascade RoPE: coords_q and coords_src are required")
            # rotate Q by the target position and K by the source position: the score then
            # depends only on the offset target - source. Angles in float32.
            with torch.no_grad(), torch.autocast(device_type=Q.device.type, enabled=False):
                freqs = rope_2d_wavelength_freqs(
                    self.dh, self.h, self.rope_wavelengths[0], self.rope_wavelengths[1],
                    device=Q.device,
                )
                cq, sq = rotary_cos_sin_2d_wavelengths(coords_q.float(), freqs)   # (N, h, dh/2)
                ck, sk = rotary_cos_sin_2d_wavelengths(coords_src.float().reshape(-1, 2), freqs)
                half = self.dh // 2
                cq, sq = cq.view(1, N, 1, self.h, half), sq.view(1, N, 1, self.h, half)
                ck = ck.view(1, N, S, 1, self.h, half)
                sk = sk.view(1, N, S, 1, self.h, half)
            Q = apply_rotary_2d_single(Q, cq, sq)
            K = apply_rotary_2d_single(K, ck, sk)
        # attention over the S neighbour axis, per (cell, query, head)
        # Q: (B,N,q,h,dh) ; K,V: (B,N,S,q,h,dh)
        scores = torch.einsum("bnqhd,bnsqhd->bnqhs", Q, K) / (self.dh ** 0.5)
        if bias is not None:
            scores = scores + bias.to(scores.dtype)[None, :, None, None, :]
        row_all_masked = None
        if mask is not None:
            # mask: (B,N,S) -> (B,N,1,1,S)
            neg = torch.finfo(scores.dtype).min
            m = mask[:, :, None, None, :]
            scores = scores.masked_fill(~m, neg)
            # A cell whose neighbour set is ENTIRELY invalid (e.g. a coarse cell
            # with no active children -- here 354 of 467 coarse cells) has an
            # all-masked score row. softmax over all -inf yields NaN, which then
            # poisons gradients and (under FSDP) can stall the collective reduce
            # into a watchdog hang rather than a clean error. Detect those rows so
            # we can zero their attention output below instead of producing NaN.
            # row_all_masked: (B,N,q,h) True where every neighbour was masked out.
            row_all_masked = (~mask).all(dim=2)[:, :, None, None].expand(
                scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[3]
            )
        attn = torch.softmax(scores, dim=-1)
        if row_all_masked is not None:
            # replace NaN softmax rows with 0 attention (cell receives no update).
            attn = torch.where(row_all_masked.unsqueeze(-1), torch.zeros_like(attn), attn)
        out = torch.einsum("bnqhs,bnsqhd->bnqhd", attn, V).reshape(B, N, q, D)
        return self.o_proj(out)


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
        num_heads: int = 8,
        down_ring: bool = False,
        down_rope: bool = False,
        down_rope_wavelengths=(0.25, 8.0),
        down_sigma: float = 0.0,
    ):
        super().__init__()
        self.down_sigma = float(down_sigma or 0.0) if operator == "attention" else 0.0
        # down_ring / down_rope: see the module docstring. Attention operator only.
        self.down_ring = bool(down_ring) and operator == "attention"
        self.down_rope = bool(down_rope) and operator == "attention"
        if (down_ring or down_rope) and operator != "attention":
            _logger.warning(
                "latent cascade: down_ring / down_rope need operator 'attention'; ignored "
                f"for operator '{operator}'"
            )
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
            self._ops[key] = _PairOperator(
                dim, operator, num_heads=num_heads,
                down_rope_wavelengths=down_rope_wavelengths if self.down_rope else None,
            )
            poc = self.pyramid.parent_of_child(lc, lf)        # (n_fine,) coarse idx or -1
            cop = self.pyramid.child_of_parent(lc, lf)        # (n_coarse, K) fine idx or -1
            cv = self.pyramid.child_valid(lc, lf)             # (n_coarse, K) bool
            self.register_buffer(f"poc_{key}", torch.as_tensor(poc, dtype=torch.long), persistent=False)
            self.register_buffer(f"cop_{key}", torch.as_tensor(cop, dtype=torch.long), persistent=False)
            self.register_buffer(f"cv_{key}", torch.as_tensor(cv, dtype=torch.bool), persistent=False)

            if self.down_ring or self.down_rope or self.down_sigma > 0.0:
                poc_t = torch.as_tensor(poc, dtype=torch.long)
                has_parent = poc_t >= 0
                pidx = poc_t.clamp(min=0)
                # sources of every fine cell: its parent (column 0) and, with down_ring, the
                # parent's 8 neighbours. Neighbours missing from the domain are returned as
                # the parent itself (Domain.neighbours_compact convention): mask them, so
                # that the parent is not counted several times.
                if self.down_ring:
                    nb = torch.as_tensor(
                        self.pyramid.domain(lc).neighbours_compact(), dtype=torch.long
                    )                                                       # (n_coarse, 8)
                    ring = torch.cat(
                        [torch.arange(nb.shape[0]).view(-1, 1), nb], dim=1
                    )                                                       # (n_coarse, 9)
                    distinct = torch.ones_like(ring, dtype=torch.bool)
                    distinct[:, 1:] = nb != torch.arange(nb.shape[0]).view(-1, 1)
                    src = ring[pidx]                                        # (n_fine, 9)
                    src_valid = distinct[pidx] & has_parent.view(-1, 1)
                else:
                    src = pidx.view(-1, 1)                                  # (n_fine, 1)
                    src_valid = has_parent.view(-1, 1)
                self.register_buffer(f"dsrc_{key}", src, persistent=False)
                self.register_buffer(f"dsrcv_{key}", src_valid, persistent=False)
                if self.down_rope or self.down_sigma > 0.0:
                    # cell-centre positions on one flat map of the fine domain, in units
                    # of COARSE cells (a fine cell is at most ~3 coarse cells from a source)
                    from weathergen.datasets.utils import healpix_verts_rots

                    dom_f, dom_c = self.pyramid.domain(lf), self.pyramid.domain(lc)
                    frame = dec_rope_frame(lf, dom_f.active_cells)
                    scale = 2.0 ** (int(lf) - int(lc))                      # fine cells per coarse cell
                    vf, _ = healpix_verts_rots(lf, 0.5, 0.5)
                    vc, _ = healpix_verts_rots(lc, 0.5, 0.5)
                    xy_f = dec_rope_project(
                        vf[torch.from_numpy(np.asarray(dom_f.active_cells))], frame
                    ) / scale
                    xy_c = dec_rope_project(
                        vc[torch.from_numpy(np.asarray(dom_c.active_cells))], frame
                    ) / scale
                    self.register_buffer(f"dxyq_{key}", xy_f, persistent=False)        # (n_fine, 2)
                    self.register_buffer(f"dxys_{key}", xy_c[src], persistent=False)   # (n_fine, S, 2)
                    dist = (xy_c[src] - xy_f.unsqueeze(1)).norm(dim=-1)                # (n_fine, S)
                    if self.down_sigma > 0.0:
                        self.register_buffer(
                            f"dbias_{key}", -(dist**2) / (2.0 * self.down_sigma**2),
                            persistent=False,
                        )
                    off = dist[src_valid]
                    _logger.info(
                        f"latent cascade {lc}->{lf}: down_ring={self.down_ring} "
                        f"down_rope={self.down_rope} down_sigma={self.down_sigma:g}; "
                        f"{src.shape[1]} source(s) per fine cell, "
                        f"{int(src_valid.sum())} valid; distance fine cell to source "
                        f"{off.min():.2f} .. {off.max():.2f} coarse cells; wavelengths "
                        f"{down_rope_wavelengths[0]:g} .. {down_rope_wavelengths[1]:g}"
                    )

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
#            print(f"[cascade] val={not self.training} lvl={lvl} latent.shape={tuple(lat.shape)}", flush=True)
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
                valid = (poc >= 0) & (poc < coarse.shape[1])
                idx = poc.clamp(min=0, max=coarse.shape[1] - 1)
                parent = coarse[:, idx, :, :]
                parent = parent * valid.view(1, -1, 1, 1).to(parent.dtype)
                if op.reduces_source:
                    # none/linear: pass the (validity-zeroed) parent directly.
                    parent = parent * valid.view(1, -1, 1, 1).to(parent.dtype)
                    cells[lf] = cells[lf] + op.down(cells[lf], parent)
                elif self.down_ring or self.down_rope or self.down_sigma > 0.0:
                    # attention over the parent (and, with down_ring, its 8 neighbours),
                    # optionally with the fine-to-coarse offsets (down_rope)
                    sidx = getattr(self, f"dsrc_{key}")        # (n_fine, S)
                    sval = getattr(self, f"dsrcv_{key}")       # (n_fine, S)
                    sval = sval & (sidx < coarse.shape[1])
                    sidx = sidx.clamp(max=coarse.shape[1] - 1)
                    Bc, _, qc, Dc = coarse.shape
                    src = coarse[:, sidx.reshape(-1), :, :].reshape(
                        Bc, sidx.shape[0], sidx.shape[1], qc, Dc
                    )                                          # (B, n_fine, S, q, D)
                    m = sval.unsqueeze(0).expand(Bc, -1, -1)
                    cq = getattr(self, f"dxyq_{key}") if self.down_rope else None
                    cs = getattr(self, f"dxys_{key}") if self.down_rope else None
                    bias = getattr(self, f"dbias_{key}") if self.down_sigma > 0.0 else None
                    cells[lf] = cells[lf] + op.down(cells[lf], src, m, cq, cs, bias)
                else:
                    # attention: query = fine cell's own latent; source = its single
                    # parent as a 1-neighbour set; mask out unpaired fine cells.
                    src = parent.unsqueeze(2)                  # (B, n_fine, 1, q, D)
                    m = valid.view(1, -1, 1).expand(parent.shape[0], -1, 1)
                    cells[lf] = cells[lf] + op.down(cells[lf], src, m)

            # UP: fine -> coarse, from finest pair back to coarsest
            for (lc, lf) in reversed(self.pairs):
                key = f"{lc}_{lf}"
                op = self._ops[key]
                cop = getattr(self, f"cop_{key}")             # (n_coarse, K)
                cv = getattr(self, f"cv_{key}")               # (n_coarse, K)   <-- must exist first
                fine = cells[lf]                              # (B, n_fine, q, D)
                B, _, q, D = fine.shape
                n_coarse = cop.shape[0]
                K = cop.shape[1]
                cv = cv & (cop < fine.shape[1])               # now safe: cv already bound
                idx = cop.clamp(min=0, max=fine.shape[1] - 1)
                gathered = fine[:, idx.reshape(-1), :, :].reshape(
                    B, n_coarse, K, q, D
                )                                             # (B, n_coarse, K, q, D)
                if op.reduces_source:
                    maskf = cv.view(1, n_coarse, K, 1, 1).to(gathered.dtype)
                    summed = (gathered * maskf).sum(dim=2)
                    count = cv.sum(dim=1).clamp(min=1).view(1, -1, 1, 1).to(gathered.dtype)
                    pooled = summed / count                   # mean over active children
                    cells[lc] = cells[lc] + op.up(cells[lc], pooled)
                else:
                    # attention: query = coarse cell; source = its K children set;
                    # mask out inactive children.
                    m = cv.view(1, n_coarse, K).expand(B, -1, -1)
                    cells[lc] = cells[lc] + op.up(cells[lc], gathered, m)

        # re-merge aux + updated cells
        out = {}
        for lvl in tokens_by_level:
            out[lvl] = self._merge(aux[lvl], cells[lvl])
        return out
