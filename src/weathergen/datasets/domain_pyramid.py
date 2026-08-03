# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Multi-resolution HEALPix latent support: a *pyramid* of `Domain` objects.

Background
----------
The single-level `Domain` (see domain.py) is the tested primitive: it owns the
mapping between global nested HEALPix indices and a compact [0, num_active)
indexing for ONE healpix level, optionally cropped to a regional bounding box.

For multi-resolution latents we need SEVERAL such domains at different levels
(e.g. a coarse hl5 domain covering a large area for reanalysis streams, and a
fine hl8 domain covering a small nested area for high-resolution streams), plus
the exact geometric maps that let information move BETWEEN levels:

    coarse compact cell  --(children)-->  fine compact cells   (upsample / down-pass)
    fine   compact cell  --(parent)--->   coarse compact cell  (pool / up-pass)

This module provides `DomainPyramid`, which:

  * holds one `Domain` per level, keyed by healpix level;
  * is torch-free, picklable, and safe to send to dataloader workers (it holds
    only numpy arrays and scalars, exactly like `Domain`);
  * builds the parent<->child compact-index maps between adjacent *configured*
    levels using pure HEALPix NESTED arithmetic (no neighbour search, no
    interpolation);
  * restricts every cross-level map to the OVERLAP of the two domains, so a fine
    domain that is smaller than (and nested inside) the coarse domain is handled
    naturally: cells of the coarse domain with no active fine children, and fine
    cells whose coarse parent is outside the coarse domain, are recorded as
    "unpaired" rather than silently mismapped.

HEALPix nested parent/child arithmetic
--------------------------------------
In the NESTED scheme, refining by one level splits each cell into 4 children
whose indices are the parent index shifted left by 2 bits plus {0,1,2,3}. More
generally, going from level `Lc` to a finer level `Lf` (Lf > Lc) multiplies the
index space by `4**(Lf - Lc)`:

    parent(g_f, Lc, Lf)   = g_f // 4**(Lf - Lc)
    children(g_c, Lc, Lf) = g_c * 4**(Lf - Lc) + [0 .. 4**(Lf - Lc) - 1]

These are exact and vectorised. The whole point of using HEALPix here is that
this makes the cross-level operators index gathers, not interpolations.

Design note: this module deliberately does NOT decide which streams attach to
which level, nor does it do any pooling/upsampling of *data* -- it only owns the
geometry (index maps). The model code consumes `child_of_parent` /
`parent_of_child` to build the actual latent up/down operators. Keeping streams
out of here is what makes adding more streams pure configuration.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from weathergen.datasets.domain import Domain

_logger = logging.getLogger(__name__)


def cells_parent(global_idxs: NDArray, level_from: int, level_to: int) -> NDArray[np.int64]:
    """
    Global nested index of the parent at coarser `level_to` (< level_from).

    Vectorised, exact. `level_to` must be <= `level_from`.
    """
    assert level_to <= level_from, "parent must be at a coarser (<=) level"
    factor = 4 ** (level_from - level_to)
    return np.asarray(global_idxs, dtype=np.int64) // factor


def cells_children(global_idxs: NDArray, level_from: int, level_to: int) -> NDArray[np.int64]:
    """
    Global nested indices of the children at finer `level_to` (> level_from).

    Returns shape (len(global_idxs), 4**(level_to - level_from)). Row i lists the
    children of global_idxs[i], in ascending order.
    """
    assert level_to >= level_from, "children must be at a finer (>=) level"
    n_children = 4 ** (level_to - level_from)
    base = np.asarray(global_idxs, dtype=np.int64)[:, None] * n_children
    return base + np.arange(n_children, dtype=np.int64)[None, :]


class DomainPyramid:
    """
    An ordered set of `Domain` objects at increasing healpix level, with exact
    parent/child compact-index maps between adjacent configured levels.

    Attributes
    ----------
    levels : list[int]
        The configured healpix levels, sorted ascending (coarse -> fine).
    domains : dict[int, Domain]
        `domains[L]` is the `Domain` at level L.
    """

    def __init__(self, domains: dict[int, Domain]) -> None:
        assert len(domains) >= 1, "DomainPyramid needs at least one level"
        self.levels: list[int] = sorted(domains.keys())
        self.domains: dict[int, Domain] = dict(domains)

        # sanity: coarse domain should geometrically contain the finer ones,
        # otherwise the cascade overlap will be surprising. We do not enforce
        # strict containment (disjoint fine nests inside a global coarse domain
        # are legitimate), but we DO check level ordering is unique.
        assert len(set(self.levels)) == len(self.levels), "duplicate levels"

        # Build parent<->child maps between every ADJACENT configured level pair.
        # Keyed by (coarse_level, fine_level).
        self._child_of_parent: dict[tuple[int, int], NDArray] = {}
        self._parent_of_child: dict[tuple[int, int], NDArray] = {}
        self._child_valid: dict[tuple[int, int], NDArray] = {}
        for lc, lf in zip(self.levels[:-1], self.levels[1:], strict=True):
            self._build_level_pair(lc, lf)

    # -----------------------------------------------------------------------
    # constructors
    # -----------------------------------------------------------------------

    @classmethod
    def single(cls, healpix_level: int, domain: Domain | None = None) -> "DomainPyramid":
        """
        A degenerate one-level pyramid. This is the backward-compatibility path:
        with a single level the pyramid is just a thin wrapper around one Domain
        and introduces no cross-level maps, so existing single-latent behaviour
        is reproduced exactly.
        """
        dom = domain if domain is not None else Domain.global_(healpix_level)
        assert dom.healpix_level == healpix_level
        return cls({healpix_level: dom})

    @classmethod
    def from_config(cls, cf) -> "DomainPyramid":
        """
        Build a pyramid from config.

        Reads an optional `latent_levels` block:

            latent_levels:
              - level: 5
                domain: {enabled: True, lon_min: ..., ..., pad_rings: 1}
              - level: 8
                domain: {enabled: True, lon_min: ..., ..., pad_rings: 1}

        If `latent_levels` is absent, falls back to a single-level pyramid at
        `cf.healpix_level` using the existing `Domain.from_config(cf)` -- i.e.
        exactly the current behaviour.
        """
        levels_cfg = cf.get("latent_levels", None)
        if not levels_cfg:
            base = Domain.from_config(cf)
            return cls.single(base.healpix_level, base)

        domains: dict[int, Domain] = {}
        for entry in levels_cfg:
            lvl = int(entry["level"])
            dom_cfg = entry.get("domain", None)
            if dom_cfg is None or not dom_cfg.get("enabled", True):
                domains[lvl] = Domain.global_(lvl)
            else:
                domains[lvl] = Domain.from_bbox(
                    lvl,
                    lon_min=dom_cfg["lon_min"],
                    lon_max=dom_cfg["lon_max"],
                    lat_min=dom_cfg["lat_min"],
                    lat_max=dom_cfg["lat_max"],
                    pad_rings=dom_cfg.get("pad_rings", 0),
                )
        return cls(domains)

    # -----------------------------------------------------------------------
    # level access
    # -----------------------------------------------------------------------

    @property
    def coarsest(self) -> int:
        return self.levels[0]

    @property
    def finest(self) -> int:
        return self.levels[-1]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def is_single(self) -> bool:
        return len(self.levels) == 1

    def domain(self, level: int) -> Domain:
        return self.domains[level]

    def __len__(self) -> int:
        return len(self.levels)

    def __repr__(self) -> str:
        parts = ", ".join(f"hl{L}:{len(self.domains[L])}" for L in self.levels)
        return f"DomainPyramid({parts})"

    # -----------------------------------------------------------------------
    # cross-level maps (built once, in compact indexing)
    # -----------------------------------------------------------------------

    def _build_level_pair(self, lc: int, lf: int) -> None:
        """
        Build the parent<->child compact-index maps for the adjacent pair
        (coarse level lc, fine level lf), restricted to the overlap.

        Produces three arrays, all in COMPACT indexing:

        parent_of_child[(lc, lf)] : shape (num_fine_active,)
            For each active fine compact cell, the compact index of its parent in
            the coarse domain, or -1 if that parent is not active in the coarse
            domain (fine cell has no coarse counterpart -> "unpaired").

        child_of_parent[(lc, lf)] : shape (num_coarse_active, n_children)
            For each active coarse compact cell, the compact indices of its
            children in the fine domain, with -1 where a child is not active in
            the fine domain. n_children = 4**(lf - lc).

        child_valid[(lc, lf)] : shape (num_coarse_active, n_children) bool
            True where the corresponding child_of_parent entry is a real (>=0)
            fine cell. Convenience mask so callers can do masked pooling without
            re-testing for -1.
        """
        dom_c = self.domains[lc]
        dom_f = self.domains[lf]
        n_children = 4 ** (lf - lc)

        # --- parent_of_child (fine -> coarse) ---
        # global parent index of every active fine cell, then map through the
        # coarse domain's bounds-safe global->compact table (-1 if parent inactive).
        fine_global = dom_f.active_cells                       # (num_fine,)
        parent_global = cells_parent(fine_global, lf, lc)      # (num_fine,)
        parent_compact = dom_c.to_compact_safe(parent_global)  # -1 where parent inactive
        self._parent_of_child[(lc, lf)] = parent_compact

        # --- child_of_parent (coarse -> fine) ---
        coarse_global = dom_c.active_cells                          # (num_coarse,)
        children_global = cells_children(coarse_global, lc, lf)     # (num_coarse, n_children)
        # remap every child through the fine domain's global->compact table
        children_compact = dom_f.to_compact_safe(children_global.reshape(-1)).reshape(
            children_global.shape
        )
        child_valid = children_compact >= 0
        self._child_of_parent[(lc, lf)] = children_compact.astype(np.int64)
        self._child_valid[(lc, lf)] = child_valid

        n_paired_fine = int((parent_compact >= 0).sum())
        n_paired_coarse = int(child_valid.any(axis=1).sum())
        _logger.info(
            "DomainPyramid pair hl%d<->hl%d: coarse=%d fine=%d children/parent=%d | "
            "fine cells with active parent: %d/%d | coarse cells with >=1 active child: %d/%d",
            lc, lf, len(dom_c), len(dom_f), n_children,
            n_paired_fine, len(dom_f), n_paired_coarse, len(dom_c),
        )

    def _pair_key(self, lc: int, lf: int) -> tuple[int, int]:
        assert (lc, lf) in self._child_of_parent, (
            f"({lc},{lf}) is not an adjacent configured level pair; "
            f"configured pairs: {list(self._child_of_parent.keys())}"
        )
        return (lc, lf)

    def parent_of_child(self, coarse_level: int, fine_level: int) -> NDArray[np.int64]:
        """(num_fine_active,) compact parent index per fine cell, -1 if unpaired."""
        return self._parent_of_child[self._pair_key(coarse_level, fine_level)]

    def child_of_parent(self, coarse_level: int, fine_level: int) -> NDArray[np.int64]:
        """(num_coarse_active, 4**(lf-lc)) compact child indices, -1 where inactive."""
        return self._child_of_parent[self._pair_key(coarse_level, fine_level)]

    def child_valid(self, coarse_level: int, fine_level: int) -> NDArray[np.bool_]:
        """(num_coarse_active, 4**(lf-lc)) bool mask of valid children."""
        return self._child_valid[self._pair_key(coarse_level, fine_level)]


def build_domain_pyramid(cf) -> DomainPyramid:
    """
    Single entry point for constructing the latent-domain pyramid from config.

    Phase 2 contract
    ----------------
    With no `latent_levels` block in the config, this returns a one-level pyramid
    whose single domain is exactly `Domain.from_config(cf)`. Model code that asks
    for `pyramid.domain(pyramid.finest)` therefore receives the *identical* Domain
    object it would have built directly today -- so introducing the pyramid at the
    construction boundary changes nothing observable until additional levels are
    configured.

    All three model construction sites (ModelParams, Model, EncoderModule) should
    build the pyramid through THIS function so that Step 3+ can add levels in one
    place.
    """
    return DomainPyramid.from_config(cf)


def model_domain(cf) -> Domain:
    """
    The single Domain the current (single-latent) model consumes.

    This is the finest level of the pyramid. With one configured level it is
    `Domain.from_config(cf)` unchanged; once multiple levels exist, the finest
    level is the one the existing single-latent code path corresponds to (the
    fine latent), and coarser levels are added around it by later steps.
    """
    pyr = build_domain_pyramid(cf)
    return pyr.domain(pyr.finest)
