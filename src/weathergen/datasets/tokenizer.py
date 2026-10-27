# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings

import astropy_healpix as hp
import numpy as np
import torch

from weathergen.datasets.utils import (
    healpix_verts_rots,
    r3tos2,
)

def _subset(t: torch.Tensor, cells) -> torch.Tensor:
    """Take rows of a per-cell table for the active cells (identity if global)."""
    return t[torch.from_numpy(cells)]

class Tokenizer:
    """
    Base class for tokenizers.
    """

    def __init__(self, healpix_level: int, domain=None, normalize_lcoords: bool = True):
        ref = torch.tensor([1.0, 0.0, 0.0])

        if domain is None:
            from weathergen.datasets.domain import Domain
            domain = Domain.global_(healpix_level)
        self.domain = domain

        self.healpix_level = healpix_level
        self.hl_source = healpix_level
        self.hl_target = healpix_level

        self.num_healpix_cells_source = len(domain)
        self.num_healpix_cells_target = len(domain)

        self.size_time_embedding = 6

        cells = domain.active_cells

        ###########################################################################
        # source
        ###########################################################################
        verts00, verts00_rots = healpix_verts_rots(self.hl_source, 0.0, 0.0)
        verts10, verts10_rots = healpix_verts_rots(self.hl_source, 1.0, 0.0)
        verts11, verts11_rots = healpix_verts_rots(self.hl_source, 1.0, 1.0)
        verts01, verts01_rots = healpix_verts_rots(self.hl_source, 0.0, 1.0)
        vertsmm, vertsmm_rots = healpix_verts_rots(self.hl_source, 0.5, 0.5)

        # keep only active cells
        verts00, verts00_rots = _subset(verts00, cells), _subset(verts00_rots, cells)
        verts10, verts10_rots = _subset(verts10, cells), _subset(verts10_rots, cells)
        verts11, verts11_rots = _subset(verts11, cells), _subset(verts11_rots, cells)
        verts01, verts01_rots = _subset(verts01, cells), _subset(verts01_rots, cells)
        vertsmm, vertsmm_rots = _subset(vertsmm, cells), _subset(vertsmm_rots, cells)

        self.hpy_verts = [
            verts00.to(torch.float32),
            verts10.to(torch.float32),
            verts11.to(torch.float32),
            verts01.to(torch.float32),
            vertsmm.to(torch.float32),
        ]
        self.hpy_verts_rots_source = [
            verts00_rots.to(torch.float32),
            verts10_rots.to(torch.float32),
            verts11_rots.to(torch.float32),
            verts01_rots.to(torch.float32),
            vertsmm_rots.to(torch.float32),
        ]

        ###########################################################################
        # target
        ###########################################################################
        # Vertices and rotations are first built on the FULL healpix grid so that the
        # normalization statistics of the local relative coordinates are computed over
        # all healpix cells (and therefore do not depend on the active domain). The
        # tables are subset to the active cells afterwards.
        verts00, verts00_rots = healpix_verts_rots(self.hl_target, 0.0, 0.0)
        verts10, verts10_rots = healpix_verts_rots(self.hl_target, 1.0, 0.0)
        verts11, verts11_rots = healpix_verts_rots(self.hl_target, 1.0, 1.0)
        verts01, verts01_rots = healpix_verts_rots(self.hl_target, 0.0, 1.0)
        vertsmm, vertsmm_rots = healpix_verts_rots(self.hl_target, 0.5, 0.5)

        # relative position of the cell vertices w.r.t. each reference vertex:
        # rotate the reference vertex to `ref` and take ref - rotated(other vertices)
        transforms = [
            ([verts10, verts11, verts01, vertsmm], verts00_rots),
            ([verts00, verts11, verts01, vertsmm], verts10_rots),
            ([verts00, verts10, verts01, vertsmm], verts11_rots),
            ([verts00, verts11, verts10, vertsmm], verts01_rots),
            ([verts00, verts10, verts11, verts01], vertsmm_rots),
        ]

        # The full-grid table is never materialized as a whole (at hl8 it would be
        # ~786k x 5 x 4 x 3 float64, several hundred MB per tokenizer, and one tokenizer
        # is built per latent level): statistics are accumulated per transform in
        # float64 and only the active-cell subset is kept.
        cells_t = torch.from_numpy(cells)
        stat_n = 0
        stat_sum = torch.zeros(3, dtype=torch.float64)
        stat_sumsq = torch.zeros(3, dtype=torch.float64)
        verts_local = []
        for _verts, rot in transforms:
            # shape: <4, healpix, 3> -> <healpix, 4, 3>
            verts = torch.stack(_verts).transpose(0, 1)
            # Batch multiplication by the 3x3 rotation matrices.
            # shape: <healpix, 3, 3> @ <healpix, 3, 4> -> <healpix, 3, 4> -> <healpix, 4, 3>
            t1 = torch.bmm(rot, verts.transpose(-1, -2)).transpose(-2, -1)
            # shape: <healpix, 4, 3>
            t2 = (ref - t1).to(torch.float64)
            t2_flat = t2.reshape(-1, 3)
            stat_n += t2_flat.shape[0]
            stat_sum += t2_flat.sum(0)
            stat_sumsq += (t2_flat * t2_flat).sum(0)
            # keep only active cells
            verts_local.append(t2[cells_t].to(torch.float32))
            del verts, t1, t2, t2_flat

        # z-score normalization factors (per xyz component), computed over the relative
        # positions of all vertices of all healpix cells, considering all rotations
        mean = stat_sum / stat_n
        var = (stat_sumsq / stat_n - mean * mean).clamp_min(0.0)
        self.hpy_verts_diff_mean = mean.to(torch.float32)
        self.hpy_verts_diff_std = var.sqrt().clamp_min(1e-12).to(torch.float32)
        # passed to tokenize_apply_mask_target(); None disables the normalization
        self.hpy_verts_diff_norm_factors = (
            (self.hpy_verts_diff_mean, self.hpy_verts_diff_std) if normalize_lcoords else None
        )

        # shape: <healpix_active, 5 (reference vertex), 4 * 3>
        # (the table is kept un-normalized; normalization is applied in
        #  get_target_coords_local together with the other local coords)
        self.hpy_verts_local_target = torch.stack(verts_local).transpose(0, 1).flatten(2, 3)
        self.verts_local = list(self.hpy_verts_local_target.transpose(0, 1))

        # keep only active cells
        verts00, verts00_rots = _subset(verts00, cells), _subset(verts00_rots, cells)
        verts10, verts10_rots = _subset(verts10, cells), _subset(verts10_rots, cells)
        verts11, verts11_rots = _subset(verts11, cells), _subset(verts11_rots, cells)
        verts01, verts01_rots = _subset(verts01, cells), _subset(verts01_rots, cells)
        vertsmm, vertsmm_rots = _subset(vertsmm, cells), _subset(vertsmm_rots, cells)

        self.hpy_verts = [
            verts00.to(torch.float32),
            verts10.to(torch.float32),
            verts11.to(torch.float32),
            verts01.to(torch.float32),
            vertsmm.to(torch.float32),
        ]
        self.hpy_verts_rots_target = [
            verts00_rots.to(torch.float32),
            verts10_rots.to(torch.float32),
            verts11_rots.to(torch.float32),
            verts01_rots.to(torch.float32),
            vertsmm_rots.to(torch.float32),
        ]

        # add local coords wrt to center of neighboring cells
        # (since the neighbors are used in the prediction)
        # Domain.neighbours_compact() already: computes global neighbours, remaps to
        # compact indexing, and falls back to self for both healpix-corner (-1) and
        # out-of-domain neighbours -- i.e. the same convention as the loop it replaces.
        num_healpix_cells = len(domain)
        temp = domain.neighbours_compact()
        self.hpy_nctrs_target = (
            vertsmm[temp.flatten()]                # vertsmm is already subset above
            .reshape((num_healpix_cells, 8, 3))
            .transpose(1, 0)
            .to(torch.float32)
        )

    def compute_source_centroids(self, source_tokens_cells: list[torch.Tensor]) -> torch.Tensor:
        source_means = [
            (
                self.hpy_verts[-1][i].unsqueeze(0).repeat(len(s), 1)
                if len(s) > 0
                else torch.tensor([])
            )
            for i, s in enumerate(source_tokens_cells)
        ]
        source_means_lens = [len(s) for s in source_means]
        # merge and split to vectorize computations
        source_means = torch.cat(source_means)
        # TODO: precompute also source_means_r3 and then just cat
        source_centroids = torch.cat(
            [source_means.to(torch.float32), r3tos2(source_means).to(torch.float32)], -1
        )
        source_centroids = torch.split(source_centroids, source_means_lens)

        return source_centroids

    def get_size_time_embedding(self) -> int:
        """
        Get size of time embedding
        """
        return self.size_time_embedding
