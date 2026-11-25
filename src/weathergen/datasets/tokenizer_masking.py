# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.batch import SampleMetaData
from weathergen.datasets.masking import Masker
from weathergen.datasets.tokenizer import Tokenizer
from weathergen.datasets.tokenizer_utils import (
    encode_times_source,
    encode_times_target,
    tokenize_apply_mask_source,
    tokenize_apply_mask_target,
    tokenize_space,
    tokenize_spacetime,
)


_logger = logging.getLogger(__name__)
_n_keep_one = 0


def keep_one_source_cell(cell_mask, idxs_cells_lens, rng=None, stream_name="?"):
    """
    Source mask safeguard for random (cell) masking.

    If the mask keeps some cells (cell_mask.any()) but none of the cells that hold tokens,
    the stream would have no source tokens on this rank: its embedding is skipped on this
    rank only, the FSDP collectives get out of step and training deadlocks (NCCL watchdog
    timeout). This happens e.g. for spoofed windows (2 points in 2 cells, MEPS before
    2020) or sparse streams (few stations / a small satellite swath) when part of the
    cells is masked. Then one random cell with tokens is kept.

    A mask that keeps no cell at all (diagnostic stream: no source by design) and windows
    without tokens are returned unchanged. Returns the (possibly new) mask; the input mask
    is not modified.
    """
    global _n_keep_one
    if cell_mask is None or idxs_cells_lens is None:
        return cell_mask
    km = np.asarray(cell_mask.cpu() if isinstance(cell_mask, torch.Tensor) else cell_mask,
                    dtype=bool)
    if not km.any():
        return cell_mask
    has_tok = np.array([len(lens) > 0 for lens in idxs_cells_lens], dtype=bool)
    if not has_tok.any() or (km & has_tok).any():
        return cell_mask
    cand = np.flatnonzero(has_tok)
    # the Masker's generator (np.random.Generator) when available, for reproducibility
    pick = cand[rng.integers(len(cand))] if rng is not None else cand[np.random.randint(len(cand))]
    _n_keep_one += 1
    if _n_keep_one in (1, 10, 100, 1000) or _n_keep_one % 10000 == 0:
        _logger.info(
            "%s: random source mask kept no cell with data -> kept 1 of %d cells "
            "(%d times so far in this process)", stream_name, len(cand), _n_keep_one,
        )
    if isinstance(cell_mask, torch.Tensor):
        new = cell_mask.clone()
        new[int(pick)] = True
    else:
        new = km.copy()
        new[pick] = True
    return new


def readerdata_to_torch(rdata: IOReaderData) -> IOReaderData:
    """
    Convert data, coords, and geoinfos to torch tensor
    """
    if type(rdata.coords) is not torch.Tensor:
        rdata.coords = torch.tensor(rdata.coords)
    if type(rdata.geoinfos) is not torch.Tensor:
        rdata.geoinfos = torch.tensor(rdata.geoinfos)
    if type(rdata.data) is not torch.Tensor:
        rdata.data = torch.tensor(rdata.data)

    return rdata


class TokenizerMasking(Tokenizer):
    def __init__(
        self, healpix_level: int, masker: Masker, domain=None, normalize_lcoords: bool = True,
        normalize_source_lcoords: bool = False,
    ):
        super().__init__(
            healpix_level, domain, normalize_lcoords=normalize_lcoords,
            normalize_source_lcoords=normalize_source_lcoords,
        )
        self.masker = masker
        self.rng = None
        self.token_size = None

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.masker.reset_rng(rng)
        self.rng = rng

    def get_tokens_windows(self, stream_info, data, pad_tokens):
        """
        Tokenize data (to amortize over the different views that are generated)

        """

        tok_spacetime = stream_info.get("tokenize_spacetime", False)
        tok = tokenize_spacetime if tok_spacetime else tokenize_space
        hl = self.healpix_level
        token_size = stream_info["token_size"]

        tokens = []
        for rdata in data:
            # skip empty data
            if rdata.is_empty():
                tokens += [(None, None)]
                continue
            # tokenize data
            idxs_cells, idxs_cells_lens = tok(
                readerdata_to_torch(rdata), token_size, hl, pad_tokens, domain=self.domain
            )
            tokens += [(idxs_cells, idxs_cells_lens)]

        return tokens

    def build_samples_for_stream(
        self,
        training_mode: str,
        num_cells: int,
        stream_info: dict,
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Create masks for samples
        """
        return self.masker.build_samples_for_stream(training_mode, num_cells, stream_info)

    def cell_to_token_mask(self, idxs_cells, idxs_cells_lens, mask):
        """ """

        mask_tokens, mask_channels = None, None
        num_tokens = torch.tensor([len(t) for t in idxs_cells_lens]).sum().item()

        # If there are no tokens, return empty lists.
        if num_tokens == 0:
            return (mask_tokens, mask_channels)

        # TODO, TODO, TODO: use np.repeat
        # https://stackoverflow.com/questions/26038778/repeat-each-values-of-an-array-different-times
        # build token level mask: for each cell replicate the keep flag across its tokens
        token_level_flags: list[np.typing.NDArray] = []
        for km, lens_cell in zip(mask, idxs_cells_lens, strict=True):
            num_tokens_cell = len(lens_cell)
            if num_tokens_cell == 0:
                continue
            token_level_flags.append(
                np.ones(num_tokens_cell, dtype=bool)
                if km
                else np.zeros(num_tokens_cell, dtype=bool)
            )
        if token_level_flags:
            mask_tokens = np.concatenate(token_level_flags)
        else:
            mask_tokens = np.array([], dtype=bool)

        return (mask_tokens, mask_channels)

    def get_source(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        idxs_cells_data,
        time_win: tuple,
        cell_mask: torch.Tensor,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = idxs_cells_data

        # a random source mask must not remove ALL cells that hold tokens (see
        # keep_one_source_cell): the stream would be skipped on this rank only -> deadlock
        cell_mask = keep_one_source_cell(
            cell_mask, idxs_cells_lens, getattr(self.masker, "rng", None),
            stream_info.get("name", "?"),
        )

        # select strategy from XXX depending on stream and if student or teacher

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        source_tokens_cells, source_tokens_lens = tokenize_apply_mask_source(
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            stream_info["stream_id"],
            rdata,
            time_win,
            self.hpy_verts_rots_source[-1],
            encode_times_source,
            coords_local_scale=self.source_lcoords_scale,
        )

        return (source_tokens_cells, source_tokens_lens)

    def get_target_coords(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        # TODO: split up
        _, _, _, coords_local, coords_per_cell = tokenize_apply_mask_target(
            stream_info["stream_id"],
            self.hl_target,
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            rdata,
            time_win,
            self.hpy_verts_rots_target,
            self.hpy_verts_local_target,
            self.hpy_nctrs_target,
            encode_times_target,
            verts_diff_norm_factors=self.hpy_verts_diff_norm_factors,
        )

        return (coords_local, coords_per_cell)

    def get_target_values(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        data, datetimes, coords, _, _ = tokenize_apply_mask_target(
            stream_info["stream_id"],
            self.hl_target,
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            rdata,
            time_win,
            self.hpy_verts_rots_target,
            self.hpy_verts_local_target,
            self.hpy_nctrs_target,
            encode_times_target,
            verts_diff_norm_factors=self.hpy_verts_diff_norm_factors,
        )

        idxs_ord_inv = None
        if data.numel() > 0:
            # flatten per-token indices into one flat list
            idxs_flat = torch.cat([idxs for idxs_cell in idxs_cells for idxs in idxs_cell])
            # compute indices for inversion
            _, idxs_ord_inv = torch.sort(idxs_flat)

        return (data, datetimes, coords, idxs_ord_inv)
