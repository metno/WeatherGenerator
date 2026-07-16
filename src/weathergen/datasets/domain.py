# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Regional HEALPix subdomain support.

The rest of the code base addresses healpix cells by their *global nested index*
in [0, 12 * 4**hl). When a regional bounding box is configured, only a subset of
those cells is active, and we want every per-cell array (masks, token lists,
pe_global, neighbour tables, ...) to have length `num_active` instead of
12 * 4**hl.

This module provides the single source of truth for that mapping:

    global nested index  --(to_compact)-->  compact index in [0, num_active)
    compact index        --(active_cells)-> global nested index

Design notes
------------
* `Domain` is deliberately a plain, picklable, torch-free dataclass-like object.
  It is constructed once in the sampler and in ModelParams, and it must survive
  being sent to dataloader worker processes (multiprocessing_method: "fork" or
  "spawn"), so it holds only numpy arrays and scalars.

* The global domain is a first-class case, not a special case: `Domain.global_(hl)`
  returns a domain where `to_compact` is the identity. This is what makes the
  change safe -- with no bbox in the config you get bit-identical behaviour to
  the current code, and that is directly testable (see the note at the bottom of
  the change-set document).

* Cell selection is by cell *centre* by default. A cell whose centre lies just
  outside the box but which still contains observations inside the box would then
  be dropped, and those observations are silently discarded. Use `pad_rings` to
  grow the domain by N rings of neighbours if you want a margin; 1 ring is a
  sensible default for a downscaling setup where the boundary matters.
"""

import logging

import astropy_healpix as hp
import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


class Domain:
    """
    A set of active healpix cells at a fixed healpix level.

    Attributes
    ----------
    healpix_level : int
        The healpix level (unchanged from cf.healpix_level).
    active_cells : NDArray[np.int64]
        Sorted global nested indices of the active cells. Shape (num_active,).
        `active_cells[c]` is the global index of compact index `c`.
    to_compact : NDArray[np.int64]
        Lookup of length 12 * 4**hl. `to_compact[g]` is the compact index of
        global index `g`, or -1 if `g` is outside the domain.
    is_global : bool
        True if the domain covers the whole sphere (then to_compact is identity).
    bbox : tuple | None
        (lon_min, lon_max, lat_min, lat_max) in degrees, or None if global.
    """

    def __init__(
        self,
        healpix_level: int,
        active_cells: NDArray[np.int64],
        is_global: bool = False,
        bbox: tuple | None = None,
    ) -> None:
        num_total = 12 * 4**healpix_level

        self.healpix_level = healpix_level
        self.active_cells = np.asarray(active_cells, dtype=np.int64)
        self.is_global = is_global
        self.bbox = bbox

        assert self.active_cells.ndim == 1, "active_cells must be 1D"
        assert len(self.active_cells) > 0, (
            "Domain is empty: no healpix cells fall inside the configured bounding box. "
            "Either the box is too small for this healpix level, or lat/lon are swapped."
        )
        assert np.all(np.diff(self.active_cells) > 0), (
            "active_cells must be sorted and unique"
        )
        assert self.active_cells[0] >= 0 and self.active_cells[-1] < num_total, (
            "active_cells out of range for this healpix level"
        )

        self.to_compact = np.full(num_total, -1, dtype=np.int64)
        self.to_compact[self.active_cells] = np.arange(len(self.active_cells), dtype=np.int64)

    def __len__(self) -> int:
        """Number of active cells. This replaces `12 * 4**healpix_level` everywhere."""
        return len(self.active_cells)

    @property
    def num_cells(self) -> int:
        """Alias for len(self), for call sites that read better with a name."""
        return len(self.active_cells)

    @property
    def num_total_cells(self) -> int:
        """Number of cells the full sphere would have at this level."""
        return 12 * 4**self.healpix_level

    # -----------------------------------------------------------------------
    # constructors
    # -----------------------------------------------------------------------

    @classmethod
    def global_(cls, healpix_level: int) -> "Domain":
        """The whole sphere. to_compact is the identity; behaviour is unchanged."""
        num_total = 12 * 4**healpix_level
        return cls(
            healpix_level,
            np.arange(num_total, dtype=np.int64),
            is_global=True,
            bbox=None,
        )

    @classmethod
    def from_bbox(
        cls,
        healpix_level: int,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        pad_rings: int = 0,
    ) -> "Domain":
        """
        Build a domain from a lon/lat bounding box in degrees.

        Parameters
        ----------
        lon_min, lon_max :
            Longitude bounds in degrees, in [-180, 180]. Wrap-around is supported:
            if lon_min > lon_max the box is taken to cross the antimeridian, e.g.
            lon_min=170, lon_max=-170 selects a 20-degree band around 180.
        lat_min, lat_max :
            Latitude bounds in degrees, in [-90, 90]. Must have lat_min < lat_max.
        pad_rings :
            Grow the selected set by this many rings of healpix neighbours. Use >=1
            if you want cells that straddle the boundary to be retained.
        """

        assert lat_min < lat_max, f"lat_min must be < lat_max, got {lat_min}, {lat_max}"
        assert -90.0 <= lat_min and lat_max <= 90.0, "latitudes must be in [-90, 90]"
        assert -180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0, (
            "longitudes must be in [-180, 180]"
        )

        nside = 2**healpix_level
        num_total = 12 * 4**healpix_level
        # Determine each cell's centre by INVERTING the codebase's own convention,
        # so that the cells selected here are exactly the cells ang2pix will assign
        # data to in hpy_cell_splits(). That convention (theta_phi_to_standard_coords)
        # is phi = ((lon_deg + 180)/360) * 2*pi, i.e. it shifts longitude by +180 deg
        # before calling ang2pix. Selecting cells by true geographic longitude instead
        # picks a set disjoint from where the data actually lands.
        from astropy_healpix.healpy import pix2ang

        theta, phi = pix2ang(nside, np.arange(num_total), nest=True)
        lats_deg = 90.0 - np.degrees(theta)
        # invert phi = ((lon + 180)/360)*2*pi  ->  lon = degrees(phi) - 180
        lons_deg = np.degrees(phi) - 180.0

        in_lat = (lats_deg >= lat_min) & (lats_deg <= lat_max)
        if lon_min <= lon_max:
            in_lon = (lons_deg >= lon_min) & (lons_deg <= lon_max)
        else:
            # box crosses the antimeridian
            in_lon = (lons_deg >= lon_min) | (lons_deg <= lon_max)

        selected = np.flatnonzero(in_lat & in_lon).astype(np.int64)

        assert len(selected) > 0, (
            f"No healpix cells at level {healpix_level} fall inside bbox "
            f"lon=[{lon_min}, {lon_max}] lat=[{lat_min}, {lat_max}]. "
            "The box is likely smaller than a single cell, or lat/lon are swapped."
        )

        for _ in range(pad_rings):
            selected = _grow_one_ring(selected, nside)

        selected = np.unique(selected)

        _logger.info(
            "Domain: healpix level %d, bbox lon=[%g, %g] lat=[%g, %g], pad_rings=%d "
            "-> %d of %d cells active (%.2f%% of the sphere)",
            healpix_level,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            pad_rings,
            len(selected),
            num_total,
            100.0 * len(selected) / num_total,
        )

        return cls(
            healpix_level,
            selected,
            is_global=(len(selected) == num_total),
            bbox=(lon_min, lon_max, lat_min, lat_max),
        )

    @classmethod
    def from_config(cls, cf) -> "Domain":
        """
        Build the domain from the top-level config.

        Reads the optional `domain` block:

            domain:
              lon_min: 0.0
              lon_max: 32.0
              lat_min: 53.0
              lat_max: 72.0
              pad_rings: 1

        If the block is absent or `domain.enabled` is False, returns the global
        domain, i.e. exactly the current behaviour.
        """
        dom_cfg = cf.get("domain", None)
        if dom_cfg is None or not dom_cfg.get("enabled", True):
            return cls.global_(cf.healpix_level)

        return cls.from_bbox(
            cf.healpix_level,
            lon_min=dom_cfg["lon_min"],
            lon_max=dom_cfg["lon_max"],
            lat_min=dom_cfg["lat_min"],
            lat_max=dom_cfg["lat_max"],
            pad_rings=dom_cfg.get("pad_rings", 0),
        )

    # -----------------------------------------------------------------------
    # index mapping
    # -----------------------------------------------------------------------

    def remap(self, global_idxs: NDArray) -> NDArray[np.int64]:
        """
        Map global nested indices to compact indices. Out-of-domain -> -1.

        This is the hot path: it is called once per stream per time window on
        the full point set, so it must stay a single vectorised lookup.
        """
        if self.is_global:
            return global_idxs.astype(np.int64, copy=False)
        return self.to_compact[global_idxs]

    def inside_mask(self, global_idxs: NDArray) -> NDArray[np.bool_]:
        """Boolean mask of which points fall in the domain."""
        if self.is_global:
            return np.ones(len(global_idxs), dtype=bool)
        return self.to_compact[global_idxs] >= 0

    def bbox_point_mask(self, lats: NDArray, lons: NDArray) -> NDArray[np.bool_]:
        """
        Cheap pre-filter on raw lat/lon in degrees, for use in the data readers.

        This is NOT the authoritative filter -- the authoritative one is the cell
        remap in the tokenizer, which is what actually decides what the model sees.
        This is only to avoid reading/normalising points that are certainly
        irrelevant. It is deliberately permissive at the boundary: with pad_rings > 0
        the active cell set extends beyond the box, so we widen the box here by a
        generous margin to avoid dropping points the tokenizer would have kept.
        """
        if self.is_global or self.bbox is None:
            return np.ones(len(lats), dtype=bool)

        lon_min, lon_max, lat_min, lat_max = self.bbox

        # cell diagonal at this level, times number of padding rings, plus slack
        cell_deg = 58.6 / (2**self.healpix_level)
        margin = cell_deg * 2.0

        in_lat = (lats >= lat_min - margin) & (lats <= lat_max + margin)
        if lon_min <= lon_max:
            in_lon = (lons >= lon_min - margin) & (lons <= lon_max + margin)
        else:
            in_lon = (lons >= lon_min - margin) | (lons <= lon_max + margin)

        return in_lat & in_lon

    # -----------------------------------------------------------------------
    # geometry, remapped to compact indexing
    # -----------------------------------------------------------------------

    def neighbours_compact(self) -> NDArray[np.int64]:
        """
        The 8-neighbour table of the active cells, in compact indexing.

        Returns shape (num_active, 8). Neighbours that are missing in healpix
        (the -1 returned at the base-pixel corners) OR that fall outside the
        domain are replaced by the cell itself.

        The "replace by self" convention is exactly what the existing code already
        does for missing neighbours (`temp[i][row == -1] = i`), so a domain-edge
        cell behaves like a healpix-corner cell: the readout pools over fewer
        distinct neighbours rather than reading garbage.
        """
        import warnings

        nside = 2**self.healpix_level
        with warnings.catch_warnings(action="ignore"):
            nb = hp.neighbours(self.active_cells, nside, order="nested").transpose()
        # nb has shape (num_active, 8) in GLOBAL indices, with -1 for missing

        nb_compact = np.full(nb.shape, -1, dtype=np.int64)
        valid = nb >= 0
        nb_compact[valid] = self.to_compact[nb[valid]]

        # missing (-1 from healpix) and out-of-domain (-1 from to_compact) both
        # fall back to self
        self_idx = np.arange(len(self), dtype=np.int64)[:, None]
        nb_compact = np.where(nb_compact < 0, self_idx, nb_compact)

        return nb_compact

    def centres_lonlat(self) -> tuple[NDArray, NDArray]:
        """(lons, lats) in degrees of the active cell centres, compact order."""
        from astropy_healpix.healpy import pix2ang

        theta, phi = pix2ang(2**self.healpix_level, self.active_cells, nest=True)
        lats_deg = 90.0 - np.degrees(theta)
        # invert the codebase convention: phi = ((lon + 180)/360)*2*pi
        lons_deg = np.degrees(phi) - 180.0
        return lons_deg, lats_deg

    def __repr__(self) -> str:
        if self.is_global:
            return f"Domain(global, hl={self.healpix_level}, num_cells={len(self)})"
        return (
            f"Domain(bbox={self.bbox}, hl={self.healpix_level}, "
            f"num_cells={len(self)}/{self.num_total_cells})"
        )


def _grow_one_ring(cells: NDArray[np.int64], nside: int) -> NDArray[np.int64]:
    """Add one ring of healpix neighbours around the given cell set."""
    import warnings

    with warnings.catch_warnings(action="ignore"):
        nb = hp.neighbours(cells, nside, order="nested")
    nb = nb[nb >= 0]
    return np.unique(np.concatenate([cells, nb.astype(np.int64)]))
