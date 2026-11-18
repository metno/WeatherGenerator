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

* Alternatively (`type: points`, `points_file: ...`) the domain is the footprint of
  a point set, e.g. a CARRA2/MEPS template: every cell containing at least one point
  is active (plus `pad_rings`). `points_file` can be a `.npz` (`lat`/`lon`), a NetCDF
  file or an anemoi-style zarr; optional `lat_var` / `lon_var` name the variables and
  `cells_sha256` pins the resulting cell set (printed in the log at start-up).
"""

import hashlib
import logging
import os

import astropy_healpix as hp
import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# Domains are built from the config in several places (sampler, model, pyramid, ...);
# cache them so the construction (and its log line, and reading a points file) happens
# once per process.
_DOMAIN_CACHE: dict[tuple, "Domain"] = {}
# coordinates read from a points file, keyed by (abs path, lat_var, lon_var): a file used
# by several latent levels is read once per process
_POINTS_CACHE: dict[tuple, tuple[NDArray, NDArray]] = {}


def _lonlat_to_thetaphi(lats_deg: NDArray, lons_deg: NDArray) -> tuple[NDArray, NDArray]:
    """Same convention as tokenizer_utils.theta_phi_to_standard_coords."""
    thetas = ((90.0 - lats_deg) / 180.0) * np.pi
    phis = ((lons_deg + 180.0) / 360.0) * 2.0 * np.pi
    return thetas, phis


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
        pad_rings: int = 0,
        points_file: str | None = None,
    ) -> None:
        num_total = 12 * 4**healpix_level

        self.healpix_level = healpix_level
        self.active_cells = np.asarray(active_cells, dtype=np.int64)
        self.is_global = is_global
        self.bbox = bbox
        self.pad_rings = pad_rings
        # absolute path of the points file of a footprint domain (type: points), else None
        self.points_file = points_file
        # lazily built lookup for the reader pre-filter of a points domain
        self._prefilter_lookup: NDArray[np.bool_] | None = None

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

    def to_compact_safe(self, global_idxs: NDArray) -> NDArray[np.int64]:
        """
        Bounds-safe global->compact lookup: returns the compact index for each
        global nested index, or -1 if the index is inactive OR out of range.

        Unlike `remap()` (which assumes valid in-range indices and returns the
        identity for a global domain), this never indexes out of bounds and always
        consults `to_compact`, so it is correct for cross-level parent/child maps
        where a computed parent/child index may fall outside this level's active
        set. Works identically for global and regional domains, since `to_compact`
        is a full-length table in both cases (identity for global).
        """
        g = np.asarray(global_idxs, dtype=np.int64)
        out = np.full(g.shape, -1, dtype=np.int64)
        in_range = (g >= 0) & (g < self.num_total_cells)
        out[in_range] = self.to_compact[g[in_range]]
        return out

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
            pad_rings=pad_rings,
        )

    @classmethod
    def from_points(
        cls,
        healpix_level: int,
        lats: NDArray,
        lons: NDArray,
        pad_rings: int = 0,
        points_file: str | None = None,
    ) -> "Domain":
        """
        Build a domain from the footprint of a point set (e.g. a CARRA2/MEPS template):
        every cell containing at least one point (same cell assignment as the
        tokenizer) is active, grown by `pad_rings` rings of neighbours. The domain then
        follows the shape of the grid instead of its lon/lat bounding box.

        Parameters
        ----------
        lats, lons :
            Point coordinates in degrees, any shape (flattened). NaN points are ignored.
        pad_rings :
            Grow the selected set by this many rings of healpix neighbours.
        points_file :
            Where the points came from (for logging and the domain identity only).
        """
        from astropy_healpix.healpy import ang2pix

        assert pad_rings >= 0, "pad_rings must be >= 0"
        lats = np.asarray(lats, dtype=np.float64).ravel()
        lons = np.asarray(lons, dtype=np.float64).ravel()
        assert lats.shape == lons.shape, "lats and lons must have the same number of points"
        finite = np.isfinite(lats) & np.isfinite(lons)
        assert finite.any(), f"No finite lat/lon points to build a domain from ({points_file})"

        nside = 2**healpix_level
        num_total = 12 * 4**healpix_level
        thetas, phis = _lonlat_to_thetaphi(lats[finite], lons[finite])
        selected = np.unique(ang2pix(nside, thetas, phis, nest=True).astype(np.int64))
        num_footprint = len(selected)

        for _ in range(pad_rings):
            selected = _grow_one_ring(selected, nside)

        domain = cls(
            healpix_level,
            selected,
            is_global=(len(selected) == num_total),
            bbox=None,
            pad_rings=pad_rings,
            points_file=points_file,
        )
        _logger.info(
            "Domain: healpix level %d, footprint of %d points from %s, pad_rings=%d "
            "-> %d of %d cells active (%d with points, %.2f%% of the sphere), cells_sha256=%s",
            healpix_level,
            int(finite.sum()),
            points_file,
            pad_rings,
            len(selected),
            num_total,
            num_footprint,
            100.0 * len(selected) / num_total,
            domain.cells_sha256,
        )
        return domain

    @classmethod
    def from_config(cls, cf) -> "Domain":
        """
        Build the domain from the top-level config.

        Reads the optional `domain` block, either a lon/lat box

            domain:
              lon_min: 0.0
              lon_max: 32.0
              lat_min: 53.0
              lat_max: 72.0
              pad_rings: 1

        or the footprint of a point set

            domain:
              type: points                      # optional, inferred from points_file
              points_file: /abs/path/template.npz
              pad_rings: 1

        If the block is absent or `domain.enabled` is False, returns the global
        domain, i.e. exactly the current behaviour.
        """
        return cls.from_level_config(cf.healpix_level, cf.get("domain", None))

    @classmethod
    def from_level_config(cls, healpix_level: int, dom_cfg) -> "Domain":
        """
        Build (or fetch from the per-process cache) the domain at `healpix_level` for a
        `domain` config block. A missing block or `enabled: false` gives the global
        domain. Shared by `from_config` (top-level `domain`) and
        `DomainPyramid.from_config` (one `domain` block per entry of `latent_levels`).
        """
        key = domain_config_key(healpix_level, dom_cfg)

        domain = _DOMAIN_CACHE.get(key)
        if domain is None:
            if key[1] is None:
                domain = cls.global_(key[0])
            elif key[1] == "points":
                _, _, path, pad_rings, lat_var, lon_var = key
                if not os.path.isabs(os.path.expanduser(str(dom_cfg["points_file"]))):
                    _logger.warning(
                        "domain points_file %r is a relative path, resolved to %s from the "
                        "current directory. A continued run or inference started from another "
                        "directory would read a different file or fail: use an absolute path.",
                        str(dom_cfg["points_file"]),
                        path,
                    )
                lats, lons = _load_points(path, lat_var, lon_var)
                domain = cls.from_points(key[0], lats, lons, pad_rings, points_file=path)
            else:
                _, _, lon_min, lon_max, lat_min, lat_max, pad_rings = key
                domain = cls.from_bbox(
                    key[0],
                    lon_min=lon_min,
                    lon_max=lon_max,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    pad_rings=pad_rings,
                )
            _DOMAIN_CACHE[key] = domain

        expected = dom_cfg.get("cells_sha256", None) if dom_cfg is not None else None
        if expected is not None and str(expected) != domain.cells_sha256:
            raise ValueError(
                f"Domain at healpix level {healpix_level} ({domain!r}) has cells_sha256="
                f"{domain.cells_sha256}, but the config pins {expected}. The points file or "
                "the domain settings changed since the value was recorded."
            )
        return domain

    @property
    def cells_sha256(self) -> str:
        """sha256 of the active cells (level + sorted global ids): identifies the domain."""
        h = hashlib.sha256(str(self.healpix_level).encode())
        h.update(self.active_cells.astype("<i8").tobytes())
        return h.hexdigest()

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

        For a points (footprint) domain there is no bbox; the filter is then the cell
        lookup of `_points_prefilter_mask` (active cells plus one ring).

        This is NOT the authoritative filter -- the authoritative one is the cell
        remap in the tokenizer, which is what actually decides what the model sees.
        This is only to avoid reading/normalising points that are certainly
        irrelevant. It is deliberately permissive at the boundary: with pad_rings > 0
        the active cell set extends beyond the box, so we widen the box here by a
        generous margin to avoid dropping points the tokenizer would have kept.
        """
        if self.is_global:
            return np.ones(len(lats), dtype=bool)
        if self.points_file is not None:
            return self._points_prefilter_mask(lats, lons)
        if self.bbox is None:
            return np.ones(len(lats), dtype=bool)

        lon_min, lon_max, lat_min, lat_max = self.bbox

        # cell diagonal at this level, times number of padding rings, plus slack
#        cell_deg = 58.6 / (2**self.healpix_level)
#        margin = cell_deg * 2.0
        cell_deg = 58.6 / (2**self.healpix_level)
        margin = cell_deg * (self.pad_rings + 2)

        in_lat = (lats >= lat_min - margin) & (lats <= lat_max + margin)
        if lon_min <= lon_max:
            in_lon = (lons >= lon_min - margin) & (lons <= lon_max + margin)
        else:
            in_lon = (lons >= lon_min - margin) | (lons <= lon_max + margin)

        return in_lat & in_lon

    def point_mask(self, lats: NDArray, lons: NDArray) -> NDArray[np.bool_]:
        """
        Exact mask of which points (lat/lon in degrees) fall in an ACTIVE cell, using
        the same cell assignment as the tokenizer. Works for bbox and points domains.
        """
        return self._cell_lookup_mask(lats, lons, self.to_compact >= 0)

    def _points_prefilter_mask(self, lats: NDArray, lons: NDArray) -> NDArray[np.bool_]:
        """
        Reader pre-filter for a points (footprint) domain. A footprint has no bbox to
        widen, so instead keep every point whose cell is active OR one ring outside the
        active set. Like the bbox margin this is deliberately permissive (it absorbs
        float32/float64 differences at cell edges); the tokenizer's cell remap stays the
        authoritative filter.
        """
        if self._prefilter_lookup is None:
            grown = _grow_one_ring(self.active_cells, 2**self.healpix_level)
            lookup = np.zeros(self.num_total_cells, dtype=bool)
            lookup[grown] = True
            self._prefilter_lookup = lookup
        return self._cell_lookup_mask(lats, lons, self._prefilter_lookup)

    def _cell_lookup_mask(
        self, lats: NDArray, lons: NDArray, lookup: NDArray[np.bool_]
    ) -> NDArray[np.bool_]:
        from astropy_healpix.healpy import ang2pix

        lats = np.asarray(lats)
        lons = np.asarray(lons)
        mask = np.zeros(len(lats), dtype=bool)
        finite = np.isfinite(lats) & np.isfinite(lons)
        if finite.any():
            thetas, phis = _lonlat_to_thetaphi(
                lats[finite].astype(np.float64), lons[finite].astype(np.float64)
            )
            cells = ang2pix(2**self.healpix_level, thetas, phis, nest=True)
            mask[finite] = lookup[cells]
        return mask

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
        if self.points_file is not None:
            return (
                f"Domain(points={self.points_file}, pad_rings={self.pad_rings}, "
                f"hl={self.healpix_level}, num_cells={len(self)}/{self.num_total_cells})"
            )
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


_BBOX_KEYS = ("lon_min", "lon_max", "lat_min", "lat_max")


def domain_config_key(healpix_level: int, dom_cfg) -> tuple:
    """
    Hashable key of a `domain` config block at `healpix_level`; also says which kind of
    domain it is. Global: (hl, None). Box: (hl, "bbox", lon_min, lon_max, lat_min,
    lat_max, pad_rings). Points: (hl, "points", abs path, pad_rings, lat_var, lon_var).
    """
    hl = int(healpix_level)
    if dom_cfg is None or not dom_cfg.get("enabled", True):
        return (hl, None)

    has_points = dom_cfg.get("points_file", None) is not None
    has_bbox = any(dom_cfg.get(k, None) is not None for k in _BBOX_KEYS)
    kind = dom_cfg.get("type", None)
    if kind is None:
        if has_points and has_bbox:
            raise ValueError(
                "domain: both `points_file` and lon/lat bounds are given; set "
                "`type: points` or `type: bbox` to say which one to use."
            )
        kind = "points" if has_points else "bbox"
    pad_rings = int(dom_cfg.get("pad_rings", 0))

    if kind == "points":
        if not has_points:
            raise ValueError("domain: `type: points` needs a `points_file`.")
        path = os.path.abspath(os.path.expanduser(str(dom_cfg["points_file"])))
        lat_var = dom_cfg.get("lat_var", None)
        lon_var = dom_cfg.get("lon_var", None)
        return (hl, "points", path, pad_rings, lat_var, lon_var)
    if kind == "bbox":
        missing = [k for k in _BBOX_KEYS if dom_cfg.get(k, None) is None]
        if missing:
            raise ValueError(f"domain: `type: bbox` is missing {missing}.")
        return (hl, "bbox") + tuple(float(dom_cfg[k]) for k in _BBOX_KEYS) + (pad_rings,)
    raise ValueError(f"domain: unknown type {kind!r} (use 'bbox' or 'points').")


_LAT_NAMES = ("latitude", "latitudes", "lat", "XLAT")
_LON_NAMES = ("longitude", "longitudes", "lon", "XLONG")


def _pick(names, available, given, what, path):
    if given is not None:
        if given not in available:
            raise KeyError(f"{what} variable {given!r} not found in {path}")
        return given
    for n in names:
        if n in available:
            return n
    raise KeyError(
        f"No {what} variable in {path} (looked for {list(names)}); set `lat_var` / `lon_var`."
    )


def _load_points(path: str, lat_var: str | None = None, lon_var: str | None = None):
    """
    Read point coordinates (degrees, flattened) from a NetCDF file, an anemoi-style
    zarr or a `.npz`. Only the coordinate variables are read.
    """
    key = (path, lat_var, lon_var)
    if key in _POINTS_CACHE:
        return _POINTS_CACHE[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"domain points_file not found: {path}")

    lower = path.lower().rstrip("/")
    if lower.endswith(".npz"):
        with np.load(path) as f:
            vlat = _pick(_LAT_NAMES + ("lats",), f.files, lat_var, "latitude", path)
            vlon = _pick(_LON_NAMES + ("lons",), f.files, lon_var, "longitude", path)
            lats, lons = np.asarray(f[vlat]), np.asarray(f[vlon])
    elif lower.endswith(".zarr") or os.path.isdir(path):
        import zarr

        z = zarr.open(path, mode="r")
        names = list(z.array_keys())
        vlat = _pick(_LAT_NAMES, names, lat_var, "latitude", path)
        vlon = _pick(_LON_NAMES, names, lon_var, "longitude", path)
        lats, lons = np.asarray(z[vlat][:]), np.asarray(z[vlon][:])
    else:  # NetCDF
        try:
            import xarray as xr

            with xr.open_dataset(path) as ds:
                names = list(ds.variables)
                vlat = _pick(_LAT_NAMES, names, lat_var, "latitude", path)
                vlon = _pick(_LON_NAMES, names, lon_var, "longitude", path)
                lats, lons = np.asarray(ds[vlat].values), np.asarray(ds[vlon].values)
                if lats.ndim == 1 and lons.ndim == 1 and ds[vlat].dims != ds[vlon].dims:
                    lons, lats = np.meshgrid(lons, lats)  # regular lat/lon grid
        except ImportError:
            try:
                import netCDF4
            except ImportError as e:
                raise ImportError(
                    f"Reading the domain points_file {path} needs xarray or netCDF4; "
                    "alternatively save the coordinates as a .npz with `lat` and `lon`."
                ) from e
            with netCDF4.Dataset(path) as ds:
                names = list(ds.variables)
                vlat = _pick(_LAT_NAMES, names, lat_var, "latitude", path)
                vlon = _pick(_LON_NAMES, names, lon_var, "longitude", path)
                lats = np.ma.filled(ds[vlat][:].astype(np.float64), np.nan)
                lons = np.ma.filled(ds[vlon][:].astype(np.float64), np.nan)
                if lats.ndim == 1 and lons.ndim == 1 and ds[vlat].dimensions != ds[vlon].dimensions:
                    lons, lats = np.meshgrid(lons, lats)

    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    if lats.shape != lons.shape:
        raise ValueError(
            f"{path}: latitude has {lats.size} values but longitude {lons.size}; they must "
            "describe the same points."
        )
    _POINTS_CACHE[key] = (lats, lons)
    return lats, lons
