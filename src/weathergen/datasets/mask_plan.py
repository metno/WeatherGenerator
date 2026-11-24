# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Mask plans: student / teacher masks for SSL (JEPA) training on a multi-resolution latent.

The `Masker` draws one mask per stream and per latent level, independently. For JEPA on a
multi-resolution latent this is not what we want: a cell hidden for one stream can be
visible in another one, and the masks of the latent levels are uncorrelated. A *mask plan*
is instead drawn ONCE per sample (data item) and shared by every stream and every latent
level of that sample:

1. draw a strategy from the configured mixture (one strategy = mixture with one entry);
2. build, for every latent level l, the student keep mask K_l (True = visible to the
   student) and the teacher view T_l (True = visible to the teacher, full domain);
3. every stream gets the masks of the level it is encoded at, filtered by its SSL view
   role (stream config key `ssl_view`: both | student | teacher).

A strategy gives one spec per latent level:

    pattern spec (dict)  : {pattern: random | blocks | crop | none,
                            unit_level: u (<= l, default l),
                            hidden: fraction or [min, max] of hidden unit cells,
                            blocks: {...} (pattern blocks only, overrides the defaults)}
                           The pattern selects cells of the *unit* level u; every level-l
                           cell is hidden iff its ancestor at level u is hidden.
    "follow:<c>"         : hidden iff the ancestor at the (coarser) latent level c is hidden
    "visible"            : nothing hidden at this level
    "drop"               : the whole level is hidden (modality / level dropout)

Unit cells at level u are the cells of the latent level u when u is a latent level, else
the parents of the level-l cells. Level-l cells whose unit ancestor lies outside the unit
domain follow `unit_outside` (visible | hidden).

The plan is stored in the sample metadata (`params["mask_plan"]`), so that the loss can
use the masks of every level, and the strategy name for per-strategy logging.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.datasets.domain import Domain
from weathergen.datasets.domain_pyramid import DomainPyramid, cells_parent

logger = logging.getLogger(__name__)

# name of the masking strategy (model_input / target_input) that uses the mask plan
MASK_PLAN_STRATEGY = "mask_plan"

SSL_VIEW_ROLES = ("both", "student", "teacher")

_DEFAULT_BLOCKS = {
    # V-JEPA style multi-block masking: families of blocks, each block a disk of `rings`
    # neighbour rings (0 = one unit cell) around a random centre. Blocks are added
    # (family drawn with `prob`) until the hidden fraction is reached.
    "families": {
        "short": {"prob": 0.7, "rings": [0, 1]},
        "long": {"prob": 0.3, "rings": [2, 3]},
    },
}


@dataclasses.dataclass
class MaskPlan:
    """Masks of one sample, shared by all streams."""

    strategy: str
    strategy_idx: int
    # keep masks per latent level, True = cell visible to the student
    student: dict[int, torch.Tensor]
    # view masks per latent level, True = cell visible to the teacher
    teacher: dict[int, torch.Tensor]

    def as_params(self) -> dict:
        return {
            "strategy": self.strategy,
            "strategy_idx": self.strategy_idx,
            "student": self.student,
            "teacher": self.teacher,
        }


def _to_dict(cfg) -> dict:
    """OmegaConf node / dict / None -> plain dict (recursively)."""
    if cfg is None:
        return {}
    try:
        import omegaconf

        if isinstance(cfg, omegaconf.DictConfig | omegaconf.ListConfig):
            return omegaconf.OmegaConf.to_container(cfg, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    return dict(cfg)


def _as_range(value, name: str) -> tuple[float, float]:
    if isinstance(value, int | float):
        return float(value), float(value)
    lo, hi = (float(v) for v in value)
    assert 0.0 <= lo <= hi <= 1.0, f"mask_plan: {name} must be in [0, 1], got {value}"
    return lo, hi


def _as_int_range(value, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    lo, hi = (int(v) for v in value)
    assert 0 <= lo <= hi, f"mask_plan: {name} must be non-negative, got {value}"
    return lo, hi


def get_stream_roles(streams) -> dict[str, str]:
    """SSL view role of every stream (stream config key `ssl_view`, default both)."""
    roles = {}
    for name, si in streams.items():
        role = si.get("ssl_view", "both")
        assert role in SSL_VIEW_ROLES, (
            f"stream {name!r}: ssl_view={role!r}, expected one of {SSL_VIEW_ROLES}"
        )
        roles[name] = role
    return roles


def uses_mask_plan(mode_cfg) -> bool:
    """True if a model_input or target_input strategy of the stage config uses the plan."""
    for section in ("model_input", "target_input"):
        for scfg in (mode_cfg.get(section, None) or {}).values():
            if scfg.get("masking_strategy", None) == MASK_PLAN_STRATEGY:
                return True
    return False


class MaskPlanner:
    """
    Draws one MaskPlan per sample and source sample index (see module docstring).

    Picklable (numpy / python state only), so it can live in dataloader workers.
    """

    def __init__(self, domain_pyramid: DomainPyramid, plan_cfg, stream_roles: dict[str, str]):
        self.pyramid = domain_pyramid
        self.levels = sorted(domain_pyramid.levels)
        self.stream_roles = dict(stream_roles)
        self.rng: np.random.Generator | None = None
        self._cache: dict[int, MaskPlan] = {}
        # unit domains and BFS neighbour tables, cached
        self._unit_domains: dict[tuple[int, int], Domain] = {}
        self._unit_maps: dict[tuple[int, int], NDArray[np.int64]] = {}
        self._nbours: dict[tuple, NDArray[np.int64]] = {}

        cfg = _to_dict(plan_cfg)
        assert len(cfg) > 0, (
            "masking_strategy 'mask_plan' requires a `mask_plan` block in the stage config "
            "(training_config / validation_config)."
        )
        self.unit_outside = cfg.get("unit_outside", "visible")
        assert self.unit_outside in ("visible", "hidden")
        self.default_blocks = {**_DEFAULT_BLOCKS, **(cfg.get("blocks", None) or {})}

        strategies = cfg.get("strategies", None)
        assert strategies, "mask_plan.strategies must define at least one strategy"
        self.names: list[str] = []
        probs = []
        self.specs: list[dict[int, object]] = []
        for name, scfg in strategies.items():
            scfg = scfg or {}
            if not scfg.get("enabled", True):
                continue
            self.names.append(str(name))
            probs.append(float(scfg.get("prob", 1.0)))
            self.specs.append(self._parse_levels(str(name), scfg.get("levels", None) or {}))
        assert len(self.names) > 0, "mask_plan: all strategies are disabled"
        probs = np.asarray(probs, dtype=np.float64)
        assert (probs >= 0).all() and probs.sum() > 0, "mask_plan: invalid strategy probs"
        self.probs = probs / probs.sum()

        if logger.isEnabledFor(logging.INFO):
            desc = ", ".join(
                f"{n} (p={p:.2f})" for n, p in zip(self.names, self.probs, strict=True)
            )
            logger.info(f"mask_plan strategies: {desc}; stream roles: {self.stream_roles}")

    # ------------------------------------------------------------------ config ----

    def _parse_levels(self, name: str, levels_cfg: dict) -> dict[int, object]:
        specs: dict[int, object] = {}
        given = {int(k): v for k, v in levels_cfg.items()}
        for lvl in given:
            assert lvl in self.levels, (
                f"mask_plan strategy {name!r}: level {lvl} is not a latent level {self.levels}"
            )
        for lvl in self.levels:
            spec = given.get(lvl, "visible")
            if isinstance(spec, str):
                if spec.startswith("follow:"):
                    src = int(spec.split(":", 1)[1])
                    assert src in self.levels and src < lvl, (
                        f"mask_plan strategy {name!r}: level {lvl} can only follow a coarser "
                        f"latent level, got {spec!r}"
                    )
                    specs[lvl] = ("follow", src)
                else:
                    assert spec in ("visible", "drop"), (
                        f"mask_plan strategy {name!r}, level {lvl}: unknown policy {spec!r}"
                    )
                    specs[lvl] = (spec, None)
            else:
                spec = dict(spec)
                pattern = spec.get("pattern", "blocks")
                assert pattern in ("random", "blocks", "crop", "none"), (
                    f"mask_plan strategy {name!r}, level {lvl}: unknown pattern {pattern!r}"
                )
                unit = int(spec.get("unit_level", lvl))
                assert unit <= lvl, (
                    f"mask_plan strategy {name!r}, level {lvl}: unit_level {unit} must be <= {lvl}"
                )
                hidden = _as_range(spec.get("hidden", 0.5), "hidden")
                blocks = {**self.default_blocks, **(spec.get("blocks") or {})}
                specs[lvl] = (
                    "own",
                    {"pattern": pattern, "unit": unit, "hidden": hidden, "blocks": blocks},
                )
        return specs

    # ---------------------------------------------------------------- per item ----

    def start_item(self, rng: np.random.Generator) -> None:
        """Start a new sample: forget the plans of the previous one."""
        self.rng = rng
        self._cache = {}

    def get_plan(self, i_sample: int) -> MaskPlan:
        """Plan of source sample i_sample of the current item (drawn on first use)."""
        if i_sample not in self._cache:
            self._cache[i_sample] = self.draw()
        return self._cache[i_sample]

    def student_mask(self, i_sample: int, level: int, stream_name: str):
        """(keep mask, params) of a stream's student (source) sample."""
        plan = self.get_plan(i_sample)
        if self.stream_roles.get(stream_name, "both") == "teacher":
            mask = torch.zeros_like(plan.student[level])
        else:
            mask = plan.student[level]
        return mask, {"mask_plan": plan.as_params()}

    def teacher_mask(self, i_target: int, level: int, stream_name: str):
        """(view mask, params) of a stream's teacher (target) sample."""
        # the teacher sees the full domain (V-JEPA: the target encoder sees the whole clip);
        # it does not depend on the student strategy
        n = len(self.pyramid.domain(level))
        visible = self.stream_roles.get(stream_name, "both") != "student"
        mask = torch.full((n,), visible, dtype=torch.bool)
        return mask, {"mask_plan_teacher": True}

    # ------------------------------------------------------------------- draw ----

    def draw(self) -> MaskPlan:
        assert self.rng is not None, "MaskPlanner.start_item() has not been called"
        idx = int(self.rng.choice(len(self.names), p=self.probs))
        specs = self.specs[idx]

        keep: dict[int, NDArray[np.bool_]] = {}
        for lvl in self.levels:  # coarse -> fine: `follow` sources are drawn first
            n = len(self.pyramid.domain(lvl))
            kind, arg = specs[lvl]
            if kind == "visible":
                keep[lvl] = np.ones(n, dtype=bool)
            elif kind == "drop":
                keep[lvl] = np.zeros(n, dtype=bool)
            elif kind == "follow":
                anc = self.pyramid.ancestor_of(lvl, arg)
                k_src = keep[arg]
                keep[lvl] = np.where(anc >= 0, k_src[np.maximum(anc, 0)], True)
            else:
                keep[lvl] = self._draw_own(lvl, arg)

        student = {lvl: torch.from_numpy(k.copy()) for lvl, k in keep.items()}
        teacher = {
            lvl: torch.ones(len(self.pyramid.domain(lvl)), dtype=torch.bool) for lvl in self.levels
        }
        return MaskPlan(self.names[idx], idx, student, teacher)

    def _draw_own(self, lvl: int, spec: dict) -> NDArray[np.bool_]:
        unit = spec["unit"]
        dom_u = self._unit_domain(lvl, unit)
        n_u = len(dom_u)
        lo, hi = spec["hidden"]
        frac = lo if lo == hi else float(self.rng.uniform(lo, hi))
        num_hidden = int(np.round(frac * n_u))

        pattern = spec["pattern"]
        hidden_u = np.zeros(n_u, dtype=bool)
        if pattern == "none" or num_hidden == 0:
            pass
        elif num_hidden >= n_u:
            hidden_u[:] = True
        elif pattern == "random":
            hidden_u[self.rng.choice(n_u, num_hidden, replace=False)] = True
        elif pattern == "blocks":
            hidden_u = self._blocks(dom_u, num_hidden, spec["blocks"])
        elif pattern == "crop":
            # keep one contiguous region of n_u - num_hidden unit cells, hide the rest
            centre = int(self.rng.integers(n_u))
            dist = self._bfs_distance(dom_u, centre)
            # random tie breaking among cells at the same ring distance
            order = np.lexsort((self.rng.random(n_u), dist))
            hidden_u[:] = True
            hidden_u[order[: n_u - num_hidden]] = False

        # unit cells -> level-l cells
        if unit == lvl:
            return ~hidden_u
        umap = self._unit_map(lvl, unit)
        outside_hidden = self.unit_outside == "hidden"
        hidden_l = np.where(umap >= 0, hidden_u[np.maximum(umap, 0)], outside_hidden)
        return ~hidden_l

    def _blocks(self, dom_u: Domain, num_hidden: int, blocks_cfg: dict) -> NDArray[np.bool_]:
        n_u = len(dom_u)
        families = list((blocks_cfg.get("families") or {}).items())
        assert len(families) > 0, "mask_plan: pattern blocks needs at least one block family"
        fprobs = np.asarray([float(f.get("prob", 1.0)) for _, f in families], dtype=np.float64)
        fprobs = fprobs / fprobs.sum()
        rings = [_as_int_range(f.get("rings", [0, 1]), "rings") for _, f in families]

        hidden = np.zeros(n_u, dtype=bool)
        # bounded number of attempts (a block can fall onto already hidden cells)
        for _ in range(10 * n_u + 10):
            if hidden.sum() >= num_hidden:
                break
            fam = int(self.rng.choice(len(families), p=fprobs))
            r = int(self.rng.integers(rings[fam][0], rings[fam][1] + 1))
            # centres preferably on visible cells, so that every block adds hidden cells
            visible = np.flatnonzero(~hidden)
            centre = int(visible[self.rng.integers(len(visible))])
            dist = self._bfs_distance(dom_u, centre, max_rings=r)
            new = np.flatnonzero((dist <= r) & ~hidden)
            need = num_hidden - int(hidden.sum())
            if len(new) > need:
                # last block: keep its cells closest to the centre (random among equal
                # ring distance) so that the hidden fraction is met exactly
                order = np.lexsort((self.rng.random(len(new)), dist[new]))
                new = new[order[:need]]
            hidden[new] = True
        return hidden

    # --------------------------------------------------------------- geometry ----

    def _unit_domain(self, lvl: int, unit: int) -> Domain:
        """Domain of the unit cells used to mask latent level lvl."""
        if unit in self.levels:
            return self.pyramid.domain(unit)
        key = (lvl, unit)
        if key not in self._unit_domains:
            parents = np.unique(cells_parent(self.pyramid.domain(lvl).active_cells, lvl, unit))
            self._unit_domains[key] = Domain(unit, parents)
        return self._unit_domains[key]

    def _unit_map(self, lvl: int, unit: int) -> NDArray[np.int64]:
        """(num_cells(lvl),) compact index of each level-lvl cell's unit cell, -1 outside."""
        key = (lvl, unit)
        if key not in self._unit_maps:
            dom_u = self._unit_domain(lvl, unit)
            parents = cells_parent(self.pyramid.domain(lvl).active_cells, lvl, unit)
            self._unit_maps[key] = dom_u.to_compact_safe(parents)
        return self._unit_maps[key]

    def _neighbours(self, dom: Domain) -> NDArray[np.int64]:
        key = (dom.healpix_level, len(dom), int(dom.active_cells[0]), int(dom.active_cells[-1]))
        if key not in self._nbours:
            self._nbours[key] = dom.neighbours_compact()
        return self._nbours[key]

    def _bfs_distance(self, dom: Domain, centre: int, max_rings: int | None = None):
        """Ring distance (neighbour steps inside the domain) of every cell to `centre`."""
        nb = self._neighbours(dom)
        n = len(dom)
        dist = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        dist[centre] = 0
        frontier = np.array([centre], dtype=np.int64)
        d = 0
        while len(frontier) > 0 and (max_rings is None or d < max_rings):
            d += 1
            cand = np.unique(nb[frontier].ravel())
            cand = cand[dist[cand] > d]
            dist[cand] = d
            frontier = cand
        return dist
