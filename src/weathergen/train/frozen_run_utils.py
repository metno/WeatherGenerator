# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Helpers to run (a part of) a model from another, frozen run on the current run's batch:
the frozen-teacher encoder (FrozenTeacher) and the frozen autoencoder decoder used to
decode predicted latents (latent_decoder).
"""

import logging

import numpy as np
from omegaconf import OmegaConf

from weathergen.common.config import Config
from weathergen.datasets.domain_pyramid import build_domain_pyramid
from weathergen.datasets.stream_levels import assign_stream_levels

logger = logging.getLogger(__name__)


class DatasetStreamSubset:
    """Sizes of a subset of the dataset streams (to build a frozen teacher on that subset)."""

    def __init__(self, dataset, idxs: list[int]):
        self._dataset, self._idxs = dataset, idxs

    def get_sources_size(self):
        return [self._dataset.get_sources_size()[i] for i in self._idxs]

    def get_targets_num_channels(self):
        return [self._dataset.get_targets_num_channels()[i] for i in self._idxs]

    def get_targets_coords_size(self):
        return [self._dataset.get_targets_coords_size()[i] for i in self._idxs]


def frozen_run_streams(cf: Config, teacher_config: Config) -> tuple[Config, list[int], list[int]]:
    """
    Stream set of a model from a frozen run (frozen-teacher encoder, latent decoder) that runs
    on the current (student) run's batch.

    Returns (frozen run config with `streams` replaced, indices of these streams in the
    student's stream list, latent levels of the frozen run).

    Streams = the student's streams encoded at one of the frozen run's latent levels, in the
    student's order. A stream that also exists in the frozen run keeps that run's stream
    config (architecture of the checkpoint's embedder / decoder); the others get the
    student's config and stay untrained (for a frozen teacher they must carry no teacher
    tokens: set `ssl_view: student` on them). The frozen run's latent levels must have the
    same domain (cells and order) as the student's levels.
    """
    student_pyr = build_domain_pyramid(cf)
    teacher_pyr = build_domain_pyramid(teacher_config)
    teacher_levels = list(teacher_pyr.levels)
    for lvl in teacher_levels:
        assert lvl in student_pyr.levels, (
            f"frozen run latent level {lvl} is not a latent level of the student "
            f"{student_pyr.levels}"
        )
        a = teacher_pyr.domain(lvl).active_cells
        b = student_pyr.domain(lvl).active_cells
        assert len(a) == len(b) and np.array_equal(a, b), (
            f"frozen run and student have different domains at latent level {lvl} "
            f"({len(a)} vs {len(b)} cells): use the same latent_levels / domain entry "
            "(points_file, pad_rings, ...) as the frozen run"
        )

    student_names = list(cf.streams.keys())
    student_level, _ = assign_stream_levels(cf, ladder=student_pyr.levels)
    names = [n for n in student_names if student_level[n] in teacher_levels]
    teacher_run_streams = teacher_config.get("streams", None) or {}
    missing = [n for n in teacher_run_streams.keys() if n not in names]
    assert not missing, (
        f"streams {missing} of the teacher run are not streams of the student at the teacher's "
        f"latent levels {teacher_levels}: the student run must load them (with the same name, "
        "channels and normalisation; set `ssl_view: teacher` to hide them from the student)"
    )
    streams = {}
    for n in names:
        si = teacher_run_streams[n] if n in teacher_run_streams else cf.streams[n]
        si = OmegaConf.to_container(si, resolve=True) if not isinstance(si, dict) else dict(si)
        if len(teacher_levels) == 1:
            si.pop("latent_level", None)
        streams[n] = si
    teacher_config = teacher_config.copy()
    teacher_config.streams = OmegaConf.create(streams)
    idxs = [student_names.index(n) for n in names]
    logger.info(
        f"Frozen teacher: latent levels {teacher_levels}, streams {names} "
        f"(from teacher run: {[n for n in names if n in teacher_run_streams]})"
    )
    return teacher_config, idxs, teacher_levels
